"""The optimizer every stage and the reference build (RUNTIME-I20).

`optimizer` "adamw" (the default) is torch.optim.AdamW(lr, **ADAMW). "adamw-host"
is the same update with the moments m and v in host memory: the device keeps the
fp32 parameters and the step's folded gradient only, 8 of AdamW's 16 bytes per
trainable element, so a stage holds about twice the parameters. The moments
stream through the parameter's device in chunks of CHUNK elements (pinned host
buffers where the host allows), so the step adds a few chunks to the device's
peak, never a parameter's moments.

The update is written in single-rounding IEEE operations only: mul, add, sub,
tensor-by-tensor div and sqrt, each its own kernel. No lerp, addcmul or addcdiv
(a kernel may fuse their multiply into the add), and no division by a scalar
(CUDA turns it into a multiplication by the rounded reciprocal; the CPU divides):
the scalar factors are fp32 constants multiplied in. The sqrt is taken in fp64
and rounded once to fp32, because torch's fp32 sqrt is not correctly rounded on
every CPU build. Every element is therefore
rounded once per operation, and the result depends on neither the device, the
vector width, the thread count nor the chunking. It is a recipe pin all the same
(JobSpec.pins): its bits differ from torch's AdamW in the last place, so a run and
its reference must name the same optimizer, which `build` gives both.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable

import torch

OPTIMIZERS = ("adamw", "adamw-host")
CHUNK = 1 << 24  # elements per streamed chunk: 64 MiB per fp32 buffer
# Device bytes per trainable element the optimizer and the fold keep: the fp32
# parameter and the step's gradient, plus m and v when they live on the device.
STATE_BYTES = {"adamw": 16, "adamw-host": 8}


def state_bytes_per_param(optimizer: str = "adamw") -> int:
    if optimizer not in STATE_BYTES:
        raise ValueError(f"optimizer {optimizer!r} is not one of {OPTIMIZERS}")
    return STATE_BYTES[optimizer]


def build(optimizer: str, params: Iterable[torch.Tensor], lr: float):
    """The optimizer a stage (over its trainable parameters, in order) and the
    reference (over all of them) construct."""
    from wantrain.runtime.spec import ADAMW

    if optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, **ADAMW)
    if optimizer == "adamw-host":
        return HostAdamW(params, lr=lr, **ADAMW)
    raise ValueError(f"optimizer {optimizer!r} is not one of {OPTIMIZERS}")


def hparams(lr: float, optimizer: str = "adamw") -> dict:
    """The param-group hyperparameters `build` checkpoints, read off a probe
    over one detached scalar that never steps: torch's own defaults, so an
    owner-less ZeRO-1 slot records exactly what its peers record."""
    from wantrain.runtime.spec import ADAMW

    probe = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=lr, **ADAMW)
    group = {k: v for k, v in probe.state_dict()["param_groups"][0].items() if k != "params"}
    if optimizer == "adamw-host":
        group["moments"] = "host"
    elif optimizer != "adamw":
        raise ValueError(f"optimizer {optimizer!r} is not one of {OPTIMIZERS}")
    return group


def _f32(x: float) -> float:
    """x rounded to fp32: the value every kernel would cast the scalar to."""
    return float(torch.tensor(x, dtype=torch.float32))


class HostAdamW:
    """AdamW (decoupled weight decay, no amsgrad) with host-resident moments.
    The torch.optim surface the runtime uses: step, zero_grad, state_dict,
    load_state_dict, param_groups. state_dict has torch's AdamW layout (state[i]
    = step, exp_avg, exp_avg_sq; one param group naming 0..n-1) plus
    "moments": "host" in the group, so a checkpoint names its recipe."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        chunk: int = CHUNK,
        pipeline: bool | None = None,
        slots: int = 3,
    ):
        self.params = list(params)
        for p in self.params:
            # A frozen tensor (a LoRA base) never has a gradient and is never stepped.
            if p.requires_grad and p.dtype != torch.float32:
                raise ValueError(f"adamw-host trains fp32 parameters, got {p.dtype}")
        self.lr, self.betas, self.eps, self.weight_decay = lr, tuple(betas), eps, weight_decay
        self.chunk = int(chunk)
        group = hparams(lr, "adamw-host")
        group.update(lr=lr, betas=self.betas, eps=eps, weight_decay=weight_decay)
        self._group = group
        self.param_groups = [{**group, "params": self.params}]
        self.state: dict = {}  # index -> {"step", "exp_avg", "exp_avg_sq"}
        # Streamed moments go through `slots` staging buffers on the device: a chunk's copy in, its update and its
        # copy out run on three CUDA streams, so the next chunk's copy in and the last one's copy out overlap this
        # one's update (PCIe is full duplex). None pipelines on CUDA only; True forces the staging on any device
        # (tests run it on the CPU).
        if pipeline is None and os.environ.get("WANTRAIN_OPT_STAGED") == "off":
            pipeline = False  # diagnosis: the serial copies of RUNTIME-I20
        self.pipeline = pipeline
        self.slots = int(slots)
        self._staging: dict = {}  # device -> [(m, v)] buffers of `chunk` elements
        self._copy_streams: dict = {}  # device -> (h2d, d2h)
        self.resident: set = (
            set()
        )  # parameter indices whose moments live on the parameter's device (promote)

    # ------------------------------------------------------------ host memory

    def _host_zeros(self, p: torch.Tensor) -> torch.Tensor:
        """fp32 zeros shaped like p in host memory; pinned when p lives on CUDA
        and the host grants it (pageable otherwise: slower copies, same bits)."""
        if p.device.type == "cuda":
            try:
                return torch.zeros(p.shape, dtype=torch.float32, pin_memory=True)
            except RuntimeError:
                pass
        return torch.zeros(p.shape, dtype=torch.float32)

    def _slot(self, i: int, p: torch.Tensor) -> dict:
        st = self.state.get(i)
        if st is None:
            st = self.state[i] = {
                "step": torch.tensor(0.0, dtype=torch.float32),
                "exp_avg": self._host_zeros(p),
                "exp_avg_sq": self._host_zeros(p),
            }
        return st

    # ------------------------------------------------------------------ step

    @torch.no_grad()
    def step(self, on_param=None) -> None:
        """One AdamW step over every parameter with a gradient, in order. `on_param(i)` (optional) runs right after
        parameter i's update is issued, so a caller can release what reads it (RUNTIME-I23)."""
        b1, b2 = self.betas
        pipes: dict = {}  # device -> _Staging, for this step
        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue  # as torch: a parameter without a gradient is not stepped
            st = self._slot(i, p)
            st["step"] += 1
            t = float(st["step"])
            c = {
                "decay": _f32(1.0 - self.lr * self.weight_decay),
                "b1": _f32(b1),
                "one_b1": _f32(1.0 - b1),
                "b2": _f32(b2),
                "one_b2": _f32(1.0 - b2),
                "inv_bc2_sqrt": _f32(1.0 / math.sqrt(1.0 - b2**t)),
                "eps": _f32(self.eps),
                "step_size": _f32(self.lr / (1.0 - b1**t)),
            }
            self._update(p, g, st["exp_avg"], st["exp_avg_sq"], c, pipes)
            if on_param is not None:
                on_param(i)
        if any(p.device.type == "cuda" for p in self.params):
            # The last chunks' moments travel back asynchronously; the host
            # copies are the state, so they land before the step returns.
            torch.cuda.synchronize()

    def _update(self, p, g, m_host, v_host, c, pipes) -> None:
        if not p.is_contiguous():
            raise ValueError("adamw-host steps contiguous parameters only")
        pf = p.view(-1)
        gf = g.reshape(-1)
        mf, vf = m_host.view(-1), v_host.view(-1)
        dev = p.device
        staged = mf.device != dev and (
            self.pipeline if self.pipeline is not None else dev.type == "cuda"
        )
        if self.pipeline and mf.device == dev and dev.type == "cpu":
            staged = True  # forced: the CPU exercises the staging's indexing
        for s in range(0, pf.numel(), self.chunk):
            e = min(s + self.chunk, pf.numel())
            pc, gc = pf[s:e], gf[s:e]
            if staged:
                pipe = pipes.get(dev)
                if pipe is None:
                    pipe = pipes[dev] = _Staging(self, dev)
                m, v, slot = pipe.load(mf[s:e], vf[s:e])
                _adamw_math(pc, gc, m, v, c, self.weight_decay)
                pipe.store(m, v, mf[s:e], vf[s:e], slot)
                continue
            m = mf[s:e].to(dev, non_blocking=True)
            v = vf[s:e].to(dev, non_blocking=True)
            _adamw_math(pc, gc, m, v, c, self.weight_decay)
            if m.data_ptr() != mf[s:e].data_ptr():
                mf[s:e].copy_(m, non_blocking=True)
                vf[s:e].copy_(v, non_blocking=True)

    # ------------------------------------------------------------ residency

    def promote(self, budget_bytes: int) -> int:
        """Move the moments of parameters, in order, onto their parameter's device while they fit `budget_bytes`
        (8 bytes per element: m and v in fp32); a parameter that does not fit is skipped. The update reads and
        writes the same values wherever they live, so the bits do not change; a resident parameter's step skips
        the PCIe round trip. -> bytes moved."""
        moved = 0
        for i, p in enumerate(self.params):
            if i in self.resident or not p.requires_grad or p.device.type == "cpu":
                continue
            need = 8 * p.numel()
            if moved + need > budget_bytes:
                continue
            st = self._slot(i, p)
            st["exp_avg"] = st["exp_avg"].to(p.device)
            st["exp_avg_sq"] = st["exp_avg_sq"].to(p.device)
            self.resident.add(i)
            moved += need
        if moved and any(p.device.type == "cuda" for p in self.params):
            torch.cuda.synchronize()
        return moved

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    # ------------------------------------------------------------ checkpoint

    def state_dict(self) -> dict:
        return {
            "state": {
                i: {
                    "step": st["step"].clone(),
                    # a resident moment is checkpointed as the host copy it would otherwise be
                    "exp_avg": st["exp_avg"].cpu(),
                    "exp_avg_sq": st["exp_avg_sq"].cpu(),
                }
                for i, st in sorted(self.state.items())
            },
            "param_groups": [{**self._group, "params": list(range(len(self.params)))}],
        }

    def load_state_dict(self, sd: dict) -> None:
        groups = sd.get("param_groups") or []
        if len(groups) != 1:
            raise ValueError("adamw-host state holds one param group")
        theirs = {k: v for k, v in groups[0].items() if k != "params"}
        mine = json_norm(self._group)
        if json_norm(theirs) != mine:
            raise ValueError(
                f"optimizer state of another recipe: {json_norm(theirs)} is not {mine}"
            )
        if list(groups[0]["params"]) != list(range(len(self.params))):
            raise ValueError("optimizer state names other parameters")
        self.state = {}
        for i, st in sd.get("state", {}).items():
            i = int(i)
            p = self.params[i]
            for k in ("exp_avg", "exp_avg_sq"):
                if tuple(st[k].shape) != tuple(p.shape):
                    raise ValueError(f"optimizer state {i}.{k} has shape {tuple(st[k].shape)}")
            slot = self._slot(i, p)
            slot["step"] = torch.as_tensor(st["step"], dtype=torch.float32).clone().cpu()
            slot["exp_avg"].copy_(st["exp_avg"])
            slot["exp_avg_sq"].copy_(st["exp_avg_sq"])


def _adamw_math(pc, gc, m, v, c, weight_decay) -> None:
    """One chunk's update, in place on pc (parameter), m and v (moments, on pc's device)."""
    if weight_decay:
        pc.mul_(c["decay"])
    m.mul_(c["b1"])
    m.add_(gc.mul(c["one_b1"]))
    v.mul_(c["b2"])
    sq = gc.mul(gc)
    sq.mul_(c["one_b2"])
    v.add_(sq)
    del sq
    # sqrt in fp64, rounded once to fp32: the correctly rounded fp32 sqrt on every device. torch's fp32
    # sqrt is correctly rounded on CUDA but not on every CPU build (an AVX-512 host: 1 ulp off on 17% of
    # elements); fp64 holds 2*24+2 bits, so its sqrt rounds to fp32 exactly once.
    d = v.double().sqrt_().float()
    d.mul_(c["inv_bc2_sqrt"])
    d.add_(c["eps"])
    u = m.div(d)
    del d
    u.mul_(c["step_size"])
    pc.sub_(u)
    del u


class _Staging:
    """One step's moment staging on one device: chunk k goes through slot k % slots. On CUDA its copy in runs on
    the h2d stream, its update on the step's (current) stream, its copy out on the d2h stream, ordered by events:
    a slot is refilled only after its previous chunk has been copied out, and the update waits for its copy in.
    The caller's stream therefore never waits for a copy out, which a layer's release does not need either."""

    def __init__(self, opt: HostAdamW, dev: torch.device):
        self.cuda = dev.type == "cuda"
        n = max(1, opt.slots)
        bufs = opt._staging.get(dev)
        if bufs is None:
            bufs = opt._staging[dev] = [
                (
                    torch.empty(opt.chunk, dtype=torch.float32, device=dev),
                    torch.empty(opt.chunk, dtype=torch.float32, device=dev),
                )
                for _ in range(n)
            ]
        self.bufs = bufs
        self.k = 0
        if self.cuda:
            if dev not in opt._copy_streams:
                opt._copy_streams[dev] = (torch.cuda.Stream(dev), torch.cuda.Stream(dev))
            self.h2d, self.d2h = opt._copy_streams[dev]
            self.compute = torch.cuda.current_stream(dev)
            self.ev_in = [torch.cuda.Event() for _ in bufs]
            self.ev_done = [torch.cuda.Event() for _ in bufs]
            self.ev_back = [torch.cuda.Event() for _ in bufs]
            self.used = [False] * len(bufs)
            # the slots may still be read by the previous step's copies out: step() synchronized, so they are not

    def load(self, mh: torch.Tensor, vh: torch.Tensor):
        slot = self.k % len(self.bufs)
        self.k += 1
        n = mh.numel()
        m, v = self.bufs[slot][0][:n], self.bufs[slot][1][:n]
        if not self.cuda:
            m.copy_(mh)
            v.copy_(vh)
            return m, v, slot
        if self.used[slot]:
            self.h2d.wait_event(self.ev_back[slot])
        with torch.cuda.stream(self.h2d):
            m.copy_(mh, non_blocking=True)
            v.copy_(vh, non_blocking=True)
            self.ev_in[slot].record(self.h2d)
        self.compute.wait_event(self.ev_in[slot])
        return m, v, slot

    def store(
        self, m: torch.Tensor, v: torch.Tensor, mh: torch.Tensor, vh: torch.Tensor, slot: int
    ) -> None:
        if not self.cuda:
            mh.copy_(m)
            vh.copy_(v)
            return
        self.ev_done[slot].record(self.compute)
        self.d2h.wait_event(self.ev_done[slot])
        with torch.cuda.stream(self.d2h):
            mh.copy_(m, non_blocking=True)
            vh.copy_(v, non_blocking=True)
            self.ev_back[slot].record(self.d2h)
        self.used[slot] = True


def json_norm(group: dict) -> dict:
    """Tuples as lists, so a group read back from JSON or torch.load compares
    equal to a fresh one."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(group.items())}

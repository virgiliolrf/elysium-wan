"""Run one pipeline stage and make its progress and its exit visible.

This is the process a worker agent supervises: one (rank, peer) slot of one
spec. The engine writes losses to a stream file and prints nothing; this
wrapper tails that stream and re-emits one `WT_METRIC` line per optimizer
step, then always prints `WT_DONE <rc>` last. The done marker is how a
supervisor tells "the engine raised" (a deterministic error a replacement node
would hit again) from "the process was killed from outside" (a dead box).

    python -m wantrain.worker.stage_main --spec spec.json --rank 0 [--peer 0]
        [--launcher local|coordinator]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time


def _metric(d: dict) -> str:
    return "WT_METRIC " + json.dumps(
        {"step": d.get("step"), "loss": d.get("loss"), "t": d.get("t")}
    )


def stream_losses(path: str, stop: threading.Event, poll_s: float = 0.5) -> None:
    """Tail the engine's loss stream; one line per step, not per microbatch.
    Binary reads and byte offsets: tell() is disabled on a text handle being
    iterated, and a partial last line is left for the next pass."""
    pos, last_step = 0, None
    while True:
        done = stop.is_set()  # sample first, so the final pass sees the tail
        try:
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    fh.seek(pos)
                    for raw in fh:
                        if not raw.endswith(b"\n"):
                            break
                        pos += len(raw)
                        try:
                            d = json.loads(raw.decode("utf-8", "replace"))
                        except ValueError:
                            continue
                        if d.get("step") != last_step:
                            last_step = d.get("step")
                            print(_metric(d), flush=True)
        except OSError:
            pass
        if done:
            return
        time.sleep(poll_s)


def reset_loss_stream(path: str, resume: bool) -> None:
    """A fresh (non-resumed) launch starts from an empty loss stream. Every slot
    process of the spec runs this on the same shared path, so another slot may
    remove the file between any check and the unlink: absence is the goal, not
    an error."""
    if resume:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def run_stage(spec_path: str, rank: int, peer: int) -> int:
    from wantrain.runtime.spec import JobSpec
    from wantrain.runtime.stage import StageRuntime

    spec = JobSpec.load(spec_path)
    stream = os.path.join(spec.out_dir, f"{spec.tag}_loss_stream.jsonl")
    reset_loss_stream(stream, spec.resume)
    stop = threading.Event()
    t = threading.Thread(target=stream_losses, args=(stream, stop), daemon=True)
    print("WT_PHASE train", flush=True)
    print(
        f"WT_STAGE rank={rank} peer={peer} device={spec.device_for(rank, peer)} "
        f"partition={spec.partition} compute={spec.model.compute_dtype} "
        f"boundary={spec.model.boundary_dtype} codec={spec.wire_codec} "
        f"resume={spec.resume}",
        flush=True,
    )
    t.start()
    rc = 0
    try:
        StageRuntime(spec, rank, peer).run()
    except BaseException as e:  # noqa: BLE001 -- reported through the exit code
        import traceback

        # Traceback before the marker: a supervisor surfaces the marker, and an
        # operator who sees a nonzero exit needs the reason next to it.
        traceback.print_exc()
        sys.stdout.flush()
        rc = 130 if isinstance(e, KeyboardInterrupt) else 1
    finally:
        stop.set()
        t.join(timeout=5)
        print(f"WT_DONE {rc}", flush=True)
    return rc


LAUNCHERS = ("local", "coordinator")


def refuse_unchecked_pieces(spec_path: str, launcher: str | None) -> str | None:
    """BA-58: a one-slot launch cannot verify its peers' device class, so a
    spec whose JSON has shard "pieces" runs only under a launcher that checked
    it (run_local's class check, or the coordinator's admission). The field is
    read before JobSpec.load, as `wantrain stage` does (I-9). -> the reason, or
    None when the launch may go on."""
    if launcher in LAUNCHERS:
        return None
    with open(spec_path) as f:
        shard = json.load(f).get("shard", "zero1")
    if shard == "pieces":
        return (
            "shard 'pieces' refused: a one-slot launch cannot verify its peers' device "
            "class; launch through the coordinator or 'wantrain local' (--launcher)"
        )
    return None


def refuse_unchecked_hf(spec_path: str, launcher: str | None) -> str | None:
    """BA-445: an arch "hf" slot starts only under a launcher that ran the gate
    and the admission (run_local or the coordinator); a one-slot launch did
    neither. model.arch is read from the JSON before JobSpec.load, as for
    pieces. -> the reason, or None when the launch may go on."""
    if launcher in LAUNCHERS:
        return None
    with open(spec_path) as f:
        model = json.load(f).get("model")
    if isinstance(model, dict) and model.get("arch") == "hf":
        return (
            "arch 'hf' refused: a one-slot launch runs no gate and no admission; launch "
            "through the coordinator or 'wantrain local', which run the gate and admission "
            "(--launcher)"
        )
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="run one pipeline stage from a spec")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--peer", type=int, default=0)
    ap.add_argument("--launcher", choices=LAUNCHERS, default=None)
    a = ap.parse_args(argv)
    reason = refuse_unchecked_pieces(a.spec, a.launcher) or refuse_unchecked_hf(a.spec, a.launcher)
    if reason is not None:
        print(f"WT_REFUSED {reason}", flush=True)
        print("WT_DONE 2", flush=True)
        return 2
    return run_stage(a.spec, a.rank, a.peer)


if __name__ == "__main__":
    sys.exit(main())

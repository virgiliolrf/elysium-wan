"""Data provisioning: a data-reading slot holds, verified, the bytes its microbatches draw.

Only the first stage (inputs) and the last stage (targets) read the corpus; the stages between them never touch it.
Microbatch mb draws from logical shard mb % data_shards (model/data.py), and a slot owns the microbatches
spec.owned(rank, peer) gives it. Its need is therefore a set of shards, and that set moves when the plan moves
ownership (a heal, a rebalance): the shard is tied to the microbatch, not to the node. Before a data-reading slot
connects, provision():

1. makes the manifest local (fetched from the source when missing) and checks it against the digest the coordinator
   planned with (spec.dataset_sha256). A node holding another corpus, tokenizer or shard layout refuses at startup,
   where it used to train on other bytes or fail at its first microbatch;
2. for each needed shard, finds its bytes (the shard file, or that range of the whole file) and verifies them against
   the manifest's per-shard sha256. When the recipe's shard count differs from the prepared one, or the manifest
   predates per-shard digests, the whole file is verified against the manifest's sha256 instead;
3. fetches missing or corrupt bytes from spec.dataset_source (a directory or an http(s) URL holding the prepared
   files: the manifest, the whole file and/or its shard files), into a temp file renamed into place once verified;
4. refuses, naming the shards and the fix, when bytes are missing or wrong and there is no source.

A verified range leaves a stamp (size, mtime, digest) in `<file>.verified.json`, so a restart does not hash the corpus
again. Integrity does not depend on the source: every fetched byte is checked against the pinned manifest.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from wantrain.model import dataset as ds

CHUNK = 1 << 20
FETCH_TIMEOUT_S = 60


class DataRefused(RuntimeError):
    """This node cannot hold the verified bytes its slot needs."""


def needed_shards(spec, rank: int, peer: int) -> list[int]:
    """The logical shards slot (rank, peer) reads: those of the microbatches it owns, on the first or last stage."""
    if not spec.dataset_path or rank not in (0, spec.stages - 1):
        return []
    n = max(1, int(spec.data_shards or 1))
    return sorted({mb % n for mb in spec.owned(rank, peer)})


def _sha_range(path: str, lo: int = 0, hi: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(lo)
        left = (os.path.getsize(path) if hi is None else hi) - lo
        while left > 0:
            b = f.read(min(CHUNK, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def manifest_digest(dataset: str) -> str | None:
    """sha256 of the manifest file: it pins tokenizer, vocabulary, dtype, token count and every digest."""
    p = ds.manifest_path(dataset)
    return _sha_range(p) if os.path.exists(p) else None


def _stamps_path(path: str) -> str:
    return f"{path}.verified.json"


def _read_stamps(path: str) -> dict:
    try:
        with open(_stamps_path(path)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _verified(path: str, lo: int, hi: int, want: str) -> bool:
    """Whether bytes [lo, hi) of path have sha256 `want`, from a fresh stamp or by hashing them."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    if st.st_size < hi:
        return False
    key = f"{lo}-{hi}"
    stamps = _read_stamps(path)
    s = stamps.get(key)
    if (
        s
        and s.get("size") == st.st_size
        and s.get("mtime_ns") == st.st_mtime_ns
        and s.get("sha256") == want
    ):
        return True
    if _sha_range(path, lo, hi) != want:
        return False
    stamps[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": want}
    tmp = f"{_stamps_path(path)}.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(stamps, f)
        os.replace(tmp, _stamps_path(path))
    except OSError:  # a read-only corpus directory: verify again next time
        pass
    return True


def _is_url(source: str) -> bool:
    return urllib.parse.urlparse(source).scheme in ("http", "https")


def _fetch(source: str, name: str, dest: str, byte_range: tuple[int, int] | None = None) -> bool:
    """Copy `name`, or bytes [lo, hi) of it, from source to dest. False when the source does not hold it."""
    tmp = f"{dest}.part.{os.getpid()}"
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    try:
        if _is_url(source):
            req = urllib.request.Request(source.rstrip("/") + "/" + urllib.parse.quote(name))
            if byte_range:
                req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1] - 1}")
            try:
                r = urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return False
                raise
            with r, open(tmp, "wb") as f:
                if byte_range and r.status != 206:
                    raise DataRefused(f"{source} ignored the byte-range request for {name}")
                shutil.copyfileobj(r, f, CHUNK)
        else:
            src = os.path.join(source, name)
            if not os.path.exists(src):
                return False
            with open(src, "rb") as s, open(tmp, "wb") as f:
                if byte_range is None:
                    shutil.copyfileobj(s, f, CHUNK)
                else:
                    s.seek(byte_range[0])
                    left = byte_range[1] - byte_range[0]
                    while left > 0:
                        b = s.read(min(CHUNK, left))
                        if not b:
                            raise DataRefused(f"{src} ends before byte {byte_range[1]}")
                        f.write(b)
                        left -= len(b)
        os.replace(tmp, dest)
        return True
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def ensure_manifest(dataset: str, pinned: str | None = None, source: str | None = None) -> dict:
    """The local manifest, fetched from source when missing or not the pinned one; refused when it cannot match."""
    mp = ds.manifest_path(dataset)
    name = os.path.basename(mp)
    have = manifest_digest(dataset)
    if source and (have is None or (pinned and have != pinned)):
        os.makedirs(os.path.dirname(os.path.abspath(mp)), exist_ok=True)
        if _fetch(source, name, mp):
            have = manifest_digest(dataset)
    if pinned and have != pinned:
        where = "is missing" if have is None else f"has sha256 {have[:12]}…"
        raise DataRefused(
            f"the dataset manifest {mp} {where}, but the job was planned with {pinned[:12]}…: this node holds "
            f"another corpus, tokenizer or shard layout. Copy the coordinator's {name} here, or set dataset_source."
        )
    if have is None and not os.path.exists(dataset):
        raise DataRefused(
            f"no dataset at {dataset!r} and no manifest at {mp!r}; copy them here or set dataset_source"
        )
    return ds.read_manifest(dataset)


def _retire_divergent_shard_files(
    dataset: str, shards: list[int], n_shards: int, total: int, isz: int
) -> None:
    """With the whole file as the checked unit, a `.shardKofN` file (stale, corrupt, or from another layout) is
    served only if its bytes equal that range of the whole file: shard_view prefers a shard file."""
    if n_shards < 2:
        return
    for k in shards:
        sp = f"{dataset}.shard{k}of{n_shards}"
        if not os.path.exists(sp):
            continue
        lo, hi = (total * k) // n_shards, (total * (k + 1)) // n_shards
        if not _verified(sp, 0, (hi - lo) * isz, _sha_range(dataset, lo * isz, hi * isz)):
            os.replace(sp, f"{sp}.bad")


@contextlib.contextmanager
def _host_lock(dataset: str):
    """One provisioner per corpus on this host at a time. Slots sharing a host (a wide first stage on a multi-GPU
    machine, or the first and last stage together) share the corpus files, and one slot's quarantine could move aside
    the shard another had just fetched and verified. A read-only corpus directory takes no lock: nothing there can be
    rewritten, so there is nothing to race."""
    path = f"{dataset}.provision.lock"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        f = open(path, "a")
    except OSError:
        yield
        return
    with f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def provision(
    dataset: str,
    shards: list[int],
    n_shards: int,
    pinned: str | None = None,
    source: str | None = None,
) -> dict:
    """Make every shard in `shards` (of the recipe's n_shards) present and verified here; fetch what is missing or
    corrupt from `source`. Returns {"verified": [...], "fetched": [...], "unverified": bool}. Serialized per corpus
    on this host (_host_lock)."""
    with _host_lock(dataset):
        return _provision(dataset, shards, n_shards, pinned, source)


def _provision(
    dataset: str, shards: list[int], n_shards: int, pinned: str | None, source: str | None
) -> dict:
    meta = ensure_manifest(dataset, pinned, source)
    base = os.path.basename(dataset)
    total, isz = int(meta["total_tokens"]), np.dtype(ds.DTYPES[meta["dtype"]]).itemsize
    per = (
        meta.get("shard_sha256")
        if int(meta.get("n_shards", 1)) == n_shards and n_shards > 1
        else None
    )
    verified: list[int] = []
    fetched: list[int] = []
    report: dict = {"verified": verified, "fetched": fetched, "unverified": False}

    if per is None:
        # One digest covers the corpus: the whole file is the unit.
        whole = meta.get("sha256")
        if whole is None:
            if not os.path.exists(dataset):
                raise DataRefused(f"no dataset at {dataset!r}; copy it here or set dataset_source")
            _retire_divergent_shard_files(dataset, shards, n_shards, total, isz)
            report["unverified"] = True  # a legacy manifest records no digest to check
            report["verified"] = list(shards)
            return report
        size = total * isz
        if not _verified(dataset, 0, size, whole):
            if not (
                source and _fetch(source, base, dataset) and _verified(dataset, 0, size, whole)
            ):
                raise DataRefused(
                    f"{dataset!r} is missing or differs from its manifest (sha256 {whole[:12]}…), and "
                    f"{'the source does not hold a matching copy' if source else 'no dataset_source is set'}: "
                    f"copy the coordinator's file here or set dataset_source"
                )
            report["fetched"] = list(shards)
        _retire_divergent_shard_files(dataset, shards, n_shards, total, isz)
        report["verified"] = list(shards)
        return report

    missing = []
    for k in shards:
        lo, hi = (total * k) // n_shards, (total * (k + 1)) // n_shards
        want, sp = per[k], f"{dataset}.shard{k}of{n_shards}"
        span = (hi - lo) * isz
        if os.path.exists(sp) and not _verified(sp, 0, span, want):
            os.replace(
                sp, f"{sp}.bad"
            )  # shard_view prefers a shard file: never let it serve wrong bytes
        if (os.path.exists(sp) and _verified(sp, 0, span, want)) or _verified(
            dataset, lo * isz, hi * isz, want
        ):
            verified.append(k)
            continue
        got = False
        if source:
            got = _fetch(source, os.path.basename(sp), sp) or _fetch(
                source, base, sp, (lo * isz, hi * isz)
            )
            if got and not _verified(sp, 0, span, want):
                os.replace(sp, f"{sp}.bad")
                got = False
        if got:
            verified.append(k)
            fetched.append(k)
        else:
            missing.append(k)
    if missing:
        names = ", ".join(f"{base}.shard{k}of{n_shards}" for k in missing[:6]) + (
            " …" if len(missing) > 6 else ""
        )
        raise DataRefused(
            f"this slot reads shards {missing} of {dataset!r} ({n_shards} shards) and this node does not hold them "
            f"verified against the manifest; copy {names} (or the whole file) here, or set dataset_source to a "
            f"directory or URL that holds them"
            + (" (the source did not hold matching bytes)" if source else "")
        )
    return report


def provision_slot(spec, rank: int, peer: int) -> dict | None:
    """provision() for the shards slot (rank, peer) reads; None for a slot that reads no data."""
    need = needed_shards(spec, rank, peer)
    if not need:
        return None
    return provision(
        spec.dataset_path,
        need,
        max(1, int(spec.data_shards or 1)),
        pinned=spec.dataset_sha256,
        source=spec.dataset_source,
    )

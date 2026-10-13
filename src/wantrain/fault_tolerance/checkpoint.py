"""Checkpoint coordination: shard waves held off the workers, and the resume.

Each slot's stage writes its own shard (its layers plus its optimizer state)
through save_shard, a seqlock: the old sidecar is removed before the new shard
replaces the old one and the new sidecar is written last, so a sidecar naming
step N proves the shard on disk is the one for step N. Workers ship every new wave to the
coordinator, verified by sha256 on both ends, so the death of a worker no
longer takes its shard with it. Recovery resumes every slot from the newest
wave *every* slot holds (a survivor may hold a newer save than the one the
dead slot shipped), under the original spec: the partition, microbatching and
seeds are pinned, because a shard sliced under one partition and loaded under
another is the wrong weights in the wrong stages with every digest matching.

Resume is exact: data is a pure function of (seed, step, microbatch) and the
optimizer state is in the shard, so a healed run continues the loss sequence
bit for bit when the replacement is the same device model.

Waves are immutable directories, one per slot per step:
out_dir/waves/<tag>/r<rank>p<peer>/step-<step:08d>/ holds state.pt and
wave.json, which is written last and names the step and every file with its
sha256 and size. validate_wave is the only step binding. The legacy pair
<tag>_ckpt_rank<r>p<p>.pt + .step.json is an alias derived from a wave by
link_legacy, for the oracle readers; every alias writer replaces the directory
entry and never opens the alias for writing, because the alias is a hard link
to a committed wave's state.pt (BA-46).
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading

# Waves newer than the common one kept per slot. A slot that never ships (it
# died before its first save) must not let the others grow without bound.
MAX_WAVES_AHEAD = 4


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def engine_shard_path(out_dir: str, tag: str, rank: int, peer: int) -> str:
    """Where the stage runtime writes (and on resume reads) a slot's shard."""
    return os.path.join(out_dir, f"{tag}_ckpt_rank{rank}p{peer}.pt")


def engine_sidecar_path(out_dir: str, tag: str, rank: int, peer: int) -> str:
    return engine_shard_path(out_dir, tag, rank, peer)[:-3] + ".step.json"


def read_sidecar(path: str) -> int | None:
    try:
        with open(path) as f:
            v = json.load(f).get("step")
        return int(v) if isinstance(v, int) else None
    except (OSError, ValueError, AttributeError):
        return None


def sidecar_of(shard_path: str) -> str:
    """The sidecar naming the step of the shard at shard_path."""
    return shard_path[:-3] + ".step.json"


def invalidate_sidecar(side: str) -> None:
    try:
        os.unlink(side)
    except FileNotFoundError:
        pass


def write_sidecar(side: str, step: int) -> None:
    """Atomic: a reader sees the old sidecar, none, or this one, never a part."""
    tmp = side + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"step": int(step)}, f)
    os.replace(tmp, side)


def save_shard(path: str, step: int, save_fn) -> None:
    """Write the shard for `step` at `path` and its sidecar, in seqlock order.

    The sidecar is gone before the shard is replaced, so a reader that read
    the old step, copied the shard and reads the sidecar again sees a change
    whenever the copy may hold the new bytes (CONTROL-I10)."""
    tmp = path + ".tmp"
    # A stale temp left by a crash may be a hard link to a committed wave's
    # state.pt (link_legacy); save_fn must create a new file, never open it.
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
    save_fn(tmp)
    invalidate_sidecar(sidecar_of(path))
    os.replace(tmp, path)
    write_sidecar(sidecar_of(path), step)


def check_resume_step(side: str, shard_step: int) -> None:
    """Refuse a shard its sidecar does not name: a resume from it would start
    the slot at a step the rest of the fleet is not at."""
    named = read_sidecar(side)
    if named is None:
        raise RuntimeError(
            f"resume refused: sidecar {side} is missing or unreadable, so the shard's "
            f"step {shard_step} is unproven"
        )
    if named != shard_step:
        raise RuntimeError(
            f"resume refused: sidecar {side} names step {named}, the shard holds step "
            f"{shard_step}: mislabeled wave"
        )


WAVE_MANIFEST = "wave.json"
WAVE_STATE = "state.pt"
WAVE_KEEP = 3
_WAVE_DIR_RE = re.compile(r"^step-(\d{8,})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_index(v) -> bool:
    # Bounded so a step from the wire names a path, never ENAMETOOLONG.
    return type(v) is int and 0 <= v < 1 << 63


def _slot_waves_dir(out_dir: str, tag: str, rank: int, peer: int) -> str:
    if (
        not isinstance(tag, str)
        or tag in ("", ".", "..")
        or any(c in tag for c in ("/", "\\", "\0"))
    ):
        raise ValueError(f"bad wave tag {tag!r}")
    if not (_is_index(rank) and _is_index(peer)):
        raise ValueError(f"bad wave slot r{rank!r}p{peer!r}")
    return os.path.join(out_dir, "waves", tag, f"r{rank}p{peer}")


def wave_dir(out_dir: str, tag: str, rank: int, peer: int, step: int) -> str:
    """out_dir/waves/<tag>/r<rank>p<peer>/step-<step:08d>/ (outside the oracle's
    out_dir/*_ckpt_rank*p*.pt glob)."""
    if not _is_index(step):
        raise ValueError(f"bad wave step {step!r}")
    return os.path.join(_slot_waves_dir(out_dir, tag, rank, peer), f"step-{step:08d}") + os.sep


def _fsync(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _file_entry(path: str, name: str) -> dict:
    return {"name": name, "sha256": sha256_file(path), "bytes": os.path.getsize(path)}


def _check_manifest(m, step: int, where: str) -> list:
    """The listed files of a well-formed manifest naming `step`; raises otherwise."""
    if not isinstance(m, dict) or set(m) != {"v", "step", "files"}:
        raise ValueError(f"wave {where}: manifest is not {{v, step, files}}")
    if type(m["v"]) is not int or m["v"] != 1:
        raise ValueError(f"wave {where}: manifest version {m['v']!r} is not 1")
    if type(m["step"]) is not int or m["step"] != step:
        raise ValueError(f"wave {where}: manifest names step {m['step']!r}, not step {step}")
    files = m["files"]
    if not isinstance(files, list) or not files:
        raise ValueError(f"wave {where}: manifest lists no files")
    names = set()
    for e in files:
        ok = (
            isinstance(e, dict)
            and set(e) == {"name", "sha256", "bytes"}
            and isinstance(e["name"], str)
            and e["name"] not in ("", ".", "..", WAVE_MANIFEST)
            and not any(c in e["name"] for c in ("/", "\\", "\0"))
            and isinstance(e["sha256"], str)
            and _SHA256_RE.match(e["sha256"]) is not None
            and _is_index(e["bytes"])
        )
        if not ok:
            raise ValueError(f"wave {where}: malformed file entry {e!r}")
        if e["name"] in names:
            raise ValueError(f"wave {where}: file {e['name']} listed twice")
        names.add(e["name"])
    if WAVE_STATE not in names:
        raise ValueError(f"wave {where}: manifest does not list {WAVE_STATE}")
    return files


def _check_files(d: str, files: list, where: str) -> None:
    """Every listed file is a regular file in d with its size and sha256, and
    nothing else but the manifest is there."""
    listed = {e["name"] for e in files}
    for name in sorted(set(os.listdir(d)) - listed - {WAVE_MANIFEST}):
        raise ValueError(f"wave {where}: file {name} is present but not listed")
    for e in files:
        path = os.path.join(d, e["name"])
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            raise ValueError(f"wave {where}: listed file {e['name']} is missing") from None
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"wave {where}: listed file {e['name']} is not a regular file")
        if st.st_size != e["bytes"]:
            raise ValueError(
                f"wave {where}: file {e['name']} has {st.st_size} bytes, manifest lists {e['bytes']}"
            )
        got = sha256_file(path)
        if got != e["sha256"]:
            raise ValueError(
                f"wave {where}: file {e['name']} has sha256 {got[:12]}, "
                f"manifest lists {e['sha256'][:12]}"
            )


def _read_manifest(d: str, step: int):
    try:
        with open(os.path.join(d, WAVE_MANIFEST), "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        raise ValueError(f"wave {d}: {WAVE_MANIFEST} is missing") from None
    try:
        m = json.loads(raw)
    except ValueError:
        raise ValueError(f"wave {d}: {WAVE_MANIFEST} is unparsable") from None
    _check_manifest(m, step, d)
    return m


def validate_wave(d: str, step: int) -> dict:
    """The manifest of the committed wave at d, proven to be step `step`: its
    wave.json names that step and every file in d is listed with its size and
    sha256. Raises ValueError naming the reason otherwise."""
    if not os.path.isdir(d):
        raise ValueError(f"wave {d}: no such directory")
    try:
        m = _read_manifest(d, step)
        _check_files(d, m["files"], d)
    except (FileNotFoundError, NotADirectoryError):
        # Pruned or dropped (renamed away) by a concurrent commit while read.
        raise ValueError(f"wave {d}: vanished during validation") from None
    return m


def _wave_steps(slot_dir: str) -> list:
    """[(step, dir)] ascending over every step directory of a slot (committed by
    rename; contents not checked)."""
    try:
        names = os.listdir(slot_dir)
    except FileNotFoundError:
        return []
    out = []
    for name in names:
        mt = _WAVE_DIR_RE.match(name)
        if mt and name == f"step-{int(mt.group(1)):08d}":
            out.append((int(mt.group(1)), os.path.join(slot_dir, name) + os.sep))
    return sorted(out)


def committed_waves(out_dir: str, tag: str, rank: int, peer: int) -> list:
    """[(step, dir)] ascending over the slot's wave directories that validate."""
    out = []
    for step, d in _wave_steps(_slot_waves_dir(out_dir, tag, rank, peer)):
        try:
            validate_wave(d, step)
        except ValueError:
            continue
        out.append((step, d))
    return out


def _remove_wave(slot_dir: str, d: str) -> None:
    """Rename out of the step-* namespace first, so a crash mid-removal never
    leaves a partial directory that looks committed."""
    gone = tempfile.mkdtemp(prefix=".drop-", dir=slot_dir)
    os.rename(os.path.normpath(d), os.path.join(gone, "w"))
    shutil.rmtree(gone)


def drop_waves_after(out_dir: str, tag: str, rank: int, peer: int, step: int) -> list:
    """Remove the slot's wave directories newer than `step`; returns their steps."""
    slot_dir = _slot_waves_dir(out_dir, tag, rank, peer)
    dropped = []
    for s, d in _wave_steps(slot_dir):
        if s > step:
            _remove_wave(slot_dir, d)
            dropped.append(s)
    if dropped:
        _fsync(slot_dir)
    return dropped


def _prune(slot_dir: str, keep: int, committed_step: int) -> None:
    """SPEC-control §2.3 wave retention: keep the newest `keep` wave directories
    at or below the step just committed. Newer ones (an abandoned timeline after
    a RESTORE or replay) are left to drop_waves_after (BA-6)."""
    steps = [(s, d) for s, d in _wave_steps(slot_dir) if s <= committed_step]
    drop = steps[: max(len(steps) - keep, 0)]
    for _, d in drop:
        _remove_wave(slot_dir, d)
    if drop:
        _fsync(slot_dir)


def link_legacy(wave_dir: str, out_dir: str, tag: str, rank: int, peer: int) -> str:
    """Point the alias <tag>_ckpt_rank<r>p<p>.pt at the wave's state.pt and its
    sidecar at wave.json's step, in save_shard's seqlock order. The alias is a
    hard link (a copy across devices or where links are refused) created under
    a temp name and renamed over the alias, so nothing is ever written into the
    wave. Returns the alias path."""
    name = os.path.basename(os.path.normpath(wave_dir))
    mt = _WAVE_DIR_RE.match(name)
    if mt is None:
        raise ValueError(f"wave {wave_dir}: not a step directory")
    m = _read_manifest(wave_dir, int(mt.group(1)))
    src = os.path.join(wave_dir, WAVE_STATE)
    alias = engine_shard_path(out_dir, tag, rank, peer)

    def link(tmp: str) -> None:
        try:
            os.link(src, tmp)
        except OSError as e:
            if e.errno not in (errno.EXDEV, errno.EPERM):
                raise
            shutil.copyfile(src, tmp)

    os.makedirs(out_dir, exist_ok=True)
    save_shard(alias, m["step"], link)
    return alias


def _commit_dir(tmp: str, final: str, slot_dir: str, keep: int, step: int, link) -> None:
    """wave.json is already in tmp: make it durable, rename it into place, then
    the alias and the pruning."""
    _fsync(tmp)
    os.rename(tmp, os.path.normpath(final))
    _fsync(slot_dir)
    link()
    _prune(slot_dir, keep, step)


def _alias_names(alias: str, d: str, m: dict) -> bool:
    """The alias holds the state.pt of the wave at d and its sidecar names m's step."""
    if read_sidecar(sidecar_of(alias)) != m["step"]:
        return False
    src = os.path.join(d, WAVE_STATE)
    try:
        if os.path.samefile(alias, src):
            return True
        want = next(e for e in m["files"] if e["name"] == WAVE_STATE)
        return os.path.getsize(alias) == want["bytes"] and sha256_file(alias) == want["sha256"]
    except FileNotFoundError:
        return False


def _finish_commit(
    final: str, m: dict, out_dir: str, tag: str, rank: int, peer: int, keep: int
) -> None:
    """Restart the steps after the rename of an interrupted commit of the slot's
    newest wave: a failure in link_legacy or the pruning (ENOSPC) left the wave
    committed, so a retry of the same step must converge rather than return
    early. A finished commit writes nothing here."""
    step = m["step"]
    slot_dir = _slot_waves_dir(out_dir, tag, rank, peer)
    steps = _wave_steps(slot_dir)
    if not steps or steps[-1][0] != step:
        return
    if not _alias_names(engine_shard_path(out_dir, tag, rank, peer), final, m):
        link_legacy(final, out_dir, tag, rank, peer)
    _prune(slot_dir, keep, step)


def _check_cap(slot_dir: str, keep: int, step: int, where: str) -> None:
    """SPEC-control §2.3 wave cap: refuse a new step, before anything is
    written, when the slot already holds MAX_WAVES_AHEAD step directories above
    it or keep + MAX_WAVES_AHEAD in total (a peer shipping descending steps)."""
    steps = [s for s, _ in _wave_steps(slot_dir)]
    above = sum(1 for s in steps if s > step)
    if above >= MAX_WAVES_AHEAD or len(steps) >= keep + MAX_WAVES_AHEAD:
        raise ValueError(
            f"wave {where}: slot {slot_dir} holds {len(steps)} waves, {above} above step "
            f"{step} (cap: {MAX_WAVES_AHEAD} above, {keep + MAX_WAVES_AHEAD} in total)"
        )


def _write_manifest(d: str, step: int, files: list) -> dict:
    m = {"v": 1, "step": step, "files": files}
    path = os.path.join(d, WAVE_MANIFEST)
    with open(path, "x") as f:
        json.dump(m, f)
        f.flush()
        os.fsync(f.fileno())
    return m


def _check_extra_files(extra_files) -> list:
    """[(name, bytes)] sorted by name of a commit's extra files; raises
    ValueError naming the first name that is not a plain file name of its own
    (state.pt and wave.json are the wave's) or whose content is not bytes.
    Names are compared case-folded, as a case-insensitive file system (APFS,
    NTFS) resolves them, so 'STATE.PT' or two names differing only in case
    are refused here, before save_fn runs."""
    if extra_files is None:
        return []
    if not isinstance(extra_files, dict):
        raise ValueError(f"wave extra_files is not a dict: {type(extra_files).__name__}")
    reserved = {WAVE_MANIFEST.casefold(), WAVE_STATE.casefold()}
    folded: set = set()
    for name, data in extra_files.items():
        ok = (
            isinstance(name, str)
            and name not in ("", ".", "..")
            and name.casefold() not in reserved | folded
            and not any(c in name for c in ("/", "\\", "\0"))
            and len(name.encode("utf-8", "surrogatepass")) <= 255
        )
        if isinstance(name, str):
            folded.add(name.casefold())
        if not ok:
            raise ValueError(f"wave extra file name {str(name)[:40]!r} is not a wave file name")
        if not isinstance(data, bytes):
            raise ValueError(f"wave extra file {name}: content is not bytes")
    return sorted(extra_files.items())


def _check_held_extras(held: dict, extras: list, where: str) -> None:
    """The held wave lists exactly the requested extras, by name and sha256;
    raises naming the first file that differs (I-15, BA-460)."""
    have = {e["name"]: e["sha256"] for e in held["files"] if e["name"] != WAVE_STATE}
    want = {name: hashlib.sha256(data).hexdigest() for name, data in extras}
    for name in sorted(set(have) | set(want)):
        if have.get(name) != want.get(name):
            held_sha, want_sha = have.get(name, "absent"), want.get(name, "absent")
            raise ValueError(
                f"wave {where}: the committed wave of this step holds extra file {name} "
                f"as {held_sha[:12]}, the commit requests {want_sha[:12]}"
            )


def commit_wave(
    out_dir: str,
    tag: str,
    rank: int,
    peer: int,
    step: int,
    save_fn,
    keep: int = WAVE_KEEP,
    extra_files: dict[str, bytes] | None = None,
) -> tuple:
    """Commit the slot's wave of `step`: save_fn(path) writes state.pt into a
    private temp directory, each of extra_files (name -> bytes) is written
    beside it, wave.json is written last (state.pt first, then the extras by
    name), and the directory is renamed into place before the alias follows
    it. Returns (dir, committed); a slot already holding a valid wave of
    `step` listing the same extras (name and sha256) is left untouched
    (committed=False; only an interrupted alias or pruning of the slot's newest
    wave is finished), and an invalid one there, or one with other extras,
    raises (I-15, BA-460)."""
    if not _is_index(keep) or keep < 1:
        raise ValueError(f"bad wave keep {keep!r}")
    extras = _check_extra_files(extra_files)
    final = wave_dir(out_dir, tag, rank, peer, step)
    if os.path.lexists(os.path.normpath(final)):
        held = validate_wave(final, step)
        _check_held_extras(held, extras, final)
        _finish_commit(final, held, out_dir, tag, rank, peer, keep)
        return final, False
    slot_dir = _slot_waves_dir(out_dir, tag, rank, peer)
    _check_cap(slot_dir, keep, step, f"r{rank}p{peer} step {step}")
    os.makedirs(slot_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f".tmp-step-{step:08d}-", dir=slot_dir)
    try:
        path = os.path.join(tmp, WAVE_STATE)
        save_fn(path)
        _fsync(path)
        entries = [_file_entry(path, WAVE_STATE)]
        for name, data in extras:
            extra = os.path.join(tmp, name)
            with open(extra, "xb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            entries.append(_file_entry(extra, name))
        _write_manifest(tmp, step, entries)
        _commit_dir(
            tmp,
            final,
            slot_dir,
            keep,
            step,
            lambda: link_legacy(final, out_dir, tag, rank, peer),
        )
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final, True


def _same_manifest(a: dict, b: dict) -> bool:
    def norm(m):
        return (m["v"], m["step"], sorted((e["name"], e["sha256"], e["bytes"]) for e in m["files"]))

    return norm(a) == norm(b)


def _received_step(manifest, rank, peer, keep) -> tuple:
    """(step, listed files) of a received manifest: v 1, an int step."""
    if not _is_index(keep) or keep < 1:
        raise ValueError(f"bad wave keep {keep!r}")
    raw_step = manifest.get("step") if isinstance(manifest, dict) else None
    if not isinstance(raw_step, int) or not _is_index(raw_step):
        raise ValueError(f"received wave r{rank}p{peer}: bad step {raw_step!r}")
    return raw_step, _check_manifest(manifest, raw_step, f"received r{rank}p{peer} step {raw_step}")


def _held_equal(final: str, step: int, manifest: dict, where: str) -> dict | None:
    """The manifest of the wave committed at final when it equals `manifest`,
    None when no wave is there; raises when the held one differs."""
    if not os.path.lexists(os.path.normpath(final)):
        return None
    held = validate_wave(final, step)
    if not _same_manifest(held, manifest):
        raise ValueError(f"{where}: the committed wave of this step has another manifest")
    return held


def _check_staged_listed(staged_dir: str, files: list, where: str) -> None:
    listed = {e["name"] for e in files}
    for name in sorted(set(os.listdir(staged_dir)) - listed):
        raise ValueError(f"wave {where}: file {name} is staged but not listed")


def _commit_staged_files(
    slot_dir: str, final: str, step: int, files: list, staged_dir: str, keep: int, where: str, link
) -> None:
    """Move the listed files of staged_dir into a private temp directory, check
    each there (size, sha256), write wave.json and rename it to final."""
    _check_staged_listed(staged_dir, files, where)
    _check_cap(slot_dir, keep, step, where)
    tmp = _take_staged_files(slot_dir, step, files, staged_dir, where)
    try:
        _commit_dir(tmp, final, slot_dir, keep, step, link)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _take_staged_files(slot_dir: str, step: int, files: list, staged_dir: str, where: str) -> str:
    """The listed files of staged_dir moved into a new private temp directory of
    slot_dir (outside the step-* namespace), each checked there (size, sha256)
    and synced, and wave.json written: returns that directory, for the caller
    to rename into place or remove."""
    os.makedirs(slot_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f".tmp-step-{step:08d}-", dir=slot_dir)
    try:
        for e in files:
            src = os.path.join(staged_dir, e["name"])
            dst = os.path.join(tmp, e["name"])
            if not os.path.lexists(src):
                raise ValueError(f"wave {where}: listed file {e['name']} is missing")
            try:
                os.replace(src, dst)
            except OSError as err:
                if err.errno != errno.EXDEV:
                    raise
                shutil.copyfile(src, dst, follow_symlinks=False)
        # Checked in the private directory, after the move: nothing can change
        # the bytes between this check and the commit.
        _check_files(tmp, files, where)
        for e in files:
            _fsync(os.path.join(tmp, e["name"]))
        _write_manifest(tmp, step, [dict(e) for e in files])
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return tmp


def commit_received_wave(
    out_dir: str,
    tag: str,
    rank: int,
    peer: int,
    manifest: dict,
    staged_dir: str,
    keep: int = WAVE_KEEP,
) -> tuple:
    """Commit a wave received over the network: the manifest must be v 1 with
    an int step, staged_dir must hold exactly its listed files, and each file
    is moved into a private temp directory and checked there (size, sha256)
    before wave.json is written and the directory renamed into place. Returns
    (dir, committed); an existing wave of the step with an equal manifest is
    kept untouched (committed=False), one with another manifest raises."""
    step, files = _received_step(manifest, rank, peer, keep)
    final = wave_dir(out_dir, tag, rank, peer, step)
    held = _held_equal(final, step, manifest, f"received wave r{rank}p{peer} step {step}")
    if held is not None:
        _finish_commit(final, held, out_dir, tag, rank, peer, keep)
        return final, False
    _commit_staged_files(
        _slot_waves_dir(out_dir, tag, rank, peer),
        final,
        step,
        files,
        staged_dir,
        keep,
        f"received r{rank}p{peer} step {step} ({staged_dir})",
        lambda: link_legacy(final, out_dir, tag, rank, peer),
    )
    return final, True


def check_manifest(m, step: int, where: str) -> list:
    """The listed files of a well-formed manifest naming `step`; raises
    ValueError naming the reason otherwise."""
    return _check_manifest(m, step, where)


def read_manifest(d: str, step: int) -> dict:
    """The well-formed manifest of the wave at d naming `step` (files not
    hashed: a committed wave was hashed when it was committed)."""
    return _read_manifest(d, step)


def slot_waves(out_dir: str, tag: str, rank: int, peer: int) -> list:
    """[(step, dir)] ascending over the slot's committed wave directories
    (named by rename after every check; contents not re-hashed)."""
    return _wave_steps(_slot_waves_dir(out_dir, tag, rank, peer))


_SLOT_DIR_RE = re.compile(r"^r(\d+)p(\d+)$")
# Uncommitted steps whose parts one slot may have staged at once (CONTROL-I20).
STAGED_STEPS_MAX = 2
# The store's own retention is prune() (from the common step, MAX_WAVES_AHEAD
# newer): a commit keeps this many at or below its step, so it never removes a
# wave prune() keeps (prune leaves at most 1 + MAX_WAVES_AHEAD, a commit adds 1).
STORE_KEEP = MAX_WAVES_AHEAD + 2


class CheckpointStore:
    """Waves for one run, held as units: (slot, step) ->
    root/r<r>p<p>/step-<n:08d>/ with wave.json and its files, committed (the
    same checks and rename as commit_received_wave) only after every listed
    file's sha256 and the manifest's step verify.

    A wave arrives as one CKPT_PART per file, staged under
    root/.incoming/r<r>p<p>/step-<n:08d>/ for the connection's slot, then one
    CKPT with the manifest that commits it. Staging is bounded: at most
    STAGED_STEPS_MAX uncommitted steps per slot, one part per name, every part
    exactly the bytes and sha256 its manifest lists; anything else drops the
    staged parts (CONTROL-I20, BA-43)."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._lock = threading.Lock()
        # slot -> (attempt, {step: {name: sha256 announced by the part}})
        self._staged: dict = {}

    def _slot_dir(self, slot) -> str:
        r, p = slot
        if not (_is_index(r) and _is_index(p)):
            raise ValueError(f"bad wave slot r{r!r}p{p!r}")
        return os.path.join(self.root, f"r{r}p{p}")

    def wave_path(self, slot, step: int) -> str:
        if not _is_index(step):
            raise ValueError(f"bad wave step {step!r}")
        return os.path.join(self._slot_dir(slot), f"step-{step:08d}") + os.sep

    def manifest(self, slot, step: int) -> dict:
        return _read_manifest(self.wave_path(slot, step), step)

    def index(self) -> dict:
        """slot -> ascending list of the steps of its committed waves."""
        out: dict = {}
        for name in os.listdir(self.root):
            m = _SLOT_DIR_RE.match(name)
            if m is None or name != f"r{int(m.group(1))}p{int(m.group(2))}":
                continue
            steps = [s for s, _ in _wave_steps(os.path.join(self.root, name))]
            if steps:
                out[(int(m.group(1)), int(m.group(2)))] = steps
        return out

    def has(self, slot, step: int) -> bool:
        return os.path.isdir(os.path.normpath(self.wave_path(slot, step)))

    def _staging(self, slot, step: int | None = None) -> str:
        d = os.path.join(self.root, ".incoming", f"r{slot[0]}p{slot[1]}")
        return d if step is None else os.path.join(d, f"step-{step:08d}")

    def _drop_locked(self, slot, step: int | None = None) -> None:
        if step is None:
            self._staged.pop(slot, None)
            shutil.rmtree(self._staging(slot), ignore_errors=True)
        else:
            self._staged.get(slot, (None, {}))[1].pop(step, None)
            shutil.rmtree(self._staging(slot, step), ignore_errors=True)

    def drop_staged(self, slot=None) -> None:
        """Unlink the staged parts of one slot (its connection ended), or of
        every slot (a new attempt)."""
        with self._lock:
            if slot is not None:
                self._drop_locked(slot)
                return
            self._staged.clear()
            shutil.rmtree(os.path.join(self.root, ".incoming"), ignore_errors=True)

    def stage_part(self, slot, attempt: int, step: int, name, sha256, src: str) -> None:
        """Move one received CKPT_PART body (src) into the slot's staging for
        `step`. Raises ValueError after dropping what the violation spoils: a
        part that would open a third uncommitted step drops every staged part
        of the slot, a second part of one name drops the step's parts; a
        malformed name or sha256 refuses the part alone. The caller unlinks
        src when it is still there."""
        self._slot_dir(slot)
        if not _is_index(step):
            raise ValueError(f"bad wave step {step!r}")
        ok_name = (
            isinstance(name, str)
            and name not in ("", ".", "..", WAVE_MANIFEST)
            and not any(c in name for c in ("/", "\\", "\0"))
            and len(name) <= 255
        )
        if not ok_name:
            raise ValueError(f"part name {str(name)[:40]!r} is not a wave file name")
        if not isinstance(sha256, str) or _SHA256_RE.match(sha256) is None:
            raise ValueError(f"part {name}: sha256 {str(sha256)[:40]!r} is not hex")
        with self._lock:
            held, steps = self._staged.get(slot, (attempt, {}))
            if held != attempt:
                self._drop_locked(slot)
                steps = {}
            if step not in steps and len(steps) >= STAGED_STEPS_MAX:
                self._drop_locked(slot)
                raise ValueError(
                    f"slot {slot}: part {name} of step {step} would open a third uncommitted "
                    f"step (open: {sorted(steps)}); every staged part of the slot dropped"
                )
            if name in steps.get(step, {}):
                self._drop_locked(slot, step)
                raise ValueError(
                    f"slot {slot}: part {name} of step {step} staged twice; "
                    "the step's staged parts dropped"
                )
            d = self._staging(slot, step)
            os.makedirs(d, exist_ok=True)
            shutil.move(src, os.path.join(d, name))
            steps.setdefault(step, {})[name] = sha256
            self._staged[slot] = (attempt, steps)

    def commit_staged(self, slot, attempt: int, step: int, manifest) -> bool:
        """Commit the staged parts of `step` as the slot's wave under its CKPT
        manifest. The staged parts are gone afterwards, committed or dropped.
        Returns False when an equal wave was already held; raises ValueError
        when the manifest names another step, lists a file not staged or
        staged with another sha256, or any part is not exactly its listed
        bytes (an oversized part included)."""
        where = f"slot {slot} step {step}"
        slot_dir = self._slot_dir(slot)
        with self._lock:
            held, steps = self._staged.get(slot, (None, {}))
            parts = steps.pop(step, None) if held == attempt else None
        staged = self._staging(slot, step)
        tmp = None
        try:
            if parts is None:
                raise ValueError(f"{where}: CKPT without staged parts")
            if not isinstance(manifest, dict):
                raise ValueError(f"{where}: manifest is not an object")
            files = _check_manifest(manifest, step, where)
            for e in files:
                if parts.get(e["name"]) != e["sha256"]:
                    raise ValueError(
                        f"{where}: manifest lists {e['name']}, not staged or staged "
                        "with another sha256"
                    )
            final = self.wave_path(slot, step)
            # Every hash (the held wave's, the parts') runs outside the store's
            # lock, on this connection's reader alone: the slots committing one
            # step never queue each other's readers, and with them their
            # heartbeats, behind their bytes. Only the namespace steps (cap,
            # rename, prune) take the lock.
            if _held_equal(final, step, manifest, where) is not None:
                return False
            _check_staged_listed(staged, files, where)
            _check_cap(slot_dir, STORE_KEEP, step, where)
            tmp = _take_staged_files(slot_dir, step, files, staged, where)
            with self._lock:
                if os.path.lexists(os.path.normpath(final)):
                    raise ValueError(f"{where}: committed meanwhile by another connection")
                _check_cap(slot_dir, STORE_KEEP, step, where)
                _commit_dir(tmp, final, slot_dir, STORE_KEEP, step, lambda: None)
            return True
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)  # gone once renamed into place
            shutil.rmtree(staged, ignore_errors=True)

    def replace_wave(self, slot, step: int, payload: dict) -> None:
        """RUNTIME-I26: put `payload` (a re-cut state) in place of the slot's committed wave of `step`: written and
        hashed in a private directory beside it, then swapped in by rename, the old one removed after."""
        import torch

        final = os.path.normpath(self.wave_path(slot, step))
        slot_dir = self._slot_dir(slot)
        os.makedirs(slot_dir, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix=f".recut-step-{step:08d}-", dir=slot_dir)
        try:
            path = os.path.join(tmp, WAVE_STATE)
            torch.save(payload, path)
            _fsync(path)
            _write_manifest(tmp, step, [_file_entry(path, WAVE_STATE)])
            with self._lock:
                old = None
                if os.path.lexists(final):
                    old = tempfile.mkdtemp(prefix=f".old-step-{step:08d}-", dir=slot_dir)
                    os.rmdir(old)
                    os.rename(final, old)
                os.rename(tmp, final)
            if old is not None:
                shutil.rmtree(old, ignore_errors=True)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def drop_after(self, slots, step: int) -> None:
        """Remove every committed wave newer than `step` of `slots` (a re-cut resumes from `step`; the newer ones
        belong to the old cut)."""
        idx = self.index()
        with self._lock:
            for slot in slots:
                for s in idx.get(tuple(slot), []):
                    if s > step:
                        try:
                            _remove_wave(self._slot_dir(slot), self.wave_path(slot, s))
                        except FileNotFoundError:
                            pass

    def common_step(self, slots) -> int | None:
        """The newest step for which every slot holds a committed wave, or None."""
        idx = self.index()
        sets = [set(idx.get(tuple(s), [])) for s in slots]
        if not sets or any(not s for s in sets):
            return None
        shared = set.intersection(*sets)
        return max(shared) if shared else None

    def prune(self, slots) -> None:
        """Drop waves that can no longer be the resume point: anything older
        than the newest common wave, and anything beyond MAX_WAVES_AHEAD newer
        than it."""
        common = self.common_step(slots)
        idx = self.index()
        with self._lock:
            for slot, steps in idx.items():
                newer = [s for s in steps if common is None or s > common]
                drop = [s for s in steps if common is not None and s < common]
                drop += newer[:-MAX_WAVES_AHEAD] if len(newer) > MAX_WAVES_AHEAD else []
                slot_dir = self._slot_dir(slot)
                for s in drop:
                    try:
                        _remove_wave(slot_dir, self.wave_path(slot, s))
                    except FileNotFoundError:
                        pass

    def export_wave(self, step: int, slots, out_dir: str, tag: str) -> list:
        """Write wave `step` of every slot into engine layout (for a local
        resume): the committed wave under out_dir/waves/... (every hash
        checked on both sides) and then its alias. Returns the alias paths."""
        paths = []
        os.makedirs(out_dir, exist_ok=True)
        for r, p in slots:
            src = self.wave_path((r, p), step)
            m = validate_wave(src, step)
            staged = tempfile.mkdtemp(prefix=".export-", dir=out_dir)
            try:
                for e in m["files"]:
                    shutil.copyfile(os.path.join(src, e["name"]), os.path.join(staged, e["name"]))
                d, committed = commit_received_wave(out_dir, tag, r, p, m, staged)
                if not committed:
                    link_legacy(d, out_dir, tag, r, p)
            finally:
                shutil.rmtree(staged, ignore_errors=True)
            paths.append(engine_shard_path(out_dir, tag, r, p))
        return paths


_PEER_ID_RE = re.compile(r"[0-9a-f]{32}")


def slot_peer_pins(slot_peers) -> dict:
    """The spec fields that pin each slot's key (CONTROL-I22): slot_peers
    itself, slot_peers[stage][peer] = the peer id the assigned node proved in
    its control handshake, and trusted_peers = the sorted set of those ids."""
    if not isinstance(slot_peers, list) or not all(
        isinstance(row, list)
        and row
        and all(isinstance(p, str) and _PEER_ID_RE.fullmatch(p) for p in row)
        for row in slot_peers
    ):
        raise ValueError(f"bad slot_peers {slot_peers!r}")
    return {
        "slot_peers": [list(row) for row in slot_peers],
        "trusted_peers": sorted({p for row in slot_peers for p in row}),
    }


OWNERSHIP_FIELDS = ("peer_weights", "mb_owner", "schedule_order")


def ownership_fields(ownership) -> dict:
    """The ownership a rebalance replaces (CONTROL-I8, CONTROL-I21): exactly
    peer_weights, mb_owner and schedule_order, each a non-empty list; their
    shapes are checked against the recipe by JobSpec.finalize at the stage."""
    if (
        not isinstance(ownership, dict)
        or set(ownership) != set(OWNERSHIP_FIELDS)
        or not all(isinstance(ownership[k], list) and ownership[k] for k in OWNERSHIP_FIELDS)
    ):
        raise ValueError(f"bad ownership {str(ownership)[:120]!r}")
    return {k: ownership[k] for k in OWNERSHIP_FIELDS}


def patch_spec_for_resume(
    spec_json: str,
    endpoints=None,
    devices=None,
    slot_peers=None,
    resume_step: int | None = None,
    ownership=None,
) -> str:
    """The stored spec, pointed at the healed fleet (its addresses, devices
    and slot keys) and told to resume from the wave of `resume_step` (the one
    every slot was restored to). Nothing numerical changes: partition,
    microbatches, mb_size and seeds are the original run's. A rebalance
    (CONTROL-I21) also passes `ownership` = {peer_weights, mb_owner,
    schedule_order}, the only way those fields change; the fold order is the
    microbatch index, so they move no bit."""
    if resume_step is not None and not _is_index(resume_step):
        raise ValueError(f"bad resume_step {resume_step!r}")
    pins = slot_peer_pins(slot_peers) if slot_peers is not None else {}
    if ownership is not None:
        pins.update(ownership_fields(ownership))
    d = json.loads(spec_json)
    if resume_step is not None:
        d["resume_step"] = resume_step
    if endpoints is not None:
        d["endpoints"] = endpoints
    if devices is not None:
        d["devices"] = devices
    d.update(pins)
    d["resume"] = True
    return json.dumps(d, indent=2)

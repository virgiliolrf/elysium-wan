"""MODEL-I11: a data-reading slot holds, verified, the shards its microbatches draw (model/provision.py)."""

import http.server
import os
import shutil
import threading

import numpy as np
import pytest
import torch

from wantrain.model import dataset as ds
from wantrain.model import provision
from wantrain.model.data import make_batch
from wantrain.model.gpt import make_config
from wantrain.runtime.spec import JobSpec

VOCAB, SEQ, MB = 500, 8, 2


def _corpus(d, n_shards=4, tokens=4000, name="tokens.bin"):
    """A prepared corpus without a tokenizer: the token file, its shard files and a manifest with every digest."""
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    np.random.default_rng(7).integers(0, VOCAB, tokens, dtype=np.uint16).tofile(path)
    extra = {"sha256": ds.file_digest(path)}
    if n_shards > 1:
        extra["shard_sha256"] = ds.shard_digests(path, tokens, n_shards, "uint16")
    ds.write_manifest(path, "test", VOCAB, tokens, n_shards, "uint16", extra=extra)
    if n_shards > 1:
        ds._write_shards(path, tokens, n_shards, "uint16")
    return path


def _node(d, src_path, files):
    """A node directory holding only `files` (basenames) copied from the source corpus."""
    os.makedirs(d, exist_ok=True)
    for f in files:
        shutil.copy(os.path.join(os.path.dirname(src_path), f), os.path.join(d, f))
    return os.path.join(d, os.path.basename(src_path))


def _batch(path, mb, shards):
    return make_batch(5678, 3, mb, MB, SEQ, VOCAB, dataset=path, data_shards=shards)


def _flip(path, offset):
    with open(path, "r+b") as f:
        f.seek(offset)
        b = f.read(1)
        f.seek(offset)
        f.write(bytes([b[0] ^ 0xFF]))


def test_needed_shards_follow_ownership_and_skip_the_middle_stages():
    # covers: MODEL-I11
    # WOULD-FAIL-IF: a middle stage is asked for data, a first or last stage slot's set is not the shards of exactly
    # the microbatches it owns, or a rebalance (other peer weights) leaves the set unchanged.
    kw = dict(
        model=make_config("toy"), stages=3, stage_widths=[2, 1, 2], microbatches=8, data_shards=4
    )
    spec = JobSpec(dataset_path="/d/tokens.bin", **kw).finalize()
    for rank, width in enumerate(spec.stage_widths):
        for peer in range(width):
            want = sorted({mb % 4 for mb in spec.owned(rank, peer)}) if rank in (0, 2) else []
            assert provision.needed_shards(spec, rank, peer) == want
    assert provision.needed_shards(JobSpec(**kw).finalize(), 0, 0) == [], (
        "a synthetic run reads no corpus"
    )
    moved = JobSpec(
        dataset_path="/d/tokens.bin", peer_weights=[[3.0, 1.0], [1.0], [1.0, 1.0]], **kw
    ).finalize()
    assert provision.needed_shards(moved, 0, 1) != provision.needed_shards(spec, 0, 1), (
        "ownership moved"
    )


def test_a_whole_file_serves_its_needed_shards_and_only_their_bytes_are_checked(tmp_path):
    # covers: MODEL-I11
    # WOULD-FAIL-IF: verification hashes shards the slot does not read (a corrupt unread shard refuses), or a corrupt
    # byte in a read shard passes.
    src = _corpus(str(tmp_path / "src"))
    node = _node(str(tmp_path / "node"), src, ["tokens.bin", "tokens.bin.meta.json"])
    _flip(node, 3 * 1000 * 2 + 5)  # shard 3 of 4 (1000 tokens each, uint16)
    r = provision.provision(node, [0, 1], 4, pinned=provision.manifest_digest(src))
    assert r["verified"] == [0, 1] and r["fetched"] == []
    with pytest.raises(provision.DataRefused, match=r"shards \[3\].*tokens.bin.shard3of4"):
        provision.provision(node, [3], 4)


def test_missing_shards_refuse_at_startup_without_a_source(tmp_path):
    # covers: MODEL-I11
    # WOULD-FAIL-IF: a node holding only some shard files passes for a shard it lacks (it used to fail with
    # FileNotFoundError at its first microbatch, mid-run), or the refusal does not name the shard file and the fix.
    src = _corpus(str(tmp_path / "src"))
    node = _node(str(tmp_path / "node"), src, ["tokens.bin.meta.json", "tokens.bin.shard0of4"])
    assert provision.provision(node, [0], 4)["verified"] == [0]
    with pytest.raises(provision.DataRefused, match=r"tokens.bin.shard1of4.*dataset_source"):
        provision.provision(node, [0, 1], 4)


def test_a_source_directory_fills_the_shards_and_the_batches_equal_the_whole_file(tmp_path):
    # covers: MODEL-I11
    # WOULD-FAIL-IF: provisioning from a source does not fetch the manifest and the needed shards, fetches shards
    # the slot does not read, or the fetched layout serves other tokens than the whole file for the same microbatch.
    src = _corpus(str(tmp_path / "src"))
    node = str(tmp_path / "node" / "tokens.bin")
    r = provision.provision(
        node, [1, 2], 4, pinned=provision.manifest_digest(src), source=str(tmp_path / "src")
    )
    assert r["fetched"] == [1, 2]
    assert sorted(os.listdir(tmp_path / "node")) == [
        "tokens.bin.meta.json",
        "tokens.bin.provision.lock",
        "tokens.bin.shard1of4",
        "tokens.bin.shard1of4.verified.json",
        "tokens.bin.shard2of4",
        "tokens.bin.shard2of4.verified.json",
    ]
    for mb in (1, 2, 5, 6):  # microbatches of shards 1 and 2
        for a, b in zip(_batch(node, mb, 4), _batch(src, mb, 4)):
            assert torch.equal(a, b)


def test_a_corrupt_shard_file_is_never_served_and_is_fetched_again(tmp_path):
    # covers: MODEL-I11
    # WOULD-FAIL-IF: a shard file whose bytes differ from the manifest is kept (shard_view prefers a shard file, so
    # the slot would train on it), or it is not replaced by verified bytes from the source.
    src = _corpus(str(tmp_path / "src"))
    node = _node(str(tmp_path / "node"), src, ["tokens.bin.meta.json", "tokens.bin.shard2of4"])
    _flip(node + ".shard2of4", 11)
    with pytest.raises(provision.DataRefused):
        provision.provision(node, [2], 4)
    assert not os.path.exists(node + ".shard2of4"), (
        "the corrupt shard file was left where shard_view reads it"
    )
    assert provision.provision(node, [2], 4, source=str(tmp_path / "src"))["fetched"] == [2]
    for a, b in zip(_batch(node, 2, 4), _batch(src, 2, 4)):
        assert torch.equal(a, b)


def test_a_manifest_other_than_the_planned_one_refuses_or_is_replaced_from_the_source(tmp_path):
    # covers: MODEL-I11
    # WOULD-FAIL-IF: a node whose manifest differs from the one the job was planned with (another corpus, tokenizer
    # or layout) passes, or a source does not replace it with the pinned one.
    src = _corpus(str(tmp_path / "src"))
    other = _corpus(str(tmp_path / "other"), n_shards=2)
    node = _node(str(tmp_path / "node"), other, ["tokens.bin", "tokens.bin.meta.json"])
    pinned = provision.manifest_digest(src)
    with pytest.raises(provision.DataRefused, match="planned with"):
        provision.provision(node, [0], 4, pinned=pinned)
    r = provision.provision(node, [0], 4, pinned=pinned, source=str(tmp_path / "src"))
    assert provision.manifest_digest(node) == pinned and r["verified"] == [0]


def test_a_recipe_shard_count_other_than_the_prepared_one_verifies_the_whole_file(tmp_path):
    # covers: MODEL-I11
    # WOULD-FAIL-IF: per-shard digests prepared for 4 shards are applied to a recipe of 2 logical shards, or the
    # whole file is not checked (and fetched when missing) in that case.
    src = _corpus(str(tmp_path / "src"))
    node = str(tmp_path / "node" / "tokens.bin")
    r = provision.provision(node, [0], 2, source=str(tmp_path / "src"))
    assert r["fetched"] == [0] and ds.file_digest(node) == ds.file_digest(src)
    _flip(node, 7)
    os.utime(node, ns=(1, 1))  # a stale stamp must not hide the change
    with pytest.raises(provision.DataRefused, match="differs from its manifest"):
        provision.provision(node, [1], 2)


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """A static server that honours single byte ranges, like an object store."""

    def do_GET(self):  # noqa: N802
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, "rb") as f:
            data = f.read()
        rng = self.headers.get("Range")
        if rng:
            lo, hi = (int(x) for x in rng.split("=")[1].split("-"))
            data = data[lo : hi + 1]
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_an_http_source_serves_shards_by_byte_range_of_the_whole_file(tmp_path):
    # covers: MODEL-I11
    # WOULD-FAIL-IF: an http(s) source holding only the whole file cannot provide a shard (no byte-range fetch), or
    # the fetched range is not verified against the shard's digest before use.
    src = _corpus(str(tmp_path / "src"), n_shards=4)
    for k in range(4):
        os.remove(f"{src}.shard{k}of4")  # the source holds the whole file only
    handler = lambda *a, **kw: _RangeHandler(*a, directory=str(tmp_path / "src"), **kw)  # noqa: E731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        node = str(tmp_path / "node" / "tokens.bin")
        r = provision.provision(node, [3], 4, pinned=provision.manifest_digest(src), source=url)
        assert r["fetched"] == [3]
        for a, b in zip(_batch(node, 3, 4), _batch(src, 3, 4)):
            assert torch.equal(a, b)
    finally:
        server.shutdown()
        server.server_close()


def test_provision_slot_reads_the_spec_and_a_middle_stage_needs_nothing(tmp_path):
    # covers: MODEL-I11
    # WOULD-FAIL-IF: provision_slot ignores the spec's pinned digest or source, or provisions a middle stage.
    src = _corpus(str(tmp_path / "src"))
    spec = JobSpec(
        model=make_config("toy"),
        stages=3,
        microbatches=8,
        data_shards=4,
        dataset_path=str(tmp_path / "node" / "tokens.bin"),
        dataset_sha256=provision.manifest_digest(src),
        dataset_source=str(tmp_path / "src"),
    ).finalize()
    assert provision.provision_slot(spec, 1, 0) is None
    assert provision.provision_slot(spec, 0, 0)["fetched"] == [0, 1, 2, 3]


def test_the_planner_pins_the_manifest_and_keeps_the_prepared_shard_count(tmp_path):
    # covers: MODEL-I11
    # WOULD-FAIL-IF: the plan does not pin the manifest's digest (a node with another corpus would pass), drops the
    # source, or picks a shard count other than the prepared one (no slot could use the prepared shard files).
    from wantrain.scheduler.planner import Fleet, Node, plan_job

    src = _corpus(str(tmp_path / "src"), n_shards=3)
    nodes = [
        Node(n, tflops=50.0, vram_bytes=24 << 30, compute_cap=(8, 9), device_class="cuda:sm_89")
        for n in "ab"
    ]
    fleet = Fleet(nodes=nodes, link_mbps=1000.0, rtt_ms=5.0, duplex="per_direction")
    spec = plan_job(
        make_config("gpt124m"),
        fleet,
        dataset=src,
        dataset_source="https://data.example/c",
        steps=2,
        device="cuda",
        require_margin=False,
    )["spec"]
    assert spec.data_shards == 3
    assert spec.dataset_sha256 == provision.manifest_digest(src)
    assert spec.dataset_source == "https://data.example/c"


def test_the_planner_says_when_the_shard_default_comes_from_the_corpus_or_nothing_pins_it(tmp_path):
    # covers: MODEL-I11
    # WOULD-FAIL-IF: the prepared shard count becomes the recipe's data_shards without a word in the plan's warnings
    # (it changed from the node count), an unsharded corpus does not keep the node-count default, an explicit
    # data_shards is overridden, or a corpus without a manifest is planned without a warning that nothing pins it.
    from wantrain.scheduler.planner import Fleet, Node, Overrides, plan_job

    nodes = [
        Node(n, tflops=50.0, vram_bytes=24 << 30, compute_cap=(8, 9), device_class="cuda:sm_89")
        for n in "ab"
    ]
    fleet = Fleet(nodes=nodes, link_mbps=1000.0, rtt_ms=5.0, duplex="per_direction")

    def plan(path, **ov):
        return plan_job(
            make_config("gpt124m"),
            fleet,
            dataset=path,
            steps=2,
            device="cuda",
            require_margin=False,
            overrides=Overrides(**ov),
        )

    sharded = plan(_corpus(str(tmp_path / "s"), n_shards=3))
    assert sharded["spec"].data_shards == 3
    assert any("prepared shard count" in w for w in sharded["warnings"])
    explicit = plan(_corpus(str(tmp_path / "e"), n_shards=3), data_shards=5)
    assert explicit["spec"].data_shards == 5
    assert not any("prepared shard count" in w for w in explicit["warnings"])
    whole = plan(_corpus(str(tmp_path / "w"), n_shards=1))
    assert whole["spec"].data_shards == 2, "an unsharded corpus keeps the node-count default"
    raw = str(tmp_path / "raw.bin")
    np.zeros(4000, dtype=np.uint16).tofile(raw)
    bare = plan(raw)
    assert bare["spec"].dataset_sha256 is None
    assert any("not pinned by content" in w for w in bare["warnings"])


def test_the_content_pin_is_absent_unless_the_corpus_is_pinned():
    # covers: MODEL-I11
    # WOULD-FAIL-IF: pins() always carries dataset_sha256, so a synthetic run's checkpoint, or one written before the
    # corpus was pinned by content, no longer resumes (an absent key differs from a present None).
    from wantrain.runtime.spec import first_pin_difference

    old = JobSpec(model=make_config("toy"), stages=2).finalize().pins()
    assert "dataset_sha256" not in old
    assert (
        first_pin_difference(old, JobSpec(model=make_config("toy"), stages=2).finalize().pins())
        is None
    )
    pinned = JobSpec(model=make_config("toy"), stages=2, dataset_sha256="cd" * 32).finalize().pins()
    assert first_pin_difference(old, pinned) == "dataset_sha256"


def test_a_stale_shard_file_is_retired_when_the_whole_file_is_the_checked_unit(tmp_path):
    # covers: MODEL-I11
    # WOULD-FAIL-IF: with the whole file verified (recipe shard count other than the prepared one), a local
    # .shardKofN file whose bytes differ from that range of the whole file stays in place, and shard_view, which
    # prefers shard files, serves it after provision() reported success.
    src = _corpus(str(tmp_path / "src"))
    node = _node(str(tmp_path / "node"), src, ["tokens.bin", "tokens.bin.meta.json"])
    with open(node, "rb") as f:
        whole = f.read()
    half = 2000 * 2  # 2 logical shards of 2000 uint16 tokens
    stale = bytearray(whole[half:])
    stale[3] ^= 0xFF  # one byte off the whole file's shard-1 range
    with open(node + ".shard0of2", "wb") as f:
        f.write(whole[:half])  # equal to the whole file's shard-0 range
    with open(node + ".shard1of2", "wb") as f:
        f.write(bytes(stale))
    provision.provision(node, [0, 1], 2)
    assert os.path.exists(node + ".shard0of2"), (
        "a shard file equal to the whole file's range is kept"
    )
    assert not os.path.exists(node + ".shard1of2"), (
        "the stale shard file was left where shard_view reads it"
    )
    for mb in (0, 1):
        for a, b in zip(_batch(node, mb, 2), _batch(src, mb, 2)):
            assert torch.equal(a, b)


def test_provisioners_on_one_host_are_serialized(tmp_path, monkeypatch):
    # covers: MODEL-I11
    # WOULD-FAIL-IF: two slots on one host provision the same corpus at once: one's quarantine (os.replace to .bad)
    # can move aside the shard the other just fetched and verified, and that slot then refuses spuriously.
    import fcntl

    src = _corpus(str(tmp_path / "src"))
    node = str(tmp_path / "node" / "tokens.bin")
    held = []
    real_fetch = provision._fetch

    def fetch(*a, **kw):
        with open(node + ".provision.lock", "a") as other:
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held.append(False)
                fcntl.flock(other, fcntl.LOCK_UN)
            except BlockingIOError:
                held.append(True)
        return real_fetch(*a, **kw)

    monkeypatch.setattr(provision, "_fetch", fetch)
    provision.provision(node, [0, 1], 4, source=str(tmp_path / "src"))
    assert held and all(held), "a fetch ran while another provisioner could take the corpus lock"
    with open(node + ".provision.lock", "a") as after:
        fcntl.flock(after, fcntl.LOCK_EX | fcntl.LOCK_NB)  # released once provision() returns
    assert sorted(provision.provision(node, [0, 1], 4)["verified"]) == [0, 1]
    assert src

"""Dataset layer: any corpus, any tokenizer, still byte-exact.

A run reads a pretokenized token stream plus a manifest pinning tokenizer,
vocabulary size, element dtype, token count and the sha256 of the stream and of
each shard. A data-reading slot checks its manifest against the one the job was
planned with and verifies the shards it reads before it connects
(model/provision.py), so nodes that disagree fail at startup instead of
training on different data. Sharding is
layout-independent: the recipe names the logical shard each microbatch draws
from, and a whole file or a lone `<dataset>.shardKofN` slice serve the same
bytes for the same logical offset. uint16 wraps ids above 65535 (Llama-3 has
128256), so the manifest records the dtype and the loader refuses a mismatch.
"""

import hashlib
import json
import os

import numpy as np

DTYPES = {"uint16": np.uint16, "uint32": np.uint32}
_CACHE = {}


def dtype_for_vocab(vocab_size: int) -> str:
    """Smallest exact element type for this vocabulary."""
    return "uint16" if vocab_size <= 2**16 else "uint32"


def manifest_path(dataset: str) -> str:
    return f"{dataset}.meta.json"


def write_manifest(
    dataset: str,
    tokenizer: str,
    vocab_size: int,
    total_tokens: int,
    n_shards: int,
    dtype: str = None,
    extra: dict = None,
) -> dict:
    m = dict(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        total_tokens=total_tokens,
        n_shards=n_shards,
        dtype=dtype or dtype_for_vocab(vocab_size),
        **(extra or {}),
    )
    with open(manifest_path(dataset), "w") as f:
        json.dump(m, f, indent=2)
    return m


def read_manifest(dataset: str) -> dict:
    """Manifest, or a best-effort reconstruction for legacy uint16 files."""
    p = manifest_path(dataset)
    if os.path.exists(p):
        with open(p) as f:
            m = json.load(f)
        m.setdefault("dtype", "uint16")
        m.setdefault("n_shards", 1)
        m.setdefault("tokenizer", "unknown")
        return m
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"no dataset at {dataset!r} and no manifest at {p!r}")
    n = os.path.getsize(dataset) // 2
    return dict(
        tokenizer="unknown",
        vocab_size=None,
        total_tokens=n,
        n_shards=1,
        dtype="uint16",
        legacy=True,
    )


def check_compatible(dataset: str, model_cfg) -> dict:
    """Startup guard: refuse a corpus the model cannot represent."""
    m = read_manifest(dataset)
    v = m.get("vocab_size")
    if v is not None and v > model_cfg.vocab_size:
        raise ValueError(
            f"dataset {dataset!r} was tokenized with vocab_size={v} but the "
            f"model declares {model_cfg.vocab_size}; token ids would be out of "
            f"range. Rebuild the model config or retokenize the corpus"
        )
    want = dtype_for_vocab(v) if v else m["dtype"]
    if m["dtype"] != want:
        raise ValueError(
            f"dataset {dataset!r} stores {m['dtype']} but a vocabulary of {v} "
            f"needs {want}; ids above 65535 would wrap silently"
        )
    return m


def token_array(path: str, dtype: str):
    key = (path, dtype)
    if key not in _CACHE:
        _CACHE[key] = np.memmap(path, dtype=DTYPES[dtype], mode="r")
    return _CACHE[key]


def shard_view(dataset: str, shard: int, n_shards: int, total_tokens: int, dtype: str):
    """(array, base_offset) serving the logical range of `shard`.

    Physical layout does not affect which tokens are drawn: either the whole
    file is present or only `<dataset>.shard<K>of<N>` is, and both answer the
    same logical offset with the same bytes.
    """
    lo = (total_tokens * shard) // n_shards
    sp = f"{dataset}.shard{shard}of{n_shards}"
    if os.path.exists(sp):
        return token_array(sp, dtype), lo
    return token_array(dataset, dtype), 0


def file_digest(path: str, chunk: int = 1 << 20) -> str:
    """Content digest, so every node can prove it holds the same bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ preparation


def prepare(
    source: str,
    out: str,
    tokenizer: str = "gpt2",
    source_type: str = "auto",
    text_key: str = "text",
    n_shards: int = 1,
    limit_tokens: int = None,
    eot_token: int = None,
) -> dict:
    """Tokenize a corpus into a flat token stream plus a manifest.

    source_type: auto | jsonl | txt | hf   (hf = a HuggingFace dataset id)
    tokenizer:   a tiktoken encoding name, or an HF tokenizer id/path

    Imports are lazy so a node that only trains never needs the tokenizer stack.
    """
    enc, vocab_size = _load_tokenizer(tokenizer)
    if eot_token is None:
        eot_token = _eot(tokenizer)
    dtype = dtype_for_vocab(vocab_size)
    np_dtype = DTYPES[dtype]

    total = 0
    with open(out, "wb") as f:
        for text in _iter_texts(source, source_type, text_key):
            ids = enc(text)
            ids.append(eot_token)
            arr = np.asarray(ids, dtype=np_dtype)
            if arr.max(initial=0) >= vocab_size:
                raise ValueError("tokenizer produced an id outside its vocab")
            f.write(arr.tobytes())
            total += arr.size
            if limit_tokens and total >= limit_tokens:
                break

    extra = dict(
        sha256=file_digest(out), eot_token=eot_token, source=source, source_type=source_type
    )
    if n_shards > 1:
        extra["shard_sha256"] = shard_digests(out, total, n_shards, dtype)
    m = write_manifest(out, tokenizer, vocab_size, total, n_shards, dtype, extra=extra)
    if n_shards > 1:
        _write_shards(out, total, n_shards, dtype)
    return m


def shard_digests(path: str, total_tokens: int, n_shards: int, dtype: str) -> list:
    """sha256 of each logical shard's bytes, so a node can verify only the shards it reads (model/provision.py),
    from a shard file or from that range of the whole file alike."""
    isz = np.dtype(DTYPES[dtype]).itemsize
    out = []
    with open(path, "rb") as f:
        for k in range(n_shards):
            lo = (total_tokens * k) // n_shards
            hi = (total_tokens * (k + 1)) // n_shards
            f.seek(lo * isz)
            h, left = hashlib.sha256(), (hi - lo) * isz
            while left > 0:
                b = f.read(min(1 << 20, left))
                if not b:
                    break
                h.update(b)
                left -= len(b)
            out.append(h.hexdigest())
    return out


def _write_shards(out, total, n_shards, dtype):
    a = token_array(out, dtype)
    for k in range(n_shards):
        lo = (total * k) // n_shards
        hi = (total * (k + 1)) // n_shards
        p = f"{out}.shard{k}of{n_shards}"
        with open(p, "wb") as f:
            f.write(np.asarray(a[lo:hi]).tobytes())


def _load_tokenizer(name: str):
    """-> (encode_fn, vocab_size). tiktoken name first, then HF."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding(name)
        return (lambda t: enc.encode_ordinary(t)), enc.n_vocab
    except (ImportError, ValueError, KeyError):
        pass
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            f"tokenizer {name!r} is not a tiktoken encoding and transformers "
            f"is not installed; pip install tiktoken or transformers"
        ) from e
    base, rev = _hf_ref(name)
    # pinned via name@rev
    tok = AutoTokenizer.from_pretrained(base, revision=rev)
    return (lambda t: tok(t, add_special_tokens=False)["input_ids"]), len(tok)


def _hf_ref(name: str):
    """ "org/model@revision" -> (name, revision). A pinned revision is what makes
    "every node tokenized with the same tokenizer" true rather than likely:
    nodes tokenize independently, and an unpinned hub id can move between
    two nodes' downloads."""
    base, _, rev = name.partition("@")
    return base, (rev or None)


def _eot(name: str) -> int:
    try:
        import tiktoken

        return tiktoken.get_encoding(name).eot_token
    except Exception:
        from transformers import AutoTokenizer

        base, rev = _hf_ref(name)
        # name@rev
        tok = AutoTokenizer.from_pretrained(base, revision=rev)
        return tok.eos_token_id if tok.eos_token_id is not None else 0


def _iter_texts(source: str, source_type: str, text_key: str):
    if source_type == "auto":
        source_type = (
            "jsonl"
            if source.endswith((".jsonl", ".json"))
            else "txt"
            if os.path.exists(source)
            else "hf"
        )
    if source_type == "jsonl":
        with open(source) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)[text_key]
    elif source_type == "txt":
        with open(source) as f:
            yield f.read()
    elif source_type == "hf":
        from datasets import load_dataset

        name, _, split = source.partition(":")
        base, rev = _hf_ref(name)
        # pinned via name@rev; the corpus digest is compared
        ds = load_dataset(base, split=split or "train", streaming=True, revision=rev)
        for row in ds:
            yield row[text_key]
    else:
        raise ValueError(f"unknown source_type {source_type!r}")

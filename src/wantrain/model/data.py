"""Deterministic batches.

Every batch is a pure function of (data_seed, step, microbatch_index), so any
process (stage 0 for inputs, the last stage for targets, the baseline for both)
regenerates identical data with no broadcast. Synthetic tokens by default;
windows into a pretokenized token file when a dataset is given.
"""

import numpy as np
import torch


def make_batch(
    data_seed: int,
    step: int,
    mb: int,
    mb_size: int,
    seq_len: int,
    vocab_size: int,
    dataset: str = None,
    data_shards: int = 1,
):
    """Deterministic batch as a pure function of (seed, step, mb).

    dataset=None gives synthetic random tokens; dataset=path gives windows into
    a pretokenized token stream. data_shards > 1 is part of the recipe:
    microbatch mb draws only from logical shard (mb % data_shards), and the
    single-device baseline follows the same mapping, so distributed and
    baseline see identical data however the bytes are stored.
    """
    g = torch.Generator()
    g.manual_seed(data_seed * 1_000_003 + step * 1_009 + mb)
    if dataset is None:
        toks = torch.randint(0, vocab_size, (mb_size, seq_len + 1), generator=g)
    else:
        from wantrain.model import dataset as ds

        span = seq_len + 1
        meta = ds.read_manifest(dataset)
        total, dtype = meta["total_tokens"], meta["dtype"]
        if data_shards > 1:
            shard = mb % data_shards
            lo = (total * shard) // data_shards
            hi = (total * (shard + 1)) // data_shards
            arr, base = ds.shard_view(dataset, shard, data_shards, total, dtype)
            max_start = (hi - lo) - span - 1
            if max_start <= 0:
                raise ValueError(
                    f"shard {shard} of {data_shards} holds {hi - lo} tokens, too "
                    f"few for a {span}-token window; use fewer shards, a "
                    f"shorter seq_len, or more data"
                )
            starts = torch.randint(0, max_start, (mb_size,), generator=g)
            rows = [
                torch.from_numpy(arr[lo - base + s : lo - base + s + span].astype(np.int64))
                for s in starts.tolist()
            ]
        else:
            arr = ds.token_array(dataset, dtype)
            max_start = total - span - 1
            if max_start <= 0:
                raise ValueError(f"dataset holds {total} tokens, too few for a {span}-token window")
            starts = torch.randint(0, max_start, (mb_size,), generator=g)
            rows = [torch.from_numpy(arr[s : s + span].astype(np.int64)) for s in starts.tolist()]
        toks = torch.stack(rows)
    return toks[:, :-1].contiguous(), toks[:, 1:].contiguous()

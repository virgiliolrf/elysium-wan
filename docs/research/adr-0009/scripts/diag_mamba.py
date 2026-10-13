import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
# ADR-0009 round 1: root cause of the census DIFFER for mamba / falcon_mamba / jamba / phimoe / megatron-bert / xlm-roberta-xl.
import sys, os
D = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(D, "census.py")).read()
SRC = SRC.split("class TO(Exception)")[0].replace('os.path.dirname(os.path.abspath(__file__))', repr(D))
exec(SRC)
import torch
mode = os.environ.get("MODE", "")
if mode == "noconvert":
    wgrad.convert = lambda m: m
_orig_check = check
for mt in sys.argv[1:]:
    cfg = shrunk(mt)
    os.environ.pop("DBG", None)  # DBG indexes out[k], absent when the gradient is missing entirely
    r = check(mt, None, torch.float32, cfg=cfg)
    print(mt, mode or "default", {k: r[k] for k in ("logits", "grads", "differ")}, "first_bad", r["first_bad"])

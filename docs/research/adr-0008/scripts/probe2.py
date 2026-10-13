import torch
torch.manual_seed(1)
print("threads", torch.get_num_threads())
for dt in (torch.float32, torch.bfloat16):
    W = torch.randn(256, 512).to(dt); X = torch.randn(2048, 512).to(dt)
    full = X @ W.t(); iv = torch.int32 if dt == torch.float32 else torch.int16
    bad = [m for m in list(range(1, 33)) + [100, 255, 256, 257, 1023, 2047] if int((X[:m] @ W.t()).view(iv).ne(full[:m].view(iv)).sum())]
    # same rows at an offset (row position inside the GEMM)
    off = int(((X[5:69] @ W.t()).view(iv) != full[5:69].view(iv)).sum())
    print(dt, "M values whose rows differ from the M=2048 GEMM:", bad, "| offset slice diffs:", off)
# F.linear (what nn.Linear uses) M=1 vs batched
W = torch.randn(256, 512); X = torch.randn(64, 512)
a = torch.nn.functional.linear(X, W); b = torch.nn.functional.linear(X[:1], W)
print("F.linear M=1 vs batched row0 diffs:", int((a[:1] != b).sum()))
# 3-D input (b,t,h) vs flattened
X3 = X.view(4, 16, 512); c = torch.nn.functional.linear(X3, W).reshape(64, 256)
print("F.linear 3D vs 2D diffs:", int((c != a).sum()))

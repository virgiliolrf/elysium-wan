import os, sys  # added on commit (README.md)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
# Derivation: is SplitLinear bit-identical to stock nn.Linear (fwd, dX, dW, db)?
import sys, torch, torch.nn as nn, torch.nn.functional as F

from wantrain.model import wgrad
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
res = {}
for dt in (torch.float32, torch.bfloat16):
  for (b,t,din,dout,bias) in [(2,64,128,384,False),(4,128,256,688,True),(1,512,512,512,False),(3,37,96,200,True)]:
    for ac in (False, True):
      if ac and dt is torch.float32: continue
      torch.manual_seed(0)
      wdt = torch.float32 if ac else dt
      a = nn.Linear(din,dout,bias=bias).to(wdt); s = nn.Linear(din,dout,bias=bias).to(wdt)
      s.load_state_dict(a.state_dict()); wgrad.convert(nn.ModuleList([s]))
      x = torch.randn(b,t,din,dtype=wdt); g = torch.randn(b,t,dout,dtype=dt)
      outs=[]
      for m in (a,s):
        xx = x.clone().requires_grad_(True)
        with torch.autocast("cpu", dtype=dt, enabled=ac):
          y = m(xx)
        gx, gw, *gb = torch.autograd.grad(y, [xx, m.weight] + ([m.bias] if bias else []), g)
        outs.append((y, gx, gw, *gb))
      key=f"{dt}-ac{ac}-{b}x{t}x{din}->{dout}-bias{bias}"
      res[key]=[torch.equal(u,v) for u,v in zip(*outs)]
for k,v in res.items(): print(k, "y,dX,dW,db equal:", v)

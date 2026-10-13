"""The dense GPT shape a WAN fleet runs best (SCHEDULER-I23): the square-cube law, priced from measured figures.

    python benchmarks/wan/advise.py --stages 3 --mem-gib 79 --tflops 99 --mbps 45 [--bubble 0.2] [--microbatches 32]
--tflops is the engine's effective rate on the stage GPU (99 on an H100 NVL, 2026-10-12: a stage's F+B+W time against
its FLOPs); --mbps the slowest hop's per-direction rate as the coordinator measured it; --mem-gib the fit budget.
"""

import argparse

from wantrain.scheduler.shape import advise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, required=True)
    ap.add_argument("--mem-gib", type=float, required=True)
    ap.add_argument("--tflops", type=float, required=True)
    ap.add_argument("--mbps", type=float, required=True)
    ap.add_argument("--bubble", type=float, default=0.2)
    ap.add_argument("--microbatches", type=int, default=32)
    a = ap.parse_args()
    r = advise(
        a.stages,
        a.mem_gib * 2**30,
        tflops=a.tflops,
        link_mbps=a.mbps,
        bubble_target=a.bubble,
        microbatches=a.microbatches,
    )
    top = sorted(r["fits"], key=lambda s: (s.bubble > a.bubble, -s.params))[:8]
    print("d_model  n_head  n_layer  per_stage  params_B  compute_ms  wire_ms  bubble  stage_GiB")
    for s in top:
        print(
            f"{s.d_model:7d} {s.n_head:7d} {s.n_layer:8d} {s.layers_per_stage:10d} {s.params / 1e9:9.1f} "
            f"{s.compute_s * 1e3:11.0f} {s.transfer_s * 1e3:8.0f} {s.bubble:7.2f} {s.stage_bytes / 2**30:10.1f}"
        )
    b = r["best"]
    print(
        f"BEST d_model {b.d_model}, n_head {b.n_head}, n_layer {b.n_layer} ({b.params / 1e9:.1f}B), "
        f"priced bubble {b.bubble:.0%}{'' if r['met_target'] else ' (target not met)'}"
    )


if __name__ == "__main__":
    main()

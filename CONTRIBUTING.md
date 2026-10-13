# Contributing to Elysium

Thanks for your interest in Elysium. It's early research software, and the most useful help right now is in kernel
determinism, WAN transport, pipeline scheduling, heterogeneous backends and fault tolerance.

## Before you start

For anything bigger than a small fix, please open an issue first so we can agree on the approach. Changes to
numerics, schedules or the transport usually touch an invariant, and that's easier to settle before the code is
written.

## Setting up

You need Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/virgiliolrf/elysium-wan.git
cd elysium-wan
uv sync          # also installs pytest, ruff, mypy and hypothesis
```

## Checking your change

```bash
uv run ruff format src tests
uv run ruff check src tests
uv run pytest -q tests/unit
uv run pytest -q tests/integration/test_engine_bitwise.py
uv run pytest -q tests/chaos tests/oracle tests/wan
```

The bitwise suite runs real multi-process pipelines and compares each one with a single-device reference, bit for
bit. A pull request has to keep it green.

## How we work

- **Every behavior has a numbered invariant.** For example, RUNTIME-I27 lives in a SPEC document under
  [docs/design/](docs/design/), and the code and tests that implement it cite it. When a SPEC changes, append its
  sha256 and the reason to [docs/design/FROZEN.json](docs/design/FROZEN.json).
- **Big decisions get an ADR.** See [docs/design/adr/](docs/design/adr/).
- **Tests say what they cover and what would make them fail,** in `covers:` and `WOULD-FAIL-IF:` comment lines.
- **A speedup has to keep the bits.** If it changes the numerics, it's a new recipe, and the single-device reference
  has to run it too.
- **Measure before you theorize.** GPU runs leave per-operation telemetry. [benchmarks/wan/](benchmarks/wan/) has the
  tools to turn it into charts, and the run reports show how each fix came out of it.

## Pull requests

Keep each pull request to one change. Describe what it changes, which invariants it touches, and how you tested it. If
it affects performance, include the measurement and how you made it.

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE), the license
of this project.

## Code of conduct and security

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md). Please report security problems privately, as
described in [SECURITY.md](SECURITY.md).

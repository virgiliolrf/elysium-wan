# Security policy

Elysium moves model state and training data between machines that may belong to different people, so we take security
reports seriously.

## Reporting a vulnerability

Please don't open a public issue for a security problem. Report it privately instead: open this repository's
**Security** tab and choose **Report a vulnerability**. We'll acknowledge the report, work on a fix with you, and
credit you in the fix unless you'd rather stay anonymous.

## What's in scope

The threat model is in [docs/design/THREAT_MODEL.md](docs/design/THREAT_MODEL.md). In scope:
- **Peer authentication and the sealed data plane:** ed25519 identities and AES-256-GCM links between stages (ChaCha20-Poly1305 when either end lacks hardware AES).
- **The coordinator's control channel and its token handling.**
- **Checkpoint integrity:** waves are sha256-verified.
- **Anything else that lets a worker or a network attacker change training results, or read data it shouldn't.**

## Before you run it on machines you don't control

Elysium is research software. Some of the hardening described in the threat model is still planned:
- moving encryption off the critical path;
- sealing only the links that cross sites;
- auditing against malicious workers.

Until then, run it on machines and networks you trust.

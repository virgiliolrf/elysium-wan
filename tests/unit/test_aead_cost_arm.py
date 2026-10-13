"""The AEAD cost measurement's per-arm launcher is executable by its owner only.

covers: test infrastructure of TRANSPORT-I7's cost measurement (validation/aead_cost.py)
WOULD-FAIL-IF: the launcher were group- or world-accessible (bandit B103) or not
executable by its owner (the measured stages could not start).
"""

import os
import stat

from wantrain.validation import aead_cost


def test_arm_launcher_is_owner_only_and_executable(tmp_path):
    path = aead_cost._arm_python(str(tmp_path), "sealed")
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o700
    assert os.access(path, os.X_OK)

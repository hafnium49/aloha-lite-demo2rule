# tests/test_demo2rules.py
import os, sys

# Make the project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

# Import the updated helpers / templates
from demo2rules import (
    detect_segments_from_matrix,
    build_extractors,
    RULE_TEMPLATE,
)

# --------------------------------------------------------------------------- #
# 1.  detect_segments_from_matrix
# --------------------------------------------------------------------------- #
def test_detect_segments_from_matrix():
    """
    Synthetic trajectory:
        – t = 0‒0.5 s  : moving
        – t = 1.0‒2.0 s: static   (plateau should start here)
        – t = 2.5 s    : moving   (ends at final index)
    We expect a single segment whose *start* index is inside the plateau and
    whose *end* index is the final frame (5).
    """
    ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    mat = np.array(
        [
            [0.0, 0.0],  # moving
            [0.5, 0.0],  # moving
            [0.5, 0.0],  # static
            [0.5, 0.0],  # static
            [0.5, 0.0],  # static
            [1.0, 0.0],  # moving again
        ]
    )
    segments = detect_segments_from_matrix(ts, mat, vel_thresh=0.1, window=1)
    # Should detect exactly one plateau finishing at last frame
    assert len(segments) == 1
    start, end = segments[0]
    assert end == len(ts) - 1
    assert start < end


# --------------------------------------------------------------------------- #
# 2.  build_extractors
# --------------------------------------------------------------------------- #
def test_build_extractors():
    colnames = [
        "observation.state | motor_0",
        "observation.state | motor_1",
        "observation.state | motor_0_secondary",
        "observation.state | motor_1_secondary",
        "gripper_open | action",
        "gripper_open_secondary | action",
    ]

    extract_left, extract_right, extract_misc = build_extractors(colnames)

    row = np.array([0.1, 0.2, 0.3, 0.4, 1.0, 0.0])
    left_j = extract_left(row)
    right_j = extract_right(row)
    misc = extract_misc(row)

    assert left_j == [0.1, 0.2]
    assert right_j == [0.3, 0.4]
    assert misc["left_grip"] is True
    assert misc["right_grip"] is False


# --------------------------------------------------------------------------- #
# 3.  RULE_TEMPLATE placeholder integrity
# --------------------------------------------------------------------------- #
def test_rule_template_filling():
    filled = RULE_TEMPLATE.format(
        idx=0,
        next_idx=1,
        left_j=[1, 2, 3],
        right_move_j="",
        left_cart_block="",
        right_cart_block="",
        grip_block="# grips",
    )
    # Basic sanity checks on the rendered snippet
    assert "stage_0" in filled
    assert "@when_all(m.stage == 0)" in filled
    assert "left_arm.move_j([1, 2, 3])" in filled

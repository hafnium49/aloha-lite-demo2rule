# tests/test_demo2rules.py
"""
Test suite for demo2rules.py

Includes original functionality tests plus additional edge case tests
that cover issues discovered and fixed during debugging:

1. Data structure handling (object dtype columns with nested arrays)
2. Matrix dimension issues (1D vs 2D arrays)
3. Empty or single-column matrices
4. Mixed data types and conversion edge cases
5. Very short time sequences
6. Missing or partial column matches in extractors

These tests ensure robustness against various LeRobot dataset formats
and prevent regression of the debugging fixes.
"""

import os, sys

# Make the project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

# Import the updated helpers / templates
from demo2rules import (
    detect_segments_from_matrix,
    build_extractors,
    load_pose_matrix,
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


# --------------------------------------------------------------------------- #
# 4.  Edge cases discovered during debugging
# --------------------------------------------------------------------------- #
def test_detect_segments_empty_matrix():
    """
    Test handling of empty matrices (no columns) - should return single segment.
    This was an edge case discovered during debugging.
    """
    ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    mat = np.empty((5, 0))  # Empty matrix with no columns
    segments = detect_segments_from_matrix(ts, mat, vel_thresh=0.1, window=3)
    assert len(segments) == 1
    assert segments[0] == (0, len(ts) - 1)


def test_detect_segments_1d_matrix():
    """
    Test handling of 1D matrices that need reshaping to 2D.
    This was the root cause of the "axis 1 is out of bounds" error.
    """
    ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # 1D array with plateau then movement
    segments = detect_segments_from_matrix(ts, mat, vel_thresh=0.1, window=3)
    # Should handle reshaping without error (main goal is no crash)
    assert isinstance(segments, list)


def test_detect_segments_single_column():
    """
    Test handling of single-column matrices.
    """
    ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    mat = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])  # Single column with plateau
    segments = detect_segments_from_matrix(ts, mat, vel_thresh=0.1, window=3)
    # Should handle single column without error
    assert isinstance(segments, list)


def test_detect_segments_mixed_data_types():
    """
    Test that the segment detection works with different numeric data types.
    This tests the float64 conversion that was added during debugging.
    """
    ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    # Mix of int and float data with clear plateau
    mat = np.array([
        [0, 0.0],
        [0, 0.0], 
        [0, 0.0],
        [0, 0.0],
        [0, 0.0],
        [1, 1.0]  # Movement at the end
    ])
    segments = detect_segments_from_matrix(ts, mat, vel_thresh=0.1, window=3)
    # Main goal is no crash due to data type issues
    assert isinstance(segments, list)


def test_build_extractors_no_matches():
    """
    Test extractor building when no matching columns are found.
    This tests the robustness improvements made during debugging.
    """
    colnames = ["timestamp", "frame_index", "some_other_column"]
    extract_left, extract_right, extract_misc = build_extractors(colnames)
    
    row = np.array([1.0, 2.0, 3.0])
    left_j = extract_left(row)
    right_j = extract_right(row)
    misc = extract_misc(row)
    
    assert left_j == []
    assert right_j == []
    assert misc == {}


def test_build_extractors_partial_matches():
    """
    Test extractor building with only some expected columns present.
    """
    colnames = [
        "observation.state | motor_0",
        "observation.state | motor_1",
        "some_other_column",
        "gripper_open | action",
    ]
    extract_left, extract_right, extract_misc = build_extractors(colnames)
    
    row = np.array([0.1, 0.2, 999.0, 1.0])
    left_j = extract_left(row)
    right_j = extract_right(row)
    misc = extract_misc(row)
    
    assert left_j == [0.1, 0.2]
    assert right_j == []  # No secondary motors
    assert misc["left_grip"] is True
    assert "right_grip" not in misc  # No secondary gripper


def test_detect_segments_very_short_sequence():
    """
    Test handling of very short time sequences.
    This tests the len(ts) < 2 check that prevents errors.
    """
    # Single frame
    ts = np.array([0.0])
    mat = np.array([[0.0, 0.0]])
    segments = detect_segments_from_matrix(ts, mat, vel_thresh=0.1, window=3)
    assert segments == []
    
    # Two frames - should not crash, may or may not detect segments
    ts = np.array([0.0, 0.5])
    mat = np.array([[0.0, 0.0], [0.1, 0.1]])
    segments = detect_segments_from_matrix(ts, mat, vel_thresh=0.1, window=3)
    assert isinstance(segments, list)  # Just ensure it doesn't crash


@pytest.mark.skipif(
    True,  # Skip by default since this requires network access
    reason="Requires network access to Hugging Face"
)
def test_load_pose_matrix_integration():
    """
    Integration test for the data structure handling fixes.
    This would test the actual fix for object-dtype columns with nested arrays.
    """
    # This test would use a real dataset to verify the fixes
    # Skip by default to avoid network dependencies in CI
    ts, mat, colnames, info = load_pose_matrix("Hafnium49/aloha_lite", 0)
    assert ts.shape[0] > 0
    assert mat.ndim == 2
    assert mat.shape[0] == ts.shape[0]
    assert len(colnames) == mat.shape[1]
    assert mat.dtype == np.float64

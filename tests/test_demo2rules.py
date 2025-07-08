import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from demo2rules import detect_segments, summarise, generate_rules

def test_detect_segments():
    ds = [
        {'qdot': [0.1, 0, 0], 'q': np.zeros(3)},
        {'qdot': [0.1, 0, 0], 'q': np.zeros(3)},
        {'qdot': [0.0, 0, 0], 'q': np.zeros(3)},
        {'qdot': [0.0, 0, 0], 'q': np.zeros(3)},
        {'qdot': [0.0, 0, 0], 'q': np.zeros(3)},
        {'qdot': [0.1, 0, 0], 'q': np.zeros(3)},
        {'qdot': [0.1, 0, 0], 'q': np.zeros(3)},
        {'qdot': [0.0, 0, 0], 'q': np.zeros(3), 'gripper_open': True},
        {'qdot': [0.0, 0, 0], 'q': np.zeros(3)},
        {'qdot': [0.0, 0, 0], 'q': np.zeros(3)},
    ]
    segments = detect_segments(ds, vel_thresh=0.05, window=2)
    segs = [tuple(int(x) for x in s) for s in segments]
    assert segs == [(3, 8)]

def test_summarise():
    ds = [
        {'q': np.array([0, 0, 0]), 'qdot': [0.1, 0, 0]},
        {'q': np.array([1, 1, 1]), 'qdot': [0.1, 0, 0]},
        {'q': np.array([2, 2, 2]), 'qdot': [0.0, 0, 0]},
    ]
    summary = summarise(ds, 0, 2)
    assert summary == {'pose': [2, 2, 2], 'grip': False}

def test_generate_rules():
    specs = [
        {'pose': [1, 2, 3], 'grip': True},
        {'pose': [4, 5, 6], 'grip': False},
    ]
    rules = generate_rules(specs)
    assert "@when_all(m.stage == 0)" in rules
    assert "@when_all(m.stage == 1)" in rules
    assert "def _done" in rules

#!/usr/bin/env python3
"""
Convert a LeRobot v2 Parquet episode into a durable_rules script that re‑plays
the demonstration on one or two SO‑101 arms via the Phosphobot SDK.

The column‑parsing logic follows the HTML Dataset Visualizer:
  • download meta/info.json
  • ignore non‑pose keys (timestamp, frame_index, …)
  • accept numeric 1‑D features
  • group by shared suffix after " | "
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

###############################################################################
# 1. Utilities for Hugging Face datasets
###############################################################################

DEFAULT_DS_BASE = "https://huggingface.co/datasets"
SERIES_DELIM = " | "               # identical to the visualizer


def fetch_json(url: str):
    import requests

    r = requests.get(url)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        if r.status_code == 401:
            raise RuntimeError(
                f"Unauthorized when fetching {url}. "
                "Check that the dataset slug is correct or that the repo is public."
            ) from e
        raise
    return r.json()


def download_parquet(url: str, cache_dir: Path) -> Path:
    """
    Stream a remote parquet file into the local cache and return the Path.
    Uses SHA‑1 of the URL for the filename.
    """
    import requests
    BUF = 1 << 20

    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".parquet")
    if fname.exists():
        return fname

    with requests.get(url, stream=True) as r, open(fname, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=BUF):
            f.write(chunk)
    return fname


def load_episode_table(
    repo_id: str,
    episode_id: int,
    base_url: str = DEFAULT_DS_BASE,
    cache_dir: Path | None = None,
):
    """
    Returns a pyarrow.Table for the requested episode (single Parquet shard).
    """
    import pyarrow.parquet as pq

    cache_dir = cache_dir or Path.home() / ".cache" / "demo2rules_parquet"
    info_url = f"{base_url}/{repo_id}/resolve/main/meta/info.json"
    info = fetch_json(info_url)

    # In the LeRobot spec each Parquet file stores 1k frames by default
    episode_chunk = episode_id // 1000
    parquet_relpath = info["data_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_id,
    )
    parquet_url = f"{base_url}/{repo_id}/resolve/main/{parquet_relpath}"
    fpath = download_parquet(parquet_url, cache_dir)
    return pq.read_table(fpath), info


def numeric_feature_keys(info: Dict) -> List[Tuple[str, int]]:
    """
    Returns [(column_key, length)] for all 1‑D numeric features,
    *excluding* timestamp + housekeeping fields.
    """
    EXCLUDE = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    out: List[Tuple[str, int]] = []
    for key, spec in info["features"].items():
        if key in EXCLUDE:
            continue
        if spec["dtype"] in ("float32", "int32") and len(spec["shape"]) == 1:
            out.append((key, spec["shape"][0]))
    return out


def expand_column_names(key: str, spec: Dict) -> List[str]:
    """
    Re‑implements the JS helper that adds the prefix (key) to each sub‑name
    using the SERIES_DELIM delimiter, *or* falls back to index numbers.
    """
    names = spec["names"]
    if isinstance(names, list):
        return [f"{key}{SERIES_DELIM}{n}" for n in names]
    if isinstance(names, dict):
        # find deepest list
        while isinstance(names, dict):
            names = next(iter(names.values()))
        if isinstance(names, list):
            return [f"{key}{SERIES_DELIM}{n}" for n in names]
    # fallback: numeric indices
    length = spec["shape"][0]
    return [f"{key}{SERIES_DELIM}{i}" for i in range(length)]


###############################################################################
# 2. Build a 2‑D numpy array [T, D] of pose‑related columns
###############################################################################

def load_pose_matrix(repo_id: str, episode_id: int):
    import numpy as np
    table, info = load_episode_table(repo_id, episode_id)
    keys_and_lens = numeric_feature_keys(info)

    # Build the final column name list (timestamp first for convenience)
    timestamp = table["timestamp"].to_numpy()
    colnames: List[str] = []
    arrays: List[np.ndarray] = []

    for key, _ in keys_and_lens:
        spec = info["features"][key]
        expanded = expand_column_names(key, spec)
        col = table[key].to_numpy()        # shape (T, len(expanded))
        
        # Handle PyArrow array conversion
        if hasattr(col, 'to_pylist'):
            col = np.array(col.to_pylist())
        
        # Handle different data structures
        if col.dtype == object:
            # If the column contains arrays/lists, convert to 2D numeric array
            try:
                # Try to stack the sequences into a 2D array
                col = np.stack(col)
            except (ValueError, TypeError):
                # If stacking fails, try to convert each element
                col_list = []
                for item in col:
                    if hasattr(item, '__iter__') and not isinstance(item, str):
                        col_list.append(np.array(item, dtype=np.float64))
                    else:
                        col_list.append(np.array([float(item)]))
                col = np.stack(col_list)
        
        # Ensure col is 2D and numeric
        if col.ndim == 1:
            col = col.reshape(-1, 1)
        
        # Convert to float64, handling potential conversion issues
        try:
            col = col.astype(np.float64)
        except (ValueError, TypeError):
            # If direct conversion fails, flatten and convert
            col = np.array([[float(x) for x in row] if hasattr(row, '__iter__') and not isinstance(row, str) 
                           else [float(row)] for row in col])
        
        colnames.extend(expanded)
        arrays.append(col)

    # Concatenate horizontally
    if arrays:
        pose_mat = np.hstack(arrays)           # (T, D)
        # Ensure pose_mat is 2D and numeric
        if pose_mat.ndim == 1:
            pose_mat = pose_mat.reshape(-1, 1)
        pose_mat = pose_mat.astype(np.float64)
    else:
        # No numeric features found
        pose_mat = np.empty((len(timestamp), 0), dtype=np.float64)
    
    return timestamp, pose_mat, colnames, info


###############################################################################
# 3. Segment detection (velocity plateau + dt)
###############################################################################

def detect_segments_from_matrix(
    ts: Sequence[float],
    mat: "np.ndarray",        # noqa: F821 – type only
    vel_thresh: float = 0.03,
    window: int = 15,
) -> List[Tuple[int, int]]:
    """
    mat: joint (or any) time‑series, shape (T, D)
    """
    import numpy as np

    if len(ts) < 2:
        return []
    
    # Ensure mat is 2D
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    
    # If matrix is empty (no columns), return a single segment for the entire sequence
    if mat.shape[1] == 0:
        return [(0, len(ts) - 1)]
    
    dt = float(np.median(np.diff(ts)))     # seconds

    # finite diff → speed per frame (L2 over dims)
    diff_mat = np.diff(mat, axis=0) / dt
    if diff_mat.ndim == 1:
        diff_mat = diff_mat.reshape(-1, 1)
    
    # Ensure diff_mat is float type for norm calculation
    diff_mat = diff_mat.astype(np.float64)
    
    vel = np.linalg.norm(diff_mat, axis=1, ord=2)
    vel = np.hstack([[0.0], vel])          # align shape to T

    half = window // 2
    static = np.array(
        [
            vel[max(0, i - half) : i + half + 1].max() < vel_thresh
            for i in range(len(vel))
        ],
        dtype=bool,
    )

    edges = [i + 1 for i in range(len(static) - 1) if not static[i] and static[i + 1]]
    if not edges or edges[-1] != len(static) - 1:
        edges.append(len(static) - 1)
    return [(edges[k - 1], edges[k]) for k in range(1, len(edges))]


###############################################################################
# 4. Heuristics to extract joint arrays / eef pose / gripper state
###############################################################################

MOTOR_RE = re.compile(r"motor_(\d+)(?:_secondary)?")
GRIPPER_RE = re.compile(r"gripper(_open|_open_secondary)?\s*\|\s*action", re.I)
EEF_RE = re.compile(r"(ee|eef|tcp|pose).*(x|y|z|rx|ry|rz)", re.I)

def split_left_right(colnames: List[str]) -> Tuple[List[str], List[str]]:
    left, right = [], []
    for c in colnames:
        if "_secondary" in c or "secondary" in c or "right" in c:
            right.append(c)
        else:
            left.append(c)
    return left, right


def build_extractors(colnames: List[str]):
    """
    Returns three callables that, given a row vector, return:
        • left_joints (List[float])
        • right_joints (List[float])
        • misc  (dict)
    """
    import numpy as np

    # ---- joints
    motor_cols = {
        c: MOTOR_RE.search(c).group(1)
        for c in colnames
        if MOTOR_RE.search(c) and "observation.state" in c
    }
    left_cols, right_cols = split_left_right(list(motor_cols.keys()))
    left_cols.sort(key=lambda x: int(motor_cols[x]))
    right_cols.sort(key=lambda x: int(motor_cols[x]))

    left_idx = [colnames.index(c) for c in left_cols]
    right_idx = [colnames.index(c) for c in right_cols]

    # ---- gripper
    grip_cols = [c for c in colnames if GRIPPER_RE.search(c)]
    grip_left = next((c for c in grip_cols if "secondary" not in c and "right" not in c), None)
    grip_right = next((c for c in grip_cols if c != grip_left), None)
    grip_left_idx = colnames.index(grip_left) if grip_left else None
    grip_right_idx = colnames.index(grip_right) if grip_right else None

    # ---- eef pose (x,y,z,rx,ry,rz)
    eef_cols = [c for c in colnames if EEF_RE.search(c)]
    eef_left, eef_right = split_left_right(eef_cols)
    eef_order = ["x", "y", "z", "rx", "ry", "rz"]

    def order_pose(cols: List[str]) -> List[int]:
        ordered = []
        for name in eef_order:
            for c in cols:
                if f"{SERIES_DELIM}{name}" in c:
                    ordered.append(colnames.index(c))
                    break
        return ordered

    eef_left_idx = order_pose(eef_left)
    eef_right_idx = order_pose(eef_right)

    # ------------------------------------------------------------------ #
    def extract_left(row: "np.ndarray"):  # noqa: F821
        return list(row[left_idx]) if left_idx else []

    def extract_right(row: "np.ndarray"):
        return list(row[right_idx]) if right_idx else []

    def extract_misc(row: "np.ndarray"):
        misc = {}
        if eef_left_idx:
            misc["left_cart"] = list(row[eef_left_idx])
        if eef_right_idx:
            misc["right_cart"] = list(row[eef_right_idx])
        if grip_left_idx is not None:
            misc["left_grip"] = bool(row[grip_left_idx] > 0.5)
        if grip_right_idx is not None:
            misc["right_grip"] = bool(row[grip_right_idx] > 0.5)
        return misc

    return extract_left, extract_right, extract_misc


###############################################################################
# 5. Code generation
###############################################################################

HEADER = '''"""
AUTOGENERATED by demo2rules.py
Dataset : {repo_id}  (episode {episode})
SHA‑256 : {sha}
Params  : {params}
"""
'''

RULESET_TEMPLATE = """
from durable.lang import *
from phosphobot import SkillClient

left_arm  = SkillClient("{left_arm}")
{right_arm_decl}

with ruleset("{ruleset_name}"):

    @when_all(+m.ready)
    def _start(c):
        c.post({{"stage": 0}})

{rules}

    @when_all(m.stage == {n_rules})
    def _done(c):
        print("🎉  demo finished")
"""

RULE_TEMPLATE = """
    @when_all(m.stage == {idx})
    def stage_{idx}(c):
        # joints
        left_arm.move_j({left_j})
{right_move_j}
        # cartesian (optional)
{left_cart_block}{right_cart_block}
        # grippers
{grip_block}
        c.post({{"stage": {next_idx}}})
"""


def sha_file(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()[:10]


###############################################################################
# 6. Main pipeline
###############################################################################

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="org/dataset (HF repo‑id)")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--out", default="rules_autogen.py")
    parser.add_argument("--vel-thresh", type=float, default=0.03)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--left-arm", default="so101_left")
    parser.add_argument("--right-arm", default="so101_right")
    parser.add_argument("--ruleset-name", default="so101_dual")
    parser.add_argument("--single-arm", action="store_true")
    args = parser.parse_args(argv)

    repo_id = args.dataset
    ts, mat, colnames, info = load_pose_matrix(repo_id, args.episode)
    segs = detect_segments_from_matrix(ts, mat, args.vel_thresh, args.window)
    if not segs:
        print("⚠️  No segments detected; aborting.", file=sys.stderr)
        sys.exit(1)

    # build column extractors
    extract_left, extract_right, extract_misc = build_extractors(colnames)

    # summarise each segment (use the *end* column index)
    import numpy as np
    rulespecs = []
    for a, b in segs:
        row = mat[b]
        spec = {
            "left_j": extract_left(row),
            "right_j": extract_right(row),
        }
        spec.update(extract_misc(row))
        rulespecs.append(spec)

    # ------------------------------------------------------------------ #
    # Generate durable_rules source
    rules_src_blocks = []
    for i, spec in enumerate(rulespecs):
        next_idx = i + 1
        right_move_j = (
            f"        right_arm.move_j({spec['right_j']})"
            if spec["right_j"] and not args.single_arm
            else ""
        )

        # cartesian blocks
        left_cart_block = ""
        if "left_cart" in spec:
            left_cart_block = (
                f'        left_arm.move_cart({{"x": {spec["left_cart"][0]}, "y": {spec["left_cart"][1]}, '
                f'"z": {spec["left_cart"][2]}, "rx": {spec["left_cart"][3]}, "ry": {spec["left_cart"][4]}, '
                f'"rz": {spec["left_cart"][5]}}})\n'
            )
        right_cart_block = ""
        if "right_cart" in spec and not args.single_arm:
            right_cart_block = (
                f'        right_arm.move_cart({{"x": {spec["right_cart"][0]}, "y": {spec["right_cart"][1]}, '
                f'"z": {spec["right_cart"][2]}, "rx": {spec["right_cart"][3]}, "ry": {spec["right_cart"][4]}, '
                f'"rz": {spec["right_cart"][5]}}})\n'
            )

        # gripper
        grip_cmds = []
        if "left_grip" in spec:
            grip_cmds.append(f'        left_arm.grip(open={str(spec["left_grip"]).lower()})')
        if "right_grip" in spec and not args.single_arm:
            grip_cmds.append(f'        right_arm.grip(open={str(spec["right_grip"]).lower()})')
        grip_block = "\n".join(grip_cmds) if grip_cmds else "        # no gripper info"

        rules_src_blocks.append(
            RULE_TEMPLATE.format(
                idx=i,
                next_idx=next_idx,
                left_j=spec["left_j"],
                right_move_j=right_move_j,
                left_cart_block=left_cart_block,
                right_cart_block=right_cart_block,
                grip_block=grip_block,
            )
        )

    right_decl = "" if args.single_arm else f'right_arm = SkillClient("{args.right_arm}")'
    whole_script = (
        HEADER.format(
            repo_id=repo_id,
            episode=args.episode,
            sha=sha_file(Path(__file__)),
            params=json.dumps(vars(args), separators=(",", ":")),
        )
        + RULESET_TEMPLATE.format(
            left_arm=args.left_arm,
            right_arm_decl=right_decl,
            ruleset_name=args.ruleset_name,
            rules="".join(rules_src_blocks),
            n_rules=len(rulespecs),
        )
    )

    Path(args.out).write_text(whole_script)
    print(f"✅  Wrote {len(rulespecs)} rules → {args.out}")


if __name__ == "__main__":
    main()

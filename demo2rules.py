import argparse
from pathlib import Path



def detect_segments(ds, vel_thresh=0.03, window=15):
    import numpy as np

    T = len(ds)
    velocities = np.linalg.norm([ds[i]["qdot"] for i in range(T)], axis=1)
    is_static = velocities < vel_thresh
    plateau = np.convolve(is_static.astype(int), np.ones(window, dtype=int), mode="same") == window
    edges = np.flatnonzero(np.diff(plateau.astype(int)) == 1) + 1
    if len(edges) == 0:
        return []
    return list(zip(edges, edges[1:]))


def summarise(ds, k0, k1):
    q_end = ds[k1]["q"]
    gripper = ds[k1].get("gripper_open", False)
    return {"pose": q_end.tolist(), "grip": bool(gripper)}


def generate_rules(rulespecs):
    lines = []
    lines.append("from durable.lang import *")
    lines.append("")
    lines.append("with ruleset('so101'):")
    lines.append("")
    lines.append("    @when_all(+m.ready)")
    lines.append("    def _start(c): c.post({'stage': 0})")
    lines.append("")
    for idx, spec in enumerate(rulespecs):
        lines.append(f"    @when_all(m.stage == {idx})")
        lines.append("    def _rule(c, spec=spec, idx=idx):")
        lines.append("        arm.move_j(spec['pose'])")
        lines.append("        arm.grip(open=spec['grip'])")
        lines.append("        c.post({'stage': idx + 1})")
        lines.append("")
    lines.append(f"    @when_all(m.stage == {len(rulespecs)})")
    lines.append("    def _done(c): print('🎉 Finished demo!')")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Convert a LeRobot demo to durable_rules code.')
    parser.add_argument('--dataset', required=True, help='Path to LeRobot dataset directory or HF repo ID')
    parser.add_argument('--out', default='rules_autogen.py', help='Output file for generated rules')
    parser.add_argument('--vel-thresh', type=float, default=0.03, help='Velocity threshold for static detection')
    parser.add_argument('--window', type=int, default=15, help='Window size for static detection')
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.dataset)
    segments = detect_segments(ds, args.vel_thresh, args.window)
    rulespecs = [summarise(ds, a, b) for a, b in segments]

    rules_text = generate_rules(rulespecs)
    Path(args.out).write_text(rules_text)
    print(f"Wrote {len(rulespecs)} rules to {args.out}")


if __name__ == '__main__':
    main()

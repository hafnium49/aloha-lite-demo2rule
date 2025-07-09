import argparse
from pathlib import Path



def detect_segments(ds, vel_thresh=0.03, window=15):
    from math import sqrt

    T = len(ds)
    qdots = []
    for i in range(T):
        item = ds[i]
        if "qdot" in item:
            qdot = list(item["qdot"])
        elif "dq" in item:
            qdot = list(item["dq"])
        elif "q" in item and i > 0 and "q" in ds[i-1]:
            prev_q = ds[i-1]["q"]
            qdot = [item["q"][j] - prev_q[j] for j in range(len(item["q"]))]
        elif "observation.state" in item and i > 0 and "observation.state" in ds[i-1]:
            state = item["observation.state"]
            prev_state = ds[i-1]["observation.state"]
            half = len(state) // 2
            qdot = [state[j] - prev_state[j] for j in range(half)]
        else:
            length = len(item.get("q", item.get("observation.state", [0])))
            qdot = [0.0] * length
        qdots.append(qdot)

    velocities = [sqrt(sum(v * v for v in qd)) for qd in qdots]

    half = window // 2
    plateau = []
    for i in range(T):
        start = max(0, i - half)
        end = min(T, i + half + 1)
        plateau.append(all(velocities[j] < vel_thresh for j in range(start, end)))

    edges = [i + 1 for i in range(T - 1) if not plateau[i] and plateau[i + 1]]
    if not edges:
        return []
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def summarise(ds, k0, k1):
    item = ds[k1]
    if "q" in item:
        q_end = list(item["q"])
    elif "observation.state" in item:
        state = list(item["observation.state"])
        q_end = state[: len(state) // 2]
    else:
        q_end = []
    gripper = item.get("gripper_open", False)
    return {"pose": q_end, "grip": bool(gripper)}


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

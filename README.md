# ALOHA Lite Demo to Rule

This project shows how to convert a tele-operated recording from the [LeRobot](https://github.com/phospho-ai/lerobot) format into an executable rule set for the Phosphobot SDK.  The script `demo2rules.py` implements the workflow described in the project instructions.

## Requirements

- Python 3.8+
- [LeRobot](https://pypi.org/project/lerobot/)
- [durable_rules](https://pypi.org/project/durable-rules/)
- `torch` and other dependencies used by LeRobot

## Usage

Run the CLI tool to generate a rules file from a dataset:

```bash
python demo2rules.py --dataset path/to/recording --out rules_autogen.py
```

The resulting `rules_autogen.py` can be executed alongside the Phosphobot `sensor_bridge` to replay the demonstration.

## Notes

The pipeline detects motion plateaus via joint velocities and converts each segment into a rule that moves the arm to the final pose and sets the gripper state.  Thresholds for velocity and window size can be tuned with `--vel-thresh` and `--window`.

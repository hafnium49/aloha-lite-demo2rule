# ALOHA Lite: Demo → Rule Converter

`demo2rules.py` turns a **tele‑operated LeRobot v2 recording (Parquet on Hugging Face or local)** into an **executable `durable_rules` script** that drives one *or* two **SO‑101** arms through the Phosphobot SDK.

The workflow mirrors the HTML Dataset Visualizer’s column‑parsing logic, so any dataset you can inspect in the visualizer will also compile into rules here.

---

## Requirements

| Package                                                    | Purpose                 |
| ---------------------------------------------------------- | ----------------------- |
| **Python ≥ 3.8**                                           |                         |
| [`lerobot`](https://pypi.org/project/lerobot/) `>= 0.7`    | dataset loader          |
| [`durable_rules`](https://pypi.org/project/durable-rules/) | rule execution          |
| [`pyarrow`](https://pypi.org/project/pyarrow/)             | Parquet reading         |
| [`requests`](https://pypi.org/project/requests/)           | HTTP fetch for HF files |
| **Phosphobot SDK** `>= 0.0.18`                             | sends skills to SO‑101  |
| *Optionally* `huggingface_hub`                             | faster HF caching       |

All other dependencies (NumPy, etc.) are pulled in transitively.

```bash
pip install lerobot durable_rules pyarrow requests phosphobot
```

---

## CLI usage

```bash
# convert the first episode of a HF dataset into durable_rules
python demo2rules.py \
    --dataset Hafnium49/aloha_lite \
    --episode 0 \
    --out rules_autogen.py \
    --left-arm so101_left \
    --right-arm so101_right
```

Key flags:

| Flag             | Default            | Meaning                                                        |
| ---------------- | ------------------ | -------------------------------------------------------------- |
| `--dataset`      | **required**       | `<org>/<dataset>` (HF repo‑id) *or* local path                 |
| `--episode`      | `0`                | which episode (Parquet shard) to convert                       |
| `--out`          | `rules_autogen.py` | output file name                                               |
| `--left-arm`     | `so101_left`       | Phosphobot ID of the primary arm                               |
| `--right-arm`    | `so101_right`      | secondary arm; omit or add `--single-arm` for single‑arm demos |
| `--ruleset-name` | `so101_dual`       | durable\_rules namespace                                       |
| `--vel-thresh`   | `0.03`             | rad s⁻¹ threshold for “static” detection                       |
| `--window`       | `15`               | window size (# frames) for plateau detection                   |

---

## Executing the generated rules

1. **Start the sensor bridge** (streams facts into durable\_rules):

   ```bash
   ros2 run phosphobot sensor_bridge &
   ```

2. **Run the auto‑generated script** in a separate shell:

   ```bash
   python rules_autogen.py
   ```

The arms will reproduce the demo, moving through each plateaued pose and applying the recorded gripper states (and, if available, Cartesian poses).

---

## How the converter works

1. **Fetches dataset metadata** (`meta/info.json`) and Parquet shards exactly like
   the HTML Dataset Visualizer.

2. **Filters numeric 1‑D features**, excluding housekeeping columns (`timestamp`, `frame_index`, …).

3. **Groups columns by suffix** (e.g. `observation.state | motor_0`) to rebuild:

   * joint positions for primary & secondary arms
   * end‑effector Cartesian pose (`x,y,z,rx,ry,rz`) – if present
   * gripper open/close signals

4. **Detects motion plateaus** via joint‑velocity magnitude `< vel_thresh` over a sliding window.

5. **Generates a durable\_rules script** containing:

   ```python
   left_arm.move_j([...])
   right_arm.move_j([...])              # optional
   left_arm.move_cart({...})            # optional
   left_arm.grip(open=True/False)
   ...
   ```

6. **Embeds a SHA‑256 hash and CLI parameters** in the file header for provenance.

Tweak `--vel-thresh` or `--window` if your demo has very slow or very fast motions.

---

## Running the unit tests

```bash
pytest -q
```

The updated test‑suite (`tests/test_demo2rules.py`) covers:

* plateau detection on synthetic time‑series
* column‑to‑joint/gripper extraction heuristics
* template rendering of an individual rule block

---

### Road‑map

* automatic merging of very short segments
* inverse‑kinematic fallback when only Cartesian data are available
* richer safety guards (force thresholds, timeout aborts) in generated rules

Contributions are welcome—please open an issue or PR.

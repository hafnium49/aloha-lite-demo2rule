# ALOHA Lite: Demo → Rule Converter & Robot Control Suite

A comprehensive toolkit for converting **tele‑operated LeRobot v2 recordings** into executable robot control scripts and configurations. This suite includes:

1. **`demo2rules.py`** - Converts demonstrations into executable `durable_rules` scripts for SO‑101 arms
2. **`extract_at_time.py`** - Extracts precise joint configurations at specific timestamps
3. **Robot configuration management** - JSON-based configuration system for reproducible robot poses
4. **Sequential execution tools** - Multi-step procedure automation with safety features

The workflow mirrors the HTML Dataset Visualizer's column‑parsing logic, so any dataset you can inspect in the visualizer will also compile into rules here.HA Lite: Demo → Rule Converter

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

**Dataset Compatibility**: The converter is designed to work with LeRobot v2 datasets that follow the standard format. It robustly handles various data structures including:
- Scalar numeric columns
- Columns containing sequences of numpy arrays (common in action/observation data)
- Mixed data types with automatic conversion to float64

---

## Tools Overview

### 1. demo2rules.py - Demonstration Converter
Converts LeRobot demonstrations into executable durable_rules scripts with **CSV output support**.

### 2. extract_at_time.py - Time-based Configuration Extractor
Extracts precise joint configurations at specific timestamps from demonstrations.

### 3. Robot Configuration Management
JSON-based system for storing and managing robot configurations with phosphobot API integration.

---

## CLI Usage

### Converting Demonstrations to Rules (with CSV output)

```bash
# Generate durable_rules script (default)
python demo2rules.py \
    --dataset Hafnium49/aloha_lite \
    --episode 0 \
    --out rules_autogen.py \
    --left-arm so101_left \
    --right-arm so101_right

# Generate CSV output for analysis
python demo2rules.py \
    --dataset Hafnium49/aloha_lite \
    --episode 1 \
    --out rules_episode_1.csv \
    --format csv \
    --left-arm so101_left \
    --right-arm so101_right
```

### Extracting Joint Values at Specific Times

```bash
# Extract joint configuration at 4.2 seconds from episode 1
python extract_at_time.py Hafnium49/aloha_lite 1 4.2

# This generates:
# - extracted_Hafnium49_aloha_lite_ep1_t4.2s.json (detailed configuration)
# - config_4_2s.py (executable Python configuration)
```

### Robot Configuration Management

```bash
# Execute a specific configuration from JSON file
cd ../temp_rules
python ../execute_rules.py robot_configurations.json dispensing_water_to_beaker

# Execute sequential procedures
python ../sequential_execute.py standoff_to_dispensing
```

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

3. **Handles complex data structures**: The converter robustly processes LeRobot v2 datasets where columns contain nested numpy arrays (common for action/observation data) by:
   * Detecting object-dtype columns containing sequences
   * Using `np.stack()` to convert arrays of arrays into proper 2D matrices
   * Gracefully handling PyArrow → NumPy conversions
   * Ensuring all data is properly shaped and typed as `float64`

4. **Groups columns by suffix** (e.g. `observation.state | motor_0`) to rebuild:

   * joint positions for primary & secondary arms
   * end‑effector Cartesian pose (`x,y,z,rx,ry,rz`) – if present
   * gripper open/close signals

5. **Detects motion plateaus** via joint‑velocity magnitude `< vel_thresh` over a sliding window, with robust handling of:
   * Empty or single-column matrices
   * 1D data that needs reshaping to 2D
   * Proper data type conversion for norm calculations

6. **Generates a durable\_rules script** containing:

   ```python
   left_arm.move_j([...])
   right_arm.move_j([...])              # optional
   left_arm.move_cart({...})            # optional
   left_arm.grip(open=True/False)
   ...
   ```

7. **Embeds a SHA‑256 hash and CLI parameters** in the file header for provenance.

Tweak `--vel-thresh` or `--window` if your demo has very slow or very fast motions.

---

## Troubleshooting

### Common Issues

**`ValueError: Unknown format code 'd' for object of type 'str'`**
- This occurred in earlier versions when the dataset's `data_path` format string used integer format codes (`:03d`, `:06d`) but the code passed string arguments
- Fixed by passing integer values directly to `.format()` instead of zero-padded strings

**`ValueError: setting an array element with a sequence`**
- Happens when dataset columns contain nested numpy arrays rather than scalar values
- The converter now automatically detects and handles object-dtype columns by stacking sequences into proper 2D matrices

**`AxisError: axis 1 is out of bounds for array of dimension 1`**
- Occurs when the pose matrix is 1D instead of expected 2D shape
- Fixed by ensuring all arrays are properly reshaped to 2D before velocity calculations

**No segments detected**
- Try adjusting `--vel-thresh` (lower for slower motions) or `--window` (smaller for shorter plateaus)
- Check that your dataset contains meaningful joint/pose data

**Robot execution errors**
- `ConnectionError`: Ensure phosphobot services are running and accessible
- `Robot collision during initialization`: Use default no-init mode or `--no-init` flag
- `Configuration not found`: Check JSON file path and configuration name spelling

**Time extraction issues**
- `Timestamp out of range`: Check dataset duration and use valid timestamps
- `No data at timestamp`: Dataset may have gaps; try nearby timestamps
- `Environment not found`: Ensure conda environment `aloha_lite` is activated

**CSV output problems**
- Empty CSV files: Check that dataset contains valid plateau segments
- Missing columns: Ensure dataset has expected arm/joint structure
- Format detection: Use explicit `--format csv` flag for CSV output

---

## Running the unit tests

```bash
pytest -q
```

The updated test‑suite (`tests/test_demo2rules.py`) covers:

* plateau detection on synthetic time‑series
* column‑to‑joint/gripper extraction heuristics
* template rendering of an individual rule block
* **Edge cases from debugging**:
  * empty or single-column matrices
  * 1D arrays requiring reshaping to 2D
  * mixed data types and conversion robustness
  * very short time sequences
  * missing or partial column matches

---

### Road‑map

**Recently Completed ✅**
* ✅ CSV output format for analysis and integration
* ✅ Time-based configuration extraction at specific timestamps  
* ✅ JSON-based robot configuration management system
* ✅ Sequential execution with predefined procedures
* ✅ Safety improvements with default no-initialization mode
* ✅ Precise joint value extraction from demonstrations

**Future Enhancements**
* automatic merging of very short segments
* inverse‑kinematic fallback when only Cartesian data are available
* richer safety guards (force thresholds, timeout aborts) in generated rules
* real-time demonstration recording and immediate execution
* advanced trajectory interpolation between extracted poses

Contributions are welcome—please open an issue or PR.

---

### Command-line Flags Reference

#### demo2rules.py Flags

| Flag             | Default            | Meaning                                                        |
| ---------------- | ------------------ | -------------------------------------------------------------- |
| `--dataset`      | **required**       | `<org>/<dataset>` (HF repo‑id) *or* local path                 |
| `--episode`      | `0`                | which episode (Parquet shard) to convert                       |
| `--out`          | `rules_autogen.py` | output file name                                               |
| `--format`       | `python`           | output format: `python` (durable_rules) or `csv` (analysis)    |
| `--left-arm`     | `so101_left`       | Phosphobot ID of the primary arm                               |
| `--right-arm`    | `so101_right`      | secondary arm; omit or add `--single-arm` for single‑arm demos |
| `--ruleset-name` | `so101_dual`       | durable\_rules namespace                                       |
| `--vel-thresh`   | `0.03`             | rad s⁻¹ threshold for "static" detection                       |
| `--window`       | `15`               | window size (# frames) for plateau detection                   |

#### extract_at_time.py Arguments

| Argument    | Required | Meaning                                     |
| ----------- | -------- | ------------------------------------------- |
| `dataset`   | Yes      | Dataset name (e.g., `Hafnium49/aloha_lite`) |
| `episode`   | Yes      | Episode number (0-based)                   |
| `timestamp` | Yes      | Time in seconds (e.g., `4.2`)              |

---

## Robot Configuration System

### Configuration JSON Structure

Robot configurations are stored in JSON format with the following structure:

```json
{
  "configurations": {
    "configuration_name": {
      "name": "configuration_name",
      "description": "Description of the robot pose",
      "source": {
        "dataset": "Hafnium49/aloha_lite",
        "episode": 1,
        "timestamp": 4.2,
        "extraction_method": "time_based"
      },
      "configuration": {
        "left_arm": {
          "joints": {
            "j1": 0.227, "j2": 0.746, "j3": 1.054,
            "j4": -1.670, "j5": -1.729, "j6": -0.006
          }
        },
        "right_arm": {
          "joints": {
            "j1": 0.219, "j2": 0.198, "j3": 0.910,
            "j4": -1.410, "j5": -2.065, "j6": 0.488
          }
        }
      }
    }
  }
}
```

### Execution Tools

#### execute_rules.py
Execute individual configurations with safety features:
```bash
# Execute specific configuration (safe - no initialization)
python execute_rules.py robot_configurations.json dispensing_water_to_beaker

# Execute with robot initialization (use with caution)
python execute_rules.py robot_configurations.json dispensing_water_to_beaker --init
```

#### sequential_execute.py
Execute predefined sequences of robot movements:
```bash
# Execute predefined sequences
python sequential_execute.py standoff_to_dispensing
python sequential_execute.py full_lab_procedure

# List available sequences
python sequential_execute.py --list
```

### Safety Features

- **Default no-initialization**: Prevents dangerous collision during startup
- **Configuration validation**: Ensures all required joint values are present
- **Error handling**: Graceful failure with informative error messages
- **Sequential timing**: Controlled delays between movements in procedures

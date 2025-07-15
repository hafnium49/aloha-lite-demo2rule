#!/usr/bin/env python3
"""
Extract joint values at a specific time from LeRobot v2 datasets.

This script extracts robot joint positions at a specified timestamp from a dataset episode,
which can be used to create new robot configurations or analyze specific moments in demonstrations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path to import demo2rules functions
sys.path.append(str(Path(__file__).parent))
from demo2rules import (
    load_pose_matrix, 
    build_extractors,
    numeric_feature_keys,
    expand_column_names
)

def find_closest_timestamp_index(timestamps: List[float], target_time: float) -> int:
    """Find the index of the timestamp closest to the target time."""
    import numpy as np
    timestamps_array = np.array(timestamps)
    closest_idx = np.argmin(np.abs(timestamps_array - target_time))
    return int(closest_idx)

def extract_joint_values_at_time(
    repo_id: str, 
    episode_id: int, 
    target_time: float
) -> Dict:
    """Extract joint values at a specific time from a dataset episode."""
    
    print(f"🔍 Loading dataset: {repo_id}, episode {episode_id}")
    print(f"🎯 Target time: {target_time} seconds")
    
    try:
        # Load the episode data
        ts, mat, colnames, info = load_pose_matrix(repo_id, episode_id)
        
        if len(ts) == 0:
            raise ValueError("No timestamp data found in episode")
        
        # Find closest timestamp
        closest_idx = find_closest_timestamp_index(ts.tolist(), target_time)
        actual_time = ts[closest_idx]
        
        print(f"📊 Dataset length: {len(ts)} frames")
        print(f"⏱️  Time range: {ts[0]:.3f}s to {ts[-1]:.3f}s")
        print(f"🎯 Closest timestamp: {actual_time:.3f}s (index {closest_idx})")
        print(f"📏 Time difference: {abs(actual_time - target_time):.3f}s")
        
        # Build extractors for joint data
        extract_left, extract_right, extract_misc = build_extractors(colnames)
        
        # Extract data at the target time
        row = mat[closest_idx]
        
        left_joints = extract_left(row)
        right_joints = extract_right(row)
        misc_data = extract_misc(row)
        
        # Create result dictionary
        result = {
            "metadata": {
                "dataset": repo_id,
                "episode": episode_id,
                "target_time": target_time,
                "actual_time": float(actual_time),
                "frame_index": closest_idx,
                "time_difference": float(abs(actual_time - target_time)),
                "total_frames": len(ts),
                "extracted_at": "2025-07-15"
            },
            "joint_data": {
                "left_arm": {
                    "joints": left_joints,
                    "joint_count": len(left_joints)
                },
                "right_arm": {
                    "joints": right_joints,
                    "joint_count": len(right_joints)
                }
            },
            "additional_data": misc_data,
            "configuration_template": {
                "name": f"extracted_time_{target_time}s_episode_{episode_id}",
                "description": f"Configuration extracted at {actual_time:.3f}s from {repo_id} episode {episode_id}",
                "source": {
                    "dataset": repo_id,
                    "episode": episode_id,
                    "timestamp": float(actual_time),
                    "frame_index": closest_idx,
                    "extraction_method": "time_based"
                },
                "timestamp": "2025-07-15",
                "configuration": {
                    "left_arm": {
                        "name": "SO-101 Left Arm",
                        "joints": {
                            f"j{i+1}": float(joint) for i, joint in enumerate(left_joints)
                        } if left_joints else {},
                        "joint_names": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
                        "units": "radians"
                    },
                    "right_arm": {
                        "name": "SO-101 Right Arm", 
                        "joints": {
                            f"j{i+1}": float(joint) for i, joint in enumerate(right_joints)
                        } if right_joints else {},
                        "joint_names": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
                        "units": "radians"
                    }
                }
            }
        }
        
        # Add usage examples if we have joint data
        if left_joints and right_joints:
            result["configuration_template"]["usage"] = {
                "phosphobot_api": {
                    "left_arm": f"POST /joints/write with body: {left_joints}",
                    "right_arm": f"POST /joints/write with body: {right_joints}"
                },
                "python_example": {
                    "description": "Example usage with phosphobot SkillClient",
                    "code": f"left_arm.move_j({left_joints})\nright_arm.move_j({right_joints})"
                }
            }
        
        return result
        
    except Exception as e:
        print(f"❌ Error extracting joint values: {e}")
        raise

def save_configuration(result: Dict, output_file: str) -> None:
    """Save the extracted configuration to a file."""
    output_path = Path(output_file)
    
    if output_path.suffix.lower() == '.json':
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Saved full extraction data to: {output_path}")
    else:
        # Save as Python configuration file
        config_template = result["configuration_template"]
        
        # Generate Python code
        actual_time = result["metadata"]["actual_time"]
        python_code = f'''"""
Configuration extracted from {result["metadata"]["dataset"]} episode {result["metadata"]["episode"]}
Extracted at time: {actual_time:.3f}s
Generated on: {result["metadata"]["extracted_at"]}
"""

# Configuration data
CONFIGURATION = {json.dumps(config_template, indent=4)}

# Quick access to joint values
LEFT_JOINTS = {result["joint_data"]["left_arm"]["joints"]}
RIGHT_JOINTS = {result["joint_data"]["right_arm"]["joints"]}

if __name__ == "__main__":
    print("Configuration extracted at {actual_time:.3f}s:")
    print("Left arm joints:", LEFT_JOINTS)
    print("Right arm joints:", RIGHT_JOINTS)
'''
        
        with open(output_path, 'w') as f:
            f.write(python_code)
        print(f"🐍 Saved Python configuration to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract joint values at specific time from dataset")
    parser.add_argument("dataset", help="Dataset repository ID (e.g., 'Hafnium49/aloha_lite')")
    parser.add_argument("episode", type=int, help="Episode number")
    parser.add_argument("time", type=float, help="Target time in seconds")
    parser.add_argument("--output", "-o", default=None, 
                       help="Output file (.json for full data, .py for config only)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed information")
    
    args = parser.parse_args()
    
    try:
        # Extract joint values
        result = extract_joint_values_at_time(args.dataset, args.episode, args.time)
        
        # Display results
        print("\n" + "="*60)
        print("📋 EXTRACTION RESULTS")
        print("="*60)
        
        metadata = result["metadata"]
        left_joints = result["joint_data"]["left_arm"]["joints"]
        right_joints = result["joint_data"]["right_arm"]["joints"]
        
        print(f"Dataset: {metadata['dataset']}")
        print(f"Episode: {metadata['episode']}")
        print(f"Target time: {metadata['target_time']}s")
        print(f"Actual time: {metadata['actual_time']:.3f}s")
        print(f"Frame index: {metadata['frame_index']}")
        print(f"Time accuracy: ±{metadata['time_difference']:.3f}s")
        
        print(f"\n🤖 LEFT ARM JOINTS ({len(left_joints)} joints):")
        if left_joints:
            for i, joint in enumerate(left_joints, 1):
                print(f"  j{i}: {joint:.6f}")
        else:
            print("  No left arm joints found")
        
        print(f"\n🤖 RIGHT ARM JOINTS ({len(right_joints)} joints):")
        if right_joints:
            for i, joint in enumerate(right_joints, 1):
                print(f"  j{i}: {joint:.6f}")
        else:
            print("  No right arm joints found")
        
        # Show additional data if verbose
        if args.verbose and result["additional_data"]:
            print(f"\n📊 ADDITIONAL DATA:")
            for key, value in result["additional_data"].items():
                print(f"  {key}: {value}")
        
        # Save to file if specified
        if args.output:
            save_configuration(result, args.output)
        else:
            # Generate default filename
            safe_dataset = args.dataset.replace("/", "_")
            default_filename = f"extracted_{safe_dataset}_ep{args.episode}_t{args.time}s.json"
            save_configuration(result, default_filename)
        
        print(f"\n✅ Extraction completed successfully!")
        
        # Show usage example
        print(f"\n💡 USAGE EXAMPLE:")
        print(f"python3 execute_rules.py --config extracted_time_{args.time}s_episode_{args.episode}")
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

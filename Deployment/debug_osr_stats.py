import numpy as np
import json

# Load the npz file
osr_stats_path = "./Deployment/export/osr_stats.npz"
data = np.load(osr_stats_path, allow_pickle=True)

print("=" * 80)
print("OSR STATS NPZ CONTENTS")
print("=" * 80)

# Check classes
classes = [str(c) for c in data['classes'].tolist()]
print(f"\nClasses ({len(classes)}):")
for i, cls in enumerate(classes):
    print(f"  [{i}] {cls}")

# Check centroids
centroids = data['centroids']
print(f"\nCentroids shape: {centroids.shape}")

# Check thresholds
thresholds_raw = data['thresholds']
print(f"\nThresholds raw type: {type(thresholds_raw)}, shape: {thresholds_raw.shape}")

# Try to extract as dict
try:
    thresholds_dict = thresholds_raw.item()
    print(f"Thresholds dict type: {type(thresholds_dict)}")
    print(f"Thresholds dict keys: {list(thresholds_dict.keys())}")
    print("\nThresholds values:")
    for cls in classes:
        if cls in thresholds_dict:
            print(f"  {cls:25} τ = {thresholds_dict[cls]:.6f}")
        else:
            print(f"  {cls:25} ❌ NOT FOUND")
except Exception as e:
    print(f"Error extracting dict: {e}")
    print(f"Thresholds raw value: {thresholds_raw}")

print("\n" + "=" * 80)
print("COMPARISON WITH DEBUG CSV")
print("=" * 80)

import pandas as pd
debug_csv = pd.read_csv("best_thresholds_debug.csv")
print("\nDebug CSV:")
print(debug_csv.to_string())

print("\nMatch check:")
for _, row in debug_csv.iterrows():
    cls = row['class']
    csv_threshold = row['threshold']
    if cls in thresholds_dict:
        npz_threshold = thresholds_dict[cls]
        match = "✓" if abs(csv_threshold - npz_threshold) < 0.001 else "✗"
        print(f"  {cls:25} CSV={csv_threshold:10.6f}, NPZ={npz_threshold:10.6f} {match}")
    else:
        print(f"  {cls:25} ✗ NOT IN NPZ")

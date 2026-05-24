"""
Diagnose the mismatch between predicted class index and class name
"""
import numpy as np
import json

# Check class_names from export
with open("./Deployment/export/class_names.json", "r") as f:
    class_names = json.load(f)

print("=" * 80)
print("CLASS NAMES ORDER (from model)")
print("=" * 80)
for i, cls in enumerate(class_names):
    print(f"  [{i}] {cls}")

# Check classes from OSR stats
osr_stats_path = "./Deployment/export/osr_stats.npz"
data = np.load(osr_stats_path, allow_pickle=True)
classes_osr = [str(c) for c in data['classes'].tolist()]

print("\n" + "=" * 80)
print("CLASS NAMES ORDER (from OSR stats - SORTED)")
print("=" * 80)
for i, cls in enumerate(classes_osr):
    print(f"  [{i}] {cls}")

print("\n" + "=" * 80)
print("MISMATCH CHECK")
print("=" * 80)

if class_names == classes_osr:
    print("✓ Classes are in the same order")
else:
    print("✗ CLASSES ARE IN DIFFERENT ORDERS!")
    print("\nMapping from model index to OSR index:")
    for model_idx, model_cls in enumerate(class_names):
        osr_idx = classes_osr.index(model_cls) if model_cls in classes_osr else -1
        match = "✓" if model_idx == osr_idx else "✗"
        print(f"  Model[{model_idx:2}]={model_cls:20} -> OSR[{osr_idx:2}] {match}")

print("\n" + "=" * 80)
print("IMPLICATION")
print("=" * 80)
print("""
When ONNX model predicts class index 8 (for example):
- Model expects this to map to class_names[8]
- But evaluate_osr uses classes_osr[8] which might be different!

This causes:
- Distance to wrong centroid to be retrieved
- Wrong threshold to be used
- Wrong OSR decision
""")

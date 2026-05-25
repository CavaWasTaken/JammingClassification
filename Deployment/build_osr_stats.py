import json
import os
import pandas as pd
import numpy as np
import onnxruntime as ort

from osr_stats import calibrate_thresholds_on_validation, save_osr_stats


def main():
    # Configuration
    onnx_file_path = "./Deployment/export/jamming_model.onnx"
    class_names_path = "./Deployment/export/class_names.json"
    features_csv_path = "./NMon_Dataset_10MHz/all_features_scaled.csv"
    osr_stats_output = "./Deployment/export/osr_stats.npz"
    val_split = 0.3
    max_fpr = 0.05  # Max 5% false rejection rate on known samples
    
    # Validation unknown jammer types
    unknown_jammers = {
        'FH-20': './NMon_Dataset_10MHz/features_file_FH-20.csv',
        'HOOKED-SAWTOOTH-3-20': './NMon_Dataset_10MHz/features_file_HOOKED-SAWTOOTH-3-20.csv',
        'LINEAR-WIDE-20': './NMon_Dataset_10MHz/features_file_LINEAR-WIDE-20.csv',
        'MULTITONE-NARROW-40-20': './NMon_Dataset_10MHz/features_file_MULTITONE-NARROW-40-20.csv',
    }

    # Validate file existence
    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_file_path}")
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"class_names file not found: {class_names_path}")
    if not os.path.exists(features_csv_path):
        raise FileNotFoundError(f"features csv not found: {features_csv_path}")

    with open(class_names_path, "r") as f:
        class_names = json.load(f)

    # Load known jammers dataset
    features_df_known = pd.read_csv(features_csv_path)
    print(f"✓ Loaded {len(features_df_known)} known samples from CSV")

    # Load and combine unknown jammer datasets
    features_df_unknown = pd.DataFrame()
    print("\nLoading unknown jammer types for validation:")
    for jammer_name, csv_file in unknown_jammers.items():
        if os.path.exists(csv_file):
            df_temp = pd.read_csv(csv_file)
            df_temp['is_unknown'] = True
            df_temp['unknown_type'] = jammer_name
            features_df_unknown = pd.concat([features_df_unknown, df_temp], ignore_index=True)
            print(f"  ✓ {jammer_name}: {len(df_temp)} samples")
        else:
            print(f"  ⚠ Not found: {csv_file}")
    
    if len(features_df_unknown) > 0:
        print(f"✓ Loaded {len(features_df_unknown)} unknown samples total")
    else:
        print("⚠ No unknown samples loaded - calibration may be biased")
    
    # Mark known samples
    features_df_known['is_unknown'] = False
    features_df_known['unknown_type'] = None

    print('✓ Providers disponibili:', ort.get_available_providers())
    # Initialize ONNX session (CPU only - one-time calibration task)
    try:
        sess = ort.InferenceSession(
            onnx_file_path,
            providers=['CPUExecutionProvider']
        )
        print(f"✓ ONNX Runtime is using CPU for calibration")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise
    
    # Split known data: train (70%) + validation (30%)
    val_size = int(len(features_df_known) * val_split)
    val_indices = np.random.choice(len(features_df_known), val_size, replace=False)
    train_indices = np.setdiff1d(np.arange(len(features_df_known)), val_indices)
    
    train_df = features_df_known.iloc[train_indices].reset_index(drop=True)
    val_known_df = features_df_known.iloc[val_indices].reset_index(drop=True)
    
    print("\n" + "="*70)
    
    print("THRESHOLD CALIBRATION WITH ROC CURVE")
    print("="*70)
    print(f"Known data split:")
    print(f"  Training: {len(train_df)} samples (70%)")
    print(f"  Validation: {len(val_known_df)} samples (30%)")
    print(f"Unknown data for validation: {len(features_df_unknown)} samples")
    print(f"Max FPR constraint: {max_fpr*100:.1f}%")
    print("="*70)
    
    # Calibrate thresholds using validation set with ROC curve
    stats = calibrate_thresholds_on_validation(
        sess,
        train_df,
        val_known_df,
        features_df_unknown,
        class_names,
        max_fpr=max_fpr
    )

    save_osr_stats(osr_stats_output, stats)

    print(f"\n✓ OSR stats saved to: {osr_stats_output}")

    print("\n" + "="*70)
    print("CALIBRATED PER-CLASS THRESHOLDS")
    print("="*70)
    for cls in stats['classes']:
        print(f"  {cls:25} τ = {stats['thresholds'][cls]:.6f}")
    print("="*70)


if __name__ == "__main__":
    main()

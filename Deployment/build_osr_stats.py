import argparse
import json
import os
import pandas as pd
import onnxruntime as ort

from osr_stats import build_osr_stats, save_osr_stats


def main():
    parser = argparse.ArgumentParser(description="Build and store OSR centroid statistics from training dataset")
    parser.add_argument("--onnx", default="./Deployment/export/jamming_model.onnx", help="Path to ONNX model")
    parser.add_argument("--class-names", default="./Deployment/export/class_names.json", help="Path to class names JSON")
    parser.add_argument("--features-csv", default="./NMon_Dataset_10MHz/all_features_scaled.csv", help="Path to features CSV")
    parser.add_argument("--output", default="./Deployment/export/osr_stats.npz", help="Output stats file")
    parser.add_argument("--quantile", type=float, default=0.95, help="Distance quantile threshold per class")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for calibration rows")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
    if not os.path.exists(args.class_names):
        raise FileNotFoundError(f"class_names file not found: {args.class_names}")
    if not os.path.exists(args.features_csv):
        raise FileNotFoundError(f"features csv not found: {args.features_csv}")

    with open(args.class_names, "r") as f:
        class_names = json.load(f)

    features_df = pd.read_csv(args.features_csv)
    initial_count = len(features_df)
    if 'power' in features_df.columns:
        features_df = features_df[features_df['power'] != 'P1'].reset_index(drop=True)
    removed = initial_count - len(features_df)

    print(f"Loaded {len(features_df)} rows from CSV (removed P1 rows: {removed})")
    print("Building OSR stats... this can take a while on full dataset.")

    sess = ort.InferenceSession(args.onnx)
    stats = build_osr_stats(
        sess,
        features_df,
        class_names,
        quantile=args.quantile,
        max_samples=args.max_samples,
    )

    save_osr_stats(args.output, stats)

    print(f"Saved OSR stats to: {args.output}")
    print(
        f"Rows used: {stats['used_rows']}/{stats['requested_rows']} (skipped {stats['skipped']}), "
        f"quantile={stats['quantile']:.2f}"
    )
    print("Per-class thresholds:")
    for cls, th, ct in zip(stats['classes'], stats['thresholds'], stats['counts']):
        print(f"  - {cls}: n={int(ct)}, threshold={float(th):.4f}")


if __name__ == "__main__":
    main()

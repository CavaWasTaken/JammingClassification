import pandas as pd
import numpy as np
import json
import os
import time
import argparse
import onnxruntime as ort
import warnings
from collections import defaultdict
from osr_stats import (
    run_model,
    extract_true_class,
    build_osr_stats,
    save_osr_stats,
    load_osr_stats,
    evaluate_osr,
)
warnings.filterwarnings('ignore', message='X does not have valid feature names')


def softmax_np(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)



def main(max_iterations=None, calib_samples=None, calib_quantile=0.95):
    onnx_file_path = "./Deployment/export/jamming_model.onnx"
    class_names_path = "./Deployment/export/class_names.json"
    path_to_features = "./NMon_Dataset_10MHz/all_features_scaled.csv"
    
    # Load model and class names
    sess = ort.InferenceSession(onnx_file_path)
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    
    # Load features dataset with all aligned spectrograms
    features_df = pd.read_csv(path_to_features)
    # Exclude P1 samples (not used)
    initial_count = len(features_df)
    if 'power' in features_df.columns:
        features_df = features_df[features_df['power'] != 'P1'].reset_index(drop=True)
        removed = initial_count - len(features_df)
        if removed > 0:
            print(f"✓ Removed {removed} samples with power P1; {len(features_df)} remaining")
    print(f"✓ Loaded {len(features_df)} samples from features CSV")
    print(f"✓ Classes: {features_df['class'].unique()}\n")

    # Load precomputed OSR statistics or build once and persist.
    osr_stats_path = "./Deployment/export/osr_stats.npz"
    if os.path.exists(osr_stats_path):
        osr = load_osr_stats(osr_stats_path)
        print(f"✓ Loaded OSR stats from {osr_stats_path}")
    else:
        print("--- OSR stats file not found, building from dataset...")
        osr = build_osr_stats(
            sess,
            features_df,
            class_names,
            quantile=calib_quantile,
            max_samples=calib_samples,
        )
        save_osr_stats(osr_stats_path, osr)
        print(f"✓ Saved OSR stats to {osr_stats_path}")

    print(
        f"✓ OSR stats ready: used {osr['used_rows']}/{osr['requested_rows']} rows "
        f"(skipped {osr['skipped']}) at q={osr['quantile']:.2f}"
    )
    print("✓ OSR class support:")
    for cls, th, ct in zip(osr['classes'], osr['thresholds'], osr['counts']):
        print(f"  - {cls}: n={int(ct)}, threshold={float(th):.4f}")
    print()
    
    # Load scaler
    scaler_path = "./Deployment/export/scaler_model.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Feature scaler not found: {scaler_path}")
    
    try:
        import joblib
        scaler = joblib.load(scaler_path)
    except:
        import pickle
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    
    # Setup output CSV
    csv_path = "./Deployment/predictions.csv"
    with open(csv_path, "w") as f:
        f.write(
            "iteration,timestamp,true_type,true_level,true_class_full,pred_class,osr_label,"
            "osr_distance,osr_threshold,confidence,probability,margin,energy,inference_time_ms,match_closed_set,match_osr\n"
        )
    
    # Statistics tracking
    stats = defaultdict(int)
    correct_predictions = 0
    total_predictions = 0
    inference_times = []
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # Select a random sample from the features dataframe (ensures alignment)
            random_idx = np.random.randint(0, len(features_df))
            sample_row = features_df.iloc[random_idx]
            
            spec_path = sample_row['path_spec']
            class_type = sample_row['class']
            true_class_full = extract_true_class(spec_path, class_type)
            
            # Load spectrogram
            if not os.path.exists(spec_path):
                print(f"⚠ Spectrogram not found: {spec_path}, skipping...")
                continue
            
            spec = np.load(spec_path)
            
            # Extract features from row
            features_row = sample_row.drop(['path_spec', 'power', 'class']).values.astype(np.float32)
            
            # Parse the true class to extract type and level
            if true_class_full == 'CLEAN':
                true_type = 'CLEAN'
                true_level = 'N/A'
            else:
                parts = true_class_full.rsplit('_', 1)
                true_type = parts[0] if len(parts) == 2 else true_class_full
                true_level = parts[1] if len(parts) == 2 else 'UNKNOWN'
            
            print(f"\n[Iteration {iteration}] Sampling from: {spec_path}")
            print(f"  True: {true_type} ({true_level}) | Spectrogram shape: {spec.shape} | Features: {len(features_row)}")
            
            # Prepare inputs
            start_time = time.perf_counter()
            logits, penultimate, energy = run_model(sess, spec, features_row)
            end_time = time.perf_counter()
            
            inference_time_ms = (end_time - start_time) * 1000
            inference_times.append(inference_time_ms)
            
            # Compute predictions and confidence metrics
            predicted_class = np.argmax(logits)
            pred_class_name = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"

            # OSR decision: distance to predicted class centroid vs threshold.
            osr_eval = evaluate_osr(penultimate, pred_class_name, osr)
            osr_distance = osr_eval['distance']
            threshold = osr_eval['threshold']
            osr_label = osr_eval['osr_label']
            estimated_power = osr_eval['estimated_power']
            
            # Softmax probabilities
            probabilities = softmax_np(logits)
            max_prob = np.max(probabilities)
            
            # Margin between top 2 predictions
            sorted_logits = np.sort(logits)
            margin = sorted_logits[-1] - sorted_logits[-2]
            
            # Check if prediction matches ground truth (closed-set and OSR labels)
            is_correct_closed = pred_class_name == true_class_full
            is_correct_osr = osr_label == true_class_full
            correct_predictions += is_correct_closed
            total_predictions += 1
            
            # Update statistics
            stats[f'true_{true_class_full}'] += 1
            stats[f'pred_{pred_class_name}'] += 1
            stats[f'osr_{osr_label}'] += 1
            if is_correct_closed:
                stats['correct'] += 1
            else:
                stats['incorrect'] += 1
            
            # Log prediction details
            print(f"  Closed-set: {pred_class_name} (confidence={np.max(logits):.4f}, prob={max_prob:.4f})")
            if osr_label == 'UNKNOWN':
                print(
                    f"  🔴 OSR: UNKNOWN | "
                    f"dist={osr_distance:.4f} vs thr={threshold:.4f} | "
                    f"Margin: {margin:.4f} | Energy: {energy:.4f} | Inference: {inference_time_ms:.2f}ms"
                )
            else:
                print(
                    f"  OSR: {osr_label} | dist={osr_distance:.4f} vs thr={threshold:.4f} | "
                    f"Margin: {margin:.4f} | Energy: {energy:.4f} | Inference: {inference_time_ms:.2f}ms"
                )
            print(
                f"  {'✓ CLOSED OK' if is_correct_closed else '✗ CLOSED ERR'} | "
                f"{'✓ OSR OK' if is_correct_osr else '✗ OSR ERR'} | "
                f"Closed Accuracy: {100*correct_predictions/total_predictions:.1f}%"
            )
            
            # Write to CSV
            timestamp = time.time()
            with open(csv_path, "a") as f:
                f.write(
                    f"{iteration},{timestamp},{true_type},{true_level},{true_class_full},{pred_class_name},{osr_label},"
                    f"{osr_distance:.4f},{threshold:.4f},"
                    f"{np.max(logits):.4f},{max_prob:.4f},{margin:.4f},"
                    f"{energy:.4f},{inference_time_ms:.2f},{'1' if is_correct_closed else '0'},{'1' if is_correct_osr else '0'}\n"
                )
            
            # Print summary every 10 iterations
            if iteration % 10 == 0:
                print(f"\n{'='*60}")
                print(f"SUMMARY after {iteration} iterations:")
                print(f"  Accuracy: {100*correct_predictions/total_predictions:.1f}% ({correct_predictions}/{total_predictions})")
                print(f"  Avg inference time: {np.mean(inference_times):.2f}ms (min={np.min(inference_times):.2f}ms, max={np.max(inference_times):.2f}ms)")
                print(f"{'='*60}\n")

            if max_iterations is not None and iteration >= max_iterations:
                print(f"Reached max iterations ({max_iterations}), stopping.")
                break
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("FINAL SUMMARY:")
        print(f"{'='*60}")
        print(f"Total iterations: {total_predictions}")
        print(f"Accuracy: {100*correct_predictions/total_predictions:.1f}% ({correct_predictions}/{total_predictions})")
        print(f"Avg inference time: {np.mean(inference_times):.2f}ms")
        print(f"Predictions saved to: {csv_path}")
        print("Stopping...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote test with embedding-distance Open Set Recognition")
    parser.add_argument("--max-iterations", type=int, default=None, help="Stop after N iterations")
    parser.add_argument("--calib-samples", type=int, default=None, help="Optional cap for OSR stats build (default: full dataset)")
    parser.add_argument("--calib-quantile", type=float, default=0.95, help="Distance quantile used when building OSR stats")
    args = parser.parse_args()

    main(
        max_iterations=args.max_iterations,
        calib_samples=args.calib_samples,
        calib_quantile=args.calib_quantile,
    )
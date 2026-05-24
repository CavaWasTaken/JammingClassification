"""
Remote test with validation-based Open Set Recognition (OSR)

CALIBRATION APPROACH:
  - Thresholds computed using validation-based F1 optimization
  - Each class gets its own per-class threshold
  - Thresholds calibrated to maximize discrimination between:
    * Target class known samples (should accept)
    * Unknown/out-of-distribution samples (should reject)

USAGE:
  python Deployment/main_remoteTest.py [--max-iterations N] [--known-only]

OUTPUT:
  - Real-time predictions with OSR labels (UNKNOWN or class name)
  - CSV file: predictions.csv with detailed metrics
  - Final summary with accuracy and error analysis
"""
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
    load_osr_stats,
    evaluate_osr
)
warnings.filterwarnings('ignore', message='X does not have valid feature names')


def softmax_np(x):
    """Numerically stable softmax computation."""
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)



def main(max_iterations=None, include_unknown=True):
    print("="*80)
    print("VALIDATION-BASED OPEN SET RECOGNITION (OSR) INFERENCE TEST")
    print("="*80)
    print()
    
    onnx_file_path = "./Deployment/export/jamming_model.onnx"
    class_names_path = "./Deployment/export/class_names.json"
    path_to_features = "./NMon_Dataset_10MHz/all_features_scaled.csv"
    
    # Load model and class names
    sess = ort.InferenceSession(
        onnx_file_path,
        providers=['CPUExecutionProvider']
    )
    active_provider = sess.get_providers()[0]
    print(f"✓ Using provider: {active_provider}")
    
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    
    # Load known jammers dataset with all aligned spectrograms
    features_df_known = pd.read_csv(path_to_features)
    # Exclude P1 samples (not used)
    initial_count = len(features_df_known)
    if 'power' in features_df_known.columns:
        features_df_known = features_df_known[features_df_known['power'] != 'P1'].reset_index(drop=True)
        removed = initial_count - len(features_df_known)
        if removed > 0:
            print(f"✓ Removed {removed} samples with power P1; {len(features_df_known)} remaining")
    print(f"✓ Loaded {len(features_df_known)} known samples from features CSV")
    print(f"✓ Known Classes: {features_df_known['class'].unique().tolist()}\n")
    
    # Load unknown jammer datasets
    features_df_unknown = pd.DataFrame()
    if include_unknown:
        unknown_jammers = {
            'FH-20': './NMon_Dataset_10MHz/features_file_FH-20.csv',
            'HOOKED-SAWTOOTH-3-20': './NMon_Dataset_10MHz/features_file_HOOKED-SAWTOOTH-3-20.csv',
            'LINEAR-WIDE-20': './NMon_Dataset_10MHz/features_file_LINEAR-WIDE-20.csv',
            'MULTITONE-NARROW-40-20': './NMon_Dataset_10MHz/features_file_MULTITONE-NARROW-40-20.csv',
        }
        
        print("Loading unknown jammer types:")
        for jammer_name, csv_file in unknown_jammers.items():
            if os.path.exists(csv_file):
                df_temp = pd.read_csv(csv_file)
                # Update paths to local
                df_temp['path_spec'] = df_temp['path_spec'].apply(
                    lambda x: x.replace('/content/drive/MyDrive/N-MON/ALTRO/spectograms/', 
                                      './NMon_Dataset_10MHz/image_dataset/')
                )
                df_temp['is_unknown'] = True
                df_temp['unknown_type'] = jammer_name
                features_df_unknown = pd.concat([features_df_unknown, df_temp], ignore_index=True)
                print(f"  ✓ {jammer_name}: {len(df_temp)} samples")
        
        if len(features_df_unknown) > 0:
            print(f"✓ Loaded {len(features_df_unknown)} unknown samples total\n")
        else:
            print("⚠ No unknown samples loaded\n")
    
    # Mark known samples
    features_df_known['is_unknown'] = False
    features_df_known['unknown_type'] = None
    
    # Combine known + unknown for mixed sampling
    features_df = pd.concat([features_df_known, features_df_unknown], ignore_index=True)
    print(f"✓ Total dataset: {len(features_df)} samples (Known: {len(features_df_known)}, Unknown: {len(features_df_unknown)})\n")

    # Load precomputed OSR statistics
    osr_stats_path = "./Deployment/export/osr_stats.npz"
    if os.path.exists(osr_stats_path):
        osr = load_osr_stats(osr_stats_path)
        print(f"✓ Loaded OSR stats from {osr_stats_path}")
        print(f"  Classes: {osr['classes']}")
        print(f"  Thresholds:")
        for cls in osr['classes']:
            print(f"    {cls:25} τ = {osr['thresholds'][cls]:.6f}")
        use_osr = True
    else:
        print(f"⚠ OSR stats file not found: {osr_stats_path}")
        print(f"  Run: python Deployment/build_osr_stats.py")
        use_osr = False
    
    # Load scaler for feature normalization
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
    print(f"✓ Scaler loaded")
    print(f"✓ Ready for inference testing\n")
    
    # Setup output CSV
    csv_path = "./Deployment/predictions.csv"
    with open(csv_path, "w") as f:
        f.write(
            "iteration,timestamp,"
            # Ground truth
            "is_unknown,true_type,true_level,true_class_full,"
            # Closed-set predictions
            "pred_class,pred_confidence,pred_max_prob,pred_margin,"
            # OSR
            "osr_label,osr_distance,osr_threshold,corrected,best_class,"
            # Energy/diagnostics
            "energy,inference_time_ms\n"
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
            
            # Select a random sample from the combined dataframe (ensures alignment)
            random_idx = np.random.randint(0, len(features_df))
            sample_row = features_df.iloc[random_idx]
            
            spec_path = sample_row['path_spec']
            is_unknown = sample_row['is_unknown']
            
            # Determine true class
            if is_unknown:
                # Unknown jammer type
                true_class_full = 'UNKNOWN'
                true_type = sample_row['unknown_type']
                true_level = 'N/A'
                class_type = sample_row['class']
            else:
                # Known jammer type
                class_type = sample_row['class']
                true_class_full = extract_true_class(spec_path, class_type)
                
                # Parse the true class to extract type and level
                if true_class_full == 'CLEAN':
                    true_type = 'CLEAN'
                    true_level = 'N/A'
                else:
                    parts = true_class_full.rsplit('_', 1)
                    true_type = parts[0] if len(parts) == 2 else true_class_full
                    true_level = parts[1] if len(parts) == 2 else 'UNKNOWN'
            
            # Load spectrogram
            if not os.path.exists(spec_path):
                print(f"⚠ Spectrogram not found: {spec_path}, skipping...")
                continue
            
            spec = np.load(spec_path)
            
            # Extract features from row (handle different CSV structures for unknown vs known)
            if is_unknown:
                feature_cols = [f'f{i}' for i in range(1, 17)]
                features_unscaled = sample_row[feature_cols].values.astype(np.float32)
                # Need to scale unknown features
                try:
                    import joblib
                    scaler = joblib.load(scaler_path) if 'scaler' not in locals() else scaler
                except:
                    if 'scaler' not in locals():
                        import pickle
                        with open(scaler_path, "rb") as f:
                            scaler = pickle.load(f)
                features_row = scaler.transform([features_unscaled])[0]
            else:
                features_row = sample_row.drop(['path_spec', 'power', 'class', 'is_unknown', 'unknown_type'], errors='ignore').values.astype(np.float32)
            
            # Add marker for console output
            unknown_marker = "🔷 UNKNOWN" if is_unknown else ""
            print(f"\n[Iteration {iteration}] {unknown_marker} Sampling from: {spec_path}")
            print(f"  True: {true_type} ({true_level}) | Spectrogram shape: {spec.shape} | Features: {len(features_row)}")
            
            # Prepare inputs
            start_time = time.perf_counter()
            logits, penultimate, energy = run_model(sess, spec, features_row)
            end_time = time.perf_counter()
            
            inference_time_ms = (end_time - start_time) * 1000
            inference_times.append(inference_time_ms)
            
            # Compute predictions and confidence metrics
            predicted_class = np.argmax(logits)
            pred_class_name = class_names[predicted_class]

            # OSR decision with correction logic
            if use_osr:
                osr_eval = evaluate_osr(penultimate, logits, osr, class_names)
                osr_label = osr_eval['osr_label']
                osr_distance = osr_eval['distance_to_pred']
                osr_threshold = osr_eval['threshold_pred']
                corrected = osr_eval['corrected']
                best_class = osr_eval['best_class']
            else:
                osr_label = pred_class_name
                osr_distance = 0.0
                osr_threshold = float('inf')
                corrected = False
                best_class = pred_class_name
            
            # Softmax probabilities
            probabilities = softmax_np(logits)
            max_prob = np.max(probabilities)
            
            # Margin between top 2 predictions
            sorted_logits = np.sort(logits)
            margin = sorted_logits[-1] - sorted_logits[-2]
            
            # Check if prediction matches ground truth
            is_correct_closed = pred_class_name == true_class_full
            is_correct_osr = osr_label == true_class_full
            correct_predictions += is_correct_osr
            total_predictions += 1
            
            # Update statistics
            stats[f'true_{true_class_full}'] += 1
            stats[f'pred_{pred_class_name}'] += 1
            stats[f'osr_{osr_label}'] += 1
            if is_correct_osr:
                stats['correct'] += 1
            else:
                stats['incorrect'] += 1
            
            # Log prediction details
            marker = "🔷" if is_unknown else "🟢"
            print(f"  Closed-set: {pred_class_name} | Confidence: {np.max(logits):.4f} | Prob: {max_prob:.4f}")
            
            if osr_label == 'UNKNOWN':
                print(f"  {marker} OSR: UNKNOWN | dist={osr_distance:.4f} vs τ={osr_threshold:.4f}")
            else:
                corr_str = " (CORRECTED)" if corrected else ""
                print(f"  {marker} OSR: {osr_label} | dist={osr_distance:.4f} vs τ={osr_threshold:.4f}{corr_str}")
            
            if is_unknown:
                if is_correct_osr:
                    print(f"  ✓ UNKNOWN CORRECTLY DETECTED BY OSR")
                else:
                    print(f"  ✗ UNKNOWN MISSED - OSR predicted: {osr_label}")
            
            print(f"  {'✓ CLOSED OK' if is_correct_closed else '✗ CLOSED ERR'} | {'✓ OSR OK' if is_correct_osr else '✗ OSR ERR'} | OSR Accuracy: {100*correct_predictions/total_predictions:.1f}%")
            
            # Write to CSV
            timestamp = time.time()
            
            with open(csv_path, "a") as f:
                f.write(
                    f"{iteration},{timestamp},"
                    # Ground truth
                    f"{int(is_unknown)},{true_type},{true_level},{true_class_full},"
                    # Closed-set predictions
                    f"{pred_class_name},{np.max(logits):.4f},{max_prob:.4f},{margin:.4f},"
                    # OSR
                    f"{osr_label},{osr_distance:.4f},{osr_threshold:.4f},{int(corrected)},{best_class},"
                    # Energy/diagnostics
                    f"{energy:.4f},{inference_time_ms:.2f}\n"
                )
            
            # Print summary every 10 iterations
            if iteration % 10 == 0:
                print(f"\n{'='*70}")
                print(f"SUMMARY after {iteration} iterations:")
                print(f"  OSR Accuracy: {100*correct_predictions/total_predictions:.1f}% ({correct_predictions}/{total_predictions})")
                print(f"  Avg inference time: {np.mean(inference_times):.2f}ms (min={np.min(inference_times):.2f}ms, max={np.max(inference_times):.2f}ms)")
                
                # OSR decision distribution
                if use_osr:
                    unknown_count = len([s for s in stats if s.startswith('osr_UNKNOWN')])
                    print(f"  UNKNOWN detections: {stats.get('osr_UNKNOWN', 0)}")
                
                print(f"{'='*70}\n")

            if max_iterations is not None and iteration >= max_iterations:
                print(f"Reached max iterations ({max_iterations}), stopping.")
                break
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*80}")
        print("FINAL SUMMARY:")
        print(f"{'='*80}")
        print(f"Total iterations: {total_predictions}")
        print(f"OSR Accuracy: {100*correct_predictions/total_predictions:.1f}% ({correct_predictions}/{total_predictions})")
        print(f"Avg inference time: {np.mean(inference_times):.2f}ms")
        print(f"Predictions saved to: {csv_path}")
        
        # Load and analyze the CSV
        try:
            results_df = pd.read_csv(csv_path)
            
            print(f"\n{'─'*80}")
            print("OSR DECISION DISTRIBUTION:")
            print(f"{'─'*80}")
            osr_dist = results_df['osr_label'].value_counts()
            for osr_label, count in osr_dist.items():
                pct = 100 * count / len(results_df)
                print(f"  {str(osr_label):20} {count:4} samples ({pct:5.1f}%)")
            
            # Corrections
            if use_osr:
                corrections = results_df['corrected'].sum()
                print(f"\n  Predictions corrected by distance: {corrections} ({100*corrections/len(results_df):.1f}%)")
            
            print(f"\n{'─'*80}")
            print("PERFORMANCE ANALYSIS:")
            print(f"{'─'*80}")
            
            # Known samples
            known_samples = results_df[results_df['is_unknown'] == 0]
            if len(known_samples) > 0:
                known_correct_closed = len(known_samples[known_samples['pred_class'] == known_samples['true_class_full']])
                known_correct_osr = len(known_samples[known_samples['osr_label'] == known_samples['true_class_full']])
                
                print(f"\nKnown Signals ({len(known_samples)} samples):")
                print(f"  Closed-set accuracy: {100*known_correct_closed/len(known_samples):.1f}%")
                print(f"  OSR accuracy: {100*known_correct_osr/len(known_samples):.1f}%")
            
            # Unknown samples
            unknown_samples = results_df[results_df['is_unknown'] == 1]
            if len(unknown_samples) > 0:
                unknown_detected = len(unknown_samples[unknown_samples['osr_label'] == 'UNKNOWN'])
                print(f"\nUnknown Signals ({len(unknown_samples)} samples):")
                print(f"  Correctly detected (as UNKNOWN): {unknown_detected}/{len(unknown_samples)} ({100*unknown_detected/len(unknown_samples):.1f}%)")
                
                # Detection by type
                unknown_types = unknown_samples['true_type'].unique()
                for unk_type in unknown_types:
                    unk_type_samples = unknown_samples[unknown_samples['true_type'] == unk_type]
                    detected = len(unk_type_samples[unk_type_samples['osr_label'] == 'UNKNOWN'])
                    print(f"    {unk_type:25} {detected}/{len(unk_type_samples)} ({100*detected/len(unk_type_samples):.1f}%)")
            
            print(f"\n{'─'*80}")
            print("DISTANCE STATISTICS (to predicted/corrected class):")
            print(f"{'─'*80}")
            dists = results_df['osr_distance']
            print(f"  Mean distance: {dists.mean():.6f}")
            print(f"  Std distance:  {dists.std():.6f}")
            print(f"  Min distance:  {dists.min():.6f}")
            print(f"  Max distance:  {dists.max():.6f}")
            
            print(f"\n{'─'*80}")
            print("INFERENCE TIME:")
            print(f"{'─'*80}")
            times = results_df['inference_time_ms']
            print(f"  Mean: {times.mean():.2f}ms")
            print(f"  Std:  {times.std():.2f}ms")
            print(f"  Min:  {times.min():.2f}ms")
            print(f"  Max:  {times.max():.2f}ms")
            
        except Exception as e:
            print(f"\n⚠ Could not analyze results CSV: {e}")
        
        print("Stopping...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote test with validation-based Open Set Recognition")
    parser.add_argument("--max-iterations", type=int, default=None, help="Stop after N iterations")
    parser.add_argument("--include-unknown", action="store_true", default=True, help="Include unknown jammer types in test")
    parser.add_argument("--known-only", action="store_true", help="Test only known jammers (disable unknown)")
    args = parser.parse_args()
    
    include_unknown = not args.known_only

    main(
        max_iterations=args.max_iterations,
        include_unknown=include_unknown,
    )
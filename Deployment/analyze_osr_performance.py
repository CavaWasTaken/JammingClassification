"""
Comprehensive OSR Performance Analysis
- Unknown detection capability
- Correction of wrong predictions
- Overall OSR effectiveness
"""
import pandas as pd
import numpy as np
from collections import defaultdict

# Load predictions
df = pd.read_csv("./Deployment/predictions.csv")

print("=" * 90)
print("OSR PERFORMANCE ANALYSIS")
print("=" * 90)
print(f"\nTotal iterations: {len(df)}")

# ============================================================================
# 1. UNKNOWN DETECTION PERFORMANCE
# ============================================================================
print("\n" + "=" * 90)
print("1. UNKNOWN DETECTION PERFORMANCE")
print("=" * 90)

unknown_samples = df[df['is_unknown'] == 1]
known_samples = df[df['is_unknown'] == 0]

print(f"\nTotal unknown samples: {len(unknown_samples)}")
print(f"Total known samples: {len(known_samples)}")

if len(unknown_samples) > 0:
    unknown_detected = len(unknown_samples[unknown_samples['osr_label'] == 'UNKNOWN'])
    unknown_missed = len(unknown_samples[unknown_samples['osr_label'] != 'UNKNOWN'])
    
    detection_rate = 100 * unknown_detected / len(unknown_samples)
    miss_rate = 100 * unknown_missed / len(unknown_samples)
    
    print(f"\nUnknown Detection Rate: {unknown_detected}/{len(unknown_samples)} ({detection_rate:.1f}%)")
    print(f"Unknown Miss Rate: {unknown_missed}/{len(unknown_samples)} ({miss_rate:.1f}%)")
    
    # By type
    print(f"\nDetection by jammer type:")
    for unk_type in unknown_samples['true_type'].unique():
        unk_type_samples = unknown_samples[unknown_samples['true_type'] == unk_type]
        detected = len(unk_type_samples[unk_type_samples['osr_label'] == 'UNKNOWN'])
        print(f"  {unk_type:30} {detected}/{len(unk_type_samples)} ({100*detected/len(unk_type_samples):.1f}%)")
    
    # Analyze misses
    unknown_missed_df = unknown_samples[unknown_samples['osr_label'] != 'UNKNOWN']
    if len(unknown_missed_df) > 0:
        print(f"\nWhen unknown was NOT detected, it was classified as:")
        false_labels = unknown_missed_df['osr_label'].value_counts()
        for label, count in false_labels.items():
            print(f"  {label:20} {count:3} times ({100*count/len(unknown_missed_df):.1f}%)")

# ============================================================================
# 2. CORRECTION OF WRONG PREDICTIONS ON KNOWN SAMPLES
# ============================================================================
print("\n" + "=" * 90)
print("2. CORRECTION OF WRONG PREDICTIONS ON KNOWN SAMPLES")
print("=" * 90)

# Known samples that were initially misclassified by the model
known_wrong = known_samples[known_samples['pred_class'] != known_samples['true_class_full']]
print(f"\nTotal known samples: {len(known_samples)}")
print(f"Model's closed-set errors: {len(known_wrong)} ({100*len(known_wrong)/len(known_samples):.1f}%)")

if len(known_wrong) > 0:
    # Did OSR correct these?
    corrected_to_true = len(known_wrong[known_wrong['osr_label'] == known_wrong['true_class_full']])
    rejected_as_unknown = len(known_wrong[known_wrong['osr_label'] == 'UNKNOWN'])
    still_wrong = len(known_wrong[
        (known_wrong['osr_label'] != known_wrong['true_class_full']) & 
        (known_wrong['osr_label'] != 'UNKNOWN')
    ])
    
    print(f"\n  When model was wrong, OSR:")
    print(f"    ✓ Corrected to true class:     {corrected_to_true} ({100*corrected_to_true/len(known_wrong):.1f}%)")
    print(f"    ❌ Rejected as UNKNOWN:        {rejected_as_unknown} ({100*rejected_as_unknown/len(known_wrong):.1f}%)")
    print(f"    ✗ Still predicted wrong:       {still_wrong} ({100*still_wrong/len(known_wrong):.1f}%)")
    
    # This is interesting: corrections can be rejections as unknown (conservative) 
    # or corrections to another known class (aggressive)
    print(f"\n  Conservative vs Aggressive Correction:")
    conservative = corrected_to_true + rejected_as_unknown
    print(f"    Conservative (correct or reject): {conservative}/{len(known_wrong)} ({100*conservative/len(known_wrong):.1f}%)")
    print(f"    Aggressive (preserve wrong):      {still_wrong}/{len(known_wrong)} ({100*still_wrong/len(known_wrong):.1f}%)")

# ============================================================================
# 3. CORRECTION USAGE
# ============================================================================
print("\n" + "=" * 90)
print("3. CORRECTION LOGIC USAGE")
print("=" * 90)

corrections = len(df[df['corrected'] == 1])
print(f"\nSamples where correction was applied: {corrections}/{len(df)} ({100*corrections/len(df):.1f}%)")

if corrections > 0:
    # What was corrected?
    corrected_df = df[df['corrected'] == 1]
    
    # Check if corrections helped or hurt
    corrected_correct = len(corrected_df[corrected_df['osr_label'] == corrected_df['true_class_full']])
    corrected_wrong = corrections - corrected_correct
    
    print(f"\nOutcome of corrections:")
    print(f"  ✓ Correction was beneficial:     {corrected_correct}/{corrections} ({100*corrected_correct/corrections:.1f}%)")
    print(f"  ✗ Correction made it worse:      {corrected_wrong}/{corrections} ({100*corrected_wrong/corrections:.1f}%)")

# ============================================================================
# 4. OVERALL OSR ACCURACY
# ============================================================================
print("\n" + "=" * 90)
print("4. OVERALL OSR ACCURACY")
print("=" * 90)

osr_correct = len(df[df['osr_label'] == df['true_class_full']])
osr_incorrect = len(df) - osr_correct

print(f"\nOSR Accuracy: {osr_correct}/{len(df)} ({100*osr_correct/len(df):.1f}%)")

# Separate accuracy for known vs unknown
if len(known_samples) > 0:
    known_correct = len(known_samples[known_samples['osr_label'] == known_samples['true_class_full']])
    known_acc = 100 * known_correct / len(known_samples)
    print(f"  Known samples: {known_correct}/{len(known_samples)} ({known_acc:.1f}%)")

if len(unknown_samples) > 0:
    unknown_correct = len(unknown_samples[unknown_samples['osr_label'] == unknown_samples['true_class_full']])
    unknown_acc = 100 * unknown_correct / len(unknown_samples)
    print(f"  Unknown samples: {unknown_correct}/{len(unknown_samples)} ({unknown_acc:.1f}%)")

# ============================================================================
# 5. ERROR ANALYSIS
# ============================================================================
print("\n" + "=" * 90)
print("5. ERROR ANALYSIS")
print("=" * 90)

errors_df = df[df['osr_label'] != df['true_class_full']]
print(f"\nTotal OSR errors: {len(errors_df)}")

if len(errors_df) > 0:
    # Breakdown by type
    known_errors = errors_df[errors_df['is_unknown'] == 0]
    unknown_errors = errors_df[errors_df['is_unknown'] == 1]
    
    print(f"  Known samples misclassified: {len(known_errors)}")
    print(f"  Unknown samples misclassified: {len(unknown_errors)}")
    
    # Common misclassifications
    if len(known_errors) > 0:
        print(f"\nTop known sample misclassifications:")
        known_errors['error'] = known_errors['true_class_full'] + ' → ' + known_errors['osr_label']
        error_counts = known_errors['error'].value_counts().head(5)
        for error, count in error_counts.items():
            print(f"  {error:50} {count:3} times")
    
    if len(unknown_errors) > 0:
        print(f"\nWhen unknown samples were misclassified:")
        unknown_errors['misclass_as'] = unknown_errors['osr_label']
        misclass_counts = unknown_errors['misclass_as'].value_counts().head(5)
        for cls, count in misclass_counts.items():
            print(f"  Misclassified as {cls:20} {count:3} times ({100*count/len(unknown_errors):.1f}%)")

# ============================================================================
# 6. DISTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 90)
print("6. DISTANCE ANALYSIS")
print("=" * 90)

print(f"\nDistance to predicted class centroid:")
print(f"  Mean: {df['osr_distance'].mean():.2f}")
print(f"  Std:  {df['osr_distance'].std():.2f}")
print(f"  Min:  {df['osr_distance'].min():.2f}")
print(f"  Max:  {df['osr_distance'].max():.2f}")

print(f"\nThreshold analysis:")
print(f"  Mean threshold: {df['osr_threshold'].mean():.2f}")
print(f"  Std threshold:  {df['osr_threshold'].std():.2f}")

# Distance vs threshold
within_threshold = len(df[df['osr_distance'] <= df['osr_threshold']])
beyond_threshold = len(df) - within_threshold
print(f"\nSamples within threshold: {within_threshold}/{len(df)} ({100*within_threshold/len(df):.1f}%)")
print(f"Samples beyond threshold: {beyond_threshold}/{len(df)} ({100*beyond_threshold/len(df):.1f}%)")

# For samples beyond threshold, what happened?
beyond_df = df[df['osr_distance'] > df['osr_threshold']]
if len(beyond_df) > 0:
    beyond_unknown = len(beyond_df[beyond_df['osr_label'] == 'UNKNOWN'])
    beyond_corrected = len(beyond_df[beyond_df['corrected'] == 1])
    print(f"  Of those beyond threshold:")
    print(f"    Rejected as UNKNOWN: {beyond_unknown}/{len(beyond_df)} ({100*beyond_unknown/len(beyond_df):.1f}%)")
    print(f"    Corrected to other class: {beyond_corrected}/{len(beyond_df)} ({100*beyond_corrected/len(beyond_df):.1f}%)")

# ============================================================================
# 7. EFFICIENCY METRICS
# ============================================================================
print("\n" + "=" * 90)
print("7. INFERENCE EFFICIENCY")
print("=" * 90)

print(f"\nInference time per sample:")
print(f"  Mean: {df['inference_time_ms'].mean():.2f}ms")
print(f"  Std:  {df['inference_time_ms'].std():.2f}ms")
print(f"  Min:  {df['inference_time_ms'].min():.2f}ms")
print(f"  Max:  {df['inference_time_ms'].max():.2f}ms")

print("\n" + "=" * 90)

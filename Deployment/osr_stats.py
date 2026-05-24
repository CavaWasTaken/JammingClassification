import os
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')


def get_jamming_level(power_level):
    """Map power level (P2-P8) to jamming level (LOW/MID/HIGH)."""
    level_map = {
        'P2': 'LOW', 'P3': 'LOW',
        'P4': 'MID', 'P5': 'MID', 'P6': 'MID',
        'P7': 'HIGH', 'P8': 'HIGH',
    }
    return level_map.get(power_level, 'UNKNOWN')

def extract_true_class(spec_path, class_name):
    """Extract full class label (type_level) from path and type."""
    if class_name == 'CLEAN':
        return 'CLEAN'
    path_parts = str(spec_path).split('/')
    if len(path_parts) >= 5:
        power_level = path_parts[-2]
        jamming_level = get_jamming_level(power_level)
        return f"{class_name}_{jamming_level}"
    return class_name


def run_model(sess, spec, features):
    inputs = {
        'spectrogram': spec[np.newaxis, np.newaxis, :, :].astype(np.float32),
        'features': features[np.newaxis, :].astype(np.float32),
    }
    outputs = sess.run(None, inputs)
    logits = outputs[0][0]
    penultimate = outputs[1][0]
    energy = float(outputs[2][0])
    return logits, penultimate, energy

def save_osr_stats(file_path, stats):
    """Save OSR statistics (centroids and thresholds) to npz file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savez_compressed(
        file_path,
        classes=np.array(stats['classes'], dtype=object),
        centroids=stats['centroids'].astype(np.float32),
        thresholds=np.array(stats['thresholds'], dtype=object),  # Save dict with allow_pickle
    )


def load_osr_stats(file_path):
    data = np.load(file_path, allow_pickle=True)
    classes = [str(c) for c in data['classes'].tolist()]
    centroids = data['centroids'].astype(np.float32)
    thresholds_dict = data['thresholds'].item()  # Convert dict from npz
    
    # Ensure thresholds are floats
    thresholds = {cls: float(thresholds_dict[cls]) for cls in classes}

    return {
        'classes': classes,
        'centroids': centroids,
        'thresholds': thresholds,
        'class_to_idx': {cls: i for i, cls in enumerate(classes)}
    }


def evaluate_osr(penultimate, logits, osr_stats, class_names):
    """
    Evaluate OSR prediction with correction logic.
    
    Strategy:
    1. Get predicted class from logits using class_names order (model's output order)
    2. Compute distances to all class centroids
    3. If dist(x, μ_ŷ) > τ_ŷ:
       - Check if distance to any other class k* < τ_k*
       - If yes, correct to k*
       - If no, return UNKNOWN
    4. Otherwise accept prediction
    
    Args:
        penultimate: Embedding from penultimate layer
        logits: Logits from model (ordered by class_names)
        osr_stats: Dict with 'classes', 'centroids', 'thresholds', 'class_to_idx' (sorted order)
        class_names: List of class names in model output order
    
    Returns:
        dict with:
        - 'osr_label': Final OSR prediction (class name or 'UNKNOWN')
        - 'pred_class': Original model prediction
        - 'distance_to_pred': Distance to predicted class centroid
        - 'threshold_pred': Threshold for predicted class
        - 'distance_to_best': Distance to closest class centroid
        - 'best_class': Class with minimum distance
        - 'corrected': Whether prediction was corrected
    """
    classes_sorted = osr_stats['classes']  # Sorted order
    centroids = osr_stats['centroids']
    thresholds = osr_stats['thresholds']
    
    # Map from sorted classes to indices
    class_to_sorted_idx = {cls: i for i, cls in enumerate(classes_sorted)}
    
    # Get original prediction from logits using MODEL's class_names order
    pred_class_idx = np.argmax(logits)
    pred_class = class_names[pred_class_idx]  # Use model's class order
    
    # Compute distances to all classes using the centroids (which are in sorted order)
    distances = {}
    for cls in class_names:  # Iterate through all classes in model order
        sorted_idx = class_to_sorted_idx[cls]
        dist = float(np.linalg.norm(penultimate - centroids[sorted_idx]))
        distances[cls] = dist
    
    dist_to_pred = distances[pred_class]
    threshold_pred = thresholds[pred_class]
    
    # Find class with minimum distance and its distance/threshold
    best_class = min(distances, key=distances.get)
    dist_to_best = distances[best_class]
    threshold_best = thresholds[best_class]
    
    # Decision logic
    if dist_to_pred <= threshold_pred:
        # Predicted class is within threshold - ACCEPT
        osr_label = pred_class
        corrected = False
    else:
        # Predicted class is outside threshold - CHECK CORRECTION
        if best_class != pred_class and dist_to_best <= threshold_best:
            # Closest class is different and within its threshold - CORRECT
            osr_label = best_class
            corrected = True
        else:
            # All distances exceed thresholds - UNKNOWN
            osr_label = 'UNKNOWN'
            corrected = False
    
    return {
        'osr_label': osr_label,
        'pred_class': pred_class,
        'distance_to_pred': dist_to_pred,
        'threshold_pred': threshold_pred,
        'distance_to_best': dist_to_best,
        'best_class': best_class,
        'corrected': corrected,
        'all_distances': distances,  # For debugging
    }

def compute_centroids(sess, df, class_names):
    embeddings_by_class = defaultdict(list)
    skipped = 0
    
    for idx, row in df.iterrows():
        spec_path = row['path_spec']
        true_class_full = extract_true_class(spec_path, row['class'])
        
        if true_class_full not in class_names or not os.path.exists(spec_path):
            skipped += 1
            continue
        
        try:
            spec = np.load(spec_path)
            features = row.drop(['path_spec', 'power', 'class', 'is_unknown', 'unknown_type'], errors='ignore').values.astype(np.float32)
            _, penultimate, _ = run_model(sess, spec, features)
            embeddings_by_class[true_class_full].append(penultimate.astype(np.float32))
        except Exception as e:
            skipped += 1
            continue
    
    classes = sorted(embeddings_by_class.keys())
    centroids = []
    counts = []
    
    for cls in classes:
        emb = np.vstack(embeddings_by_class[cls])
        centroid = emb.mean(axis=0)
        centroids.append(centroid)
        counts.append(len(embeddings_by_class[cls]))
    
    centroids = np.vstack(centroids).astype(np.float32)
    counts = np.array(counts, dtype=np.int32)
    
    return classes, centroids, counts, skipped


def calibrate_thresholds_on_validation(sess, train_df, val_known_df, val_unknown_df, class_names, max_fpr=0.05, random_state=42):
    """
    Calibrate per-class thresholds using ROC curve on validation set.
    
    Strategy:
    - For each class, compute distances from validation known samples
    - Use ROC curve: x=FPR on known validation, y=TPR on unknown
    - Choose operating point with max_fpr constraint
    - Returns per-class thresholds
    
    Args:
        sess: ONNX session
        train_df: Training data for computing centroids
        val_known_df: Validation known samples (30% of known data)
        val_unknown_df: All unknown samples for validation
        class_names: List of class names
        max_fpr: Maximum false positive rate to tolerate (default 5%)
        random_state: Random seed
    
    Returns:
        dict with 'classes', 'centroids', 'thresholds'
    """
    np.random.seed(random_state)
    
    # Step 1: Build centroids from training data only
    print("  Step 1: Building centroids from training data...")
    classes, centroids, counts, skipped = compute_centroids(sess, train_df, class_names)
    
    print(f"  Built {len(classes)} class centroids from {len(train_df)} training samples (skipped: {skipped})")
    
    # Debug: save centroids
    centroid_file = "centroids_debug.csv"
    with open(centroid_file, 'w') as f:
        f.write("class," + ",".join([f"dim_{i}" for i in range(centroids.shape[1])]) + "\n")
        for cls, centroid in zip(classes, centroids):
            f.write(cls + "," + ",".join([f"{c:.6f}" for c in centroid]) + "\n")
    print(f"  ✓ Centroids saved to {centroid_file}")
    
    # Step 2: Compute distances on validation set
    print(f"\n  Step 2: Computing distances on validation set...")
    print(f"    Known validation: {len(val_known_df)} samples")
    print(f"    Unknown validation: {len(val_unknown_df)} samples")
    
    # Collect distances for each class
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    val_distances_data = []
    
    # Process known validation samples
    print("    Processing known samples...")
    val_known_skipped = 0
    for idx, row in val_known_df.iterrows():
        spec_path = row['path_spec']
        true_class_full = extract_true_class(spec_path, row['class'])
        
        if not os.path.exists(spec_path) or true_class_full not in class_to_idx:
            val_known_skipped += 1
            continue
        
        try:
            spec = np.load(spec_path)
            features = row.drop(['path_spec', 'power', 'class', 'is_unknown', 'unknown_type'], errors='ignore').values.astype(np.float32)
            logits, penultimate, _ = run_model(sess, spec, features)
            
            # Compute distances to all class centroids
            distances = {}
            for cls_idx, cls in enumerate(classes):
                dist = float(np.linalg.norm(penultimate - centroids[cls_idx]))
                distances[f"dist_to_{cls}"] = dist
            
            # Get predicted class
            pred_class_idx = np.argmax(logits)
            pred_class = classes[pred_class_idx]
            
            row_data = {
                'is_unknown': False,
                'true_class': true_class_full,
                'pred_class': pred_class,
            }
            row_data.update(distances)
            val_distances_data.append(row_data)
            
        except Exception as e:
            val_known_skipped += 1
            continue
    
    # Process unknown validation samples
    print("    Processing unknown samples...")
    val_unknown_skipped = 0
    for idx, row in val_unknown_df.iterrows():
        spec_path = row['path_spec']
        
        if not os.path.exists(spec_path):
            val_unknown_skipped += 1
            continue
        
        try:
            spec = np.load(spec_path)
            features = row.drop(['path_spec', 'power', 'class', 'is_unknown', 'unknown_type'], errors='ignore').values.astype(np.float32)
            logits, penultimate, _ = run_model(sess, spec, features)
            
            # Compute distances to all class centroids
            distances = {}
            for cls_idx, cls in enumerate(classes):
                dist = float(np.linalg.norm(penultimate - centroids[cls_idx]))
                distances[f"dist_to_{cls}"] = dist
            
            pred_class_idx = np.argmax(logits)
            pred_class = classes[pred_class_idx]
            
            row_data = {
                'is_unknown': True,
                'true_class': 'UNKNOWN',
                'pred_class': pred_class,
            }
            row_data.update(distances)
            val_distances_data.append(row_data)
            
        except Exception as e:
            val_unknown_skipped += 1
            continue
    
    val_df = pd.DataFrame(val_distances_data)
    print(f"    ✓ Processed {len(val_df)} samples (skipped known: {val_known_skipped}, skipped unknown: {val_unknown_skipped})")
    
    # Debug: save distances
    val_df.to_csv("validation_distances_debug.csv", index=False)
    print(f"    ✓ Distances saved to validation_distances_debug.csv")
    
    # Step 3: Compute per-class thresholds using ROC curve
    print(f"\n  Step 3: Computing per-class thresholds using ROC curve...")
    best_thresholds = {}
    roc_data = []
    
    for cls in classes:
        print(f"    Computing threshold for class: {cls}...")
        
        # For this class:
        # - Positive samples: known validation samples where true_class == cls
        # - Negative samples: unknown samples + known samples where true_class != cls
        
        # Get distances to this class centroid
        dist_col = f"dist_to_{cls}"
        
        # Positive: known validation samples of target class
        pos_distances = val_df[(val_df['is_unknown'] == False) & (val_df['true_class'] == cls)][dist_col].values
        
        # Negative: unknown + other known samples
        neg_distances = val_df[(val_df['is_unknown'] == True) | (val_df['true_class'] != cls)][dist_col].values
        
        if len(pos_distances) == 0 or len(neg_distances) == 0:
            print(f"      ⚠ Skipping {cls}: not enough positive ({len(pos_distances)}) or negative ({len(neg_distances)}) samples")
            best_thresholds[cls] = float('inf')
            continue
        
        # Build labels: 1 for positive (should accept), 0 for negative (should reject)
        y_true = np.concatenate([np.ones(len(pos_distances)), np.zeros(len(neg_distances))])
        distances_all = np.concatenate([pos_distances, neg_distances])
        
        # Compute ROC curve: FPR vs TPR
        # FPR = false positives / all negatives (how many unknown are accepted?)
        # TPR = true positives / all positives (how many known are accepted?)
        # Note: roc_curve expects higher scores = more positive, so we negate distances
        fpr, tpr, thresholds_negated = roc_curve(y_true, -distances_all)
        
        # Find operating point with FPR <= max_fpr
        valid_idx = fpr <= max_fpr
        if not np.any(valid_idx):
            print(f"      ⚠ No valid operating point with FPR <= {max_fpr}. Using minimum FPR.")
            valid_idx = fpr == np.min(fpr)
        
        # Among valid operating points, choose the one with maximum TPR
        best_idx = np.argmax(tpr[valid_idx])
        best_idx_in_full = np.where(valid_idx)[0][best_idx]
        
        best_fpr = fpr[best_idx_in_full]
        best_tpr = tpr[best_idx_in_full]
        # Convert threshold back to positive distance space by negating
        best_threshold = -thresholds_negated[best_idx_in_full]
        best_thresholds[cls] = float(best_threshold)
        
        roc_auc = auc(fpr, tpr)
        
        print(f"      ✓ FPR={best_fpr:.4f}, TPR={best_tpr:.4f}, Threshold={best_threshold:.6f}, AUC={roc_auc:.4f}")
        roc_data.append({
            'class': cls,
            'fpr': best_fpr,
            'tpr': best_tpr,
            'threshold': best_threshold,
            'auc': roc_auc,
            'n_positive': len(pos_distances),
            'n_negative': len(neg_distances),
        })
    
    # Debug: save thresholds and ROC data
    threshold_df = pd.DataFrame([
        {'class': cls, 'threshold': best_thresholds[cls]}
        for cls in classes
    ])
    threshold_df.to_csv("best_thresholds_debug.csv", index=False)
    print(f"\n  ✓ Thresholds saved to best_thresholds_debug.csv")
    
    roc_df = pd.DataFrame(roc_data)
    roc_df.to_csv("roc_curve_debug.csv", index=False)
    print(f"  ✓ ROC data saved to roc_curve_debug.csv")
    
    print(f"\n  Per-class thresholds (max_fpr={max_fpr}):")
    for cls in classes:
        print(f"    {cls:20} τ = {best_thresholds[cls]:.6f}")
    
    return {
        'classes': classes,
        'centroids': centroids.astype(np.float32),
        'thresholds': best_thresholds,
    }
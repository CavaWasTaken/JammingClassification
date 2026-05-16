import os
import time
import numpy as np
from collections import defaultdict


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


def build_osr_stats(sess, features_df, class_names, quantile=0.95, max_samples=None, random_state=42):
    """Build per-class centroid and distance threshold from penultimate embeddings."""
    if len(features_df) == 0:
        raise ValueError("Cannot compute OSR stats with an empty dataframe")

    if max_samples is None:
        calib_df = features_df.reset_index(drop=True)
    else:
        n = min(max_samples, len(features_df))
        calib_df = features_df.sample(n=n, random_state=random_state).reset_index(drop=True)

    embeddings_by_class = defaultdict(list)
    skipped = 0

    for _, row in calib_df.iterrows():
        spec_path = row['path_spec']
        true_class_full = extract_true_class(spec_path, row['class'])

        if true_class_full not in class_names:
            skipped += 1
            continue
        if not os.path.exists(spec_path):
            skipped += 1
            continue

        try:
            spec = np.load(spec_path)
            features = row.drop(['path_spec', 'power', 'class']).values.astype(np.float32)
            _, penultimate, _ = run_model(sess, spec, features)
            embeddings_by_class[true_class_full].append(penultimate.astype(np.float32))
        except Exception:
            skipped += 1
            continue

    classes = sorted(embeddings_by_class.keys())
    if not classes:
        raise RuntimeError("OSR stats computation failed: no valid classes found")

    centroids = []
    thresholds = []
    counts = []

    for cls in classes:
        emb = np.vstack(embeddings_by_class[cls])
        centroid = emb.mean(axis=0)
        dists = np.linalg.norm(emb - centroid, axis=1)
        centroids.append(centroid)
        thresholds.append(float(np.quantile(dists, quantile)))
        counts.append(int(len(embeddings_by_class[cls])))

    return {
        'classes': classes,
        'centroids': np.vstack(centroids).astype(np.float32),
        'thresholds': np.array(thresholds, dtype=np.float32),
        'counts': np.array(counts, dtype=np.int32),
        'quantile': float(quantile),
        'skipped': int(skipped),
        'used_rows': int(sum(counts)),
        'requested_rows': int(len(calib_df)),
        'created_at': int(time.time()),
    }


def save_osr_stats(file_path, stats):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savez_compressed(
        file_path,
        classes=np.array(stats['classes'], dtype=object),
        centroids=stats['centroids'],
        thresholds=stats['thresholds'],
        counts=stats['counts'],
        quantile=np.array(stats['quantile'], dtype=np.float32),
        skipped=np.array(stats['skipped'], dtype=np.int32),
        used_rows=np.array(stats['used_rows'], dtype=np.int32),
        requested_rows=np.array(stats['requested_rows'], dtype=np.int32),
        created_at=np.array(stats['created_at'], dtype=np.int64),
    )


def load_osr_stats(file_path):
    data = np.load(file_path, allow_pickle=True)
    classes = [str(c) for c in data['classes'].tolist()]
    centroids = data['centroids'].astype(np.float32)
    thresholds = data['thresholds'].astype(np.float32)
    counts = data['counts'].astype(np.int32)

    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    return {
        'classes': classes,
        'centroids': centroids,
        'thresholds': thresholds,
        'counts': counts,
        'class_to_idx': class_to_idx,
        'quantile': float(data['quantile'].item()),
        'skipped': int(data['skipped'].item()),
        'used_rows': int(data['used_rows'].item()),
        'requested_rows': int(data['requested_rows'].item()),
        'created_at': int(data['created_at'].item()),
    }


def estimate_power_level(penultimate, stats):
    """Estimate power level (LOW/MID/HIGH) for an embedding by finding closest power-level group."""
    power_levels = ['LOW', 'MID', 'HIGH']
    power_distances = {}
    
    for power in power_levels:
        # Find all classes matching this power level
        matching_indices = [
            i for i, cls in enumerate(stats['classes'])
            if cls.endswith(f'_{power}')
        ]
        
        if not matching_indices:
            power_distances[power] = float('inf')
        else:
            # Compute mean distance to all centroids of this power level
            dists = [
                float(np.linalg.norm(penultimate - stats['centroids'][i]))
                for i in matching_indices
            ]
            power_distances[power] = float(np.mean(dists))
    
    # Return power level with minimum distance
    estimated_power = min(power_distances, key=power_distances.get)
    return estimated_power, power_distances


def evaluate_osr(penultimate, pred_class_name, stats):
    """Evaluate whether penultimate embedding is realistic for the predicted class."""
    idx = stats['class_to_idx'].get(pred_class_name)
    if idx is None:
        return {
            'osr_label': 'UNKNOWN',
            'distance': float('inf'),
            'threshold': float('inf'),
            'is_realistic': False,
            'estimated_power': None,
        }

    centroid = stats['centroids'][idx]
    threshold = float(stats['thresholds'][idx])
    distance = float(np.linalg.norm(penultimate - centroid))
    is_realistic = distance <= threshold

    return {
        'osr_label': pred_class_name if is_realistic else 'UNKNOWN',
        'distance': distance,
        'threshold': threshold,
        'is_realistic': is_realistic,
        'estimated_power': None,
    }

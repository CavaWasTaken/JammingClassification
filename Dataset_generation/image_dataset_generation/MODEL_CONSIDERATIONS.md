# Jamming Detection System: Comprehensive Analysis

---

## 1. Adding CLEAN Samples from Different Antenna

### Implications

**Positive aspects:**
- **More balanced dataset** - Reduces 1:3 imbalance (currently 2,880 vs 8,800)
- **Better CLEAN detection** - Model learns cleaner signals from multiple sources
- **Robustness** - Generalizes to different antenna characteristics

**Potential challenges:**
- **Distribution shift** - Different antenna = different signal characteristics
- **Model confusion** - May learn antenna-specific features instead of jamming patterns
- **New imbalance direction** - If you add too many, jamming classes become minority

### Recommendations

**Data collection:**
- Collect similar amount (~5,000-6,000 more CLEAN samples) to balance with jamming classes
- Document antenna specifications (gain, frequency response, polarization)
- Ensure same recording conditions (location, time, interference sources)

**Dataset strategy:**
- **Option 1 (Recommended):** Mix both antenna CLEAN samples in one class
  - Model learns "CLEAN = no jamming regardless of antenna"
  - Better generalization
  
- **Option 2:** Separate subfolders for each antenna
  - Like power levels in jamming classes
  - Helps track which antenna causes issues
  
**Training adaptation:**
- Re-compute class weights after adding samples
- May no longer need weighted loss if balanced
- Monitor per-antenna performance in validation

**Testing consideration:**
- Include both antenna types in test set
- Check confusion matrix: does model confuse antenna1-CLEAN with jamming?
- Verify model doesn't overfit to antenna characteristics

---

## 2. Open Set Recognition (OSR) Implementation

### What OSR Does

**Closed Set (Current):** Model predicts one of 6 known classes (forces choice even for unknown jamming)

**Open Set (Goal):** Model can say "unknown jamming" when encountering novel interference patterns

### OSR Strategies for Your System

#### Strategy 1: Threshold-Based (Simplest)

**How it works:**
- Model outputs probabilities for 6 classes
- If `max(probability) < threshold` → "unknown"
- If `max(probability) ≥ threshold` → predict that class

**Implementation:**
```python
outputs = model(inputs)
probs = F.softmax(outputs, dim=1)
max_prob, predicted = torch.max(probs, 1)

# Apply threshold
threshold = 0.7  # tune this
unknown_mask = max_prob < threshold
predicted[unknown_mask] = 6  # class 6 = "UNKNOWN"
```

**Advantages:**
- Easy to implement (2 lines of code)
- Works with your current ResNet-18
- Interpretable

**Disadvantages:**
- Fixed threshold may not work for all unknown types
- No learned distinction between "uncertain" and "truly unknown"

**When to use:** Quick baseline, proof of concept

---

#### Strategy 2: OpenMax (Recommended)

**How it works:**
- Fits statistical model (Weibull distribution) on known class distances
- Estimates probability that sample belongs to unknown class
- Calibrates softmax scores based on training data distribution

**Advantages:**
- Specifically designed for OSR in deep learning
- Better than simple thresholding
- Proven in research

**Disadvantages:**
- Requires additional implementation (~50 lines)
- Need to store class means from training data
- Slightly slower inference

**Implementation steps:**
1. Extract features from penultimate layer during training
2. Compute class mean activation vectors (MAVs)
3. Fit Weibull distributions on distances to MAVs
4. At test time, adjust softmax with OpenMax calibration

**When to use:** Production system, best OSR performance

---

#### Strategy 3: Distance-Based (Feature Space)

**How it works:**
- Extract feature embeddings from ResNet before final FC layer
- Compute distance to nearest known class centroid
- If distance > threshold → "unknown"

**Advantages:**
- Uses feature space geometry (more principled than softmax threshold)
- Can visualize clusters with t-SNE/UMAP
- Works well when classes are well-separated

**Disadvantages:**
- Need to choose distance metric (Euclidean, cosine, Mahalanobis)
- Threshold tuning required
- Assumes compact class clusters

**When to use:** When classes have distinct feature representations

---

#### Strategy 4: One-Class SVM on Features

**How it works:**
- Train separate one-class classifier per known class on ResNet features
- Each classifier learns boundary of that class
- Sample is unknown if rejected by all classifiers

**Advantages:**
- Principled boundary learning
- Can handle non-spherical class distributions

**Disadvantages:**
- Train 6 additional classifiers
- More complex pipeline
- Slower inference

**When to use:** When you have complex class boundaries

---

### OSR Recommendation

**Phase 1 (Immediate):** Start with **threshold-based** approach
- Quick implementation
- Establish baseline OSR performance
- Tune threshold on validation set with known unknowns

**Phase 2 (Production):** Implement **OpenMax**
- Best balance of performance and complexity
- Research-proven for CNNs
- Handles multiple unknown types better

**How to evaluate OSR:**
- Create "unknown" test set with new jamming types (not in training)
- Measure: AUROC, AUPR for unknown detection
- Check: Does model reject unknowns while accepting knowns?

---

## 3. Model Architecture Comparison

### ResNet-18 (CNN) - Current Choice

**Architecture:** Convolutional layers → Residual blocks → Spatial pooling → FC

**Best for:**
- 2D spectrograms with spatial patterns
- Frequency-time structures (jamming signatures)
- When patterns are localized in time-frequency grid

**Strengths:**
- Captures spatial hierarchies (edges → patterns → signatures)
- Proven on spectrograms (audio/RF similar to images)
- Fast inference
- Good OSR compatibility (feature space well-structured)

**Weaknesses:**
- Doesn't explicitly model temporal dependencies
- Treats time and frequency dimensions equally (may not be ideal)

**Expected performance:** 85-95% accuracy on your 6 classes

**OSR adaptation:** Excellent - OpenMax works very well with CNNs

---

### GRU (RNN) - Referenced Study's Choice

**Architecture:** Sequential processing → Hidden states → Temporal aggregation → FC

**Best for:**
- Time-series data with temporal dependencies
- When order of samples matters
- Detecting evolving jamming patterns over time

**Strengths:**
- Explicitly models temporal dynamics
- Memory mechanism captures long-range dependencies
- Can handle variable-length sequences
- Smaller model size than CNNs

**Weaknesses:**
- Slower training (sequential, not parallelizable)
- Requires reshaping 2D spectrograms to 1D sequences
- May miss spatial frequency patterns
- Needs more epochs to converge

**Expected performance:** 80-90% accuracy (slightly lower than CNN for spectrograms)

**OSR adaptation:** Moderate
- Can use threshold on final hidden state
- OpenMax possible but less common in literature
- Distance-based on hidden state embeddings works

**When GRU is better:**
- If you have raw IQ time-series (not spectrograms)
- If jamming evolves over time (e.g., sweeping, hopping)
- If temporal order is critical to detection

---

### Hybrid CNN-RNN (CNN + GRU)

**Architecture:** CNN feature extraction → Sequence of features → GRU → FC

**Best for:**
- Combining spatial (frequency patterns) and temporal (evolution) features
- Complex jamming that changes over time
- When both frequency structure and temporal dynamics matter

**How it works:**
1. Split spectrogram into time windows
2. CNN extracts features per window (spatial patterns)
3. GRU processes sequence of CNN features (temporal evolution)
4. Final prediction combines both

**Strengths:**
- Best of both worlds (spatial + temporal)
- Can detect evolving jamming signatures
- More expressive than either alone

**Weaknesses:**
- More complex architecture
- Harder to train (more hyperparameters)
- Slower inference
- Needs more data to avoid overfitting
- 2-3x longer training time

**Expected performance:** 90-95% accuracy (best, if sufficient data)

**OSR adaptation:** Good
- Can apply OSR at CNN features, GRU hidden state, or final output
- Multiple points for uncertainty estimation

**When to use:** If ResNet-18 plateaus and you need more capacity

---

### Transformer (Attention-Based)

**Architecture:** Self-attention layers → Positional encoding → Attention pooling → FC

**Best for:**
- Large datasets (10k+ samples per class)
- Long-range dependencies in spectrograms
- When you have computational resources

**Strengths:**
- Captures global dependencies (any time-frequency relationship)
- Parallel training (faster than RNNs)
- State-of-the-art on many sequence tasks

**Weaknesses:**
- Needs MUCH more data than CNNs/RNNs
- Memory intensive
- Computationally expensive
- Overkill for 46k samples

**Expected performance:** 85-92% (may not beat ResNet with your dataset size)

**OSR adaptation:** Good (attention weights can indicate uncertainty)

**When to use:** Future work if you scale to 100k+ samples

---

### 1D CNN (Direct on Time Series)

**Architecture:** 1D convolutions on raw IQ samples → Pooling → FC

**Best for:**
- Raw time-series data (not spectrograms)
- Avoiding preprocessing overhead
- Real-time embedded systems

**Strengths:**
- Learns features directly from raw signal
- No spectrogram computation needed
- Faster inference (if no preprocessing)
- End-to-end learning

**Weaknesses:**
- May need deeper network than 2D CNN
- Doesn't leverage time-frequency structure explicitly
- Requires very long input sequences

**Expected performance:** 80-90% (comparable to GRU)

**OSR adaptation:** Same as ResNet-18 (excellent)

**When to use:** Real-time systems, avoiding preprocessing

---

## 4. Comprehensive Recommendation

### For Your Current Phase

**Model choice:** **Stick with ResNet-18** ✓

**Reasons:**
1. Spectrograms are 2D → CNNs are natural fit
2. Your jamming types have spatial signatures (frequency patterns)
3. Fast training (~30 min vs 1-2 hours for RNN)
4. Excellent OSR compatibility
5. Easy to debug and interpret
6. Proven architecture

**OSR implementation:** **Threshold-based first, then OpenMax**

**Workflow:**
1. Train ResNet-18, get baseline (current step)
2. Implement threshold-based OSR with validation set
3. Collect unknown jamming samples for OSR evaluation
4. If performance good → deploy with threshold OSR
5. If performance insufficient → upgrade to OpenMax

---

### Why the Referenced Study Used GRU

**Possible reasons:**
- They had raw time-series, not spectrograms
- Temporal evolution was critical in their jamming types
- Smaller dataset (GRUs can work with less data than CNNs)
- Computational constraints (GRUs smaller than ResNets)
- Research novelty (applying RNNs to RF domain)

**For your case:** Your spectrograms favor CNNs

---

### When to Consider Other Models

**Try GRU/Hybrid if:**
- ResNet-18 accuracy plateaus < 85%
- You add time-varying jamming (frequency hopping, sweeping)
- You want to detect jamming evolution over time
- You switch to raw IQ time-series input

**Try Transformer if:**
- You scale dataset to 100k+ samples
- You have GPUs for training
- ResNet-18 can't capture long-range dependencies

**Try 1D CNN if:**
- You need real-time embedded deployment
- Spectrogram computation is bottleneck
- You have raw samples and want end-to-end learning

---

## 5. OSR Compatibility Summary

| Model | OSR Ease | Best OSR Method | Expected Unknown Detection |
|-------|----------|-----------------|---------------------------|
| **ResNet-18** | ⭐⭐⭐⭐⭐ | OpenMax, Distance | 85-95% AUROC |
| **GRU** | ⭐⭐⭐ | Threshold, Distance | 75-85% AUROC |
| **CNN-GRU** | ⭐⭐⭐⭐ | Multi-level | 85-90% AUROC |
| **Transformer** | ⭐⭐⭐⭐ | Attention-based | 80-90% AUROC |
| **1D CNN** | ⭐⭐⭐⭐⭐ | OpenMax, Distance | 85-95% AUROC |

---

## 6. Action Plan

### Immediate (Next Steps)

1. **Run your current ResNet-18** ✓
   - Evaluate confusion matrix
   - Check per-class accuracy
   - Identify problem classes

2. **Add new CLEAN antenna samples**
   - Balance dataset to ~8,000 CLEAN samples
   - Re-train with updated class weights
   - Verify no antenna-specific biases

3. **Implement threshold-based OSR**
   - Tune threshold on validation set
   - Test with simulated unknowns (leave-one-out: train without TRI, test if model rejects TRI)

### Near Future

4. **Collect unknown jamming samples**
   - New jamming types not in training set
   - Use for proper OSR evaluation

5. **Implement OpenMax**
   - Extract features, fit Weibull, calibrate
   - Compare with threshold baseline

6. **Consider GRU if:**
   - ResNet plateaus
   - You need temporal modeling
   - Compare performance side-by-side

### Long Term

7. **Explore hybrid if needed**
   - Only if single models insufficient
   - Combine spatial (CNN) + temporal (GRU) strengths

---

## Summary

**Your ResNet-18 is excellent starting point** - CNNs are ideal for spectrogram-based jamming detection.

**New CLEAN samples:** Will improve balance and robustness, just ensure proper mixing of antenna types.

**OSR:** Start simple (threshold), upgrade to OpenMax for production. ResNet-18 is perfectly suited for OSR.

**Alternative models:** GRU makes sense if you have raw time-series or need temporal modeling, but CNN is better for spectrograms.

**Next milestone:** Get ResNet results first, then make data-driven decisions about OSR and alternative architectures.

Your approach is sound - build baseline, evaluate, iterate! 🎯

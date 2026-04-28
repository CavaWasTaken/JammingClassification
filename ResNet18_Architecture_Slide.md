# CNN Architecture for GNSS Jamming Detection & Classification

## Model Overview
**ResNet-18 with Transfer Learning** adapted for single-channel spectrogram analysis

---

## Dataset Structure

### Signal Classes (16 Total)
- **CLEAN**: Clean GNSS signal (no jamming)
- **Jamming Types** with 3 power levels (LOW/MID/HIGH):
  - **LN** - Linear Noise jamming
  - **LWF** - Linear Waveform jamming  
  - **TICK** - Tick jamming
  - **TRI** - Triangle jamming
  - **TRIW** - Triangle Wave jamming

### Power Level Distribution
- **LOW**: P1, P2, P3
- **MID**: P4, P5, P6
- **HIGH**: P7, P8

---

## Data Pipeline

### Input Processing
- **Input Format**: STFT spectrograms from I/Q signal samples
- **Data Type**: `.npy` NumPy arrays (single-channel)
- **Dataset Split**:
  - Training: **50%**
  - Validation: **15%**
  - Testing: **35%**

### Batch Configuration
- **Batch Size**: 128 samples
- **Workers**: 4 parallel data loaders
- **Memory**: Pin memory enabled for faster GPU transfer

---

## Architecture Details

### Base Model: ResNet-18
- **Backbone**: Pre-trained ResNet-18 (ImageNet weights)
- **Input Adaptation**: Modified `conv1` layer
  - Original: 3 channels (RGB)
  - Modified: **1 channel** (spectrogram)
  - Kernel: 7×7, stride=2, padding=3

### Model Configuration
- **First Layer**: `Conv2d(1, 64, kernel_size=7, stride=2, padding=3)`
- **Feature Extractor**: ResNet-18 residual blocks (17 conv layers)
- **Classifier Head**: Fully connected layer
  - Input: 512 features (ResNet-18 output)
  - Output: **16 classes** (1 CLEAN + 15 jamming variants)

---

## Training Configuration

### Optimization
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Epochs**: 20 (with early stopping)
- **Device**: CUDA GPU (if available)

### Early Stopping
- **Patience**: 5 epochs without validation improvement
- **Metric**: Validation loss
- **Best Model Saved**: Checkpoint with lowest validation loss

---

## Key Features

### Technical Highlights
1. **Transfer Learning**: Leverages ImageNet pre-trained weights
2. **Single-Channel Input**: Optimized for spectrogram data
3. **Multi-Class Classification**: Discriminates between 16 signal types
4. **Class Imbalance Handling**: Weighted loss computation
5. **Efficient Data Loading**: Multi-worker parallel processing with pin memory

### Model Capabilities
- **Jamming Detection**: Identifies presence of interference
- **Jamming Classification**: Distinguishes 5 jamming types
- **Power Level Recognition**: Classifies 3 intensity levels
- **Real-time Ready**: Optimized batch processing (128 samples)

---

## Performance Monitoring

### Tracked Metrics
- Training Loss per epoch
- Validation Loss per epoch
- Validation Accuracy per epoch
- Confusion Matrix (16×16)
- Per-class Precision, Recall, F1-score

### Evaluation Analysis
- **Clean Signal Confusion**: Analysis of false CLEAN predictions
- **Jamming Misclassification**: Cross-type confusion patterns
- **Power Level Accuracy**: Performance across LOW/MID/HIGH intensities

---

## Model Output

### Final Deliverables
- **Best Model Checkpoint**: Saved with validation metrics
- **Confusion Matrix**: CSV export for detailed analysis
- **Performance Curves**: Loss and accuracy visualization
- **Per-Class Metrics**: Precision/Recall/F1 for all 16 classes

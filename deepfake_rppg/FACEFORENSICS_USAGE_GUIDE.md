# FaceForensics++ Training Guide

## Quick Start

### Test Current Model
```bash
python test_model_on_videos.py
```
Expected output shows model predictions on real.mp4 and fake.mp4.

### Train on Existing Data (Currently Running)
```bash
python train_with_all_data.py
```
- Uses 160 pre-extracted samples
- Improved training pipeline
- Estimated time: 30-60 minutes

### Train on Full FaceForensics++ Dataset
```bash
python train_on_faceforensics.py --epochs 50 --batch_size 32
```
- Processes 150 real + 150 deepfake = 300 total
- Still 1.87x larger than original 160
- Estimated time: 2-4 hours (CPU)

## FaceForensics++ Dataset Location

```
C:\Users\Avdhut\.cache\kagglehub\datasets\xdxd003\ff-c23\versions\1\FaceForensics++_C23
├── original/           # 1000 real videos
├── Deepfakes/          # 1000 deepfake videos (face-swap)
├── Face2Face/          # 1000 facial reenactment videos
├── FaceShifter/        # 1000 advanced face-swap videos
├── FaceSwap/           # 1000 face-swap videos
├── NeuralTextures/     # 1000 textured rendering videos
└── csv/                # Metadata
```

## Model Architecture

### SimpleRPPGNet (Recommended)
```
Input: (B, 1, 224, 224)
├─ Conv2d(1, 16, 3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2)
├─ Conv2d(16, 32, 3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2)
├─ Conv2d(32, 128, 3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2)
├─ Flatten (128 * 28 * 28 = 100352)
├─ Linear(100352, 256) → ReLU → Dropout(0.3)
├─ Linear(256, 128) → ReLU → Dropout(0.3)
├─ Linear(128, 1) → Sigmoid
Output: (B, 1)  # Probability of being FAKE
```

**Parameters**: ~50K (vs EfficientNet: 5M)
**Why Small**: Prevents overfitting on limited data

## Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 1e-4 | Adam optimizer |
| Weight Decay | 1e-4 | L2 regularization |
| Batch Size | 8-32 | Adjust based on memory |
| Epochs | 50 | Or until early stopping |
| Early Stopping | patience=20 | Monitored on Val AUC |
| Loss Function | BCELoss | Binary cross-entropy |
| Dropout | 0.25-0.3 | For regularization |

## Feature Extraction Pipeline

```
Video (mp4)
  ↓
[Read frames]
  ↓
[Face detection (MediaPipe)]
  ↓
[Extract ROIs: forehead + left cheek + right cheek]
  ↓
[Compute RGB color averages per frame]
  ↓
[Apply POS algorithm] → Get rPPG signal
  ↓
[Signal processing]:
  - Interpolation (fill missing frames)
  - Smoothing (moving average)
  - Bandpass filtering (0.7-4.0 Hz)
  - Normalization (z-score)
  ↓
[Generate spectrogram]:
  - Scipy.signal.spectrogram
  - Parameters: fs=30, nperseg=64, noverlap=48
  - Log scaling: 10*log10(Sxx + 1e-10)
  - Min-max normalization
  ↓
[Resize to 224×224]
  ↓
[Model Input]: (1, 224, 224)
```

## Performance Metrics Explained

### AUC-ROC
- **Range**: 0.5 (random) to 1.0 (perfect)
- **Target**: > 0.90
- **Explanation**: Area under the Receiver Operating Characteristic curve
- **Why Important**: Evaluates performance across all thresholds, not just 0.5

### Accuracy
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Target**: > 85%
- **Limitation**: Can be misleading with imbalanced data

### Sensitivity (True Positive Rate / Recall)
- **Formula**: TP / (TP + FN)
- **Interpretation**: % of actual fakes correctly identified
- **Target**: > 85%
- **Why Important**: Don't want to miss deepfakes

### Specificity (True Negative Rate)
- **Formula**: TN / (TN + FP)
- **Interpretation**: % of actual reals correctly identified
- **Target**: > 85%
- **Why Important**: Don't want to false-flag real videos

### Precision
- **Formula**: TP / (TP + FP)
- **Interpretation**: % of predicted fakes that are actually fake
- **Target**: > 85%
- **Why Important**: Minimize false alarms

## Prediction Interpretation

```python
prob_fake = model_output  # Range: [0, 1]
confidence = max(prob_fake, 1 - prob_fake) * 100

if prob_fake > 0.5:
    label = "FAKE"
    confidence = prob_fake * 100
else:
    label = "REAL"
    confidence = (1 - prob_fake) * 100
```

### Confidence Levels
- **> 90%**: High confidence (good)
- **70-90%**: Medium confidence (acceptable)
- **50-70%**: Low confidence (uncertain)
- **~50%**: No confidence (random guess)

## Troubleshooting

### "Model not found" Error
```bash
# Check if model file exists
ls -la rppg_model.pth
ls -la rppg_model_faceforensics.pth

# If not, train first:
python train_with_all_data.py
```

### CUDA out of memory
```bash
# Reduce batch size
python train_with_all_data.py --batch_size 4

# Or use CPU
python train_with_all_data.py  # Already uses CPU by default
```

### Prediction is 50% (no confidence)
This means the model is uncertain. Reasons:
1. Model not fully trained
2. Insufficient training data
3. Video quality too poor for feature extraction
4. Face not detected in video

### Training is too slow
- CPU processing: Expected 30 min - 2 hours per epoch
- Solution: Use GPU (10-50x faster)
- Or reduce dataset size (e.g., 100 vs 300 samples per class)

## Advanced: Custom Training

### Train with different model type
```bash
# Not yet implemented, but easy to add:
# python train_with_all_data.py --model_type resnet
# python train_with_all_data.py --model_type efficientnet
```

### Train with custom hyperparameters
Edit `train_with_all_data.py`:
```python
batch_size = 8          # Change here
epochs = 50             # Or here
lr = 1e-4              # Or here
weight_decay = 1e-4    # Or here
```

### Train on specific FaceForensics variants
```python
# In train_on_faceforensics.py, change:
fake_dir = os.path.join(dataset_path, 'Face2Face')  # Instead of Deepfakes
fake_dir = os.path.join(dataset_path, 'FaceShifter')  # Or this
fake_dir = os.path.join(dataset_path, 'NeuralTextures')  # Or this
```

## Expected Training Timeline

| Scenario | Time | Quality |
|----------|------|---------|
| Current (160 samples, CPU) | 30-60 min | ~55-65% AUC |
| FaceForensics (300 samples, CPU) | 2-4 hours | ~70-80% AUC |
| FaceForensics (1000 samples, CPU) | 10+ hours | ~85%+ AUC |
| FaceForensics (1000 samples, GPU) | 2-5 hours | ~85%+ AUC |
| FaceForensics Full (5000+ samples, GPU) | 10-20 hours | ~95%+ AUC |

## Next Steps

1. **Wait for current training** (`train_with_all_data.py`) to finish
2. **Test predictions**: `python test_model_on_videos.py`
3. **If results good** (>70% AUC):
   - Deploy model
   - Monitor performance
   - Collect more real videos for retraining
4. **If results poor** (<70% AUC):
   - Run `python train_on_faceforensics.py` with more samples
   - Or switch to GPU if available
   - Adjust hyperparameters

## Contact / Support

- Model weights: `rppg_model.pth`
- FaceForensics weights: `rppg_model_faceforensics.pth`
- Training logs: Check terminal output
- Detailed analysis: See `FACEFORENSICS_TRAINING_SUMMARY.md`

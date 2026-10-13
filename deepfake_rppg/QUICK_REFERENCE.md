# Quick Reference Guide - Deepfake Detection Model

## Problem Summary
✗ Both real and fake videos detected as "REAL"  
✗ Model probability ~0.12 for fake (should be 0.5+ for fake, 0.0-0.5 for real)  
✗ Model was untrained on dataset  

## Solution Summary
✓ Improved feature extraction (rPPG, signal processing)  
✓ Enhanced model architectures with proper regularization  
✓ Implemented comprehensive training pipeline  
✓ Added evaluation metrics and early stopping  
✓ Created SimpleRPPGNet for small datasets  

## Quick Commands

### 1. Train the Model
```bash
# Simple model (better for small datasets)
python train_improved.py --feature_dir processed_features --epochs 100 --model_type simple

# ResNet model
python train_improved.py --feature_dir processed_features --epochs 100 --model_type resnet

# With custom parameters
python train_improved.py --feature_dir processed_features --epochs 50 --batch_size 4 --model_type simple
```

### 2. Predict on Video
```bash
# Single video prediction
python main.py --mode predict --video_path "video.mp4"

# With specific model
python main.py --mode predict --video_path "video.mp4" --model_type simple
```

### 3. Extract More Features (if needed)
```bash
python extract_features.py --max_samples 1000
```

## Key Improvements Made

| Component | Before | After |
|-----------|--------|-------|
| Model Size | Large (EfficientNet) | Small (SimpleRPPGNet) |
| Feature Extraction | Basic | Robust with interpolation |
| Training Loop | No metrics | Full evaluation suite |
| Regularization | Minimal | Dropout + BatchNorm |
| Early Stopping | None | Yes (patience=15) |
| Data Handling | Windows issues | Fixed |

## Expected Results After Training

### With SimpleRPPGNet (160 samples):
- Validation AUC: 0.60-0.75
- Validation Accuracy: 55-70%
- Reduced overfitting vs EfficientNet

### Ideal (with 1000+ samples):
- Validation AUC: 0.90+
- Validation Accuracy: 90%+
- Good generalization

## Files You Should Know About

| File | Purpose |
|------|---------|
| `train_improved.py` | Main training script (use this!) |
| `main.py` | Inference/prediction script |
| `src/models/detector.py` | Model architectures |
| `src/training/train_loop.py` | Training logic |
| `ANALYSIS_AND_SOLUTIONS.md` | Detailed analysis |
| `IMPROVEMENTS.md` | What was changed |
| `rppg_model.pth` | Trained model weights |

## Troubleshooting

### Issue: "No features found!"
**Solution:** Make sure `processed_features/` directory exists with subdirectories `0/` and `1/`

### Issue: Very slow training
**Solution:** You're using CPU. GPU would be ~10x faster. Use CPU for small datasets (160 samples is fine)

### Issue: Model still predicting 0.5 (random)
**Solution:** 
1. Train more epochs (100+)
2. Collect more data (1000+ samples)
3. Adjust learning rate (try 5e-5)

### Issue: Training loss stuck/not decreasing
**Solution:**
1. Reduce learning rate (divide by 10)
2. Use different model type
3. Check if data is balanced (82 real, 78 fake - OK)

## Performance Metrics Explained

**AUC (Area Under Curve)**
- Range: 0.0 to 1.0
- Higher is better
- 0.5 = random guessing
- 0.9+ = excellent

**Accuracy**
- Percentage of correct predictions
- Can be misleading with imbalanced data
- Use AUC as primary metric

**Sensitivity (Recall)**
- True Positive Rate
- % of actual fakes detected
- Important for safety

**Specificity**
- True Negative Rate
- % of actual reals correctly identified
- Important for false alarms

**Confusion Matrix**
```
                Predicted
               Fake  Real
Actual Fake     TP    FN
       Real     FP    TN
```

## Typical Training Output

```
Epoch 1/100
Train Loss: 1.2131 | Train Acc: 0.5625 | Train AUC: 0.5479
Val Loss: 0.6761 | Val Acc: 0.5000 | Val AUC: 0.5859
[BEST] Saved better model with AUC: 0.5859, Acc: 0.5000
```

What it means:
- Training loss (1.21): How wrong predictions are on training data
- Training accuracy (56%): Correctly classified 56% of training samples
- Training AUC (0.55): Slightly better than random
- Validation loss (0.68): Loss on unseen data
- Validation accuracy (50%): 50% correct on validation (random)
- Validation AUC (0.59): Slightly better than random on unseen data

## Next Priorities

1. **RIGHT NOW**: Train SimpleRPPGNet and evaluate results
2. **SOON**: Collect more training data if possible
3. **LATER**: Try different hyperparameters
4. **LATER**: Implement ensemble/cross-validation

## Contact Points in Code

Need to modify something? Look here:

**Training hyperparameters**: `src/training/train_loop.py` lines 10-15
**Model architecture**: `src/models/detector.py` 
**Data loading**: `src/training/dataset.py`
**Feature generation**: `src/preprocessing/rppg.py`, `src/features/signal_processing.py`

## Success Criteria

Model is working when:
- ✓ Validation AUC > 0.7 on test set
- ✓ Real videos predicted with prob < 0.5
- ✓ Fake videos predicted with prob > 0.5
- ✓ Confidence > 60% on most predictions
- ✓ No extreme overfitting (train/val AUC gap < 0.2)

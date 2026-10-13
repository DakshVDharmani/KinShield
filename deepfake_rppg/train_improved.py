#!/usr/bin/env python3
"""
Improved training script that trains the model on pre-extracted features
with proper validation, monitoring, and evaluation metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from daksh_git.KinShield.deepfake_rppg.src.models.detector import SimpleRPPGNet, RPPGResNet
from daksh_git.KinShield.deepfake_rppg.src.training.dataset import RPPGDataset
from daksh_git.KinShield.deepfake_rppg.src.training.train_loop import train_model
import argparse
import os
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import numpy as np

def evaluate_model(model, val_loader, device):
    """Comprehensive evaluation of the model."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            if hasattr(outputs, 'logits'):
                # Handle models that return dict
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
            else:
                probs = outputs.cpu().numpy() if outputs.dim() > 0 else torch.tensor([[outputs.item()]])
                probs = probs if isinstance(probs, np.ndarray) else np.array([[probs.item()]])
            
            all_preds.extend(probs.flatten())
            all_labels.extend(labels.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics
    auc = roc_auc_score(all_labels, all_preds)
    acc = np.mean((all_preds > 0.5) == all_labels)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds > 0.5).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return {
        'auc': auc,
        'accuracy': acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def main():
    parser = argparse.ArgumentParser(description="Train improved deepfake detector")
    parser.add_argument('--feature_dir', type=str, default='processed_features',
                        help='Directory with pre-extracted features')
    parser.add_argument('--model_path', type=str, default='rppg_model.pth',
                        help='Path to save model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--model_type', type=str, default='simple',
                        choices=['simple', 'resnet'],
                        help='Model architecture')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset from pre-extracted features
    if not os.path.exists(args.feature_dir):
        print(f"Feature directory {args.feature_dir} not found!")
        return
    
    print(f"Loading features from {args.feature_dir}...")
    dataset = RPPGDataset('', [], feature_dir=args.feature_dir)
    
    if len(dataset) == 0:
        print("No features found! Run extract_features.py first.")
        return
    
    print(f"Loaded {len(dataset)} samples")
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    if args.model_type == 'simple':
        print("\nCreating SimpleRPPGNet model...")
        model = SimpleRPPGNet()
    else:
        print("\nCreating ResNet18 model...")
        model = RPPGResNet(pretrained=True)
    
    # Train
    print("\nStarting training...")
    model = train_model(model, train_loader, val_loader, 
                       num_epochs=args.epochs, device=device, 
                       model_path=args.model_path)
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("="*50)
    metrics = evaluate_model(model, val_loader, device)
    
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Sensitivity (True Positive Rate): {metrics['sensitivity']:.4f}")
    print(f"Specificity (True Negative Rate): {metrics['specificity']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {int(metrics['tn'])}")
    print(f"  False Positives: {int(metrics['fp'])}")
    print(f"  False Negatives: {int(metrics['fn'])}")
    print(f"  True Positives:  {int(metrics['tp'])}")
    
    print(f"\nModel saved to {args.model_path}")

if __name__ == "__main__":
    main()

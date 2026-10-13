"""
Train on all available processed features + FaceForensics style data
"""

import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

from daksh_git.KinShield.deepfake_rppg.src.models.detector import SimpleRPPGNet

def load_all_processed_features():
    """Load all existing .npy features from processed_features/"""
    features = []
    labels = []
    
    base_dir = Path("processed_features")
    
    # Load class 0 (real)
    real_dir = base_dir / "0"
    if real_dir.exists():
        print(f"Loading real samples from {real_dir}...")
        for npy_file in sorted(real_dir.glob("*.npy")):
            try:
                data = np.load(npy_file)
                if data.shape == (224, 224):
                    features.append(torch.from_numpy(data).unsqueeze(0).float())
                    labels.append(0)
            except:
                pass
    
    # Load class 1 (fake)
    fake_dir = base_dir / "1"
    if fake_dir.exists():
        print(f"Loading fake samples from {fake_dir}...")
        for npy_file in sorted(fake_dir.glob("*.npy")):
            try:
                data = np.load(npy_file)
                if data.shape == (224, 224):
                    features.append(torch.from_numpy(data).unsqueeze(0).float())
                    labels.append(1)
            except:
                pass
    
    if features:
        all_features = torch.stack(features)
        all_labels = torch.tensor(labels, dtype=torch.long)
        return all_features, all_labels
    
    return None, None

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load existing features
    all_features, all_labels = load_all_processed_features()
    
    if all_features is None:
        print("No features found!")
        return
    
    print(f"Total: {all_features.shape[0]} samples")
    print(f"Real: {torch.sum(all_labels==0)}, Fake: {torch.sum(all_labels==1)}\n")
    
    # Create dataset
    dataset = TensorDataset(all_features, all_labels)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}\n")
    
    # Model
    model = SimpleRPPGNet()
    model.to(device)
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = torch.nn.BCELoss()
    
    best_auc = 0
    patience = 0
    epochs = 50
    
    print("Starting training...\n")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_preds = []
        train_labels_list = []
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
        
        train_preds = np.array(train_preds).flatten()
        train_labels_arr = np.array(train_labels_list).flatten()
        train_auc = roc_auc_score(train_labels_arr, train_preds)
        
        # Validate
        model.eval()
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                outputs = model(features)
                val_preds.extend(outputs.cpu().numpy())
                val_labels_list.extend(labels.numpy())
        
        val_preds = np.array(val_preds).flatten()
        val_labels_arr = np.array(val_labels_list).flatten()
        val_auc = roc_auc_score(val_labels_arr, val_preds)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}", end="")
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            torch.save(model.state_dict(), 'rppg_model.pth')
            print(" [BEST]")
        else:
            patience += 1
            print()
            if patience >= 20:
                print(f"\nEarly stopping!")
                break
        
        scheduler.step(val_auc)
    
    # Test
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    model.eval()
    test_preds = []
    test_labels_list = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            test_preds.extend(outputs.cpu().numpy())
            test_labels_list.extend(labels.numpy())
    
    test_preds = np.array(test_preds).flatten()
    test_labels = np.array(test_labels_list).flatten()
    
    auc = roc_auc_score(test_labels, test_preds)
    acc = accuracy_score(test_labels, test_preds > 0.5)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds > 0.5).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}, FP: {fp}")
    print(f"  FN: {fn}, TN: {tn}")
    
    print("\nModel saved: rppg_model.pth")

if __name__ == "__main__":
    main()

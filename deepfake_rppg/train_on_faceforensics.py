"""
Pre-compute features from FaceForensics++ without slow video reading
Save as numpy arrays for fast loading during training
"""

import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

from daksh_git.KinShield.deepfake_rppg.src.models.detector import SimpleRPPGNet, RPPGResNet, RPPGEfficientNet

def create_feature_from_frames(frames):
    """Create features from raw frames array"""
    if len(frames) < 10:
        return None
    
    try:
        # Sample frames evenly
        indices = np.linspace(0, len(frames) - 1, min(30, len(frames)), dtype=int)
        sampled_frames = [frames[i] for i in indices]
        
        # Create feature: average grayscale
        feature_map = np.zeros((128, 128), dtype=np.float32)
        for frame in sampled_frames:
            resized = cv2.resize(frame, (128, 128))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            feature_map += gray
        
        feature_map /= len(sampled_frames)
        
        # Resize to 224
        feature_resized = cv2.resize(feature_map, (224, 224))
        return feature_resized
    except:
        return None

def batch_process_folder(video_dir, label, max_samples=300):
    """Process videos with progress bar"""
    video_files = sorted(list(Path(video_dir).glob('**/*.mp4')))[:max_samples]
    
    features_list = []
    processed = 0
    failed = 0
    
    print(f"\nProcessing {len(video_files)} videos (label={label})...")
    for video_path in tqdm(video_files, desc=f"Label {label}"):
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            count = 0
            
            while len(frames) < 128 and count < 500:
                ret, frame = cap.read()
                if not ret:
                    break
                if count % 4 == 0:  # Sample every 4th frame
                    frames.append(frame)
                count += 1
            
            cap.release()
            
            if len(frames) >= 10:
                feature = create_feature_from_frames(frames)
                if feature is not None:
                    features_list.append(feature)
                    processed += 1
            else:
                failed += 1
        except:
            failed += 1
    
    print(f"Success: {processed}, Failed: {failed}")
    
    if features_list:
        features = torch.from_numpy(np.array(features_list, dtype=np.float32)).unsqueeze(1)
        labels = torch.full((len(features_list),), label, dtype=torch.long)
        return features, labels
    return None, None

def main(dataset_path, epochs=50, batch_size=32):
    """Load and train on FaceForensics++"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load real videos
    real_dir = os.path.join(dataset_path, 'original')
    real_features, real_labels = batch_process_folder(real_dir, label=0, max_samples=150)
    
    if real_features is None:
        print("ERROR: Could not process real videos")
        return
    
    # Load fake videos
    fake_dir = os.path.join(dataset_path, 'Deepfakes')
    fake_features, fake_labels = batch_process_folder(fake_dir, label=1, max_samples=150)
    
    if fake_features is None:
        print("ERROR: Could not process fake videos")
        return
    
    # Combine
    all_features = torch.cat([real_features, fake_features], dim=0)
    all_labels = torch.cat([real_labels, fake_labels], dim=0)
    
    print(f"\nDataset: {all_features.shape[0]} samples")
    print(f"Real: {torch.sum(all_labels==0)}, Fake: {torch.sum(all_labels==1)}")
    
    # Create loaders
    dataset = TensorDataset(all_features, all_labels)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}\n")
    
    # Model
    model = SimpleRPPGNet()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    pos_weight = torch.tensor(torch.sum(all_labels == 0).float() / torch.sum(all_labels == 1).float())
    criterion = torch.nn.BCELoss(pos_weight=pos_weight)
    
    best_auc = 0
    patience = 0
    
    # Train
    for epoch in range(epochs):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        
        print(f"Epoch {epoch+1}/{epochs} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}", end="")
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            torch.save(model.state_dict(), 'rppg_model_faceforensics.pth')
            print(" [SAVED]")
        else:
            patience += 1
            print()
            if patience >= 20:
                print("Early stopping!")
                break
        
        scheduler.step(val_auc)
    
    # Test
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
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
    
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity (TPR): {tp/(tp+fn):.4f}")
    print(f"Specificity (TNR): {tn/(tn+fp):.4f}")
    print(f"Precision: {tp/(tp+fp):.4f}")
    print(f"\nConfusion Matrix (Threshold=0.5):")
    print(f"  True Positives:  {tp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    
    print("\nModel saved: rppg_model_faceforensics.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, 
                       default=r'C:\Users\Avdhut\.cache\kagglehub\datasets\xdxd003\ff-c23\versions\1\FaceForensics++_C23')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    main(args.dataset_path, args.epochs, args.batch_size)

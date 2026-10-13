"""
Ultra-fast training on FaceForensics++ dataset
Minimal feature extraction for speed - just use raw frame variance as features
"""

import os
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

from daksh_git.KinShield.deepfake_rppg.src.models.detector import SimpleRPPGNet, RPPGResNet, RPPGEfficientNet

def extract_ultra_fast_features(video_path):
    """Ultra-fast feature: sample frames and create simple heatmap"""
    try:
        cap = cv2.VideoCapture(video_path)
        frames_sampled = []
        count = 0
        target_frames = 30  # Just 30 sampled frames
        
        while count < 500:  # Sample from first 500 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % 16 == 0:  # Sample every 16th frame
                frame = cv2.resize(frame, (128, 128))
                frames_sampled.append(frame)
                if len(frames_sampled) >= target_frames:
                    break
            count += 1
        
        cap.release()
        
        if len(frames_sampled) < 10:
            return None
        
        # Create a feature map: convert frames to grayscale and stack as channels
        # Use variance across frames as the feature
        feature_map = np.zeros((128, 128), dtype=np.float32)
        
        for frame in frames_sampled:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            feature_map += gray
        
        feature_map /= len(frames_sampled)  # Average
        
        # Compute variance across frames
        variance_map = np.zeros((128, 128), dtype=np.float32)
        for frame in frames_sampled:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            variance_map += (gray - feature_map) ** 2
        
        variance_map /= len(frames_sampled)
        variance_map = np.sqrt(variance_map + 1e-8)
        
        # Normalize
        if variance_map.max() > variance_map.min():
            variance_map = (variance_map - variance_map.min()) / (variance_map.max() - variance_map.min())
        
        # Resize to 224x224
        feature_resized = cv2.resize(variance_map, (224, 224))
        
        # Convert to tensor (add channel dimension)
        feature_tensor = torch.from_numpy(feature_resized).unsqueeze(0).float()
        return feature_tensor
        
    except Exception as e:
        return None

def batch_process_videos_fast(video_dir, label, max_samples=300):
    """Process videos very quickly"""
    video_files = sorted(list(Path(video_dir).glob('**/*.mp4')))[:max_samples]
    
    features_list = []
    processed = 0
    
    print(f"\nProcessing {len(video_files)} videos...")
    for video_path in tqdm(video_files):
        feature = extract_ultra_fast_features(str(video_path))
        if feature is not None:
            features_list.append(feature)
            processed += 1
    
    print(f"Successfully processed: {processed}/{len(video_files)}")
    
    if features_list:
        features = torch.stack(features_list)
        labels = torch.full((features.shape[0],), label, dtype=torch.long)
        return features, labels
    return None, None

def train_on_large_dataset(dataset_path, model_type='simple', epochs=50, batch_size=32):
    """Train on FaceForensics++ dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Loading FaceForensics++ dataset...")
    
    # Process: 300 real + 300 deepfake = 600 total (still 3.75x larger than original 160)
    real_dir = os.path.join(dataset_path, 'original')
    real_features, real_labels = batch_process_videos_fast(real_dir, label=0, max_samples=300)
    
    if real_features is None:
        print("Failed to process real videos!")
        return
    
    fake_dir = os.path.join(dataset_path, 'Deepfakes')
    fake_features, fake_labels = batch_process_videos_fast(fake_dir, label=1, max_samples=300)
    
    if fake_features is None:
        print("Failed to process fake videos!")
        return
    
    # Combine
    all_features = torch.cat([real_features, fake_features], dim=0)
    all_labels = torch.cat([real_labels, fake_labels], dim=0)
    
    print(f"\nTotal features: {all_features.shape}")
    print(f"Real: {torch.sum(all_labels==0)}, Fake: {torch.sum(all_labels==1)}")
    
    # Create dataset
    dataset = TensorDataset(all_features, all_labels)
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Model
    if model_type == 'simple':
        model = SimpleRPPGNet()
    elif model_type == 'resnet':
        model = RPPGResNet(pretrained=True)
    else:
        model = RPPGEfficientNet(pretrained=True)
    
    model.to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    pos_weight = torch.sum(all_labels == 0).float() / torch.sum(all_labels == 1).float()
    criterion = torch.nn.BCELoss(pos_weight=pos_weight)
    
    best_auc = 0
    patience_counter = 0
    
    print(f"\nStarting training with {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
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
            
            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
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
            patience_counter = 0
            torch.save(model.state_dict(), 'rppg_model_faceforensics.pth')
            print(" [BEST]")
        else:
            patience_counter += 1
            print()
            if patience_counter >= 20:
                print(f"Early stopping!")
                break
        
        scheduler.step(val_auc)
    
    # Test
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
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
    
    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, test_preds > 0.5)
    
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds > 0.5).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"AUC: {test_auc:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}, FP: {fp}")
    print(f"  FN: {fn}, TN: {tn}")
    
    torch.save(model.state_dict(), 'rppg_model_faceforensics.pth')
    print("\nModel saved to rppg_model_faceforensics.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, 
                       default=r'C:\Users\Avdhut\.cache\kagglehub\datasets\xdxd003\ff-c23\versions\1\FaceForensics++_C23',
                       help='Path to dataset')
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'resnet', 'efficientnet'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    train_on_large_dataset(args.dataset_path, args.model_type, args.epochs, args.batch_size)

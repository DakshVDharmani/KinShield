"""
Large-scale training on FaceForensics++ dataset (2000 videos)
Extracts features from all videos and trains the model
"""

import os
import sys
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
from daksh_git.KinShield.deepfake_rppg.src.preprocessing.rppg import extract_raw_trace
from daksh_git.KinShield.deepfake_rppg.src.preprocessing.face_detection import FaceDetector
from daksh_git.KinShield.deepfake_rppg.src.features.signal_processing import (
    normalize_signal, filter_signal, generate_spectrogram, 
    smooth_signal, interpolate_missing_frames
)

def load_video(video_path, max_frames=300):
    """Load video and extract frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break
        # Resize for faster processing
        frame = cv2.resize(frame, (224, 224))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
    
    cap.release()
    return frames

def extract_features_from_video(video_path, detector, device='cpu'):
    """Extract rPPG features from a single video"""
    try:
        frames = load_video(video_path)
        if len(frames) < 30:
            return None
        
        # Extract rPPG signals
        traces = extract_raw_trace(frames, detector)
        
        # Process signals
        signals = {}
        for region, trace_list in traces.items():
            if not trace_list or len(trace_list) < 10:
                return None
            
            # Green channel rPPG (standard)
            signal = np.array([t[1] for t in trace_list], dtype=np.float32)  # Green channel
            
            # Handle zeros and invalid values
            signal = interpolate_missing_frames(signal)
            signal = smooth_signal(signal, window_size=5)
            
            # Filter and normalize
            signal = filter_signal(signal, fps=30)
            signal = normalize_signal(signal)
            signals[region] = signal
        
        # Combine signals from multiple regions
        if all(v is not None and len(v) > 0 for v in signals.values()):
            combined_signal = np.mean([signals['forehead'], signals['left_cheek'], signals['right_cheek']], axis=0)
            
            # Generate spectrogram
            spectrogram = generate_spectrogram(combined_signal, fps=30)
            if spectrogram is not None:
                # Ensure correct shape
                if spectrogram.shape != (224, 224):
                    spectrogram = cv2.resize(spectrogram, (224, 224))
                return torch.from_numpy(spectrogram).unsqueeze(0).float()  # Add channel dimension
        
        return None
    except Exception as e:
        return None

def batch_extract_features(video_dir, label, max_samples=None, skip_count=0):
    """Extract features from all videos in a directory"""
    detector = FaceDetector()
    video_files = sorted(list(Path(video_dir).glob('**/*.mp4')))
    
    if max_samples:
        video_files = video_files[:max_samples]
    
    features_list = []
    failed_count = 0
    processed_count = 0
    
    print(f"\n[Label: {label}] Extracting features from {len(video_files)} videos...")
    for i, video_path in enumerate(tqdm(video_files)):
        if i < skip_count:
            continue
            
        spectrogram = extract_features_from_video(str(video_path), detector)
        if spectrogram is not None:
            features_list.append(spectrogram)
            processed_count += 1
        else:
            failed_count += 1
        
        # Limit for memory
        if max_samples and len(features_list) >= max_samples:
            break
    
    print(f"Successfully extracted: {processed_count}, Failed: {failed_count}")
    
    if features_list:
        features = torch.stack(features_list)
        labels = torch.full((features.shape[0],), label, dtype=torch.long)
        return features, labels
    return None, None

def train_on_faceforensics(dataset_path, model_type='simple', epochs=50, batch_size=32):
    """Train model on FaceForensics++ dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    real_dir = os.path.join(dataset_path, 'original')
    fake_dir = os.path.join(dataset_path, 'Deepfakes')  # Can also use Face2Face, FaceShifter, etc.
    
    # Extract features from both classes
    real_features, real_labels = batch_extract_features(real_dir, label=0, max_samples=None)
    fake_features, fake_labels = batch_extract_features(fake_dir, label=1, max_samples=None)
    
    if real_features is None or fake_features is None:
        print("Failed to extract features!")
        return
    
    # Combine datasets
    all_features = torch.cat([real_features, fake_features], dim=0)
    all_labels = torch.cat([real_labels, fake_labels], dim=0)
    
    print(f"\nTotal features: {all_features.shape}")
    print(f"Class distribution: Real={torch.sum(all_labels==0)}, Fake={torch.sum(all_labels==1)}")
    
    # Create dataset and split
    dataset = TensorDataset(all_features, all_labels)
    
    # 80/10/10 split
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
    
    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    if model_type == 'simple':
        model = SimpleRPPGNet()
        print("Using SimpleRPPGNet")
    elif model_type == 'resnet':
        model = RPPGResNet(pretrained=True)
        print("Using RPPGResNet")
    else:
        model = RPPGEfficientNet(pretrained=True)
        print("Using RPPGEfficientNet")
    
    model.to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Compute class weights
    pos_count = torch.sum(all_labels == 1)
    neg_count = torch.sum(all_labels == 0)
    pos_weight = neg_count / pos_count
    criterion = torch.nn.BCELoss(pos_weight=pos_weight)
    
    best_auc = 0
    patience_counter = 0
    patience_limit = 20
    
    # Training loop
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
        train_acc = accuracy_score(train_labels_arr, train_preds > 0.5)
        
        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_preds = np.array(val_preds).flatten()
        val_labels_arr = np.array(val_labels_list).flatten()
        val_auc = roc_auc_score(val_labels_arr, val_preds)
        val_acc = accuracy_score(val_labels_arr, val_preds > 0.5)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), 'rppg_model_faceforensics.pth')
            print(f"[BEST] Saved better model with AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        scheduler.step(val_auc)
    
    # Test on held-out set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    
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
    
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Sensitivity (TPR): {sensitivity:.4f}")
    print(f"Specificity (TNR): {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    # Save final model
    torch.save(model.state_dict(), 'rppg_model_faceforensics.pth')
    print("\nModel saved to rppg_model_faceforensics.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on FaceForensics++ dataset")
    parser.add_argument('--dataset_path', type=str, 
                       default=r'C:\Users\Avdhut\.cache\kagglehub\datasets\xdxd003\ff-c23\versions\1\FaceForensics++_C23',
                       help='Path to FaceForensics++ dataset')
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'resnet', 'efficientnet'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    train_on_faceforensics(args.dataset_path, args.model_type, args.epochs, args.batch_size)

"""
Simplified large-scale training on FaceForensics++ dataset
Uses faster video processing and simpler feature extraction
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
from scipy import signal as scipy_signal

from daksh_git.KinShield.deepfake_rppg.src.models.detector import SimpleRPPGNet, RPPGResNet, RPPGEfficientNet

def extract_simple_features(video_path):
    """Fast feature extraction from video using green channel and spectrogram"""
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        max_frames = 150  # Reduced from 300 for speed
        count = 0
        frame_interval = 2  # Skip every other frame for speed
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % frame_interval == 0:
                frame = cv2.resize(frame, (128, 128))  # Smaller size for speed
                frames.append(frame)
                if len(frames) >= max_frames:
                    break
            
            count += 1
        
        cap.release()
        
        if len(frames) < 15:
            return None
        
        # Extract green channel (rPPG proxy)
        green_signal = np.zeros(len(frames), dtype=np.float32)
        for i, frame in enumerate(frames):
            green_signal[i] = np.mean(frame[:, :, 1])  # Green channel
        
        # Normalize
        if np.std(green_signal) > 1e-6:
            green_signal = (green_signal - np.mean(green_signal)) / (np.std(green_signal) + 1e-8)
        else:
            return None
        
        # Generate spectrogram with smaller parameters for speed
        frequencies, times, Sxx = scipy_signal.spectrogram(
            green_signal, fs=15, nperseg=32, noverlap=24  # Reduced parameters
        )
        
        # Log scaling
        Sxx = 10 * np.log10(Sxx + 1e-10)
        
        # Normalize to [0, 1]
        if Sxx.min() != Sxx.max():
            Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min() + 1e-6)
        
        # Resize to 224x224
        Sxx_resized = cv2.resize(Sxx, (224, 224))
        
        # Convert to tensor
        spectrogram = torch.from_numpy(Sxx_resized).unsqueeze(0).float()
        return spectrogram
        
    except Exception as e:
        return None

def batch_process_videos(video_dir, label, max_samples=None):
    """Process all videos in directory"""
    video_files = sorted(list(Path(video_dir).glob('**/*.mp4')))
    
    if max_samples:
        video_files = video_files[:max_samples]
    
    features_list = []
    processed = 0
    failed = 0
    
    print(f"\nProcessing {len(video_files)} videos (Label: {label})...")
    for video_path in tqdm(video_files, desc=f"Label {label}"):
        spectrogram = extract_simple_features(str(video_path))
        if spectrogram is not None:
            features_list.append(spectrogram)
            processed += 1
        else:
            failed += 1
    
    print(f"Processed: {processed}, Failed: {failed}")
    
    if features_list:
        features = torch.stack(features_list)
        labels = torch.full((features.shape[0],), label, dtype=torch.long)
        return features, labels
    return None, None

def train_on_large_dataset(dataset_path, model_type='simple', epochs=50, batch_size=32):
    """Train on FaceForensics++ with large-scale dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Process real videos (sample 500 for speed, still much larger than original 160)
    real_dir = os.path.join(dataset_path, 'original')
    real_features, real_labels = batch_process_videos(real_dir, label=0, max_samples=500)
    
    if real_features is None:
        print("Failed to process real videos!")
        return
    
    # Process fake videos (Deepfakes - sample 500)
    fake_dir = os.path.join(dataset_path, 'Deepfakes')
    fake_features, fake_labels = batch_process_videos(fake_dir, label=1, max_samples=500)
    
    if fake_features is None:
        print("Failed to process fake videos!")
        return
    
    # Optionally add more fake videos from other methods (smaller sample)
    face2face_dir = os.path.join(dataset_path, 'Face2Face')
    if os.path.exists(face2face_dir):
        f2f_features, f2f_labels = batch_process_videos(face2face_dir, label=1, max_samples=100)
        if f2f_features is not None:
            fake_features = torch.cat([fake_features, f2f_features], dim=0)
            fake_labels = torch.cat([fake_labels, f2f_labels], dim=0)
    
    # Combine
    all_features = torch.cat([real_features, fake_features], dim=0)
    all_labels = torch.cat([real_labels, fake_labels], dim=0)
    
    print(f"\nTotal features: {all_features.shape}")
    print(f"Real: {torch.sum(all_labels==0)}, Fake: {torch.sum(all_labels==1)}")
    
    # Split dataset
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
    
    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
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
    
    # Compute class weights
    pos_count = torch.sum(all_labels == 1)
    neg_count = torch.sum(all_labels == 0)
    pos_weight = neg_count / pos_count
    criterion = torch.nn.BCELoss(pos_weight=pos_weight)
    
    best_auc = 0
    patience_counter = 0
    
    # Training
    for epoch in range(epochs):
        # Train phase
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
        
        # Validation phase
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
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), 'rppg_model_faceforensics.pth')
            print(f"  [BEST] Saved model with Val AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        scheduler.step(val_auc)
    
    # Test evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
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
    print(f"Sensitivity (TPR): {sensitivity:.4f}")
    print(f"Specificity (TNR): {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    torch.save(model.state_dict(), 'rppg_model_faceforensics.pth')
    print("\nModel saved to rppg_model_faceforensics.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on FaceForensics++ (Simplified)")
    parser.add_argument('--dataset_path', type=str, 
                       default=r'C:\Users\Avdhut\.cache\kagglehub\datasets\xdxd003\ff-c23\versions\1\FaceForensics++_C23',
                       help='Path to dataset')
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'resnet', 'efficientnet'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    train_on_large_dataset(args.dataset_path, args.model_type, args.epochs, args.batch_size)

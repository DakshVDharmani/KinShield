import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from daksh_git.KinShield.deepfake_rppg.src.preprocessing.face_detection import FaceDetector
from daksh_git.KinShield.deepfake_rppg.src.preprocessing.rppg import extract_raw_trace, get_rppg_signal
from daksh_git.KinShield.deepfake_rppg.src.features.signal_processing import normalize_signal, filter_signal, generate_spectrogram

class RPPGDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, max_samples=None, feature_dir=None):
        """
        Args:
            video_paths: List of paths or directory.
            labels: List of labels.
            transform: Analysis transform.
            max_samples: If set, limits the number of videos loaded (for testing).
            feature_dir: If set, loads .npy files from here instead of processing videos.
        """
        self.video_paths = []
        self.labels = []
        self.max_samples = max_samples
        self.feature_dir = feature_dir
        
        if feature_dir and os.path.exists(feature_dir):
            self._load_from_features(feature_dir)
        elif isinstance(video_paths, list):
            self.video_paths = video_paths
            self.labels = labels
        elif isinstance(video_paths, str) and os.path.isdir(video_paths):
            self._load_from_directory(video_paths)
            
        self.transform = transform
        self.detector = FaceDetector() if not feature_dir else None

    def _load_from_features(self, feat_dir):
        # feat_dir/0/*.npy, feat_dir/1/*.npy
        for label in [0, 1]:
            label_dir = os.path.join(feat_dir, str(label))
            if not os.path.exists(label_dir): continue
            for f in os.listdir(label_dir):
                if f.endswith('.npy'):
                    self.video_paths.append(os.path.join(label_dir, f))
                    self.labels.append(label)
        print(f"Loaded {len(self.video_paths)} features from disk.")

    def _load_from_directory(self, root_dir):
        # ... (rest of old _load_from_directory logic)

        # Assumes structure: root_dir/class_name/video.mp4
        # OR root_dir/real/... and root_dir/fake/...
        
        all_real = []
        all_fake = []
        
        print(f"Scanning directory {root_dir}...")
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    path = os.path.join(root, file)
                    path_lower = path.lower()
                    
                    # FaceForensics++ Specific Logic
                    # Real
                    if 'original' in path_lower or 'real' in path_lower or 'youtube' in path_lower:
                        all_real.append(path)
                    # Fakes
                    elif any(x in path_lower for x in ['deepfake', 'face2face', 'faceshifter', 'faceswap', 'neuraltextures', 'manipulated']):
                        all_fake.append(path)
        
        print(f"Found {len(all_real)} Real and {len(all_fake)} Fake videos total.")
        
        if self.max_samples:
            # Take even split
            limit_per_class = self.max_samples // 2
            
            # Shuffle to get random variance
            # Using random.sample if available or shuffle
            import random
            random.shuffle(all_real)
            random.shuffle(all_fake)
            
            selected_real = all_real[:limit_per_class]
            selected_fake = all_fake[:limit_per_class]
            
            self.video_paths = selected_real + selected_fake
            self.labels = [0]*len(selected_real) + [1]*len(selected_fake)
        else:
            self.video_paths = all_real + all_fake
            self.labels = [0]*len(all_real) + [1]*len(all_fake)
            
        # Shuffle final dataset
        import random
        combined = list(zip(self.video_paths, self.labels))
        random.shuffle(combined)
        if combined:
            self.video_paths[:], self.labels[:] = zip(*combined) 
        
        print(f"Dataset loaded: {len(self.video_paths)} videos ({sum(self.labels)} Fake, {len(self.labels)-sum(self.labels)} Real)")


    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Check if we are loading pre-extracted features
        if self.feature_dir:
            try:
                spec_resized = np.load(video_path)
                spec_tensor = torch.tensor(spec_resized, dtype=torch.float32).unsqueeze(0)
                return spec_tensor, torch.tensor(label, dtype=torch.float32)
            except Exception as e:
                print(f"Error loading {video_path}: {e}")
                return torch.zeros(1, 224, 224), torch.tensor(label, dtype=torch.float32)

        # ELSE: Normal Video Processing (slow)
        cap = cv2.VideoCapture(video_path)

        frames = []
        frame_count = 0
        MAX_FRAMES = 300 # Limit to 10 seconds at 30fps to save memory/time
        
        while cap.isOpened() and frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1
        cap.release()
        
        if len(frames) < 30:
            # Handle too short video
            return torch.zeros(1, 128, 128), torch.tensor(label, dtype=torch.float32)

        # 2. Extract rPPG Signal
        # Using a simplified version here. In production, do this offline!
        traces = extract_raw_trace(frames, self.detector)
        
        # Combine ROIs - average signals from all detected regions for robustness
        signals = []
        for key in traces:
             if len(traces[key]) > 0:
                 # traces[key] is list of [R,G,B], so shape (Frames, 3)
                 sig = get_rppg_signal(traces[key], method='pos')
                 signals.append(sig)
        
        if not signals:
             return torch.zeros(1, 224, 224), torch.tensor(label, dtype=torch.float32)
             
        # Merge multiple region signals
        final_signal = np.mean(signals, axis=0) if len(signals) > 1 else signals[0]
        
        # 3. Process Signal with improved filtering
        from daksh_git.KinShield.deepfake_rppg.src.features.signal_processing import interpolate_missing_frames, smooth_signal
        final_signal = interpolate_missing_frames(final_signal)
        final_signal = smooth_signal(final_signal, window_size=3)
        
        norm_sig = normalize_signal(final_signal)
        filt_sig = filter_signal(norm_sig)
        
        # 4. Generate Spectrogram
        spectrogram = generate_spectrogram(filt_sig)
        
        # Ensure proper shape and resize
        if spectrogram.size == 0:
            return torch.zeros(1, 224, 224), torch.tensor(label, dtype=torch.float32)
        
        spec_resized = cv2.resize(spectrogram, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Add channel dim and ensure valid range
        spec_tensor = torch.tensor(spec_resized, dtype=torch.float32).unsqueeze(0)
        spec_tensor = torch.clamp(spec_tensor, 0, 1)  # Ensure [0, 1] range
        
        return spec_tensor, torch.tensor(label, dtype=torch.float32)

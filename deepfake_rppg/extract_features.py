import os
import numpy as np
import cv2
from tqdm import tqdm
from daksh_git.KinShield.deepfake_rppg.src.preprocessing.face_detection import FaceDetector
from daksh_git.KinShield.deepfake_rppg.src.preprocessing.rppg import extract_raw_trace, get_rppg_signal
from daksh_git.KinShield.deepfake_rppg.src.features.signal_processing import normalize_signal, filter_signal, generate_spectrogram
from daksh_git.KinShield.deepfake_rppg.src.training.dataset import RPPGDataset

def extract_and_save_features(data_dir, output_dir, max_samples=500):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "0")) # Real
        os.makedirs(os.path.join(output_dir, "1")) # Fake
    
    # Use the existing RPPGDataset to find and balance files
    dataset = RPPGDataset(data_dir, [], max_samples=max_samples)
    detector = FaceDetector()
    
    print(f"Starting feature extraction for {len(dataset)} videos...")
    
    for i in range(len(dataset)):
        video_path = dataset.video_paths[i]
        label = dataset.labels[i]
        
        file_name = os.path.basename(video_path).split('.')[0] + f"_{i}.npy"
        target_path = os.path.join(output_dir, str(label), file_name)
        
        if os.path.exists(target_path):
            continue
            
        try:
            # 1. Load Video
            cap = cv2.VideoCapture(video_path)
            frames = []
            while len(frames) < 300:
                ret, frame = cap.read()
                if not ret: break
                # Resize to 480p width to speed up MediaPipe
                h, w = frame.shape[:2]
                new_w = 480
                new_h = int(h * (new_w / w))
                frame = cv2.resize(frame, (new_w, new_h))
                frames.append(frame)
            cap.release()

            
            if len(frames) < 60: continue
            
            # 2. Extract rPPG Signal (POS method)
            traces = extract_raw_trace(frames, detector)
            signals = []
            for key in traces:
                if len(traces[key]) > 0:
                    sig = get_rppg_signal(traces[key], method='pos')
                    signals.append(sig)
            
            if not signals: continue
            
            final_signal = np.mean(signals, axis=0)
            norm_sig = normalize_signal(final_signal)
            filt_sig = filter_signal(norm_sig)
            
            # 3. Generate Spectrogram
            spectrogram = generate_spectrogram(filt_sig)
            spec_resized = cv2.resize(spectrogram, (224, 224))
            
            # 4. Save
            np.save(target_path, spec_resized)
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    DATA_DIR = "C:/Users/Avdhut/.cache/kagglehub/datasets/xdxd003/ff-c23/versions/1/FaceForensics++_C23"
    OUT_DIR = "processed_features"
    extract_and_save_features(DATA_DIR, OUT_DIR, max_samples=500)


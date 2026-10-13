import numpy as np
import cv2
from daksh_git.KinShield.deepfake_rppg.src.preprocessing.face_detection import FaceDetector

def extract_raw_trace(frames, detector: FaceDetector):
    """
    Process a list of frames (or video file) and extract raw RGB traces 
    for each ROI (Forehead, Left Cheek, Right Cheek).
    
    Returns:
        traces: Dictionary of {'forehead': [], 'left_cheek': [], 'right_cheek': []} 
                where each is a list of [R, G, B] means per frame.
    """
    traces = {
        'forehead': [],
        'left_cheek': [],
        'right_cheek': []
    }
    
    face_detected_count = 0
    
    for frame in frames:
        landmarks = detector.process_frame(frame)
        if landmarks is None:
            # Face not found, append 0s to keep timing consistent
            for key in traces:
                traces[key].append([0, 0, 0])
            continue
        
        face_detected_count += 1
        # Get ROI masks from landmarks
        masks = detector.get_roi_coordinates(landmarks, frame.shape)
        
        for region_name, mask in masks.items():
            # Apply mask to get mean color
            mean_color = cv2.mean(frame, mask=mask)[:3] # BGR
            # Ensure values are valid
            if any(np.isnan(c) or np.isinf(c) for c in mean_color):
                traces[region_name].append([0, 0, 0])
            else:
                # Convert to RGB and append
                traces[region_name].append(mean_color[::-1])
    
    # Debug: Check how many frames had face detection
    detection_rate = face_detected_count / len(frames) if frames else 0
    if detection_rate < 0.3:
        print(f"Warning: Low face detection rate ({detection_rate*100:.1f}%). Signal quality may be poor.")
            
    return traces


def get_rppg_signal(raw_traces, method='pos'):
    """
    Convert raw RGB traces into a single rPPG signal.
    
    Args:
        raw_traces: List of [R, G, B] values (N, 3).
        method: 'green', 'pos', 'chrom'.
    
    Returns:
        signal: 1D numpy array of the rPPG signal.
    """
    traces = np.array(raw_traces, dtype=np.float32) # Shape (Frames, 3)
    
    # Handle NaNs and zeros
    invalid_mask = np.any((traces == 0) | np.isnan(traces), axis=1)
    if np.sum(invalid_mask) > len(traces) * 0.5:
        # More than 50% invalid, use green channel
        return traces[:, 1]
    
    # Interpolate missing frames
    for i in range(traces.shape[1]):
        valid_indices = np.where(~invalid_mask)[0]
        if len(valid_indices) > 1:
            invalid_indices = np.where(invalid_mask)[0]
            traces[invalid_indices, i] = np.interp(invalid_indices, valid_indices, traces[valid_indices, i])
    
    if method == 'green':
        # Green channel is index 1 (RGB)
        return traces[:, 1]
    
    elif method == 'pos':
        # Plane-Orthogonal-to-Skin (POS) Algorithm - More robust for deepfakes
        # 1. Temporal normalization
        win_size = min(32, len(traces) // 4)  # Adaptive window size
        if win_size < 2:
            return traces[:, 1]  # Fall back to green channel
        
        n_frames = len(traces)
        H = np.zeros(n_frames)
        
        for i in range(n_frames - win_size):
            C = traces[i:i+win_size, :].T # 3 x win_size
            mean_C = np.mean(C, axis=1, keepdims=True)
            C_norm = C / (mean_C + 1e-6)
            
            # Projection matrix
            S = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float32)
            P = S @ C_norm
            
            # Compute alpha based on variance
            std_p1 = np.std(P[0, :])
            std_p2 = np.std(P[1, :])
            if std_p2 > 1e-6:
                alpha = std_p1 / std_p2
            else:
                alpha = 1.0
            
            # Resulting signal window
            h = P[0, :] + alpha * P[1, :]
            H[i:i+win_size] += (h - np.mean(h))
        
        # Normalize output
        if np.std(H) > 1e-6:
            H = (H - np.mean(H)) / np.std(H)
        
        return H
    
    return traces[:, 1] # Default to Green


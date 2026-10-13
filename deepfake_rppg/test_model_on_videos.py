"""
Quick test of improved model on real.mp4 and fake.mp4
"""

import torch
import numpy as np
import cv2
from daksh_git.KinShield.deepfake_rppg.src.models.detector import SimpleRPPGNet

def extract_frame_features(video_path):
    """Extract simple frame-based features for testing"""
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        
        while count < 60:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            count += 1
        
        cap.release()
        
        if len(frames) < 10:
            return None
        
        # Simple feature: sample frames
        indices = np.linspace(0, len(frames) - 1, 30, dtype=int)
        feature_map = np.zeros((128, 128), dtype=np.float32)
        
        for idx in indices:
            gray = cv2.cvtColor(frames[int(idx)], cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128)).astype(np.float32) / 255.0
            feature_map += gray
        
        feature_map /= len(indices)
        feature_resized = cv2.resize(feature_map, (224, 224))
        
        return torch.from_numpy(feature_resized).unsqueeze(0).float()
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_model():
    """Test the trained model"""
    device = torch.device('cpu')
    
    # Load SimpleRPPGNet
    model = SimpleRPPGNet()
    
    try:
        model.load_state_dict(torch.load('rppg_model_faceforensics.pth', map_location=device))
        print("Loaded FaceForensics-trained model\n")
    except:
        # Use the small trained model
        try:
            model.load_state_dict(torch.load('rppg_model.pth', map_location=device))
            print("Loaded standard model\n")
        except:
            print("No trained model found, using random weights\n")
    
    model.to(device)
    model.eval()
    
    # Test on your videos
    test_videos = [
        ("real.mp4", "REAL"),
        ("fake.mp4", "FAKE")
    ]
    
    print("="*70)
    print("TESTING DEEPFAKE DETECTION MODEL")
    print("="*70)
    
    for video_path, expected_label in test_videos:
        print(f"\nTesting: {video_path}")
        print(f"Expected: {expected_label}")
        
        features = extract_frame_features(video_path)
        
        if features is not None:
            features = features.unsqueeze(0) if features.dim() == 3 else features  # Add batch dim if needed
            features = features.to(device)
            
            with torch.no_grad():
                output = model(features)
                prob_fake = output.item()
                confidence = max(prob_fake, 1 - prob_fake) * 100
                predicted_label = "FAKE" if prob_fake > 0.5 else "REAL"
            
            print(f"Predicted: {predicted_label}")
            print(f"Confidence: {confidence:.2f}%")
            print(f"Prob Fake: {prob_fake:.4f}")
            
            match = "✓ CORRECT" if predicted_label == expected_label else "✗ WRONG"
            print(f"Result: {match}")
        else:
            print("Failed to extract features")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    test_model()

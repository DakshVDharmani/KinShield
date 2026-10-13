import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import torch
from daksh_git.KinShield.deepfake_rppg.src.training.dataset import RPPGDataset
from daksh_git.KinShield.deepfake_rppg.src.models.detector import RPPGResNet

def create_dummy_video(path, frames=60):
    height, width = 480, 640
    # Create a video with a moving red square to simulate "face" roughly or just noise
    # To be safe for face detector, let's just make it noise, but detector will likely fail and return zeros.
    # To make detector work, we'd need a real face.
    # Our code handles face detection failure: "if rois is None ... traces append [0,0,0]"
    # So noise is fine for pipeline stability test.
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
    
    for _ in range(frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()

def test_pipeline():
    print("1. Creating dummy video...")
    video_path = "dummy_test.mp4"
    create_dummy_video(video_path)
    
    print("2. Initializing Dataset...")
    # Label 0 (Real)
    dataset = RPPGDataset([video_path], [0])
    
    print("3. Loading sample...")
    # This invokes face detection (which will likely fail on noise) and signal extraction
    spec, label = dataset[0]
    
    print(f"   Spectrogram shape: {spec.shape}")
    print(f"   Label: {label}")
    
    assert spec.shape == (1, 224, 224), f"Expected (1, 224, 224), got {spec.shape}"
    
    print("4. Initializing Model...")
    model = RPPGResNet(pretrained=False) # No need to download weights for test
    model.eval()
    
    print("5. Running Inference...")
    input_tensor = spec.unsqueeze(0) # Batch dim
    print(f"   Input shape: {input_tensor.shape}")
    output = model(input_tensor)
    
    print(f"   Output: {output.item()}")
    
    print("Pipeline Test PASSED!")
    
    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)

if __name__ == "__main__":
    test_pipeline()

import sys
import os
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from daksh_git.KinShield.deepfake_rppg.src.training.dataset import RPPGDataset

def create_mock_dataset(root):
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    
    # Structure similar to FaceForensics++
    os.makedirs(os.path.join(root, "original_sequences/youtube/c23/videos"))
    os.makedirs(os.path.join(root, "manipulated_sequences/Deepfakes/c23/videos"))
    
    # Create dummy files
    open(os.path.join(root, "original_sequences/youtube/c23/videos/001.mp4"), 'w').close()
    open(os.path.join(root, "manipulated_sequences/Deepfakes/c23/videos/001_002.mp4"), 'w').close()
    
    # Some other structure
    os.makedirs(os.path.join(root, "random_folder/real"))
    os.makedirs(os.path.join(root, "random_folder/fake"))
    open(os.path.join(root, "random_folder/real/vid1.mp4"), 'w').close()
    open(os.path.join(root, "random_folder/fake/vid2.mp4"), 'w').close()

def test_dataset_loading():
    mock_dir = "mock_data"
    create_mock_dataset(mock_dir)
    
    print("Testing Dataset Loading...")
    try:
        dataset = RPPGDataset(mock_dir, [])
        
        print(f"Total videos found: {len(dataset)}")
        print(f"Labels: {dataset.labels}")
        
        # Expect 4 videos
        # original -> 0
        # real -> 0
        # manipulated -> 1
        # fake -> 1
        
        assert len(dataset) == 4, f"Expected 4 videos, found {len(dataset)}"
        assert dataset.labels.count(0) == 2, f"Expected 2 real, found {dataset.labels.count(0)}"
        assert dataset.labels.count(1) == 2, f"Expected 2 fake, found {dataset.labels.count(1)}"
        
        print("Dataset Loading Test PASSED!")
    finally:
        if os.path.exists(mock_dir):
            shutil.rmtree(mock_dir)

if __name__ == "__main__":
    test_dataset_loading()

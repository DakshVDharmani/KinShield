import torch
from torch.utils.data import DataLoader
from daksh_git.KinShield.deepfake_rppg.src.models.detector import SimpleRPPGNet, RPPGResNet, RPPGEfficientNet
from daksh_git.KinShield.deepfake_rppg.src.training.dataset import RPPGDataset
from daksh_git.KinShield.deepfake_rppg.src.training.train_loop import train_model
import glob
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="rPPG Deepfake Detection")
    parser.add_argument('--mode', type=str, default='predict', choices=['train', 'predict'], help='Mode: train or predict')
    parser.add_argument('--data_dir', type=str, help='Directory containing videos for training')
    parser.add_argument('--video_path', type=str, help='Path to single video for prediction')
    parser.add_argument('--model_path', type=str, default='rppg_model.pth', help='Path to save/load model')
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'resnet', 'efficientnet'], help='Model architecture')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit number of training samples for quick testing')
    parser.add_argument('--feature_dir', type=str, default=None, help='Directory containing pre-extracted .npy features')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--use_faceforensics_weights', action='store_true', help='Use model trained on FaceForensics++ dataset')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        if not args.data_dir and not args.feature_dir:
            print("Please provide either --data_dir or --feature_dir for training")
            return
            
        print(f"Loading dataset...")
        dataset = RPPGDataset(args.data_dir or '', [], max_samples=args.max_samples, feature_dir=args.feature_dir) 

        if len(dataset) == 0:
            print("No videos found! Check your --data_dir path and ensure videos are in subfolders with 'real'/'fake' in names.")
            return
            
        # Split train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        print(f"Starting training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples...")
        
        # Create model based on type
        if args.model_type == 'simple':
            model = SimpleRPPGNet()
            print("Using SimpleRPPGNet model")
        elif args.model_type == 'efficientnet':
            model = RPPGEfficientNet(pretrained=True)
            print("Using EfficientNet-B0 model")
        else:
            model = RPPGResNet(pretrained=True)
            print("Using ResNet18 model")
            
        model = train_model(model, train_loader, val_loader, num_epochs=args.epochs, device=device, model_path=args.model_path)
        print(f"Model saved to {args.model_path}")

    elif args.mode == 'predict':
        if not args.video_path:
            print("Please provide --video_path for prediction")
            return
        
        # Use FaceForensics++ trained model if available
        if args.use_faceforensics_weights and os.path.exists('rppg_model_faceforensics.pth'):
            model_path = 'rppg_model_faceforensics.pth'
            print("Using FaceForensics++ trained model")
        else:
            model_path = args.model_path
            
        # Load model based on type
        if args.model_type == 'simple':
            model = SimpleRPPGNet()
            print("Using SimpleRPPGNet model")
        elif args.model_type == 'efficientnet':
            model = RPPGEfficientNet(pretrained=False)
            print("Using EfficientNet-B0 model")
        else:
            model = RPPGResNet(pretrained=False)
            print("Using ResNet18 model")
            
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except FileNotFoundError:
            print("Model not found! Using random weights (Prediction will be random)")
            
        model.to(device)
        model.eval()
        
        # Create dataset for single video
        dataset = RPPGDataset([args.video_path], [0])
        loader = DataLoader(dataset, batch_size=1)
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                output = model(inputs)
                
                if isinstance(output, tuple):
                    # Some models return multiple outputs
                    output = output[0] if isinstance(output[0], torch.Tensor) else output
                
                if output.dim() > 1:
                    output = output.squeeze()
                    
                prob_fake = output.item() if output.dim() == 0 else output[0].item()
                label = "FAKE" if prob_fake > 0.5 else "REAL"
                confidence = max(prob_fake, 1 - prob_fake) * 100
                print(f"Prediction: {label} (Confidence: {confidence:.2f}%, Prob Fake: {prob_fake:.4f})")

if __name__ == "__main__":
    main()

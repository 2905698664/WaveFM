import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse

def load_model(model_path, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load trained model"""
    from WaveMF import WaveMF
    model = WaveMF(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device

def load_sample(hsi_path, rgb_path, device):
    """Load single sample using MyDataset logic"""
    # Load HSI data 
    hsi_data = np.load(hsi_path)
    hsi_data = np.transpose(hsi_data, (2, 0, 1))
    hsi_data = torch.from_numpy(hsi_data).float()
    hsi_data = hsi_data[:25, :, :]
    
    # Load RGB image 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(rgb_path).convert('RGB')
    rgb_image = transform(image)
    
    # Random crop 
    H, W = hsi_data.shape[1], hsi_data.shape[2]
    if 16 < H and 16 < W:
        top = torch.randint(0, H - 16 + 1, (1,)).item()
        left = torch.randint(0, W - 16 + 1, (1,)).item()
        hsi_data = hsi_data[:, top:top+16, left:left+16]
        rgb_image = rgb_image[:, top:top+16, left:left+16]
    
    # Add batch dimension
    hsi_data = hsi_data.unsqueeze(0).to(device)
    rgb_image = rgb_image.unsqueeze(0).to(device)
    
    return hsi_data, rgb_image

def predict(model, hsi_data, rgb_image, class_names):
    """Make prediction"""
    with torch.no_grad():
        outputs = model(hsi_data, rgb_image)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(outputs, dim=1).item()
        confidence = torch.max(probabilities).item()
    
    return class_names[predicted_idx], confidence

def main():
    parser = argparse.ArgumentParser(description='Test model on single sample')
    parser.add_argument('--model', type=str, required=True, help='Model weights path')
    parser.add_argument('--hsi', type=str, required=True, help='HSI .npy file path')
    parser.add_argument('--rgb', type=str, required=True, help='RGB image path')
    parser.add_argument('--dataset', type=str, default='maturity', 
                       choices=['color', 'maturity', 'structure', 'chroma', 'oil_content', 'identity'],
                       help='Dataset type')
    
    args = parser.parse_args()
    
    # Class names 
    class_names = {
        'color': ['2-4', '0-2', '6-8', '8-10', '4-6'],
        'maturity': ['6-8', '4-6', '8-10', '0-4'],
        'structure': ['5-8', '8-10', '3-5', '0-3'],
        'chroma': ['4-6', '2-4', '6-8', '8-10', '0-2'],
        'oil_content': ['8-10', '0-3', '5-8', '3-5'],
        'identity': ['7-10', '4-7', '0-4']
    }[args.dataset]
    
    # Load model
    model, device = load_model(args.model, len(class_names))
    print(f"Model loaded: {args.model}")
    
    # Load sample using MyDataset logic
    hsi_data, rgb_image = load_sample(args.hsi, args.rgb, device)
    print(f"Data loaded: HSI shape {hsi_data.shape}, RGB shape {rgb_image.shape}")
    
    # Predict
    predicted_class, confidence = predict(model, hsi_data, rgb_image, class_names)
    
    # Output
    print(f"\nPrediction: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    
    return predicted_class

if __name__ == "__main__":
    main()
    '''
python Predict.py --model weights/color_best_model.pth --hsi data\color_30_test\8-10\sample_010.npy --rgb data\color_30_test\8-10\sample_010.png --dataset color
    '''
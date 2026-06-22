import os
import numpy as np
from PIL import Image
import torch
import json
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.functional import interpolate

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_mean_std=True, augment=False, 
                 crop_size=16,pca=25):  

        self.root_dir = root_dir
        self.transform = transform
        self.use_mean_std = use_mean_std
        self.augment = augment  
        self.crop_size = crop_size  
        self.pca = pca

       
        self.data = []
        self.class_to_idx = {}
        class_idx = 0

        # Traverse each class subdirectory
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                # Record class name to index mapping
                self.class_to_idx[class_name] = class_idx
                class_idx += 1
                # Traverse files in the class subdirectory
                for filename in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, filename)
                    if filename.endswith('.npy'):
                        # Find corresponding image file
                        base_name = os.path.splitext(filename)[0]
                        img_extensions = ['.png', '.jpg', '.jpeg']
                        img_path = None
                        for ext in img_extensions:
                            possible_img_path = os.path.join(class_dir, base_name + ext)
                            if os.path.exists(possible_img_path):
                                img_path = possible_img_path
                                break
                        if img_path:
                            self.data.append((file_path, img_path, class_name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        hsi_path, img_path, class_name = self.data[idx]

        # 加载高光谱数据
        hsi_data = np.load(hsi_path)
        hsi_data = np.transpose(hsi_data, (2, 0, 1))
        hsi_data = torch.from_numpy(hsi_data).float()

        # 加载 RGB 图像
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.augment:
            hsi_data, image = self._apply_spatial_augmentation(hsi_data, image)
        if self.pca:
            hsi_data = hsi_data[:self.pca, :, :]
         # 4. Random crop + Resize (maintain original size)
        H, W = hsi_data.shape[1], hsi_data.shape[2]  # Get spatial dimensions
        if self.crop_size is not None and self.crop_size < H and self.crop_size < W and torch.rand(1) < 1:
            # Randomly select top-left coordinate of crop region
            top = torch.randint(0, H - self.crop_size + 1, (1,)).item()
            left = torch.randint(0, W - self.crop_size + 1, (1,)).item()
            # Crop
            hsi_cropped = hsi_data[:, top:top+self.crop_size, left:left+self.crop_size]

            image_cropped = image[:, top:top+self.crop_size, left:left+self.crop_size]
            # Resize back to original size
            # hsi = interpolate(hsi_cropped.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
            # image = interpolate(image_cropped.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
            hsi_data = hsi_cropped
            image = image_cropped


        # 获取类别索引
        class_idx = self.class_to_idx[class_name]

        return hsi_data, image, class_idx

    def load_mean_std(self, file_path="mean_std.json"):
        """Load mean and standard deviation from JSON file"""
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
            return data["mean"], data["std"]
        return None, None

    def _apply_spatial_augmentation(self, hsi, image):
        """
        Apply only spatial augmentation (synchronously process hsi and image)
        Args:
            hsi: Hyperspectral tensor, shape: [C, H, W]
            image: RGB tensor, shape: [3, H, W]
        Returns:
            Augmented hsi and image, shape unchanged
        """
        H, W = hsi.shape[1], hsi.shape[2]  # Get spatial dimensions

        # 1. Random horizontal flip (50% probability)
        if torch.rand(1) < 0.5:
            hsi = torch.flip(hsi, dims=[2])  # Flip width dimension
            image = torch.flip(image, dims=[2])

        # 2. Random vertical flip (50% probability)
        if torch.rand(1) < 0.5:
            hsi = torch.flip(hsi, dims=[1])  # Flip height dimension
            image = torch.flip(image, dims=[1])

        # 3. Random rotation (90° multiples, 50% probability)
        if torch.rand(1) < 0.5:
            k = torch.randint(1, 4, (1,)).item()  # Rotation times (1→90°, 2→180°, 3→270°)
            hsi = torch.rot90(hsi, k=k, dims=[1, 2])  # Rotate height and width dimensions
            image = torch.rot90(image, k=k, dims=[1, 2])


        return hsi, image


if __name__ == "__main__":
    # Define image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize dataset
    root_dir = '成熟度5000_PCA40_Reduced_3D/train'
    dataset = MyDataset(root_dir=root_dir, transform=transform, augment=False)
    print('Class to index mapping:', dataset.class_to_idx)
    # Test sample retrieval
    hsi_data, rgb_image, class_idx = dataset[0]
    print(f"Hyperspectral data shape: {hsi_data.shape}")
    print(f"RGB image shape: {rgb_image.shape}")
    print(f"Class index: {class_idx}")
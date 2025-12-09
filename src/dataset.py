import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from .config import Config 

class FaceVerificationDataset(Dataset):
    def __init__(self, pairs_csv, split='train'):
        self.df = pd.read_csv(pairs_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
        # Data augmentation for training
        if split == 'train':
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(degrees=10),
            ])
        else:
            self.augment = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load images
        img1 = Image.open(row['image1']).convert('RGB')
        img2 = Image.open(row['image2']).convert('RGB')
        
        # Apply augmentation
        if self.augment:
            img1 = self.augment(img1)
            img2 = self.augment(img2)
        
        # Transform to tensor
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return img1, img2, label

if __name__ == "__main__":
    # Test dataset
    dataset = FaceVerificationDataset(Config.PAIRS_CSV, split='train')
    print(f"Dataset size: {len(dataset)}")
    
    img1, img2, label = dataset[0]
    print(f"Image1 shape: {img1.shape}")
    print(f"Image2 shape: {img2.shape}")
    print(f"Label: {label}")
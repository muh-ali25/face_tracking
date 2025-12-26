import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import json
from pathlib import Path
from .config import Config

class FaceVerificationDataset(Dataset):
    def __init__(self, split='train'):
        """
        Dataset that stores person_id -> images mapping
        No pre-made pairs - pairs are created dynamically during sampling
        """
        # Load CSV with image paths and person IDs
        self.df = pd.read_csv(Config.PAIRS_CSV)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        # Load person mapping
        mapping_file = Path(Config.PAIRS_CSV).parent / 'person_mapping.json'
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        self.person_to_images = mapping[split]
        self.person_ids = list(self.person_to_images.keys())
        
        # Transform
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
        
        print(f"{split.upper()} dataset: {len(self.person_ids)} persons, "
              f"{len(self.df)} total images")
    
    def __len__(self):
        # Return total number of images (not pairs)
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        This won't be called directly - BalancedBatchSampler will handle pair creation
        But keep it for compatibility
        """
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        
        if self.augment:
            img = self.augment(img)
        
        img = self.transform(img)
        person_id = row['person_id']
        
        return img, person_id
    
    def get_image(self, image_path):
        """Load and transform a single image by path"""
        img = Image.open(image_path).convert('RGB')
        
        if self.augment:
            img = self.augment(img)
        
        img = self.transform(img)
        return img


class BalancedBatchSampler(Sampler):
    """
    Custom sampler that creates balanced batches:
    - Half positive pairs (same person)
    - Half negative pairs (different persons)
    """
    def __init__(self, dataset, batch_size, samples_per_person=2):
        """
        Args:
            dataset: FaceVerificationDataset
            batch_size: Total pairs per batch (must be even)
            samples_per_person: How many images to sample per person (usually 2)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_person = samples_per_person
        
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even for balanced sampling")
        
        self.person_ids = dataset.person_ids
        self.person_to_images = dataset.person_to_images
        
        # Calculate how many persons needed per batch to generate required pairs
        # Each person with 2 images = 1 positive pair
        # Need batch_size/2 positive pairs, so need batch_size/2 persons
        self.persons_per_batch = batch_size // 2
        
        if self.persons_per_batch > len(self.person_ids):
            raise ValueError(f"Not enough persons ({len(self.person_ids)}) "
                           f"for batch size {batch_size}")
    
    def __iter__(self):
        """
        Generate batches of balanced pairs
        Each batch contains 50% positive, 50% negative pairs
        """
        # Shuffle persons for each epoch
        shuffled_persons = np.random.permutation(self.person_ids)
        
        # Create batches
        for i in range(0, len(shuffled_persons), self.persons_per_batch):
            batch_persons = shuffled_persons[i:i + self.persons_per_batch]
            
            if len(batch_persons) < self.persons_per_batch:
                # Skip incomplete batch at the end
                continue
            
            batch_pairs = []
            
            # Generate positive pairs (same person)
            for person_id in batch_persons:
                images = self.person_to_images[person_id]
                
                # Sample 2 different images from this person
                if len(images) >= 2:
                    sampled = np.random.choice(images, size=2, replace=False)
                    batch_pairs.append({
                        'img1': sampled[0],
                        'img2': sampled[1],
                        'label': 1
                    })
            
            # Generate negative pairs (different persons)
            num_negative = len(batch_pairs)  # Equal to positive pairs
            
            for _ in range(num_negative):
                # Pick 2 different persons
                p1, p2 = np.random.choice(batch_persons, size=2, replace=False)
                
                # Pick random image from each
                img1 = np.random.choice(self.person_to_images[p1])
                img2 = np.random.choice(self.person_to_images[p2])
                
                batch_pairs.append({
                    'img1': img1,
                    'img2': img2,
                    'label': 0
                })
            
            # Shuffle pairs within batch
            np.random.shuffle(batch_pairs)
            
            yield batch_pairs
    
    def __len__(self):
        """Number of batches per epoch"""
        return len(self.person_ids) // self.persons_per_batch


def collate_fn(batch):
    """
    Custom collate function to handle batch of pairs
    batch is a list containing one element (the batch_pairs list from sampler)
    """
    batch_pairs = batch[0]  # Get the pairs from sampler
    
    imgs1 = []
    imgs2 = []
    labels = []
    
    # We need access to dataset to load images
    # This will be set by the DataLoader
    dataset = collate_fn.dataset
    
    for pair in batch_pairs:
        img1 = dataset.get_image(pair['img1'])
        img2 = dataset.get_image(pair['img2'])
        
        imgs1.append(img1)
        imgs2.append(img2)
        labels.append(pair['label'])
    
    imgs1 = torch.stack(imgs1)
    imgs2 = torch.stack(imgs2)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return imgs1, imgs2, labels


if __name__ == "__main__":
    # Test the dataset and sampler
    from torch.utils.data import DataLoader
    
    dataset = FaceVerificationDataset(split='train')
    sampler = BalancedBatchSampler(dataset, batch_size=32)
    
    # Set dataset for collate_fn
    collate_fn.dataset = dataset
    
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"\nDataLoader created with {len(loader)} batches per epoch")
    
    # Test one batch
    for imgs1, imgs2, labels in loader:
        print(f"\nBatch shapes:")
        print(f"  Images1: {imgs1.shape}")
        print(f"  Images2: {imgs2.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Positive pairs: {labels.sum().item()}")
        print(f"  Negative pairs: {(len(labels) - labels.sum()).item()}")
        break
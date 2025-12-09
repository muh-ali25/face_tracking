import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from itertools import combinations
from .config import Config

class PairGenerator:
    def __init__(self):
        self.processed_dir = Path(Config.PROCESSED_DATA_DIR)
        self.poses = ['front.jpg', 'left.jpg', 'right.jpg']
        
    def get_person_images(self):
        """Get all person directories and their images"""
        person_dirs = sorted([d for d in self.processed_dir.iterdir() if d.is_dir()])
        
        person_images = {}
        for person_dir in person_dirs:
            person_id = person_dir.name
            images = {}
            
            for pose in self.poses:
                img_path = person_dir / pose
                if img_path.exists() and img_path.stat().st_size > 0:  # Check file exists and not empty
                    images[pose] = str(img_path)
            
            # Accept persons with at least 2 images
            if len(images) >= 2:
                person_images[person_id] = images
                
        return person_images
    
    def generate_positive_pairs(self, person_images):
        """Generate positive pairs (same person, different poses)"""
        positive_pairs = []
        
        for person_id, images in person_images.items():
            available_poses = list(images.keys())
            
            # Generate all combinations of available poses
            for pose1, pose2 in combinations(available_poses, 2):
                positive_pairs.append({
                    'image1': images[pose1],
                    'image2': images[pose2],
                    'label': 1,
                    'person1': person_id,
                    'person2': person_id
                })
        
        return positive_pairs
    
    def generate_negative_pairs(self, person_images, num_pairs):
        """Generate negative pairs (different persons)"""
        negative_pairs = []
        person_ids = list(person_images.keys())
        
        while len(negative_pairs) < num_pairs:
            # Random person pair
            p1, p2 = np.random.choice(person_ids, 2, replace=False)
            
            # Random poses from available poses
            pose1 = np.random.choice(list(person_images[p1].keys()))
            pose2 = np.random.choice(list(person_images[p2].keys()))
            
            negative_pairs.append({
                'image1': person_images[p1][pose1],
                'image2': person_images[p2][pose2],
                'label': 0,
                'person1': p1,
                'person2': p2
            })
        
        return negative_pairs
    
    def generate_and_save_pairs(self):
        """Generate all pairs and save to CSV"""
        person_images = self.get_person_images()
        print(f"Found {len(person_images)} persons with complete image sets")
        
        # Generate positive pairs
        positive_pairs = self.generate_positive_pairs(person_images)
        print(f"Generated {len(positive_pairs)} positive pairs")
        
        # Generate equal number of negative pairs
        negative_pairs = self.generate_negative_pairs(person_images, len(positive_pairs))
        print(f"Generated {len(negative_pairs)} negative pairs")
        
        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        df = pd.DataFrame(all_pairs)
        df = df.sample(frac=1, random_state=Config.RANDOM_SEED).reset_index(drop=True)
        
        # Split by person (ensure no person overlap between train/val)
        unique_persons = list(person_images.keys())
        train_persons, val_persons = train_test_split(
            unique_persons, 
            test_size=1-Config.TRAIN_SPLIT, 
            random_state=Config.RANDOM_SEED
        )
        
        # Filter pairs based on person split
        train_df = df[df['person1'].isin(train_persons) & df['person2'].isin(train_persons)]
        val_df = df[df['person1'].isin(val_persons) | df['person2'].isin(val_persons)]
        
        # Save
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        
        final_df = pd.concat([train_df, val_df], ignore_index=True)
        final_df.to_csv(Config.PAIRS_CSV, index=False)
        
        print(f"\nDataset split:")
        print(f"Train: {len(train_df)} pairs ({len(train_persons)} persons)")
        print(f"Val: {len(val_df)} pairs ({len(val_persons)} persons)")
        print(f"Saved to {Config.PAIRS_CSV}")
        
        return final_df

if __name__ == "__main__":
    generator = PairGenerator()
    generator.generate_and_save_pairs()
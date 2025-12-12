import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
from .config import Config

class PairGenerator:
    def __init__(self):
        self.processed_dir = Path(Config.PROCESSED_DATA_DIR)
        self.poses = ['front.jpg']  # Only front images
        
    def get_person_images(self):
        """Get all person directories and their front images"""
        person_dirs = sorted([d for d in self.processed_dir.iterdir() if d.is_dir()])
        
        person_images = {}
        for person_dir in person_dirs:
            person_id = person_dir.name
            
            front_path = person_dir / 'front.jpg'
            if front_path.exists() and front_path.stat().st_size > 0:
                person_images[person_id] = str(front_path)
                
        return person_images
    
    def create_augmented_version(self, image_path, aug_type):
        """
        Create augmented version of image for positive pairs
        aug_type: 'flip', 'bright', 'dark', 'rotate_left', 'rotate_right', 'crop_left', 'crop_right'
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        h, w = image.shape[:2]
        
        if aug_type == 'flip':
            # Horizontal flip
            augmented = cv2.flip(image, 1)
            
        elif aug_type == 'bright':
            # Increase brightness
            augmented = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
            
        elif aug_type == 'dark':
            # Decrease brightness
            augmented = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
            
        elif aug_type == 'rotate_left':
            # Rotate 5 degrees left
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, 5, 1.0)
            augmented = cv2.warpAffine(image, matrix, (w, h))
            
        elif aug_type == 'rotate_right':
            # Rotate 5 degrees right
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, -5, 1.0)
            augmented = cv2.warpAffine(image, matrix, (w, h))
            
        elif aug_type == 'crop_left':
            # Crop and resize - slight left shift
            crop = image[:, int(w*0.05):, :]
            augmented = cv2.resize(crop, (w, h))
            
        elif aug_type == 'crop_right':
            # Crop and resize - slight right shift
            crop = image[:, :int(w*0.95), :]
            augmented = cv2.resize(crop, (w, h))
        
        else:
            augmented = image
        
        # Save augmented version temporarily
        temp_dir = Path(Config.PROCESSED_DATA_DIR) / '_augmented_temp'
        temp_dir.mkdir(exist_ok=True)
        
        person_id = Path(image_path).parent.name
        aug_path = temp_dir / f"{person_id}_{aug_type}.jpg"
        cv2.imwrite(str(aug_path), augmented)
        
        return str(aug_path)
    
    def generate_positive_pairs(self, person_images):
        """Generate positive pairs using original + augmented versions"""
        positive_pairs = []
        
        aug_types = ['flip', 'bright', 'dark', 'rotate_left', 'rotate_right', 'crop_left', 'crop_right']
        
        for person_id, image_path in person_images.items():
            # Create multiple positive pairs per person using different augmentations
            for aug_type in aug_types:
                aug_path = self.create_augmented_version(image_path, aug_type)
                
                if aug_path:
                    positive_pairs.append({
                        'image1': image_path,
                        'image2': aug_path,
                        'label': 1,
                        'person1': person_id,
                        'person2': person_id,
                        'augmentation': aug_type
                    })
        
        return positive_pairs
    
    def generate_negative_pairs(self, person_images, num_pairs):
        """Generate negative pairs (different persons)"""
        negative_pairs = []
        person_ids = list(person_images.keys())
        
        if len(person_ids) < 2:
            print("Not enough persons for negative pairs!")
            return negative_pairs
        
        while len(negative_pairs) < num_pairs:
            # Random person pair
            p1, p2 = np.random.choice(person_ids, 2, replace=False)
            
            negative_pairs.append({
                'image1': person_images[p1],
                'image2': person_images[p2],
                'label': 0,
                'person1': p1,
                'person2': p2,
                'augmentation': 'none'
            })
        
        return negative_pairs
    
    def generate_and_save_pairs(self):
        """Generate all pairs and save to CSV"""
        person_images = self.get_person_images()
        print(f"Found {len(person_images)} persons with front images")
        
        if len(person_images) < 2:
            print("Error: Need at least 2 persons to create pairs!")
            return None
        
        # Generate positive pairs (7 augmentations per person)
        positive_pairs = self.generate_positive_pairs(person_images)
        print(f"Generated {len(positive_pairs)} positive pairs (with augmentation)")
        
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
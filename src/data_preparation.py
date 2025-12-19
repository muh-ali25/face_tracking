import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from itertools import combinations
from .config import Config

class PairGenerator:
    def __init__(self):
        self.processed_dir = Path(Config.PROCESSED_DATA_DIR)
        
    def get_person_images(self):
        """Get all person directories and their images"""
        person_dirs = sorted([d for d in self.processed_dir.iterdir() if d.is_dir()])
        
        person_images = {}
        
        for person_dir in person_dirs:
            person_id = person_dir.name
            
            # Get all processed images for this person
            image_files = sorted(list(person_dir.glob("image_*.jpg")))
            
            if len(image_files) >= 2:  # Need at least 2 images to create pairs
                person_images[person_id] = [str(img) for img in image_files]
        
        return person_images
    
    def generate_positive_pairs(self, person_images):
        """Generate positive pairs using real different images of same person"""
        positive_pairs = []
        
        for person_id, images in person_images.items():
            # Create all possible combinations of this person's images
            # If person has 3 images: (img1,img2), (img1,img3), (img2,img3)
            for img1, img2 in combinations(images, 2):
                positive_pairs.append({
                    'image1': img1,
                    'image2': img2,
                    'label': 1,
                    'person1': person_id,
                    'person2': person_id
                })
        
        return positive_pairs
    
    def generate_negative_pairs(self, person_images, num_pairs):
        """Generate negative pairs (different persons)"""
        negative_pairs = []
        person_ids = list(person_images.keys())
        
        if len(person_ids) < 2:
            print("Not enough persons for negative pairs!")
            return negative_pairs
        
        attempts = 0
        max_attempts = num_pairs * 10  # Prevent infinite loop
        
        while len(negative_pairs) < num_pairs and attempts < max_attempts:
            attempts += 1
            
            # Random person pair
            p1, p2 = np.random.choice(person_ids, 2, replace=False)
            
            # Random image from each person
            img1 = np.random.choice(person_images[p1])
            img2 = np.random.choice(person_images[p2])
            
            # Check if this pair already exists (avoid duplicates)
            pair_key = tuple(sorted([img1, img2]))
            if not any(tuple(sorted([p['image1'], p['image2']])) == pair_key 
                      for p in negative_pairs):
                negative_pairs.append({
                    'image1': img1,
                    'image2': img2,
                    'label': 0,
                    'person1': p1,
                    'person2': p2
                })
        
        return negative_pairs
    
    def generate_and_save_pairs(self):
        """Generate all pairs and save to CSV"""
        person_images = self.get_person_images()
        print(f"Found {len(person_images)} persons with sufficient images")
        
        if len(person_images) < 2:
            print("Error: Need at least 2 persons to create pairs!")
            return None
        
        # Show distribution
        image_counts = [len(imgs) for imgs in person_images.values()]
        print(f"Images per person - Min: {min(image_counts)}, "
              f"Max: {max(image_counts)}, Avg: {sum(image_counts)/len(image_counts):.1f}")
        
        # Generate positive pairs (real combinations)
        positive_pairs = self.generate_positive_pairs(person_images)
        print(f"Generated {len(positive_pairs)} positive pairs (real image combinations)")
        
        # Generate equal number of negative pairs
        negative_pairs = self.generate_negative_pairs(person_images, len(positive_pairs))
        print(f"Generated {len(negative_pairs)} negative pairs")
        
        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        df = pd.DataFrame(all_pairs)
        df = df.sample(frac=1, random_state=Config.RANDOM_SEED).reset_index(drop=True)
        
        # Split by person (ensure no person overlap between train/val)
        unique_persons = list(person_images.keys())
        
        # Ensure we have enough persons for split
        if len(unique_persons) < 5:
            print("Warning: Very few persons. Using 70/30 split.")
            train_split = 0.7
        else:
            train_split = Config.TRAIN_SPLIT
        
        train_persons, val_persons = train_test_split(
            unique_persons, 
            test_size=1-train_split, 
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
        print(f"Positive/Negative ratio - Train: {sum(train_df['label'])}/"
              f"{len(train_df)-sum(train_df['label'])}, "
              f"Val: {sum(val_df['label'])}/{len(val_df)-sum(val_df['label'])}")
        print(f"Saved to {Config.PAIRS_CSV}")
        
        return final_df

if __name__ == "__main__":
    generator = PairGenerator()
    generator.generate_and_save_pairs()
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from .config import Config

class PairGenerator:
    def __init__(self):
        self.processed_dir = Path(Config.PROCESSED_DATA_DIR)
        
    def get_person_images(self):
        """Get all person directories and their images - returns dictionary"""
        person_dirs = sorted([d for d in self.processed_dir.iterdir() if d.is_dir()])
        
        person_images = {}
        
        for person_dir in person_dirs:
            person_id = person_dir.name
            
            # Get all jpg/jpeg/png images
            image_files = list(person_dir.glob("*.jpg")) + \
                         list(person_dir.glob("*.jpeg")) + \
                         list(person_dir.glob("*.png"))
            
            # Need at least 2 images per person
            if len(image_files) >= 2:
                person_images[person_id] = [str(img) for img in sorted(image_files)]
        
        return person_images
    
    def split_persons_and_save(self):
        """
        Split persons into train/val sets
        No pair generation - just save person lists
        """
        person_images = self.get_person_images()
        print(f"Found {len(person_images)} persons")
        
        if len(person_images) < 2:
            print("Error: Need at least 2 persons!")
            return None
        
        # Show statistics
        image_counts = [len(imgs) for imgs in person_images.values()]
        print(f"Images per person - Min: {min(image_counts)}, "
              f"Max: {max(image_counts)}, Avg: {sum(image_counts)/len(image_counts):.1f}")
        print(f"Total images: {sum(image_counts)}")
        
        # Split persons into train/val
        unique_persons = list(person_images.keys())
        train_persons, val_persons = train_test_split(
            unique_persons,
            test_size=1-Config.TRAIN_SPLIT,
            random_state=Config.RANDOM_SEED
        )
        
        # Create data structure for train
        train_data = []
        for person_id in train_persons:
            for img_path in person_images[person_id]:
                train_data.append({
                    'person_id': person_id,
                    'image_path': img_path,
                    'split': 'train'
                })
        
        # Create data structure for val
        val_data = []
        for person_id in val_persons:
            for img_path in person_images[person_id]:
                val_data.append({
                    'person_id': person_id,
                    'image_path': img_path,
                    'split': 'val'
                })
        
        # Combine and save
        all_data = train_data + val_data
        df = pd.DataFrame(all_data)
        df.to_csv(Config.PAIRS_CSV, index=False)
        
        print(f"\nDataset split:")
        print(f"Train: {len(train_data)} images ({len(train_persons)} persons)")
        print(f"Val: {len(val_data)} images ({len(val_persons)} persons)")
        print(f"Saved to {Config.PAIRS_CSV}")
        
        # Also save person-to-images mapping for easy access
        import json
        mapping = {
            'train': {pid: person_images[pid] for pid in train_persons},
            'val': {pid: person_images[pid] for pid in val_persons}
        }
        
        mapping_file = Path(Config.PAIRS_CSV).parent / 'person_mapping.json'
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"Person mapping saved to {mapping_file}")
        
        return df

if __name__ == "__main__":
    generator = PairGenerator()
    generator.split_persons_and_save()
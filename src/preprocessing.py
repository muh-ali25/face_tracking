import cv2
import numpy as np
from pathlib import Path
from .config import Config

class FaceAligner:
    def __init__(self):
        """No detector needed - using geometric cropping"""
        pass
    
    def extract_periocular_region(self, image):
        """
        Extract eye region (eyes, eyebrows, forehead) from image
        Works without face detection - uses geometric approach
        """
        h, w = image.shape[:2]
        
        # Focus on upper-middle portion of image where eyes typically are
        # Vertical: Take upper 65% of image (forehead to nose bridge)
        top = int(h * 0.15)
        bottom = int(h * 0.65)
        
        # Horizontal: Take middle 80% (both eyes + surrounding area)
        left = int(w * 0.10)
        right = int(w * 0.90)
        
        # Crop periocular region
        periocular = image[top:bottom, left:right]
        
        if periocular.size == 0:
            return None
        
        return periocular
    
    def enhance_image(self, image):
        """Enhance image quality for better feature extraction"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def process_image(self, image_path, save_path=None):
        """Complete pipeline: load, extract periocular region, enhance, save"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load: {image_path}")
            return None
        
        # Extract periocular region
        periocular = self.extract_periocular_region(image)
        if periocular is None:
            print(f"Failed to extract region: {image_path}")
            return None
        
        # Enhance image quality
        enhanced = self.enhance_image(periocular)
        
        # Resize to target size
        resized = cv2.resize(enhanced, Config.IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        
        if save_path:
            cv2.imwrite(str(save_path), resized)
        
        return resized
    
    def process_dataset(self):
        """Process all images in raw data directory - ALL images per person"""
        raw_dir = Path(Config.RAW_DATA_DIR)
        processed_dir = Path(Config.PROCESSED_DATA_DIR)
        
        # Clear processed directory for fresh start
        import shutil
        if processed_dir.exists():
            print("Cleaning previous processed data...")
            shutil.rmtree(processed_dir)
        processed_dir.mkdir(exist_ok=True)
        
        person_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
        
        print(f"Processing {len(person_dirs)} persons with periocular extraction...")
        print("Processing ALL images per person (multiple front images)")
        print("=" * 60)
        
        success_count = 0
        fail_count = 0
        total_images_processed = 0
        
        person_image_counts = []
        
        for person_dir in person_dirs:
            person_id = person_dir.name
            output_person_dir = processed_dir / person_id
            output_person_dir.mkdir(exist_ok=True)
            
            # Get all jpg/jpeg/png images
            image_files = list(person_dir.glob("*.jpg")) + \
                         list(person_dir.glob("*.jpeg")) + \
                         list(person_dir.glob("*.png"))
            
            if len(image_files) == 0:
                print(f"✗ {person_id}: No images found")
                fail_count += 1
                continue
            
            processed_for_person = 0
            
            for idx, img_path in enumerate(sorted(image_files)):
                # Save with numbered names: image_001.jpg, image_002.jpg, etc.
                save_path = output_person_dir / f"image_{idx+1:03d}.jpg"
                result = self.process_image(img_path, save_path)
                
                if result is not None:
                    processed_for_person += 1
                    total_images_processed += 1
            
            if processed_for_person >= 2:  # Need at least 2 images
                success_count += 1
                person_image_counts.append(processed_for_person)
                print(f"✓ {person_id}: {processed_for_person} images processed")
            else:
                fail_count += 1
                print(f"✗ {person_id}: Only {processed_for_person} image(s) - need at least 2")
        
        print("=" * 60)
        print(f"\nProcessing complete!")
        print(f"✓ Successfully processed: {success_count} persons")
        print(f" Failed to process: {fail_count} persons")
        print(f" Total images processed: {total_images_processed}")
        
        if person_image_counts:
            print(f" Images per person - Min: {min(person_image_counts)}, "
                  f"Max: {max(person_image_counts)}, "
                  f"Avg: {sum(person_image_counts)/len(person_image_counts):.1f}")

if __name__ == "__main__":
    aligner = FaceAligner()
    aligner.process_dataset()
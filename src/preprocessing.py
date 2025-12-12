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
        # Assuming images are roughly centered portraits
        
        # Vertical: Take upper 65% of image (forehead to nose bridge)
        top = int(h * 0.15)  # Start 15% from top (captures forehead)
        bottom = int(h * 0.65)  # End at 65% (captures to nose bridge)
        
        # Horizontal: Take middle 80% (both eyes + surrounding area)
        left = int(w * 0.10)
        right = int(w * 0.90)
        
        # Crop periocular region
        periocular = image[top:bottom, left:right]
        
        # Ensure we got a valid crop
        if periocular.size == 0:
            return None
        
        return periocular
    
    def enhance_image(self, image):
        """
        Enhance image quality for better feature extraction
        """
        # Convert to LAB color space for better contrast adjustment
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def process_image(self, image_path, save_path=None):
        """
        Complete pipeline: load, extract periocular region, enhance, save
        """
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
        """Process all images in raw data directory - FRONT ONLY"""
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
        print("Processing FRONT images only (ignoring left/right)")
        print("=" * 60)
        
        success_count = 0
        fail_count = 0
        total_images_processed = 0
        
        for person_dir in person_dirs:
            person_id = person_dir.name
            output_person_dir = processed_dir / person_id
            output_person_dir.mkdir(exist_ok=True)
            
            person_success = False
            
            # Only process front.jpg
            for pose in ['front.jpg']:
                img_path = person_dir / pose
                if not img_path.exists():
                    print(f" {person_id}: Missing front.jpg")
                    continue
                
                save_path = output_person_dir / pose
                result = self.process_image(img_path, save_path)
                
                if result is not None:
                    person_success = True
                    total_images_processed += 1
                    print(f" {person_id}: front.jpg processed")
                else:
                    print(f" {person_id}: front.jpg failed")
            
            if person_success:
                success_count += 1
            else:
                fail_count += 1
        
        print("=" * 60)
        print(f"\nProcessing complete!")
        print(f" Successfully processed: {success_count} persons")
        print(f" Failed to process: {fail_count} persons")
        print(f" Total images processed: {total_images_processed}")
        print(f" Success rate: {success_count}/{len(person_dirs)} ({success_count/len(person_dirs)*100:.1f}%)")

if __name__ == "__main__":
    aligner = FaceAligner()
    aligner.process_dataset()
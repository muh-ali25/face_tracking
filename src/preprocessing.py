import cv2
import numpy as np
from pathlib import Path
from mtcnn import MTCNN
from .config import Config

class FaceAligner:
    def __init__(self):
        self.detector = MTCNN()
        
    def detect_landmarks(self, image):
        """Detect facial landmarks using MTCNN"""
        try:
            # MTCNN expects RGB
            faces = self.detector.detect_faces(image)
            
            if not faces or len(faces) == 0:
                return None
            
            # Get first face
            face = faces[0]
            
            # Extract bounding box
            x, y, w, h = face['box']
            
            # Extract landmarks
            keypoints = face['keypoints']
            landmarks = np.array([
                keypoints['left_eye'],
                keypoints['right_eye'],
                keypoints['nose'],
                keypoints['mouth_left'],
                keypoints['mouth_right']
            ])
            
            facial_area = [x, y, x+w, y+h]
            
            return landmarks, facial_area
            
        except Exception as e:
            print(f"Detection error: {e}")
            return None
    
    def align_face(self, image, landmarks, facial_area):
        """Align face based on eye positions"""
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # Calculate angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Get eye midpoint
        eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                    (left_eye[1] + right_eye[1]) / 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, 1.0)
        
        # Apply rotation
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                flags=cv2.INTER_CUBIC)
        
        # Crop using facial area with padding
        x, y, x2, y2 = facial_area
        padding = 50
        x = max(0, x - padding)
        y = max(0, y - padding)
        x2 = min(aligned.shape[1], x2 + padding)
        y2 = min(aligned.shape[0], y2 + padding)
        
        cropped = aligned[y:y2, x:x2]
        
        if cropped.size == 0:
            return None
        
        # Resize to target size
        resized = cv2.resize(cropped, Config.IMG_SIZE)
        
        return resized
    
    def process_image(self, image_path, save_path=None):
        """Complete pipeline: load, detect, align, save"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load: {image_path}")
            return None
        
        # Convert BGR to RGB for MTCNN
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result = self.detect_landmarks(image_rgb)
        if result is None:
            print(f"No face detected: {image_path}")
            return None
        
        landmarks, facial_area = result
        
        aligned = self.align_face(image, landmarks, facial_area)
        if aligned is None:
            print(f"Alignment failed: {image_path}")
            return None
        
        if save_path:
            cv2.imwrite(str(save_path), aligned)
        
        return aligned
    
    def process_dataset(self):
        """Process all images in raw data directory"""
        raw_dir = Path(Config.RAW_DATA_DIR)
        processed_dir = Path(Config.PROCESSED_DATA_DIR)
        
        person_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
        
        print(f"Processing {len(person_dirs)} persons...")
        
        success_count = 0
        fail_count = 0
        
        for person_dir in person_dirs:
            person_id = person_dir.name
            output_person_dir = processed_dir / person_id
            output_person_dir.mkdir(exist_ok=True)
            
            person_success = True
            
            for pose in ['front.jpg', 'left.jpg', 'right.jpg']:
                img_path = person_dir / pose
                if not img_path.exists():
                    print(f"Missing {pose} for {person_id}")
                    person_success = False
                    continue
                
                save_path = output_person_dir / pose
                result = self.process_image(img_path, save_path)
                
                if result is None:
                    person_success = False
            
            if person_success:
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\nProcessing complete!")
        print(f"✓ Successfully processed: {success_count} persons")
        print(f"✗ Failed to process: {fail_count} persons")
        print(f"Total: {success_count + fail_count} persons")

if __name__ == "__main__":
    aligner = FaceAligner()
    aligner.process_dataset()
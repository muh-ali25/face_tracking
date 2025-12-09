import torch
from PIL import Image
import torchvision.transforms as transforms
from .config import Config
from .model import SiameseNetwork
from .preprocessing import FaceAligner 

class FaceVerifier:
    def __init__(self, model_path):
        self.device = torch.device(Config.DEVICE)
        
        # Load model
        self.model = SiameseNetwork(
            Config.EMBEDDING_DIM, 
            Config.MLP_HIDDEN_DIM
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Face aligner
        self.aligner = FaceAligner()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """Align and preprocess a single image"""
        aligned = self.aligner.process_image(image_path)
        if aligned is None:
            raise ValueError(f"Could not process image: {image_path}")
        
        # Convert BGR to RGB and to PIL
        aligned_rgb = aligned[:, :, ::-1]
        pil_image = Image.fromarray(aligned_rgb)
        
        # Transform to tensor
        tensor = self.transform(pil_image).unsqueeze(0)
        return tensor
    
    def verify(self, image1_path, image2_path):
        """Verify if two images are of the same person"""
        try:
            img1 = self.preprocess_image(image1_path).to(self.device)
            img2 = self.preprocess_image(image2_path).to(self.device)
            
            with torch.no_grad():
                similarity = self.model(img1, img2).item()
            
            is_same = similarity > Config.SIMILARITY_THRESHOLD
            
            return {
                'similarity_score': similarity,
                'is_same_person': is_same,
                'confidence': abs(similarity - 0.5) * 2  # 0 to 1 scale
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'similarity_score': None,
                'is_same_person': None
            }

if __name__ == "__main__":
    import glob
    
    # Find latest model
    models = glob.glob(f"{Config.MODEL_SAVE_DIR}/*.pth")
    if not models:
        print("No trained model found!")
        exit()
    
    latest_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Loading model: {latest_model}")
    
    verifier = FaceVerifier(latest_model)
    
    # Example usage
    image1 = "data/raw/person_001/front.jpg"
    image2 = "data/raw/person_001/left.jpg"
    
    result = verifier.verify(image1, image2)
    print(f"\nVerification result:")
    
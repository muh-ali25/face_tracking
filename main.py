import argparse
from src.preprocessing import FaceAligner
from src.data_preparation import PairGenerator
from src.train import Trainer
from src.evaluate import Evaluator
from src.inference import FaceVerifier
from src.config import Config
import glob

def preprocess():
    """Step 1: Preprocess all images"""
    print("=" * 50)
    print("STEP 1: Preprocessing images with MediaPipe")
    print("=" * 50)
    aligner = FaceAligner()
    aligner.process_dataset()

def prepare_data():
    """Step 2: Generate training pairs"""
    print("\n" + "=" * 50)
    print("STEP 2: Generating training pairs")
    print("=" * 50)
    generator = PairGenerator()
    generator.generate_and_save_pairs()

def train():
    """Step 3: Train the model"""
    print("\n" + "=" * 50)
    print("STEP 3: Training Siamese Network")
    print("=" * 50)
    trainer = Trainer()
    trainer.train()

def evaluate():
    """Step 4: Evaluate and analyze false positives"""
    print("\n" + "=" * 50)
    print("STEP 4: Evaluating model")
    print("=" * 50)
    
    models = glob.glob(f"{Config.MODEL_SAVE_DIR}/*.pth")
    if not models:
        print("No trained model found! Train the model first.")
        return
    
    latest_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Using model: {latest_model}")
    
    evaluator = Evaluator(latest_model)
    evaluator.analyze_false_positives()

def predict(image1, image2):
    """Step 5: Predict on new image pair"""
    print("\n" + "=" * 50)
    print("STEP 5: Running inference")
    print("=" * 50)
    
    models = glob.glob(f"{Config.MODEL_SAVE_DIR}/*.pth")
    if not models:
        print("No trained model found! Train the model first.")
        return
    
    latest_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Using model: {latest_model}")
    
    verifier = FaceVerifier(latest_model)
    result = verifier.verify(image1, image2)
    
    if 'error' in result:
        print(f"\nError: {result['error']}")
        return
    
    print(f"\nResults:")
    print(f"  Image 1: {image1}")
    print(f"  Image 2: {image2}")
    print(f"  Similarity Score: {result['similarity_score']:.4f}")
    print(f"  Same Person: {result['is_same_person']}")
    print(f"  Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Verification Pipeline')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['preprocess', 'prepare', 'train', 'evaluate', 'predict', 'all'],
                       help='Mode to run')
    parser.add_argument('--image1', type=str, help='First image path for prediction')
    parser.add_argument('--image2', type=str, help='Second image path for prediction')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        preprocess()
        prepare_data()
        train()
        evaluate()
    elif args.mode == 'preprocess':
        preprocess()
    elif args.mode == 'prepare':
        prepare_data()
    elif args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        evaluate()
    elif args.mode == 'predict':
        if not args.image1 or not args.image2:
            print("Error: --image1 and --image2 are required for predict mode")
        else:
            predict(args.image1, args.image2)
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from .config import Config
from .dataset import FaceVerificationDataset
from .model import SiameseNetwork 

class Evaluator:
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
        
        # Load validation data
        self.val_dataset = FaceVerificationDataset(Config.PAIRS_CSV, split='val')
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        
    def evaluate(self):
        all_predictions = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for img1, img2, labels in self.val_loader:
                img1, img2 = img1.to(self.device), img2.to(self.device)
                
                outputs = self.model(img1, img2).squeeze()
                predictions = (outputs > Config.SIMILARITY_THRESHOLD).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_scores.extend(outputs.cpu().numpy())
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, 
                                target_names=['Different', 'Same']))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Different', 'Same'],
                yticklabels=['Different', 'Same'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        print("\nConfusion matrix saved to confusion_matrix.png")
        
        return all_predictions, all_labels, all_scores
    
    def analyze_false_positives(self):
        """Find and analyze false positive cases"""
        df = pd.read_csv(Config.PAIRS_CSV)
        df = df[df['split'] == 'val'].reset_index(drop=True)
        
        predictions, labels, scores = self.evaluate()
        
        # Find false positives (predicted same but actually different)
        false_positives = []
        for i, (pred, label, score) in enumerate(zip(predictions, labels, scores)):
            if pred == 1 and label == 0:  # Predicted same but different people
                false_positives.append({
                    'index': i,
                    'image1': df.iloc[i]['image1'],
                    'image2': df.iloc[i]['image2'],
                    'person1': df.iloc[i]['person1'],
                    'person2': df.iloc[i]['person2'],
                    'score': score
                })
        
        fp_df = pd.DataFrame(false_positives)
        fp_df.to_csv('false_positives.csv', index=False)
        
        print(f"\nFound {len(false_positives)} false positives")
        print("Saved to false_positives.csv")
        print("\nTop 5 false positives (highest confidence):")
        print(fp_df.nlargest(5, 'score')[['person1', 'person2', 'score']])

if __name__ == "__main__":
    # Find latest model
    import glob
    models = glob.glob(f"{Config.MODEL_SAVE_DIR}/*.pth")
    if not models:
        print("No trained model found!")
        exit()
    
    latest_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Loading model: {latest_model}")
    
    evaluator = Evaluator(latest_model)
    evaluator.analyze_false_positives()
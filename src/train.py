import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from .config import Config
from .dataset import FaceVerificationDataset
from .model import SiameseNetwork

class Trainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        print(f"Using device: {self.device}")
        
        # Model
        self.model = SiameseNetwork(
            Config.EMBEDDING_DIM, 
            Config.MLP_HIDDEN_DIM
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        
        # Datasets
        self.train_dataset = FaceVerificationDataset(Config.PAIRS_CSV, split='train')
        self.val_dataset = FaceVerificationDataset(Config.PAIRS_CSV, split='val')
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True,
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False,
            num_workers=2
        )
        
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for img1, img2, labels in pbar:
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(img1, img2).squeeze()
            
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for img1, img2, labels in tqdm(self.val_loader, desc="Validation"):
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                
                outputs = self.model(img1, img2).squeeze()
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        path = os.path.join(Config.MODEL_SAVE_DIR, f'model_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def train(self):
        print(f"Training dataset: {len(self.train_dataset)} pairs")
        print(f"Validation dataset: {len(self.val_dataset)} pairs")
        
        for epoch in range(Config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                print("New best model saved!")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
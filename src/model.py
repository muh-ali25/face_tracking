import torch
import torch.nn as nn
import torchvision.models as models
from config import Config

class EmbeddingNetwork(nn.Module):
    """Feature extraction backbone"""
    def __init__(self, embedding_dim=512):
        super(EmbeddingNetwork, self).__init__()
        
        # Use pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add embedding layer
        self.fc = nn.Linear(2048, embedding_dim)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SimilarityNetwork(nn.Module):
    """MLP to compute similarity from concatenated embeddings"""
    def __init__(self, embedding_dim=512, hidden_dim=256):
        super(SimilarityNetwork, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, emb1, emb2):
        # Concatenate embeddings
        combined = torch.cat([emb1, emb2], dim=1)
        similarity = self.fc(combined)
        return similarity

class SiameseNetwork(nn.Module):
    """Complete Siamese network"""
    def __init__(self, embedding_dim=512, hidden_dim=256):
        super(SiameseNetwork, self).__init__()
        
        # Shared embedding network
        self.embedding_net = EmbeddingNetwork(embedding_dim)
        
        # Similarity network
        self.similarity_net = SimilarityNetwork(embedding_dim, hidden_dim)
    
    def forward(self, img1, img2):
        # Extract embeddings (shared weights)
        emb1 = self.embedding_net(img1)
        emb2 = self.embedding_net(img2)
        
        # Compute similarity
        similarity = self.similarity_net(emb1, emb2)
        
        return similarity
    
    def get_embeddings(self, img):
        """Extract embeddings for inference"""
        return self.embedding_net(img)

if __name__ == "__main__":
    # Test model
    model = SiameseNetwork(Config.EMBEDDING_DIM, Config.MLP_HIDDEN_DIM)
    
    img1 = torch.randn(2, 3, 224, 224)
    img2 = torch.randn(2, 3, 224, 224)
    
    output = model(img1, img2)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
import torch
import torch.nn as nn
import torch.nn.functional as F

class StormPredictor(nn.Module):
    """
    Hybrid CNN-Transformer model for storm prediction.
    Combines CNN for processing satellite imagery and Transformer for temporal sequence data.
    """
    def __init__(
        self,
        image_channels: int,
        sequence_length: int,
        sequence_features: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float
    ):
        super().__init__()
        
        # Store configuration
        self.image_channels = image_channels
        self.sequence_length = sequence_length
        self.sequence_features = sequence_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # CNN for satellite imagery processing
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Transformer for temporal sequence processing
        self.sequence_projection = nn.Linear(sequence_features, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, sequence_length, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final prediction layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            image: Satellite imagery tensor of shape (batch_size, channels, height, width)
            sequence: Temporal sequence tensor of shape (batch_size, sequence_length, features)
            
        Returns:
            torch.Tensor: Probability of storm event
        """
        # Process satellite imagery
        img_features = self.cnn(image)
        img_features = img_features.view(img_features.size(0), -1)
        
        # Process temporal sequence
        seq_features = self.sequence_projection(sequence)
        seq_features = seq_features + self.pos_encoder
        seq_features = self.transformer(seq_features)
        seq_features = seq_features.mean(dim=1)  # Global average pooling
        
        # Combine features
        combined = torch.cat((img_features, seq_features), dim=1)
        
        # Final prediction
        logits = self.fc(combined)
        return logits

class StormPredictorConfig:
    """Configuration class for StormPredictor model."""
    def __init__(
        self,
        image_channels: int = 3,
        sequence_length: int = 24,
        sequence_features: int = 10,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        self.image_channels = image_channels
        self.sequence_length = sequence_length
        self.sequence_features = sequence_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

def create_model(config: StormPredictorConfig) -> StormPredictor:
    """Factory function to create a StormPredictor model."""
    return StormPredictor(
        image_channels=config.image_channels,
        sequence_length=config.sequence_length,
        sequence_features=config.sequence_features,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout
    ) 
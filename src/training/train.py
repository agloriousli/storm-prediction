import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import logging
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from sklearn.metrics import roc_auc_score

from ..models.storm_predictor import StormPredictor, StormPredictorConfig
from ..data.data_processor import StormDataProcessor, StormDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StormModelTrainer:
    """Trainer class for the StormPredictor model."""
    
    def __init__(
        self,
        model: StormPredictor,
        device: torch.device,
        config: Dict[str, Any]
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Set default model directory if not provided
        if 'model_dir' not in config:
            config['model_dir'] = os.path.join(os.getcwd(), 'models')
            os.makedirs(config['model_dir'], exist_ok=True)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.criterion = nn.BCELoss()
        
        # Initialize metrics
        self.best_val_loss = float('inf')
        self.best_model_path = os.path.join(
            config['model_dir'],
            f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        
        # Initialize learning rate scheduler if configured
        if 'lr_scheduler' in config:
            scheduler_config = config['lr_scheduler']
            if scheduler_config['type'] == 'ReduceLROnPlateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=scheduler_config['factor'],
                    patience=scheduler_config['patience']
                )
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
        else:
            self.scheduler = None
        
        # Initialize early stopping if configured
        if 'early_stopping' in config:
            self.early_stopping = {
                'patience': config['early_stopping']['patience'],
                'min_delta': config['early_stopping']['min_delta'],
                'counter': 0,
                'best_loss': float('inf')
            }
        else:
            self.early_stopping = None
    
    def _train_step(self, images: torch.Tensor, sequences: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Perform a single training step.
        
        Args:
            images: Batch of satellite images
            sequences: Batch of temporal sequences
            labels: Batch of target labels
            
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(images, sequences)
        loss = self.criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _validate_step(self, images: torch.Tensor, sequences: torch.Tensor, labels: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """
        Perform a single validation step.
        
        Args:
            images: Batch of satellite images
            sequences: Batch of temporal sequences
            labels: Batch of target labels
            
        Returns:
            Tuple of (loss value, metrics dictionary)
        """
        self.model.eval()
        with torch.no_grad():
            # Forward pass
            outputs = self.model(images, sequences)
            loss = self.criterion(outputs, labels)
            
            # Calculate metrics
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == labels).float().mean()
            
            # Calculate AUC
            try:
                auc = roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())
            except ValueError:
                auc = 0.0
            
            metrics = {
                'accuracy': accuracy.item(),
                'auc': auc
            }
            
            return loss.item(), metrics
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            images, sequences, labels = [b.to(self.device) for b in batch]
            loss = self._train_step(images, sequences, labels)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average validation loss, metrics dictionary)
        """
        total_loss = 0.0
        total_metrics = {'accuracy': 0.0, 'auc': 0.0}
        num_batches = 0
        
        for batch in val_loader:
            images, sequences, labels = [b.to(self.device) for b in batch]
            loss, metrics = self._validate_step(images, sequences, labels)
            
            total_loss += loss
            for key in metrics:
                total_metrics[key] += metrics[key]
            num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int) -> None:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train for
        """
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, metrics = self.validate(val_loader)
            
            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Metrics: {metrics}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Early stopping
            if self.early_stopping is not None:
                if val_loss < self.early_stopping['best_loss'] - self.early_stopping['min_delta']:
                    self.early_stopping['best_loss'] = val_loss
                    self.early_stopping['counter'] = 0
                    # Save best model
                    torch.save(self.model.state_dict(), self.best_model_path)
                else:
                    self.early_stopping['counter'] += 1
                    if self.early_stopping['counter'] >= self.early_stopping['patience']:
                        print("Early stopping triggered")
                        break
    
    def upload_to_s3(self, bucket: str, key: str):
        """
        Upload the best model to S3.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
        """
        try:
            s3_client = boto3.client('s3')
            s3_client.upload_file(self.best_model_path, bucket, key)
            logger.info(f'Successfully uploaded model to s3://{bucket}/{key}')
        except ClientError as e:
            logger.error(f'Failed to upload model to S3: {e}')

def main():
    # Configuration
    config = {
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'num_epochs': 100,
        'log_interval': 10,
        'model_dir': 'models',
        'upload_to_s3': True,
        's3_bucket': 'your-bucket-name'
    }
    
    # Create model
    model_config = StormPredictorConfig()
    model = StormPredictor(
        image_channels=model_config.image_channels,
        sequence_length=model_config.sequence_length,
        sequence_features=model_config.sequence_features,
        hidden_dim=model_config.hidden_dim,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize trainer
    trainer = StormModelTrainer(model, device, config)
    
    # TODO: Load and prepare data
    # This would typically involve:
    # 1. Loading NOAA data
    # 2. Processing satellite imagery
    # 3. Creating train/val datasets and dataloaders
    
    # Train model
    # trainer.train(train_loader, val_loader, config['num_epochs'])

if __name__ == '__main__':
    main() 
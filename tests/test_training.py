import pytest
import torch
import torch.nn as nn
import pandas as pd
import os
from src.training.train import StormModelTrainer
from src.models.storm_predictor import StormPredictor
from src.data.data_processor import StormDataProcessor, create_data_loaders

@pytest.fixture
def model():
    """Create a sample model for testing."""
    config = {
        'image_channels': 3,
        'sequence_length': 24,
        'sequence_features': 8,
        'hidden_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'dropout': 0.1
    }
    return StormPredictor(**config)

@pytest.fixture
def trainer(model):
    """Create a trainer instance for testing."""
    device = torch.device('cpu')
    config = {
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'model_dir': 'test_models',
        'lr_scheduler': {
            'type': 'ReduceLROnPlateau',
            'factor': 0.1,
            'patience': 5
        },
        'early_stopping': {
            'patience': 10,
            'min_delta': 0.001
        }
    }
    return StormModelTrainer(model, device, config)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample weather data
    weather_data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'temperature': torch.randn(100),
        'humidity': torch.rand(100),
        'pressure': torch.randn(100),
        'wind_speed': torch.rand(100),
        'wind_direction': torch.rand(100),
        'precipitation': torch.rand(100),
        'cloud_cover': torch.rand(100),
        'visibility': torch.rand(100),
        'storm_event': torch.randint(0, 2, (100,))
    }
    weather_df = pd.DataFrame(weather_data)
    
    # Create sample satellite data
    satellite_images = torch.randn(100, 3, 64, 64)
    
    # Process data
    processor = StormDataProcessor()
    processed_weather = processor.process_noaa_data(weather_df)
    processed_satellite = processor.process_satellite_data(str(satellite_images))
    
    # Create sequences
    sequence_length = 24
    satellite_images, weather_sequences, labels = processor.create_sequences(
        processed_weather,
        processed_satellite,
        sequence_length
    )
    
    return satellite_images, weather_sequences, labels

def test_training_step(trainer, sample_data):
    """Test a single training step."""
    satellite_images, weather_sequences, labels = sample_data
    loss = trainer._train_step(satellite_images, weather_sequences, labels)
    assert isinstance(loss, float)
    assert loss >= 0

def test_validation_step(trainer, sample_data):
    """Test a single validation step."""
    satellite_images, weather_sequences, labels = sample_data
    loss, metrics = trainer._validate_step(satellite_images, weather_sequences, labels)
    assert isinstance(loss, float)
    assert loss >= 0
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'auc' in metrics

def test_training_epoch(trainer, sample_data):
    """Test training for one epoch."""
    satellite_images, weather_sequences, labels = sample_data
    batch_size = 32
    train_loader, val_loader, _ = create_data_loaders(
        satellite_images,
        weather_sequences,
        labels,
        batch_size
    )
    loss = trainer.train_epoch(train_loader)
    assert isinstance(loss, float)
    assert loss >= 0

def test_validation(trainer, sample_data):
    """Test validation."""
    satellite_images, weather_sequences, labels = sample_data
    batch_size = 32
    train_loader, val_loader, _ = create_data_loaders(
        satellite_images,
        weather_sequences,
        labels,
        batch_size
    )
    loss, metrics = trainer.validate(val_loader)
    assert isinstance(loss, float)
    assert loss >= 0
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'auc' in metrics

def test_training(trainer, sample_data):
    """Test full training process."""
    satellite_images, weather_sequences, labels = sample_data
    batch_size = 32
    train_loader, val_loader, _ = create_data_loaders(
        satellite_images,
        weather_sequences,
        labels,
        batch_size
    )
    num_epochs = 2
    trainer.train(train_loader, val_loader, num_epochs)
    
    # Check if model was saved
    assert os.path.exists(trainer.best_model_path)

def test_learning_rate_scheduling(trainer, sample_data):
    """Test learning rate scheduling."""
    satellite_images, weather_sequences, labels = sample_data
    batch_size = 32
    train_loader, val_loader, _ = create_data_loaders(
        satellite_images,
        weather_sequences,
        labels,
        batch_size
    )
    
    # Get initial learning rate
    initial_lr = trainer.optimizer.param_groups[0]['lr']
    
    # Train for a few epochs with high validation loss
    for _ in range(6):
        trainer.train_epoch(train_loader)
        trainer.validate(val_loader)
    
    # Check if learning rate was reduced
    current_lr = trainer.optimizer.param_groups[0]['lr']
    assert current_lr < initial_lr

def test_early_stopping(trainer, sample_data):
    """Test early stopping."""
    satellite_images, weather_sequences, labels = sample_data
    batch_size = 32
    train_loader, val_loader, _ = create_data_loaders(
        satellite_images,
        weather_sequences,
        labels,
        batch_size
    )
    
    # Train for more epochs than patience
    num_epochs = trainer.early_stopping['patience'] + 5
    trainer.train(train_loader, val_loader, num_epochs)
    
    # Check if training stopped early
    assert trainer.early_stopping['counter'] >= trainer.early_stopping['patience'] 
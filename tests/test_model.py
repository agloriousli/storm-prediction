import torch
import pytest
from src.models.storm_predictor import StormPredictor, StormPredictorConfig

def test_model_initialization():
    """Test model initialization with default parameters."""
    config = StormPredictorConfig()
    model = StormPredictor(
        image_channels=config.image_channels,
        sequence_length=config.sequence_length,
        sequence_features=config.sequence_features,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    
    assert isinstance(model, torch.nn.Module)
    assert model.image_channels == 3
    assert model.sequence_length == 24
    assert model.sequence_features == 10
    assert model.hidden_dim == 128
    assert model.num_heads == 4
    assert model.num_layers == 2
    assert model.dropout == 0.1

def test_model_forward_pass():
    """Test model forward pass with sample inputs."""
    config = StormPredictorConfig()
    model = StormPredictor(
        image_channels=config.image_channels,
        sequence_length=config.sequence_length,
        sequence_features=config.sequence_features,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    
    # Create sample inputs
    batch_size = 4
    image = torch.randn(batch_size, config.image_channels, 64, 64)
    sequence = torch.randn(batch_size, config.sequence_length, config.sequence_features)
    
    # Forward pass
    output = model(image, sequence)
    
    # Check output shape and values
    assert output.shape == (batch_size, 1)
    assert torch.all((output >= 0) & (output <= 1))  # Check if outputs are probabilities

def test_model_with_different_batch_sizes():
    """Test model with different batch sizes."""
    config = StormPredictorConfig()
    model = StormPredictor(
        image_channels=config.image_channels,
        sequence_length=config.sequence_length,
        sequence_features=config.sequence_features,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    
    batch_sizes = [1, 4, 8, 16]
    for batch_size in batch_sizes:
        image = torch.randn(batch_size, config.image_channels, 64, 64)
        sequence = torch.randn(batch_size, config.sequence_length, config.sequence_features)
        
        output = model(image, sequence)
        assert output.shape == (batch_size, 1)

def test_model_with_different_image_sizes():
    """Test model with different image sizes."""
    config = StormPredictorConfig()
    model = StormPredictor(
        image_channels=config.image_channels,
        sequence_length=config.sequence_length,
        sequence_features=config.sequence_features,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    
    image_sizes = [(32, 32), (64, 64), (128, 128)]
    batch_size = 4
    
    for height, width in image_sizes:
        image = torch.randn(batch_size, config.image_channels, height, width)
        sequence = torch.randn(batch_size, config.sequence_length, config.sequence_features)
        
        output = model(image, sequence)
        assert output.shape == (batch_size, 1)

def test_model_gradient_flow():
    """Test if gradients flow through the model."""
    config = StormPredictorConfig()
    model = StormPredictor(
        image_channels=config.image_channels,
        sequence_length=config.sequence_length,
        sequence_features=config.sequence_features,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    
    batch_size = 4
    image = torch.randn(batch_size, config.image_channels, 64, 64, requires_grad=True)
    sequence = torch.randn(batch_size, config.sequence_length, config.sequence_features, requires_grad=True)
    
    output = model(image, sequence)
    loss = output.mean()
    loss.backward()
    
    assert image.grad is not None
    assert sequence.grad is not None
    assert not torch.isnan(image.grad).any()
    assert not torch.isnan(sequence.grad).any() 
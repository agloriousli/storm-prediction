import pytest
import torch
import numpy as np
from src.models.storm_predictor import StormPredictor, StormPredictorConfig
from src.inference import StormPredictorInference

@pytest.fixture
def model():
    """Create a model for testing."""
    config = StormPredictorConfig()
    return StormPredictor(
        image_channels=config.image_channels,
        sequence_length=config.sequence_length,
        sequence_features=config.sequence_features,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout
    )

@pytest.fixture
def sample_input():
    """Create sample input data for testing."""
    return {
        'satellite_image': torch.randn(3, 64, 64),
        'weather_data': {
            'temperature': 25.5,
            'humidity': 0.7,
            'pressure': 1013.2,
            'wind_speed': 15.0,
            'wind_direction': 180.0,
            'precipitation': 0.0,
            'cloud_cover': 0.3,
            'visibility': 10.0
        }
    }

def test_inference_initialization(model):
    """Test initialization of StormPredictorInference."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference = StormPredictorInference(model, device)
    
    assert inference is not None
    assert inference.model == model
    assert inference.device == device

def test_preprocess_input(sample_input):
    """Test input preprocessing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StormPredictor(
        image_channels=3,
        sequence_length=24,
        sequence_features=10,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    inference = StormPredictorInference(model, device)
    
    # Preprocess input
    image, sequence = inference._preprocess_input(sample_input)
    
    assert isinstance(image, torch.Tensor)
    assert isinstance(sequence, torch.Tensor)
    assert image.shape == (1, 3, 64, 64)
    assert sequence.shape == (1, 24, 10)

def test_predict(sample_input):
    """Test model prediction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StormPredictor(
        image_channels=3,
        sequence_length=24,
        sequence_features=10,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    inference = StormPredictorInference(model, device)
    
    # Make prediction
    prediction = inference.predict(sample_input)
    
    assert isinstance(prediction, float)
    assert 0 <= prediction <= 1

def test_batch_predict(sample_input):
    """Test batch prediction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StormPredictor(
        image_channels=3,
        sequence_length=24,
        sequence_features=10,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    inference = StormPredictorInference(model, device)
    
    # Create batch of inputs
    batch_inputs = [sample_input] * 4
    
    # Make batch prediction
    predictions = inference.batch_predict(batch_inputs)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (4,)
    assert np.all((predictions >= 0) & (predictions <= 1))

def test_prediction_threshold(sample_input):
    """Test prediction thresholding."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StormPredictor(
        image_channels=3,
        sequence_length=24,
        sequence_features=10,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    inference = StormPredictorInference(model, device)
    
    # Make prediction with threshold
    prediction = inference.predict(sample_input, threshold=0.5)
    
    assert isinstance(prediction, bool)

def test_error_handling(sample_input):
    """Test error handling in inference."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StormPredictor(
        image_channels=3,
        sequence_length=24,
        sequence_features=10,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    inference = StormPredictorInference(model, device)
    
    # Test with invalid input
    invalid_input = {'invalid_key': 'invalid_value'}
    
    with pytest.raises(ValueError):
        inference.predict(invalid_input)

def test_model_loading():
    """Test model loading functionality."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and save a model
    model = StormPredictor(
        image_channels=3,
        sequence_length=24,
        sequence_features=10,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    torch.save(model.state_dict(), 'test_model.pt')
    
    # Load model
    loaded_model = StormPredictor(
        image_channels=3,
        sequence_length=24,
        sequence_features=10,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    loaded_model.load_state_dict(torch.load('test_model.pt'))
    
    # Create inference with loaded model
    inference = StormPredictorInference(loaded_model, device)
    
    # Clean up
    import os
    os.remove('test_model.pt')
    
    assert inference is not None
    assert inference.model == loaded_model

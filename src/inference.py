import os
import json
import torch
import numpy as np
from typing import Dict, Any, List, Union, Tuple, Optional
import logging

from models.storm_predictor import StormPredictor, StormPredictorConfig

logger = logging.getLogger(__name__)

class StormPredictorInference:
    """
    Class for making predictions using the StormPredictor model.
    Handles preprocessing, prediction, and post-processing of inputs.
    """
    def __init__(self, model: StormPredictor, device: torch.device):
        """
        Initialize the inference class.
        
        Args:
            model: Trained StormPredictor model
            device: Device to run inference on (CPU/GPU)
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def _preprocess_input(self, input_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess input data for model inference.
        
        Args:
            input_data: Dictionary containing:
                - satellite_image: Tensor of shape (C, H, W)
                - weather_data: Dictionary of weather features
                
        Returns:
            Tuple of (image_tensor, sequence_tensor)
        """
        # Process satellite image
        image = input_data['satellite_image']
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(self.device)
        
        # Process weather data
        weather_data = input_data['weather_data']
        features = [
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['pressure'],
            weather_data['wind_speed'],
            weather_data['wind_direction'],
            weather_data['precipitation'],
            weather_data['cloud_cover'],
            weather_data['visibility']
        ]
        
        # Create sequence tensor
        sequence = torch.tensor(features, dtype=torch.float32)
        sequence = sequence.unsqueeze(0)  # Add batch dimension
        sequence = sequence.unsqueeze(0)  # Add sequence dimension
        sequence = sequence.repeat(1, self.model.sequence_length, 1)  # Repeat for sequence length
        sequence = sequence.to(self.device)
        
        return image, sequence
    
    def predict(self, input_data: Dict[str, Any], threshold: Optional[float] = None) -> Union[float, bool]:
        """
        Make a prediction for a single input.
        
        Args:
            input_data: Input data dictionary
            threshold: Optional threshold for binary classification
            
        Returns:
            Prediction probability or binary classification
        """
        with torch.no_grad():
            image, sequence = self._preprocess_input(input_data)
            prediction = self.model(image, sequence)
            probability = prediction.item()
            
            if threshold is not None:
                return probability >= threshold
            return probability
    
    def batch_predict(self, input_data_list: List[Dict[str, Any]]) -> np.ndarray:
        """
        Make predictions for a batch of inputs.
        
        Args:
            input_data_list: List of input data dictionaries
            
        Returns:
            Array of prediction probabilities
        """
        predictions = []
        for input_data in input_data_list:
            prediction = self.predict(input_data)
            predictions.append(prediction)
        return np.array(predictions)

def model_fn(model_dir: str) -> StormPredictor:
    """
    Load the model for inference.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        StormPredictor: Loaded model
    """
    # Load model configuration
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = StormPredictorConfig(**config_dict)
    
    # Create model
    model = StormPredictor(
        image_channels=config.image_channels,
        sequence_length=config.sequence_length,
        sequence_features=config.sequence_features,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    
    # Load model weights
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model

def input_fn(request_body: bytes, request_content_type: str) -> Dict[str, torch.Tensor]:
    """
    Deserialize and prepare the prediction input.
    
    Args:
        request_body: The request body
        request_content_type: The request content type
        
    Returns:
        Dict[str, torch.Tensor]: Input tensors
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Convert input data to tensors
        image = torch.tensor(input_data['image'], dtype=torch.float32)
        sequence = torch.tensor(input_data['sequence'], dtype=torch.float32)
        
        return {
            'image': image,
            'sequence': sequence
        }
    else:
        raise ValueError(f'Unsupported content type: {request_content_type}')

def predict_fn(input_data: Dict[str, torch.Tensor], model: StormPredictor) -> np.ndarray:
    """
    Apply model to the input data.
    
    Args:
        input_data: Input tensors
        model: Loaded model
        
    Returns:
        np.ndarray: Model predictions
    """
    with torch.no_grad():
        predictions = model(
            input_data['image'],
            input_data['sequence']
        )
    
    return predictions.numpy()

def output_fn(prediction: np.ndarray, response_content_type: str) -> bytes:
    """
    Serialize and prepare the prediction output.
    
    Args:
        prediction: Model predictions
        response_content_type: The response content type
        
    Returns:
        bytes: Response body
    """
    if response_content_type == 'application/json':
        return json.dumps({
            'prediction': prediction.tolist(),
            'probability': float(prediction[0])
        }).encode('utf-8')
    else:
        raise ValueError(f'Unsupported content type: {response_content_type}')

def preprocess_image(image_data: np.ndarray) -> torch.Tensor:
    """
    Preprocess satellite imagery for model input.
    
    Args:
        image_data: Raw image data
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Normalize pixel values
    image = image_data.astype(np.float32) / 255.0
    
    # Add batch dimension
    image = torch.from_numpy(image).unsqueeze(0)
    
    return image

def preprocess_sequence(sequence_data: np.ndarray) -> torch.Tensor:
    """
    Preprocess temporal sequence data for model input.
    
    Args:
        sequence_data: Raw sequence data
        
    Returns:
        torch.Tensor: Preprocessed sequence tensor
    """
    # Normalize sequence data
    sequence = (sequence_data - sequence_data.mean()) / sequence_data.std()
    
    # Add batch dimension
    sequence = torch.from_numpy(sequence).unsqueeze(0)
    
    return sequence 
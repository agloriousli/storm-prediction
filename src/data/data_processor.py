import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import StandardScaler
import pickle

class StormDataProcessor:
    """Data processor for storm prediction."""
    
    def __init__(self, spark_session=None):
        self.spark = spark_session
        self.scalers = {}
    
    def process_noaa_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process NOAA weather data.
        
        Args:
            data: DataFrame containing NOAA weather data
            
        Returns:
            Processed DataFrame with normalized features
        """
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Normalize numerical features
        numerical_features = [
            'temperature', 'humidity', 'pressure',
            'wind_speed', 'wind_direction', 'precipitation',
            'cloud_cover', 'visibility'
        ]
        
        for feature in numerical_features:
            if feature in data.columns:
                scaler = StandardScaler()
                data[feature] = scaler.fit_transform(data[[feature]])
                self.scalers[feature] = scaler
        
        return data
    
    def process_satellite_data(self, data_path: str) -> Dict[str, torch.Tensor]:
        """
        Process satellite imagery data.
        
        Args:
            data_path: Path to satellite data directory
            
        Returns:
            Dictionary mapping timestamps to satellite image tensors
        """
        satellite_data = {}
        
        # Load and process each satellite image
        for filename in os.listdir(data_path):
            if filename.endswith('.npy'):
                # Extract timestamp from filename
                timestamp = pd.to_datetime(filename.split('_')[0])
                
                # Load image
                image_path = os.path.join(data_path, filename)
                image = np.load(image_path)
                
                # Convert to tensor and normalize
                image_tensor = torch.from_numpy(image).float()
                image_tensor = (image_tensor - image_tensor.mean()) / image_tensor.std()
                
                satellite_data[timestamp] = image_tensor
        
        return satellite_data
    
    def create_sequences(
        self,
        weather_data: pd.DataFrame,
        satellite_data: Dict[str, torch.Tensor],
        sequence_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create sequences for model training.
        
        Args:
            weather_data: Processed weather data
            satellite_data: Processed satellite data
            sequence_length: Length of temporal sequences
            
        Returns:
            Tuple of (satellite images, weather sequences, labels)
        """
        # Sort data by timestamp
        weather_data = weather_data.sort_values('timestamp')
        
        # Initialize lists for sequences
        satellite_images = []
        weather_sequences = []
        labels = []
        
        # Create sequences
        for i in range(len(weather_data) - sequence_length):
            # Get sequence timestamps
            sequence_timestamps = weather_data['timestamp'].iloc[i:i+sequence_length]
            
            # Get satellite images for sequence
            sequence_images = []
            for timestamp in sequence_timestamps:
                if timestamp in satellite_data:
                    sequence_images.append(satellite_data[timestamp])
            
            if len(sequence_images) == sequence_length:
                # Stack images
                satellite_images.append(torch.stack(sequence_images))
                
                # Get weather data for sequence
                sequence_data = weather_data.iloc[i:i+sequence_length]
                weather_features = sequence_data[[
                    'temperature', 'humidity', 'pressure',
                    'wind_speed', 'wind_direction', 'precipitation',
                    'cloud_cover', 'visibility'
                ]].values
                weather_sequences.append(torch.from_numpy(weather_features).float())
                
                # Get label (storm event in next timestep)
                label = weather_data['storm_event'].iloc[i+sequence_length]
                labels.append(torch.tensor(label, dtype=torch.float32))
        
        # Convert to tensors
        satellite_images = torch.stack(satellite_images)
        weather_sequences = torch.stack(weather_sequences)
        labels = torch.stack(labels)
        
        return satellite_images, weather_sequences, labels
    
    def save_scalers(self, path: str):
        """Save feature scalers to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.scalers, f)
    
    def load_scalers(self, path: str):
        """Load feature scalers from disk."""
        with open(path, 'rb') as f:
            self.scalers = pickle.load(f)

class StormDataset(Dataset):
    """Dataset class for storm prediction."""
    
    def __init__(
        self,
        satellite_images: torch.Tensor,
        weather_sequences: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Initialize dataset.
        
        Args:
            satellite_images: Tensor of satellite images (N, L, C, H, W)
            weather_sequences: Tensor of weather sequences (N, L, F)
            labels: Tensor of labels (N,)
        """
        self.satellite_images = satellite_images
        self.weather_sequences = weather_sequences
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (satellite images, weather sequence, label)
        """
        return (
            self.satellite_images[idx],
            self.weather_sequences[idx],
            self.labels[idx]
        )

def create_data_loaders(
    satellite_images: torch.Tensor,
    weather_sequences: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        satellite_images: Tensor of satellite images
        weather_sequences: Tensor of weather sequences
        labels: Tensor of labels
        batch_size: Batch size for data loaders
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Calculate split indices
    n_samples = len(labels)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # Create indices
    indices = torch.randperm(n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Create datasets
    train_dataset = StormDataset(
        satellite_images[train_indices],
        weather_sequences[train_indices],
        labels[train_indices]
    )
    val_dataset = StormDataset(
        satellite_images[val_indices],
        weather_sequences[val_indices],
        labels[val_indices]
    )
    test_dataset = StormDataset(
        satellite_images[test_indices],
        weather_sequences[test_indices],
        labels[test_indices]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 
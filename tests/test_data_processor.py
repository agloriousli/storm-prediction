import pytest
import torch
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from src.data.data_processor import StormDataProcessor, StormDataset, create_data_loaders

@pytest.fixture
def spark_session():
    """Create a Spark session for testing."""
    spark = SparkSession.builder \
        .appName("StormPredictorTest") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def sample_weather_data():
    """Create sample weather data for testing."""
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'temperature': np.random.normal(20, 5, 100),
        'humidity': np.random.uniform(0, 100, 100),
        'pressure': np.random.normal(1013, 10, 100),
        'wind_speed': np.random.uniform(0, 30, 100),
        'wind_direction': np.random.uniform(0, 360, 100),
        'precipitation': np.random.uniform(0, 50, 100),
        'cloud_cover': np.random.uniform(0, 100, 100),
        'visibility': np.random.uniform(0, 10, 100),
        'storm_event': np.random.choice([0, 1], 100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_satellite_data(tmp_path):
    """Create sample satellite data for testing."""
    satellite_data = {}
    for i in range(100):
        timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i)
        image = np.random.rand(3, 64, 64)
        image_path = tmp_path / f"{timestamp.strftime('%Y%m%d_%H%M%S')}.npy"
        np.save(image_path, image)
        satellite_data[timestamp] = torch.from_numpy(image).float()
    return satellite_data

def test_data_processor_initialization(spark_session):
    """Test initialization of StormDataProcessor."""
    processor = StormDataProcessor(spark_session)
    assert processor is not None

def test_data_processing(sample_weather_data):
    """Test data processing functionality."""
    processor = StormDataProcessor()
    processed_data = processor.process_noaa_data(sample_weather_data)
    
    # Check that all numerical features are normalized
    numerical_features = [
        'temperature', 'humidity', 'pressure',
        'wind_speed', 'wind_direction', 'precipitation',
        'cloud_cover', 'visibility'
    ]
    
    for feature in numerical_features:
        assert feature in processed_data.columns
        assert abs(processed_data[feature].mean()) < 1e-10
        assert abs(processed_data[feature].std() - 1.0) < 1e-10

def test_satellite_processing(sample_satellite_data):
    """Test satellite data processing."""
    processor = StormDataProcessor()
    processed_data = processor.process_satellite_data(str(sample_satellite_data))
    
    # Check that all images are tensors with correct shape
    for timestamp, image in processed_data.items():
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 64, 64)
        assert abs(image.mean()) < 1e-10
        assert abs(image.std() - 1.0) < 1e-10

def test_sequence_creation(sample_weather_data, sample_satellite_data):
    """Test sequence creation."""
    processor = StormDataProcessor()
    processed_weather = processor.process_noaa_data(sample_weather_data)
    processed_satellite = processor.process_satellite_data(str(sample_satellite_data))
    
    sequence_length = 24
    satellite_images, weather_sequences, labels = processor.create_sequences(
        processed_weather,
        processed_satellite,
        sequence_length
    )
    
    # Check shapes
    assert satellite_images.shape[1] == sequence_length
    assert weather_sequences.shape[1] == sequence_length
    assert len(labels) == len(satellite_images)
    
    # Check data types
    assert isinstance(satellite_images, torch.Tensor)
    assert isinstance(weather_sequences, torch.Tensor)
    assert isinstance(labels, torch.Tensor)

def test_dataset_creation(sample_weather_data, sample_satellite_data):
    """Test dataset creation."""
    processor = StormDataProcessor()
    processed_weather = processor.process_noaa_data(sample_weather_data)
    processed_satellite = processor.process_satellite_data(str(sample_satellite_data))
    
    sequence_length = 24
    satellite_images, weather_sequences, labels = processor.create_sequences(
        processed_weather,
        processed_satellite,
        sequence_length
    )
    
    dataset = StormDataset(satellite_images, weather_sequences, labels)
    
    # Check dataset length
    assert len(dataset) == len(labels)
    
    # Check sample shape
    sample_images, sample_sequences, sample_label = dataset[0]
    assert sample_images.shape == (sequence_length, 3, 64, 64)
    assert sample_sequences.shape == (sequence_length, 8)
    assert isinstance(sample_label, torch.Tensor)

def test_data_loader_creation(sample_weather_data, sample_satellite_data):
    """Test data loader creation."""
    processor = StormDataProcessor()
    processed_weather = processor.process_noaa_data(sample_weather_data)
    processed_satellite = processor.process_satellite_data(str(sample_satellite_data))
    
    sequence_length = 24
    satellite_images, weather_sequences, labels = processor.create_sequences(
        processed_weather,
        processed_satellite,
        sequence_length
    )
    
    batch_size = 32
    train_loader, val_loader, test_loader = create_data_loaders(
        satellite_images,
        weather_sequences,
        labels,
        batch_size
    )
    
    # Check batch shapes
    for loader in [train_loader, val_loader, test_loader]:
        batch = next(iter(loader))
        images, sequences, batch_labels = batch
        assert images.shape[0] == batch_size
        assert sequences.shape[0] == batch_size
        assert batch_labels.shape[0] == batch_size 
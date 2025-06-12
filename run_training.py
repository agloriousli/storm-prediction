import os
import yaml
import logging
from pyspark.sql import SparkSession
import torch
from src.models.storm_predictor import StormPredictor, StormPredictorConfig
from src.data.data_processor import StormDataProcessor, StormDataset, create_data_loader
from src.training.train import StormModelTrainer
from src.utils.sagemaker_utils import (
    create_sagemaker_session,
    create_pytorch_estimator,
    create_hyperparameter_tuner
)

def setup_logging(config):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        filename=config['logging']['file']
    )

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("StormPredictor") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    try:
        # Process data
        data_processor = StormDataProcessor(spark)
        storm_data = data_processor.process_noaa_data(config['data']['noaa_data_path'])
        
        # Create datasets
        train_size = int(len(storm_data) * config['data']['train_test_split'])
        train_data = storm_data[:train_size]
        val_data = storm_data[train_size:]
        
        # TODO: Load and process satellite imagery
        # This would typically involve:
        # 1. Loading satellite images
        # 2. Preprocessing images
        # 3. Creating image tensors
        satellite_data = {}  # Placeholder for satellite data
        
        # Create datasets
        train_dataset = StormDataset(
            train_data,
            satellite_data,
            sequence_length=config['model']['sequence_length']
        )
        val_dataset = StormDataset(
            val_data,
            satellite_data,
            sequence_length=config['model']['sequence_length']
        )
        
        # Create data loaders
        train_loader = create_data_loader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True
        )
        val_loader = create_data_loader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False
        )
        
        # Create model
        model_config = StormPredictorConfig(**config['model'])
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
        trainer = StormModelTrainer(model, device, config['training'])
        
        # Train model
        trainer.train(
            train_loader,
            val_loader,
            config['training']['num_epochs']
        )
        
        # Upload model to S3 if specified
        if config['aws'].get('upload_to_s3', False):
            trainer.upload_to_s3(
                config['aws']['s3_bucket'],
                f"models/storm_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise
    
    finally:
        # Stop Spark session
        spark.stop()

if __name__ == '__main__':
    main() 
import os
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def create_sagemaker_session(role: str, region: str = 'us-east-1') -> sagemaker.Session:
    """
    Create a SageMaker session.
    
    Args:
        role: IAM role ARN for SageMaker
        region: AWS region
        
    Returns:
        sagemaker.Session: SageMaker session
    """
    return sagemaker.Session(
        default_bucket=f'storm-predictor-{region}',
        default_bucket_prefix='models'
    )

def create_pytorch_estimator(
    session: sagemaker.Session,
    role: str,
    instance_type: str = 'ml.p3.2xlarge',
    instance_count: int = 1,
    hyperparameters: Dict[str, Any] = None
) -> PyTorch:
    """
    Create a PyTorch estimator for SageMaker training.
    
    Args:
        session: SageMaker session
        role: IAM role ARN
        instance_type: SageMaker instance type
        instance_count: Number of instances
        hyperparameters: Model hyperparameters
        
    Returns:
        PyTorch: SageMaker PyTorch estimator
    """
    return PyTorch(
        entry_point='train.py',
        source_dir='src',
        role=role,
        framework_version='1.12.1',
        py_version='py38',
        instance_type=instance_type,
        instance_count=instance_count,
        hyperparameters=hyperparameters or {},
        output_path=f's3://{session.default_bucket()}/models',
        code_location=f's3://{session.default_bucket()}/code',
        base_job_name='storm-predictor'
    )

def create_hyperparameter_tuner(
    estimator: PyTorch,
    objective_metric_name: str = 'validation:loss',
    max_jobs: int = 20,
    max_parallel_jobs: int = 4
) -> HyperparameterTuner:
    """
    Create a hyperparameter tuner for the model.
    
    Args:
        estimator: PyTorch estimator
        objective_metric_name: Name of the objective metric
        max_jobs: Maximum number of training jobs
        max_parallel_jobs: Maximum number of parallel training jobs
        
    Returns:
        HyperparameterTuner: SageMaker hyperparameter tuner
    """
    hyperparameter_ranges = {
        'learning_rate': ContinuousParameter(1e-5, 1e-2),
        'weight_decay': ContinuousParameter(1e-6, 1e-3),
        'hidden_dim': IntegerParameter(64, 256),
        'num_heads': IntegerParameter(2, 8),
        'num_layers': IntegerParameter(1, 4),
        'dropout': ContinuousParameter(0.1, 0.5)
    }
    
    return HyperparameterTuner(
        estimator=estimator,
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy='Bayesian'
    )

def deploy_model(
    model_data: str,
    role: str,
    instance_type: str = 'ml.m5.xlarge',
    instance_count: int = 1
) -> sagemaker.predictor.Predictor:
    """
    Deploy the trained model to a SageMaker endpoint.
    
    Args:
        model_data: S3 path to the model artifacts
        role: IAM role ARN
        instance_type: SageMaker instance type
        instance_count: Number of instances
        
    Returns:
        Predictor: SageMaker predictor
    """
    model = sagemaker.pytorch.PyTorchModel(
        model_data=model_data,
        role=role,
        framework_version='1.12.1',
        py_version='py38',
        entry_point='inference.py',
        source_dir='src'
    )
    
    predictor = model.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type
    )
    
    return predictor

def create_training_job(
    session: sagemaker.Session,
    role: str,
    input_data: Dict[str, str],
    hyperparameters: Dict[str, Any] = None
) -> str:
    """
    Create and start a SageMaker training job.
    
    Args:
        session: SageMaker session
        role: IAM role ARN
        input_data: Dictionary of input data channels
        hyperparameters: Model hyperparameters
        
    Returns:
        str: Training job name
    """
    estimator = create_pytorch_estimator(
        session=session,
        role=role,
        hyperparameters=hyperparameters
    )
    
    estimator.fit(inputs=input_data)
    
    return estimator.latest_training_job.name

def create_endpoint_config(
    model_name: str,
    instance_type: str = 'ml.m5.xlarge',
    initial_instance_count: int = 1
) -> str:
    """
    Create a SageMaker endpoint configuration.
    
    Args:
        model_name: Name of the model
        instance_type: SageMaker instance type
        initial_instance_count: Number of instances
        
    Returns:
        str: Endpoint configuration name
    """
    config_name = f'{model_name}-config'
    
    sagemaker.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': initial_instance_count,
            'InstanceType': instance_type
        }]
    )
    
    return config_name 
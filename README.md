# Storm Weather Prediction Model

A deep learning model that predicts the likelihood of storm events using historical weather data, satellite imagery, and atmospheric variables. This project leverages PyTorch, PySpark, and AWS SageMaker for scalable and efficient storm prediction.

## 🚀 Features

- Deep learning model combining CNN and Transformer architectures
- Processing of NOAA storm events data and GOES satellite imagery
- Distributed training on AWS SageMaker
- Real-time prediction capabilities
- Comprehensive evaluation metrics
- Scalable data processing pipeline using PySpark

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch
- **Data Processing**: PySpark, Pandas, NumPy
- **Machine Learning**: XGBoost, Scikit-learn
- **Cloud Infrastructure**: AWS (S3, SageMaker, Lambda)
- **Model Evaluation**: ROC-AUC, PRC, F1-Score, Brier Score

## 📋 Prerequisites

- Python 3.8+
- AWS Account with SageMaker access
- PySpark environment
- CUDA-capable GPU (for local development)

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/agloriousli/storm-prediction
cd storm-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up AWS credentials:
```bash
aws configure
```

5. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your AWS credentials and configuration
```

## 📁 Project Structure

```
storm-predictor/
├── src/
│   ├── data/           # Data processing modules
│   ├── models/         # Model architectures
│   ├── training/       # Training scripts
│   └── utils/          # Utility functions
├── data/               # Data storage
├── models/             # Saved model checkpoints
├── notebooks/          # Jupyter notebooks
├── tests/              # Unit tests
└── config/             # Configuration files
```

## 🎯 Usage

1. Data Processing:
```bash
python src/data/process_data.py
```

2. Model Training:
```bash
python src/training/train.py
```

3. Model Evaluation:
```bash
python src/evaluation/evaluate.py
```

## 📊 Model Performance

- Baseline XGBoost AUC: 0.73
- PyTorch Hybrid Model AUC: 0.86
- Inference Latency: ~150ms/request

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NOAA for providing the storm events dataset
- AWS for cloud infrastructure
- PyTorch team for the deep learning framework 

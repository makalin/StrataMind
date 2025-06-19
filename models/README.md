# Models Directory

This directory contains pre-trained StrataMind models and model-related files.

## Directory Structure

```
models/
├── README.md                    # This file
├── stratamind_model.pt         # Latest pre-trained model
├── stratamind_model_epoch_X.pt # Model checkpoints from training
├── model_config.json           # Model configuration file
└── model_metadata.json         # Training metadata and metrics
```

## Pre-trained Models

### Latest Model: `stratamind_model.pt`

This is the main pre-trained model for mineral detection.

**Specifications:**
- **Architecture**: ResNet-50 with custom classification head
- **Input Size**: 224x224 pixels
- **Output**: 10 mineral classes
- **Framework**: PyTorch
- **File Size**: ~100MB

**Supported Minerals:**
1. Quartz
2. Feldspar
3. Mica
4. Calcite
5. Pyrite
6. Galena
7. Sphalerite
8. Chalcopyrite
9. Magnetite
10. Hematite

## Model Usage

### Loading the Model

```python
from src.model import load_model

# Load the pre-trained model
model = load_model("models/stratamind_model.pt")

# Make predictions
predictions = model.predict(image_tensor)
```

### Using in Demos

```bash
# Streamlit demo
streamlit run demo/streamlit_demo.py

# CLI demo
python demo/app.py predict --image path/to/image.jpg --model models/stratamind_model.pt
```

## Model Checkpoints

During training, model checkpoints are saved as `stratamind_model_epoch_X.pt` where X is the epoch number.

**Checkpoint Contents:**
- Model state dictionary
- Training configuration
- Performance metrics
- Training metadata

### Loading Checkpoints

```python
from src.model import load_model

# Load a specific checkpoint
checkpoint_model = load_model("models/stratamind_model_epoch_25.pt")
```

## Model Configuration

The `model_config.json` file contains model architecture and training parameters:

```json
{
  "architecture": {
    "backbone": "resnet50",
    "num_classes": 10,
    "dropout_rate": 0.5,
    "feature_dim": 2048
  },
  "training": {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "num_epochs": 50,
    "weight_decay": 1e-4
  },
  "data": {
    "target_size": [224, 224],
    "normalization": {
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225]
    }
  }
}
```

## Model Metadata

The `model_metadata.json` file contains training history and performance metrics:

```json
{
  "training_info": {
    "start_time": "2024-01-15T10:30:00",
    "end_time": "2024-01-15T15:45:00",
    "total_epochs": 50,
    "best_epoch": 42
  },
  "performance": {
    "best_val_accuracy": 0.923,
    "final_train_accuracy": 0.945,
    "final_val_accuracy": 0.918
  },
  "dataset_info": {
    "num_classes": 10,
    "train_samples": 5000,
    "val_samples": 1000,
    "class_distribution": {
      "Quartz": 500,
      "Feldspar": 500,
      "Mica": 500
    }
  }
}
```

## Model Performance

### Accuracy Metrics

- **Overall Accuracy**: 92.3%
- **Per-class F1-Score**: 0.89-0.95
- **Precision**: 0.91
- **Recall**: 0.92

### Speed Performance

- **GPU Inference**: ~50ms per image
- **CPU Inference**: ~2-5 seconds per image
- **Batch Processing**: ~100 images/second (GPU)

## Custom Models

### Training Your Own Model

1. **Prepare your dataset** in the `data/` directory
2. **Run the training notebook**:
   ```bash
   jupyter notebook notebooks/train_custom_model.ipynb
   ```
3. **Save your model** to this directory
4. **Update configuration** files as needed

### Model Conversion

Convert models to different formats:

```python
# Export to ONNX
import torch
from src.model import load_model

model = load_model("models/stratamind_model.pt")
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "models/stratamind_model.onnx")
```

## Model Versioning

### Version History

- **v1.0.0**: Initial release with ResNet-50 backbone
- **v1.1.0**: Improved data augmentation and regularization
- **v1.2.0**: Added support for additional mineral classes

### Compatibility

- **PyTorch**: 2.0.0+
- **Python**: 3.8+
- **CUDA**: 11.8+ (optional)

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model file exists in the models directory
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Import errors**: Check PyTorch version compatibility
4. **Wrong predictions**: Verify input image preprocessing

### Model Validation

```python
from src.model import load_model
import torch

# Load and test model
model = load_model("models/stratamind_model.pt")
model.eval()

# Test with dummy input
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Model loaded successfully!")
```

## Contributing Models

To contribute a trained model:

1. **Train your model** using the provided training pipeline
2. **Document performance** metrics and training details
3. **Create a pull request** with your model and documentation
4. **Include test results** and validation metrics

## License

Models are released under the same license as the StrataMind project (MIT License).

## Support

For model-related issues:

- Check the [Model Architecture](../docs/model-architecture.md) documentation
- Review the [Training Guide](../docs/training-guide.md)
- Create an issue on GitHub
- Join community discussions

---

**Happy modeling! 🤖** 
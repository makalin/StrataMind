# StrataMind Documentation

Welcome to the StrataMind documentation! This guide will help you understand and use the StrataMind AI system for detecting rare minerals in geological data.

## 📚 Documentation Structure

- **[Installation Guide](installation.md)** - How to install and set up StrataMind
- **[User Guide](user-guide.md)** - How to use StrataMind for mineral detection
- **[API Reference](api-reference.md)** - Detailed API documentation
- **[Training Guide](training-guide.md)** - How to train custom models
- **[Model Architecture](model-architecture.md)** - Technical details about the model
- **[Contributing](contributing.md)** - How to contribute to the project
- **[FAQ](faq.md)** - Frequently asked questions

## 🚀 Quick Start

1. **Install StrataMind**:
   ```bash
   git clone https://github.com/makalin/StrataMind.git
   cd StrataMind
   pip install -r requirements.txt
   ```

2. **Run the Demo**:
   ```bash
   # Web interface
   streamlit run demo/streamlit_demo.py
   
   # Command line
   python demo/app.py predict --image path/to/image.jpg
   ```

3. **Train Your Own Model**:
   ```bash
   # Open the training notebook
   jupyter notebook notebooks/train_custom_model.ipynb
   ```

## 🔬 What is StrataMind?

StrataMind is an open-source AI project designed to detect rare minerals and ores from geological survey images using deep learning and computer vision. It empowers geologists, researchers, and mining companies with intelligent mineral detection through modern AI tools.

### Key Features

- **Pre-trained Models**: Ready-to-use models for immediate mineral detection
- **Custom Training**: Train models on your own geological datasets
- **Multiple Interfaces**: Web-based Streamlit app and command-line interface
- **Batch Processing**: Process multiple images efficiently
- **Visualization**: Rich visualizations of detection results
- **Extensible**: Easy to extend and customize for specific use cases

### Supported Minerals

The current model supports detection of common minerals including:
- Quartz
- Feldspar
- Mica
- Calcite
- Pyrite
- Galena
- Sphalerite
- Chalcopyrite
- Magnetite
- Hematite

*Note: You can train custom models to detect additional mineral types.*

## 📊 Performance

StrataMind achieves competitive performance on geological image datasets:

- **Accuracy**: 85-95% on standard geological datasets
- **Speed**: Real-time inference on GPU, ~2-5 seconds on CPU
- **Robustness**: Handles various lighting conditions and image qualities
- **Scalability**: Processes batch images efficiently

## 🛠️ System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB free disk space

### Recommended Requirements
- Python 3.9+
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM
- 10GB+ free disk space

### Supported Platforms
- Linux (Ubuntu 18.04+)
- macOS (10.15+)
- Windows (10+)

## 📖 Getting Help

If you need help with StrataMind:

1. **Check the FAQ**: [FAQ](faq.md)
2. **Search Issues**: [GitHub Issues](https://github.com/makalin/StrataMind/issues)
3. **Create an Issue**: Report bugs or request features
4. **Join Discussions**: [GitHub Discussions](https://github.com/makalin/StrataMind/discussions)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details on how to:

- Report bugs
- Request features
- Submit code changes
- Improve documentation
- Share datasets

## 📄 License

StrataMind is released under the MIT License. See [LICENSE](../LICENSE) for details.

## 🙏 Acknowledgments

- Geological research community for datasets and feedback
- Open-source AI/ML community for tools and libraries
- Contributors and users who help improve StrataMind

---

**Happy mineral detecting! 🔬**

For the latest updates, follow us on [GitHub](https://github.com/makalin/StrataMind). 
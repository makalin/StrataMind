# Installation Guide

This guide will help you install and set up StrataMind on your system.

## Prerequisites

Before installing StrataMind, make sure you have:

- **Python 3.8 or higher** (Python 3.9+ recommended)
- **pip** (Python package installer)
- **Git** (for cloning the repository)

### Checking Your Python Version

```bash
python --version
# or
python3 --version
```

If you don't have Python installed, download it from [python.org](https://www.python.org/downloads/).

## Installation Methods

### Method 1: Clone and Install (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/makalin/StrataMind.git
   cd StrataMind
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Using venv (Python 3.3+)
   python -m venv stratamind_env
   
   # Activate the environment
   # On Windows:
   stratamind_env\Scripts\activate
   # On macOS/Linux:
   source stratamind_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Method 2: Using pip (Development)

```bash
pip install git+https://github.com/makalin/StrataMind.git
```

## GPU Support (Optional but Recommended)

For faster training and inference, we recommend using a GPU with CUDA support.

### NVIDIA GPU Setup

1. **Install CUDA Toolkit** (version 11.8 or 12.1):
   - Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
   - Follow the installation guide for your platform

2. **Install cuDNN**:
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Extract and copy files to your CUDA installation

3. **Install PyTorch with CUDA**:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Verify GPU Installation

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

## Platform-Specific Instructions

### Windows

1. **Install Visual Studio Build Tools** (if needed):
   ```bash
   # Install Microsoft C++ Build Tools
   # Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### macOS

1. **Install Xcode Command Line Tools** (if needed):
   ```bash
   xcode-select --install
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Linux (Ubuntu/Debian)

1. **Install system dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv git
   sudo apt install libgl1-mesa-glx libglib2.0-0
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Verification

After installation, verify that everything is working:

```bash
# Test basic imports
python -c "import torch; import cv2; import streamlit; print('✅ All imports successful!')"

# Test StrataMind modules
python -c "from src.model import MineralDetector; print('✅ StrataMind modules loaded!')"
```

## Running the Demos

### Streamlit Web App

```bash
streamlit run demo/streamlit_demo.py
```

The web interface will open at `http://localhost:8501`

### Command Line Interface

```bash
# Show help
python demo/app.py --help

# Test with a sample image
python demo/app.py predict --image path/to/image.jpg
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` when importing StrataMind modules

**Solution**: Make sure you're in the correct directory and have installed dependencies:
```bash
cd StrataMind
pip install -r requirements.txt
```

#### 2. CUDA/GPU Issues

**Problem**: CUDA not available or GPU not detected

**Solution**: 
- Verify CUDA installation: `nvidia-smi`
- Reinstall PyTorch with correct CUDA version
- Check GPU drivers are up to date

#### 3. OpenCV Issues

**Problem**: OpenCV installation fails

**Solution**:
```bash
# Try alternative installation
pip install opencv-python-headless
```

#### 4. Memory Issues

**Problem**: Out of memory errors during training

**Solution**:
- Reduce batch size in training configuration
- Use gradient accumulation
- Close other applications to free memory

#### 5. Permission Errors

**Problem**: Permission denied when installing packages

**Solution**:
```bash
# Use user installation
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

### Getting Help

If you encounter issues not covered here:

1. **Check the FAQ**: [FAQ](faq.md)
2. **Search existing issues**: [GitHub Issues](https://github.com/makalin/StrataMind/issues)
3. **Create a new issue**: Include your system details and error messages
4. **Join discussions**: [GitHub Discussions](https://github.com/makalin/StrataMind/discussions)

## Next Steps

After successful installation:

1. **Try the demos**: Run the Streamlit app or CLI demo
2. **Read the User Guide**: [User Guide](user-guide.md)
3. **Train a custom model**: [Training Guide](training-guide.md)
4. **Explore the API**: [API Reference](api-reference.md)

---

**Happy installing! 🔬** 
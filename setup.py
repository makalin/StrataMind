#!/usr/bin/env python3
"""
Setup script for StrataMind

This script helps users set up the StrataMind project with all necessary
dependencies and initial configuration.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "data",
        "models", 
        "notebooks",
        "src",
        "demo",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies."""
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    return True

def create_sample_data():
    """Create sample data structure."""
    try:
        from src.data import create_sample_data_structure
        create_sample_data_structure("data")
        print("✅ Created sample data structure")
        return True
    except ImportError:
        print("⚠️  Could not create sample data structure (dependencies not installed)")
        return False

def test_installation():
    """Test if the installation was successful."""
    print("\n🧪 Testing installation...")
    
    # Test basic imports
    test_imports = [
        "torch",
        "torchvision", 
        "cv2",
        "numpy",
        "streamlit"
    ]
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            return False
    
    # Test StrataMind modules
    try:
        sys.path.append('src')
        from model import MineralDetector
        print("✅ StrataMind modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import StrataMind modules: {e}")
        return False
    
    return True

def main():
    """Main setup function."""
    print("🔬 StrataMind Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating project structure...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create sample data
    print("\n📊 Creating sample data...")
    create_sample_data()
    
    # Test installation
    print("\n🧪 Testing installation...")
    if not test_installation():
        print("❌ Installation test failed")
        sys.exit(1)
    
    # Success message
    print("\n🎉 StrataMind setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the Streamlit demo:")
    print("   streamlit run demo/streamlit_demo.py")
    print("\n2. Try the CLI demo:")
    print("   python demo/app.py predict --image path/to/image.jpg")
    print("\n3. Train your own model:")
    print("   jupyter notebook notebooks/train_custom_model.ipynb")
    print("\n4. Read the documentation:")
    print("   docs/README.md")
    
    print("\n🔬 Happy mineral detecting!")

if __name__ == "__main__":
    main() 
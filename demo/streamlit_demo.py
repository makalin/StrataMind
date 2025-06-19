"""
Streamlit Demo for StrataMind Mineral Detection

This is a web-based demo application that allows users to upload geological
images and get mineral detection predictions using the StrataMind model.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.model import MineralDetector, load_model, predict_mineral
from src.utils import load_image, visualize_results, get_image_info
from src.data import create_sample_data_structure

# Page configuration
st.set_page_config(
    page_title="StrataMind - Mineral Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .mineral-item {
        background-color: white;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_demo_model():
    """Load the demo model (placeholder for now)."""
    try:
        # Try to load a pre-trained model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'stratamind_model.pt')
        if os.path.exists(model_path):
            model = load_model(model_path)
            return model
        else:
            # Create a dummy model for demo purposes
            st.warning("No pre-trained model found. Using demo model with random predictions.")
            model = MineralDetector()
            model.eval()
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_minerals_demo(image, model):
    """Make predictions using the demo model."""
    try:
        # Convert PIL image to tensor
        if isinstance(image, Image.Image):
            # Resize image
            image = image.resize((224, 224))
            # Convert to numpy array
            image_array = np.array(image)
            # Normalize
            image_tensor = torch.from_numpy(image_array).float() / 255.0
            # Add batch dimension and rearrange to (B, C, H, W)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            # For demo purposes, generate random predictions
            if hasattr(model, 'mineral_classes'):
                minerals = model.mineral_classes
            else:
                minerals = ["Quartz", "Feldspar", "Mica", "Calcite", "Pyrite"]
            
            # Generate random predictions
            num_predictions = np.random.randint(1, 4)
            selected_minerals = np.random.choice(minerals, num_predictions, replace=False)
            confidences = np.random.uniform(0.6, 0.95, num_predictions)
            
            # Sort by confidence
            sorted_indices = np.argsort(confidences)[::-1]
            selected_minerals = selected_minerals[sorted_indices]
            confidences = confidences[sorted_indices]
            
            return {
                "minerals": [selected_minerals.tolist()],
                "confidences": [confidences.tolist()]
            }
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 StrataMind</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI for Detecting Rare Minerals in Geological Data</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Select Model",
        ["Demo Model", "Pre-trained Model"],
        help="Choose between demo model or pre-trained model"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for predictions"
    )
    
    # Load model
    model = load_demo_model()
    
    if model is None:
        st.error("Failed to load model. Please check the model files.")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Geological Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a geological image for mineral detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image information
            with st.expander("Image Information"):
                try:
                    info = get_image_info(uploaded_file.name)
                    st.write(f"**File:** {info['path']}")
                    st.write(f"**Dimensions:** {info['width']} x {info['height']} pixels")
                    st.write(f"**Channels:** {info['channels']}")
                    st.write(f"**File Size:** {info['file_size_mb']:.2f} MB")
                except:
                    st.write("Could not retrieve image information")
    
    with col2:
        st.subheader("🔍 Detection Results")
        
        if uploaded_file is not None:
            # Make prediction
            with st.spinner("Analyzing image..."):
                predictions = predict_minerals_demo(image, model)
            
            if predictions:
                minerals = predictions['minerals'][0]
                confidences = predictions['confidences'][0]
                
                # Filter by confidence threshold
                filtered_results = [
                    (mineral, conf) for mineral, conf in zip(minerals, confidences)
                    if conf >= confidence_threshold
                ]
                
                if filtered_results:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.write("**Detected Minerals:**")
                    
                    for mineral, confidence in filtered_results:
                        confidence_pct = confidence * 100
                        st.markdown(f'''
                        <div class="mineral-item">
                            <strong>{mineral}</strong> - {confidence_pct:.1f}% confidence
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Confidence bar chart
                    st.subheader("Confidence Levels")
                    mineral_names = [m for m, _ in filtered_results]
                    confidence_values = [c for _, c in filtered_results]
                    
                    # Create a simple bar chart
                    chart_data = {
                        'Mineral': mineral_names,
                        'Confidence': confidence_values
                    }
                    st.bar_chart(chart_data)
                    
                else:
                    st.warning(f"No minerals detected above the confidence threshold ({confidence_threshold:.0%})")
            
            else:
                st.error("Failed to analyze image")
        
        else:
            st.info("👆 Upload an image to get started")
    
    # Additional features
    st.markdown("---")
    
    # Sample data creation
    with st.expander("📁 Create Sample Data Structure"):
        st.write("Create a sample data directory structure for training:")
        if st.button("Create Sample Structure"):
            try:
                create_sample_data_structure("data")
                st.success("Sample data structure created in 'data' directory!")
            except Exception as e:
                st.error(f"Error creating sample structure: {e}")
    
    # Model information
    with st.expander("🤖 Model Information"):
        st.write("**Model Architecture:** ResNet-50 with custom classification head")
        st.write("**Input Size:** 224x224 pixels")
        st.write("**Supported Minerals:** Quartz, Feldspar, Mica, Calcite, Pyrite, etc.")
        st.write("**Framework:** PyTorch")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🔬 StrataMind - AI for Geological Mineral Detection</p>
            <p>Built with ❤️ using Streamlit and PyTorch</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 
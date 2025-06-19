"""
Utility functions for image processing and visualization.

This module contains helper functions for loading, preprocessing, and
visualizing geological images for mineral detection.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Union, Tuple, List, Optional, Dict
import logging
import os

logger = logging.getLogger(__name__)

# Standard image preprocessing transforms
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_image(
    image_path: str, 
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    """
    Load and preprocess an image for mineral detection.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the image (height, width)
        normalize: Whether to normalize the image
        
    Returns:
        Preprocessed image as tensor or numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image = cv2.resize(image, target_size[::-1])  # OpenCV uses (width, height)
    
    # Convert to tensor if normalize is True
    if normalize:
        image = preprocess_image(image, target_size)
    
    return image


def preprocess_image(
    image: Union[np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Args:
        image: Input image as numpy array or PIL Image
        target_size: Target size for the image
        
    Returns:
        Preprocessed image tensor
    """
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Apply transforms
    tensor = transform(image)
    
    return tensor


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize a tensor back to image format.
    
    Args:
        tensor: Normalized image tensor
        
    Returns:
        Denormalized image as numpy array
    """
    # Denormalize
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    denormalized = tensor * std + mean
    
    # Convert to numpy and transpose
    image = denormalized.numpy().transpose(1, 2, 0)
    
    # Clip values to [0, 1]
    image = np.clip(image, 0, 1)
    
    return image


def visualize_results(
    image: Union[np.ndarray, torch.Tensor],
    predictions: Dict[str, List],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualize mineral detection results on the image.
    
    Args:
        image: Input image
        predictions: Dictionary with 'minerals' and 'confidences' lists
        save_path: Optional path to save the visualization
        show: Whether to display the visualization
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        image = denormalize_image(image)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title("Original Geological Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Results visualization
    ax2.imshow(image)
    ax2.set_title("Mineral Detection Results", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add prediction text
    minerals = predictions.get('minerals', [[]])[0] if predictions.get('minerals') else []
    confidences = predictions.get('confidences', [[]])[0] if predictions.get('confidences') else []
    
    if minerals:
        text = "Detected Minerals:\n"
        for mineral, conf in zip(minerals, confidences):
            text += f"• {mineral}: {conf:.2%}\n"
        
        ax2.text(
            0.02, 0.98, text,
            transform=ax2.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    else:
        ax2.text(
            0.5, 0.5, "No minerals detected\nabove threshold",
            transform=ax2.transAxes,
            fontsize=14,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8)
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_heatmap(
    image: Union[np.ndarray, torch.Tensor],
    attention_weights: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Create attention heatmap visualization.
    
    Args:
        image: Input image
        attention_weights: Attention weights for heatmap
        save_path: Optional path to save the heatmap
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        image = denormalize_image(image)
    
    # Resize attention weights to match image size
    attention_weights = cv2.resize(attention_weights, (image.shape[1], image.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(
        (attention_weights * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    
    # Overlay heatmap on image
    overlay = cv2.addWeighted(
        (image * 255).astype(np.uint8), 0.7,
        heatmap, 0.3, 0
    )
    
    # Display
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(attention_weights, cmap='jet')
    plt.title("Attention Heatmap")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Heatmap saved to {save_path}")
    
    plt.show()


def batch_predict(
    model: 'MineralDetector',
    image_paths: List[str],
    batch_size: int = 8,
    threshold: float = 0.5
) -> List[Dict]:
    """
    Perform batch prediction on multiple images.
    
    Args:
        model: Loaded MineralDetector model
        image_paths: List of image file paths
        batch_size: Batch size for processing
        threshold: Confidence threshold
        
    Returns:
        List of prediction results
    """
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        # Load batch images
        for path in batch_paths:
            try:
                image = load_image(path)
                batch_images.append(image)
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Stack images into batch
        batch_tensor = torch.stack(batch_images)
        
        # Make predictions
        batch_predictions = model.predict(batch_tensor, threshold)
        
        # Store results
        for j, path in enumerate(batch_paths):
            if j < len(batch_predictions['minerals']):
                results.append({
                    'image_path': path,
                    'minerals': batch_predictions['minerals'][j],
                    'confidences': batch_predictions['confidences'][j]
                })
    
    return results


def save_predictions(
    predictions: List[Dict],
    output_path: str,
    format: str = 'json'
) -> None:
    """
    Save prediction results to file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save the results
        format: Output format ('json' or 'csv')
    """
    import json
    import pandas as pd
    
    if format.lower() == 'json':
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
    
    elif format.lower() == 'csv':
        # Flatten predictions for CSV
        rows = []
        for pred in predictions:
            minerals = pred.get('minerals', [])
            confidences = pred.get('confidences', [])
            
            row = {'image_path': pred['image_path']}
            for i, (mineral, conf) in enumerate(zip(minerals, confidences)):
                row[f'mineral_{i+1}'] = mineral
                row[f'confidence_{i+1}'] = conf
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    
    logger.info(f"Predictions saved to {output_path}")


def get_image_info(image_path: str) -> Dict:
    """
    Get basic information about an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with image information
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Get file info
    file_size = os.path.getsize(image_path)
    
    return {
        'path': image_path,
        'height': image.shape[0],
        'width': image.shape[1],
        'channels': image.shape[2],
        'file_size_bytes': file_size,
        'file_size_mb': file_size / (1024 * 1024)
    } 
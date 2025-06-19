"""
Mineral Detection Model Module

This module contains the core MineralDetector class and related functions
for detecting rare minerals in geological images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class MineralDetector(nn.Module):
    """
    Deep learning model for detecting rare minerals in geological images.
    
    This model uses a pre-trained ResNet backbone with custom classification head
    for mineral detection tasks.
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the MineralDetector model.
        
        Args:
            num_classes: Number of mineral classes to predict
            backbone: Backbone architecture ('resnet50', 'resnet101', 'efficientnet')
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
        """
        super(MineralDetector, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.dropout_rate = dropout_rate
        
        # Initialize backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "efficientnet":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the original classification head
        if backbone.startswith("resnet"):
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Mineral class names (example - should be configurable)
        self.mineral_classes = [
            "Quartz", "Feldspar", "Mica", "Calcite", "Pyrite",
            "Galena", "Sphalerite", "Chalcopyrite", "Magnetite", "Hematite"
        ]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, Union[List[str], List[float]]]:
        """
        Predict minerals in the input image.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            threshold: Confidence threshold for predictions
            
        Returns:
            Dictionary with predicted minerals and their confidences
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)
            
            predictions = []
            confidences = []
            
            for i in range(x.size(0)):
                sample_predictions = []
                sample_confidences = []
                
                for j in range(top_probs.size(1)):
                    if top_probs[i, j] > threshold:
                        mineral_name = self.mineral_classes[top_indices[i, j].item()]
                        confidence = top_probs[i, j].item()
                        sample_predictions.append(mineral_name)
                        sample_confidences.append(confidence)
                
                predictions.append(sample_predictions)
                confidences.append(sample_confidences)
            
            return {
                "minerals": predictions,
                "confidences": confidences
            }


def load_model(model_path: str, device: str = "auto") -> MineralDetector:
    """
    Load a trained MineralDetector model.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on ('cpu', 'cuda', or 'auto')
        
    Returns:
        Loaded MineralDetector model
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model architecture
    model = MineralDetector()
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {model_path} on device {device}")
    return model


def predict_mineral(
    model: MineralDetector, 
    image: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5
) -> Dict[str, Union[List[str], List[float]]]:
    """
    Predict minerals in an image using the loaded model.
    
    Args:
        model: Loaded MineralDetector model
        image: Input image as tensor or numpy array
        threshold: Confidence threshold for predictions
        
    Returns:
        Dictionary with predicted minerals and their confidences
    """
    if isinstance(image, np.ndarray):
        # Convert numpy array to tensor
        image = torch.from_numpy(image).float()
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension
    
    # Ensure image is on the same device as model
    device = next(model.parameters()).device
    image = image.to(device)
    
    # Make prediction
    predictions = model.predict(image, threshold)
    
    return predictions


def save_model(model: MineralDetector, save_path: str, metadata: Optional[Dict] = None):
    """
    Save a trained model with optional metadata.
    
    Args:
        model: MineralDetector model to save
        save_path: Path where to save the model
        metadata: Optional metadata to save with the model
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_classes": model.num_classes,
            "backbone": model.backbone_name,
            "dropout_rate": model.dropout_rate,
            "mineral_classes": model.mineral_classes
        }
    }
    
    if metadata:
        checkpoint["metadata"] = metadata
    
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")


def create_model(
    num_classes: int = 10,
    backbone: str = "resnet50",
    pretrained: bool = True
) -> MineralDetector:
    """
    Create a new MineralDetector model.
    
    Args:
        num_classes: Number of mineral classes
        backbone: Backbone architecture
        pretrained: Whether to use pretrained weights
        
    Returns:
        New MineralDetector model
    """
    model = MineralDetector(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained
    )
    
    logger.info(f"Created new MineralDetector model with {num_classes} classes and {backbone} backbone")
    return model 
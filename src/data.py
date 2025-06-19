"""
Data loading and preprocessing utilities for mineral detection.

This module contains dataset classes and data loading functions for
training and evaluating mineral detection models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import json
from typing import List, Tuple, Dict, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MineralDataset(Dataset):
    """
    Dataset class for mineral detection training and evaluation.
    
    This dataset loads geological images and their corresponding mineral labels
    for training deep learning models.
    """
    
    def __init__(
        self,
        data_dir: str,
        annotations_file: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224),
        mode: str = "train"
    ):
        """
        Initialize the MineralDataset.
        
        Args:
            data_dir: Directory containing image files
            annotations_file: Path to annotations file (CSV or JSON)
            transform: Optional transforms to apply
            target_size: Target image size
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.mode = mode
        
        # Set up transforms
        if transform is None:
            self.transform = get_transforms(target_size, mode)
        else:
            self.transform = transform
        
        # Load annotations
        self.samples = self._load_annotations(annotations_file)
        
        # Create class mapping
        self.class_to_idx = self._create_class_mapping()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        logger.info(f"Loaded {len(self.samples)} samples for {mode} mode")
        logger.info(f"Found {len(self.class_to_idx)} mineral classes")
    
    def _load_annotations(self, annotations_file: Optional[str]) -> List[Dict]:
        """
        Load annotations from file or create from directory structure.
        
        Args:
            annotations_file: Path to annotations file
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        if annotations_file and os.path.exists(annotations_file):
            # Load from annotations file
            if annotations_file.endswith('.csv'):
                df = pd.read_csv(annotations_file)
                for _, row in df.iterrows():
                    image_path = os.path.join(self.data_dir, row['image_path'])
                    if os.path.exists(image_path):
                        samples.append({
                            'image_path': image_path,
                            'label': row['mineral_class'],
                            'confidence': row.get('confidence', 1.0)
                        })
            
            elif annotations_file.endswith('.json'):
                with open(annotations_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        image_path = os.path.join(self.data_dir, item['image_path'])
                        if os.path.exists(image_path):
                            samples.append({
                                'image_path': image_path,
                                'label': item['mineral_class'],
                                'confidence': item.get('confidence', 1.0)
                            })
        else:
            # Create from directory structure (class folders)
            for class_dir in self.data_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    for image_file in class_dir.glob('*.jpg'):
                        samples.append({
                            'image_path': str(image_file),
                            'label': class_name,
                            'confidence': 1.0
                        })
                    for image_file in class_dir.glob('*.jpeg'):
                        samples.append({
                            'image_path': str(image_file),
                            'label': class_name,
                            'confidence': 1.0
                        })
                    for image_file in class_dir.glob('*.png'):
                        samples.append({
                            'image_path': str(image_file),
                            'label': class_name,
                            'confidence': 1.0
                        })
        
        return samples
    
    def _create_class_mapping(self) -> Dict[str, int]:
        """
        Create mapping from class names to indices.
        
        Returns:
            Dictionary mapping class names to indices
        """
        unique_classes = sorted(list(set(sample['label'] for sample in self.samples)))
        return {class_name: idx for idx, class_name in enumerate(unique_classes)}
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {sample['image_path']}: {e}")
            # Return a placeholder image
            image = Image.new('RGB', self.target_size, color='gray')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.class_to_idx[sample['label']]
        
        return image, label
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return list(self.class_to_idx.keys())
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset."""
        distribution = {}
        for sample in self.samples:
            label = sample['label']
            distribution[label] = distribution.get(label, 0) + 1
        return distribution


def get_transforms(
    target_size: Tuple[int, int] = (224, 224),
    mode: str = "train",
    augment: bool = True
) -> transforms.Compose:
    """
    Get transforms for data preprocessing.
    
    Args:
        target_size: Target image size
        mode: Dataset mode ('train', 'val', 'test')
        augment: Whether to apply data augmentation
        
    Returns:
        Compose transform
    """
    if mode == "train" and augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_data_loaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (224, 224),
    augment: bool = True
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory (optional)
        test_dir: Test data directory (optional)
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        target_size: Target image size
        augment: Whether to apply data augmentation
        
    Returns:
        Dictionary containing data loaders
    """
    loaders = {}
    
    # Training loader
    train_dataset = MineralDataset(
        data_dir=train_dir,
        transform=get_transforms(target_size, "train", augment),
        mode="train"
    )
    
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation loader
    if val_dir:
        val_dataset = MineralDataset(
            data_dir=val_dir,
            transform=get_transforms(target_size, "val", False),
            mode="val"
        )
        
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Test loader
    if test_dir:
        test_dataset = MineralDataset(
            data_dir=test_dir,
            transform=get_transforms(target_size, "test", False),
            mode="test"
        )
        
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders


def split_dataset(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Dict[str, List[str]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        data_dir: Data directory
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with split file paths
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    np.random.seed(random_seed)
    
    # Collect all image files
    image_files = []
    data_path = Path(data_dir)
    
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend([str(f) for f in data_path.rglob(ext)])
    
    # Shuffle files
    np.random.shuffle(image_files)
    
    # Split files
    n_files = len(image_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    splits = {
        'train': image_files[:n_train],
        'val': image_files[n_train:n_train + n_val],
        'test': image_files[n_train + n_val:]
    }
    
    logger.info(f"Split {n_files} files: {len(splits['train'])} train, "
                f"{len(splits['val'])} val, {len(splits['test'])} test")
    
    return splits


def create_sample_data_structure(data_dir: str) -> None:
    """
    Create a sample data directory structure for demonstration.
    
    Args:
        data_dir: Directory to create structure in
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Create sample class directories
    mineral_classes = [
        "Quartz", "Feldspar", "Mica", "Calcite", "Pyrite",
        "Galena", "Sphalerite", "Chalcopyrite", "Magnetite", "Hematite"
    ]
    
    for mineral in mineral_classes:
        class_dir = data_path / mineral
        class_dir.mkdir(exist_ok=True)
        
        # Create a placeholder file
        placeholder_file = class_dir / "README.md"
        placeholder_file.write_text(f"# {mineral} Samples\n\nPlace your {mineral} images here.")
    
    # Create annotations file
    annotations = []
    for mineral in mineral_classes:
        annotations.append({
            "image_path": f"{mineral}/sample1.jpg",
            "mineral_class": mineral,
            "confidence": 1.0
        })
    
    with open(data_path / "annotations.json", 'w') as f:
        json.dump(annotations, f, indent=2)
    
    logger.info(f"Created sample data structure in {data_dir}")


def validate_dataset(data_dir: str) -> Dict:
    """
    Validate dataset structure and integrity.
    
    Args:
        data_dir: Data directory to validate
        
    Returns:
        Validation results dictionary
    """
    data_path = Path(data_dir)
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    if not data_path.exists():
        results['valid'] = False
        results['errors'].append(f"Data directory does not exist: {data_dir}")
        return results
    
    # Check for image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(data_path.rglob(ext)))
    
    if not image_files:
        results['valid'] = False
        results['errors'].append("No image files found")
    else:
        results['stats']['total_images'] = len(image_files)
    
    # Check for class directories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    if class_dirs:
        results['stats']['num_classes'] = len(class_dirs)
        results['stats']['classes'] = [d.name for d in class_dirs]
    else:
        results['warnings'].append("No class directories found")
    
    # Check for annotations file
    annotation_files = list(data_path.glob("*.json")) + list(data_path.glob("*.csv"))
    if annotation_files:
        results['stats']['annotation_files'] = [f.name for f in annotation_files]
    else:
        results['warnings'].append("No annotation files found")
    
    return results 
# Data Directory

This directory contains geological image datasets for training and testing StrataMind models.

## Directory Structure

```
data/
├── README.md                    # This file
├── annotations.json            # Dataset annotations (optional)
├── Quartz/                     # Quartz mineral images
│   ├── sample1.jpg
│   ├── sample2.jpg
│   └── ...
├── Feldspar/                   # Feldspar mineral images
│   ├── sample1.jpg
│   ├── sample2.jpg
│   └── ...
├── Mica/                       # Mica mineral images
│   └── ...
└── ...                         # Other mineral classes
```

## Data Organization

### Option 1: Class-based Directory Structure (Recommended)

Organize your images in subdirectories named after each mineral class:

```
data/
├── Quartz/
│   ├── quartz_sample_001.jpg
│   ├── quartz_sample_002.jpg
│   └── ...
├── Feldspar/
│   ├── feldspar_sample_001.jpg
│   ├── feldspar_sample_002.jpg
│   └── ...
└── ...
```

### Option 2: Annotations File

Use an annotations file (JSON or CSV) to specify image paths and labels:

```json
[
  {
    "image_path": "Quartz/sample1.jpg",
    "mineral_class": "Quartz",
    "confidence": 1.0
  },
  {
    "image_path": "Feldspar/sample1.jpg", 
    "mineral_class": "Feldspar",
    "confidence": 1.0
  }
]
```

## Supported Image Formats

- **JPEG** (.jpg, .jpeg)
- **PNG** (.png)
- **TIFF** (.tiff, .tif)

## Image Requirements

For best results, ensure your images meet these requirements:

- **Resolution**: Minimum 224x224 pixels (higher resolution recommended)
- **Format**: RGB color images
- **Quality**: Clear, well-lit images with good contrast
- **Content**: Mineral should be clearly visible and centered
- **Variety**: Include different angles, lighting conditions, and mineral specimens

## Creating Sample Data Structure

To create a sample data structure for testing:

```bash
python demo/app.py setup --data-dir data/
```

This will create placeholder directories for common mineral classes.

## Validating Your Dataset

To validate your dataset structure:

```bash
python demo/app.py validate --data-dir data/
```

This will check for:
- Valid image files
- Proper directory structure
- Class balance
- File integrity

## Dataset Guidelines

### Training Data

- **Minimum**: 50 images per class
- **Recommended**: 200+ images per class
- **Balance**: Similar number of images across classes
- **Quality**: High-quality, well-labeled images

### Validation Data

- **Split**: 20-30% of total dataset
- **Representation**: Include samples from all classes
- **Quality**: Same quality standards as training data

### Test Data

- **Split**: 10-20% of total dataset
- **Unseen**: Should not overlap with training/validation
- **Realistic**: Representative of real-world usage

## Adding Your Own Data

1. **Organize images** by mineral class in subdirectories
2. **Validate structure** using the validation tool
3. **Update annotations** if using annotation files
4. **Test loading** with the data utilities

## Example Usage

```python
from src.data import MineralDataset, create_data_loaders

# Create dataset
dataset = MineralDataset(
    data_dir="data/",
    target_size=(224, 224),
    mode="train"
)

# Create data loaders
loaders = create_data_loaders(
    train_dir="data/",
    batch_size=32,
    target_size=(224, 224)
)
```

## Data Sources

Some recommended sources for geological images:

- **Academic repositories**: University geology departments
- **Museum collections**: Natural history museums
- **Research papers**: Published geological studies
- **Field surveys**: Your own geological fieldwork
- **Online databases**: Geological image databases

## Privacy and Licensing

- Ensure you have permission to use all images
- Respect copyright and licensing requirements
- Consider data privacy implications
- Document the source of your data

## Troubleshooting

### Common Issues

1. **No images found**: Check file extensions and directory structure
2. **Import errors**: Verify image formats are supported
3. **Memory issues**: Reduce image resolution or batch size
4. **Class imbalance**: Add more images to underrepresented classes

### Getting Help

- Check the [User Guide](../docs/user-guide.md)
- Review the [FAQ](../docs/faq.md)
- Create an issue on GitHub
- Join discussions in the community

---

**Happy data organizing! 📊** 
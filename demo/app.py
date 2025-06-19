#!/usr/bin/env python3
"""
Command Line Interface for StrataMind Mineral Detection

This is a CLI application that allows users to perform mineral detection
on geological images from the command line.
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.model import MineralDetector, load_model, predict_mineral, create_model
from src.utils import load_image, visualize_results, batch_predict, save_predictions, get_image_info
from src.data import create_sample_data_structure, validate_dataset

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="StrataMind - AI for Detecting Rare Minerals in Geological Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single image
  python demo/app.py predict --image path/to/image.jpg

  # Analyze multiple images
  python demo/app.py predict --image path/to/image1.jpg path/to/image2.jpg

  # Batch prediction with output
  python demo/app.py predict --image path/to/images/ --output results.json

  # Create sample data structure
  python demo/app.py setup --data-dir data/

  # Validate dataset
  python demo/app.py validate --data-dir data/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict minerals in images')
    predict_parser.add_argument('--image', nargs='+', required=True, help='Image file(s) or directory')
    predict_parser.add_argument('--model', default='models/stratamind_model.pt', help='Model file path')
    predict_parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    predict_parser.add_argument('--output', help='Output file for results (JSON or CSV)')
    predict_parser.add_argument('--visualize', action='store_true', help='Show visualization')
    predict_parser.add_argument('--save-viz', help='Save visualization to file')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup project structure')
    setup_parser.add_argument('--data-dir', default='data', help='Data directory path')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument('--data-dir', required=True, help='Data directory path')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get image information')
    info_parser.add_argument('--image', required=True, help='Image file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'predict':
        predict_command(args)
    elif args.command == 'setup':
        setup_command(args)
    elif args.command == 'validate':
        validate_command(args)
    elif args.command == 'info':
        info_command(args)

def predict_command(args):
    """Handle predict command."""
    print("🔬 StrataMind - Mineral Detection")
    print("=" * 50)
    
    # Load model
    print(f"Loading model from {args.model}...")
    try:
        if os.path.exists(args.model):
            model = load_model(args.model)
            print("✅ Model loaded successfully")
        else:
            print("⚠️  Model file not found, creating demo model...")
            model = create_model()
            model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Process images
    image_paths = []
    for path in args.image:
        if os.path.isfile(path):
            image_paths.append(path)
        elif os.path.isdir(path):
            # Add all image files in directory
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_paths.extend([str(p) for p in Path(path).glob(ext)])
        else:
            print(f"⚠️  Warning: {path} is not a valid file or directory")
    
    if not image_paths:
        print("❌ No valid image files found")
        return
    
    print(f"Found {len(image_paths)} image(s) to process")
    
    # Process images
    if len(image_paths) == 1:
        # Single image
        result = process_single_image(image_paths[0], model, args)
    else:
        # Multiple images
        result = process_multiple_images(image_paths, model, args)
    
    # Save results
    if args.output and result:
        try:
            save_predictions(result, args.output)
            print(f"✅ Results saved to {args.output}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def process_single_image(image_path, model, args):
    """Process a single image."""
    print(f"\nProcessing: {image_path}")
    
    try:
        # Load image
        image = load_image(image_path, normalize=True)
        
        # Make prediction
        predictions = predict_mineral(model, image, args.threshold)
        
        # Display results
        minerals = predictions['minerals'][0]
        confidences = predictions['confidences'][0]
        
        if minerals:
            print("Detected Minerals:")
            for mineral, conf in zip(minerals, confidences):
                print(f"  • {mineral}: {conf:.1%}")
        else:
            print(f"No minerals detected above threshold ({args.threshold:.0%})")
        
        # Visualization
        if args.visualize or args.save_viz:
            try:
                visualize_results(
                    image, 
                    predictions, 
                    save_path=args.save_viz,
                    show=args.visualize
                )
            except Exception as e:
                print(f"⚠️  Warning: Could not create visualization: {e}")
        
        return [{
            'image_path': image_path,
            'minerals': minerals,
            'confidences': confidences
        }]
        
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

def process_multiple_images(image_paths, model, args):
    """Process multiple images."""
    print(f"\nProcessing {len(image_paths)} images...")
    
    try:
        results = batch_predict(model, image_paths, threshold=args.threshold)
        
        # Display summary
        total_detected = sum(len(r['minerals']) for r in results)
        print(f"\nSummary:")
        print(f"  • Images processed: {len(results)}")
        print(f"  • Total minerals detected: {total_detected}")
        
        # Show top predictions
        if results:
            print("\nTop predictions:")
            for result in results[:5]:  # Show first 5
                minerals = result['minerals']
                confidences = result['confidences']
                if minerals:
                    best_mineral = minerals[0]
                    best_conf = confidences[0]
                    print(f"  • {Path(result['image_path']).name}: {best_mineral} ({best_conf:.1%})")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        return None

def setup_command(args):
    """Handle setup command."""
    print("🔬 StrataMind - Setup")
    print("=" * 30)
    
    try:
        create_sample_data_structure(args.data_dir)
        print(f"✅ Sample data structure created in {args.data_dir}")
        print("\nNext steps:")
        print("1. Add your geological images to the class directories")
        print("2. Update the annotations.json file if needed")
        print("3. Run 'python demo/app.py validate --data-dir data/' to verify")
    except Exception as e:
        print(f"❌ Error creating data structure: {e}")

def validate_command(args):
    """Handle validate command."""
    print("🔬 StrataMind - Dataset Validation")
    print("=" * 40)
    
    try:
        results = validate_dataset(args.data_dir)
        
        if results['valid']:
            print("✅ Dataset is valid")
        else:
            print("❌ Dataset has issues:")
            for error in results['errors']:
                print(f"  • {error}")
        
        if results['warnings']:
            print("\n⚠️  Warnings:")
            for warning in results['warnings']:
                print(f"  • {warning}")
        
        if results['stats']:
            print("\n📊 Dataset Statistics:")
            for key, value in results['stats'].items():
                print(f"  • {key}: {value}")
                
    except Exception as e:
        print(f"❌ Error validating dataset: {e}")

def info_command(args):
    """Handle info command."""
    print("🔬 StrataMind - Image Information")
    print("=" * 35)
    
    try:
        info = get_image_info(args.image)
        
        print(f"File: {info['path']}")
        print(f"Dimensions: {info['width']} x {info['height']} pixels")
        print(f"Channels: {info['channels']}")
        print(f"File Size: {info['file_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error getting image info: {e}")

if __name__ == "__main__":
    main() 
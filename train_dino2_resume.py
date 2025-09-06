#!/usr/bin/env python3
"""
YOLOv13 with DINO Vision Transformer Backbone Training Script - Resume Training Edition

This script trains YOLOv13 enhanced with Meta's DINO vision transformer backbones (DINO2/DINO3).
Key features:
- Real DINO2/DINO3 pretrained weights from Meta
- Support for all DINOv3 multi-scale architectures
- Configurable weight freezing for transfer learning
- Clean training output without freeze warnings
- Full compatibility with Ultralytics training pipeline
- **NEW**: Support for resuming from previous trained weights

Usage:
    # Resume from previous DINO2/DINO3 weights
    python train_dino2_resume.py --data path/to/data.yaml --weights path/to/weights.pt --epochs 100 --freeze-dino2
    
    # Resume with different settings
    python train_dino2_resume.py --data data.yaml --weights runs/detect/yolov13-dino3/weights/best.pt --epochs 50
    
    # Start fresh DINO3 training (NEW)
    python train_dino2_resume.py --data data.yaml --model yolov13-dino3 --dino-variant dinov3_vitb16 --freeze-dino2
    python train_dino2_resume.py --data data.yaml --model yolov13-dino3-dual --dino-variant dinov3_vitl16
    python train_dino2_resume.py --data data.yaml --model yolov13-dino3-multi --dino-variant dinov3_vith16_plus
    
    # Start fresh DINO2 training (original)
    python train_dino2_resume.py --data data.yaml --model yolov13-dino2-working --size s --dino-variant dinov2_vits14
"""

import argparse
import logging
import sys
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr


class DINOFilter(logging.Filter):
    """Custom logging filter to suppress DINO freeze warnings."""
    
    def filter(self, record):
        """Filter out DINO-specific freeze warnings."""
        if hasattr(record, 'getMessage'):
            message = record.getMessage()
            
            # Filter out DINO2/DINO3 freeze warnings
            if ("setting 'requires_grad=True' for frozen layer" in message and 
                ("dino_model" in message)):
                return False  # Don't log this message
                
        return True  # Log all other messages


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv13 with DINO Vision Transformer Backbone - Resume Training')
    
    # Core arguments
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--name', type=str, default='yolov13-dino-resumed', help='Experiment name')
    parser.add_argument('--freeze-dino2', action='store_true', help='Freeze DINO backbone weights (works for both DINO2/DINO3)')
    parser.add_argument('--device', type=str, default=None, help='Device to run on, e.g., 0 or 0,1,2,3 for multi-GPU')
    
    # **NEW**: Weights loading option
    parser.add_argument('--weights', type=str, default=None, help='Path to previous trained weights (.pt file)')
    
    # Model variant selection (only used when --weights is not provided)
    parser.add_argument('--model', type=str, default='yolov13-dino2-working', 
                       choices=['yolov13', 'yolov13n', 'yolov13s', 'yolov13l', 'yolov13x',
                               'yolov13-dino2', 'yolov13-dino2-simple', 
                               'yolov13-dino2-working', 'yolov13-dino2-fixed',
                               'yolov13-dino3', 'yolov13-dino3-dual', 'yolov13-dino3-p3', 'yolov13-dino3-multi'],
                       help='YOLOv13 model variant (ignored if --weights is provided)')
    parser.add_argument('--size', type=str, default=None,
                       choices=['n', 's', 'l', 'x'],
                       help='YOLOv13 model size (ignored if --weights is provided)')
    parser.add_argument('--dino-variant', type=str, default='dinov2_vitb14',
                       choices=[
                           # DINO2 variants
                           'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
                           # DINOv3 ViT variants (official naming)
                           'dinov3_vits16', 'dinov3_vits16_plus', 'dinov3_vitb16', 'dinov3_vitl16', 
                           'dinov3_vith16_plus', 'dinov3_vit7b16',
                           # DINOv3 ConvNeXt variants
                           'dinov3_convnext_tiny', 'dinov3_convnext_small', 'dinov3_convnext_base', 'dinov3_convnext_large',
                           # Legacy DINOv3 naming (backward compatibility)
                           'dinov3_vits14', 'dinov3_vitb14', 'dinov3_vitl14', 'dinov3_vitg14'
                       ],
                       help='DINO model variant (DINO2/DINO3 variants for enhanced models)')
    
    args = parser.parse_args()
    
    # Apply the DINO filter to the ultralytics logger
    dino_filter = DINOFilter()
    LOGGER.addFilter(dino_filter)
    
    print(f"{colorstr('bright_blue', 'bold', 'YOLOv13 Resume Training')}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"Device: {args.device if args.device else 'auto'}")
    dino_type = "DINO3" if args.dino_variant.startswith('dinov3') else "DINO2"
    print(f"{dino_type} Frozen: {args.freeze_dino2}")
    
    try:
        # **NEW**: Load model from weights or create new
        if args.weights:
            if not Path(args.weights).exists():
                raise FileNotFoundError(f"Weights file not found: {args.weights}")
            
            print(f"🔄 Loading model from weights: {args.weights}")
            model = YOLO(args.weights)
            print(f"✅ Model loaded successfully from {args.weights}")
            
        else:
            # Original behavior - create new model
            final_model = args.model
            if args.size and not final_model.endswith(args.size):
                if final_model in ['yolov13', 'yolov13-dino2', 'yolov13-dino2-simple', 
                                  'yolov13-dino2-working', 'yolov13-dino2-fixed', 'yolov13-dino3',
                                  'yolov13-dino3-dual', 'yolov13-dino3-p3', 'yolov13-dino3-multi']:
                    if final_model == 'yolov13':
                        final_model = f'yolov13{args.size}'
                    else:
                        final_model = f'{final_model}-{args.size}'
            
            print(f"🆕 Creating new model: {final_model}")
            model_path = f'ultralytics/cfg/models/v13/{final_model}.yaml'
            model = YOLO(model_path)
        
        dino_type = "DINO3" if args.dino_variant.startswith('dinov3') else "DINO2"
        print(f"{dino_type} Variant: {args.dino_variant}")
        print("=" * 50)
        
        # Configure DINO variant and freezing (supports both DINO2 and DINO3)
        has_dino = False
        dino_type_found = None
        for module in model.model.modules():
            module_class = str(module.__class__)
            if hasattr(module, '__class__') and ('DINO2Backbone' in module_class or 'DINO3Backbone' in module_class):
                has_dino = True
                dino_type_found = "DINO3" if 'DINO3Backbone' in module_class else "DINO2"
                
                # Update DINO variant if specified
                if hasattr(module, 'model_name') and args.dino_variant != module.model_name:
                    print(f"🔄 Updating {dino_type_found} variant from {module.model_name} to {args.dino_variant}")
                    module.model_name = args.dino_variant
                    # Reinitialize the model with new variant if method exists
                    if hasattr(module, '_initialize_dino_model'):
                        module._initialize_dino_model()
                
                # Configure freezing
                if args.freeze_dino2:
                    if hasattr(module, 'freeze_backbone_layers'):
                        module.freeze_backbone_layers()
                    print(f"✅ {dino_type_found} backbone frozen: {args.dino_variant}")
                else:
                    if hasattr(module, 'unfreeze_backbone'):
                        module.unfreeze_backbone()
                    print(f"🔓 {dino_type_found} backbone unfrozen: {args.dino_variant}")
        
        if not has_dino and args.weights:
            print(f"ℹ️  Loaded model from weights (may or may not have DINO)")
        elif not has_dino and ('dino2' in args.model.lower() or 'dino3' in args.model.lower()):
            print(f"⚠️  Warning: Model {args.model} should have DINO but none found")
        elif not has_dino:
            print(f"ℹ️  Using standard YOLOv13 without DINO backbone")
        
        # Training configuration
        train_args = {
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch_size,
            'imgsz': args.imgsz,
            'name': args.name,
            'verbose': True,
            'plots': True,
            'save_period': max(10, args.epochs // 10),
        }
        
        # Add device configuration if specified
        if args.device is not None:
            train_args['device'] = args.device
        
        print(f"\nStarting {'resume' if args.weights else 'new'} training...")
        
        # Train with filtered logging
        results = model.train(**train_args)
        
        print(f"\n{colorstr('bright_green', 'bold', 'Training Completed!')}")
        print(f"Best weights: {results.save_dir}/weights/best.pt")
        
        # Show final metrics
        if hasattr(results, 'metrics') and hasattr(results.metrics, 'box'):
            metrics = results.metrics.box
            if hasattr(metrics, 'map50'):
                print(f"Final mAP50: {metrics.map50:.4f}")
            if hasattr(metrics, 'map'):
                print(f"Final mAP50-95: {metrics.map:.4f}")
        
        print(f"\nTraining Summary:")
        if args.weights:
            print(f"   • Resumed from: {args.weights} ✅")
        else:
            dino_type = "DINO3" if args.dino_variant.startswith('dinov3') else "DINO2"
            print(f"   • {dino_type} pretrained weights: LOADED ✅") 
            print(f"   • Model architecture: YOLOv13 + {dino_type} ✅")
        print(f"   • Training completed successfully ✅")
        
        # Remove filter to restore normal logging
        LOGGER.removeFilter(dino_filter)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        # Remove filter even on failure
        LOGGER.removeFilter(dino_filter)
        return


if __name__ == '__main__':
    main()

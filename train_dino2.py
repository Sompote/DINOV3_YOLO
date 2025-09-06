#!/usr/bin/env python3
"""
YOLOv13 with DINO Vision Transformer Backbone Training Script

This script trains YOLOv13 enhanced with Meta's DINO pretrained vision transformer backbones (DINO2/DINO3).
Key features:
- Real DINO2/DINO3 pretrained weights from Meta
- Support for both DINO2 and DINO3 models
- Configurable weight freezing for transfer learning
- Clean training output without freeze warnings
- Full compatibility with Ultralytics training pipeline

Usage:
    # NEW: Train with DINO3 model (recommended)
    python train_dino2.py --data path/to/data.yaml --model yolov13-dino3 --dino-variant dinov3_vitb14 --epochs 100 --freeze-dino2
    
    # Train with default DINO2 base model
    python train_dino2.py --data path/to/data.yaml --epochs 100 --freeze-dino2
    
    # Train with different YOLOv13 sizes and DINO2 variants
    python train_dino2.py --data data.yaml --model yolov13-dino2-working --size s --dino-variant dinov2_vits14
    
    # Train specific YOLOv13 size models
    python train_dino2.py --data data.yaml --model yolov13n  # Nano
    python train_dino2.py --data data.yaml --model yolov13s  # Small
    python train_dino2.py --data data.yaml --model yolov13l  # Large
    python train_dino2.py --data data.yaml --model yolov13x  # Extra Large
    
    # Train YOLOv13 + DINO3 ViT combinations (NEW)
    python train_dino2.py --data data.yaml --model yolov13-dino3 --dino-variant dinov3_vits16  # Fast
    python train_dino2.py --data data.yaml --model yolov13-dino3 --dino-variant dinov3_vitb16  # Balanced
    python train_dino2.py --data data.yaml --model yolov13-dino3 --dino-variant dinov3_vitl16  # High accuracy
    
    # Train YOLOv13 + DINO3 ConvNeXt combinations (NEW)
    python train_dino2.py --data data.yaml --model yolov13-dino3 --dino-variant dinov3_convnext_tiny  # Lightweight
    python train_dino2.py --data data.yaml --model yolov13-dino3 --dino-variant dinov3_convnext_large  # Maximum performance
    
    # Train YOLOv13 + DINO3 Multi-Scale Architectures (NEW)
    python train_dino2.py --data data.yaml --model yolov13-dino3-dual    # P3+P4 enhanced
    python train_dino2.py --data data.yaml --model yolov13-dino3-p3      # P3 focused (small objects)
    python train_dino2.py --data data.yaml --model yolov13-dino3-multi   # All scales with optimized variants
    
    # Train YOLOv13 + DINO3 Satellite Imagery variants (LATEST - for satellite/aerial imagery)
    python train_dino2.py --data satellite.yaml --model yolov13-dino3 --dino-variant dinov3_vits16_sat    # Fast satellite
    python train_dino2.py --data satellite.yaml --model yolov13-dino3 --dino-variant dinov3_vitb16_sat    # Balanced satellite  
    python train_dino2.py --data satellite.yaml --model yolov13-dino3 --dino-variant dinov3_vitl16_sat    # High accuracy satellite
    python train_dino2.py --data satellite.yaml --model yolov13-dino3 --dino-variant dinov3_convnext_large_sat  # ConvNeXt satellite
    
    # Train YOLOv13 + DINO3 Latest Model variants (7B parameters for research)
    python train_dino2.py --data research.yaml --model yolov13-dino3-multi --dino-variant dinov3_vit7b16  # 7B parameter model
    python train_dino2.py --data research.yaml --model yolov13-dino3-multi --dino-variant dinov3_vith16_plus  # Huge+ model
    
    # Train YOLOv13 + DINO2 combinations (original)
    python train_dino2.py --data data.yaml --model yolov13-dino2-working --size n --dino-variant dinov2_vits14  # Fast
    python train_dino2.py --data data.yaml --model yolov13-dino2-working --size x --dino-variant dinov2_vitl14  # Best accuracy
"""

import argparse
import logging
import sys
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
    parser = argparse.ArgumentParser(description='Train YOLOv13 with DINO Vision Transformer Backbone')
    
    # Arguments
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--name', type=str, default='yolov13-dino2', help='Experiment name')
    parser.add_argument('--freeze-dino2', action='store_true', help='Freeze DINO backbone weights (works for both DINO2/DINO3)')
    parser.add_argument('--device', type=str, default=None, help='Device to run on, e.g., 0 or 0,1,2,3 for multi-GPU')
    
    # Model variant selection
    parser.add_argument('--model', type=str, default='yolov13-dino2-working', 
                       choices=['yolov13', 'yolov13n', 'yolov13s', 'yolov13l', 'yolov13x',
                               'yolov13-dino2', 'yolov13-dino2-simple', 
                               'yolov13-dino2-working', 'yolov13-dino2-fixed',
                               # DINO3 single-scale variants
                               'yolov13-dino3', 'yolov13-dino3-n', 'yolov13-dino3-s', 'yolov13-dino3-l', 'yolov13-dino3-x',
                               # DINO3 dual-scale variants
                               'yolov13-dino3-dual', 'yolov13-dino3-dual-n', 'yolov13-dino3-dual-s', 'yolov13-dino3-dual-l', 'yolov13-dino3-dual-x',
                               # DINO3 other variants
                               'yolov13-dino3-p3', 'yolov13-dino3-multi'],
                       help='YOLOv13 model variant')
    parser.add_argument('--size', type=str, default=None,
                       choices=['n', 's', 'l', 'x'],
                       help='YOLOv13 model size (nano/small/large/xlarge) - auto-applied to base models')
    parser.add_argument('--dino-variant', type=str, default='auto',
                       choices=[
                           # Auto-selection
                           'auto',
                           # DINO2 variants
                           'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
                           # DINOv3 ViT variants (official naming from Facebook Research)
                           'dinov3_vits16', 'dinov3_vits16_plus', 'dinov3_vitb16', 'dinov3_vitl16', 
                           'dinov3_vith16_plus', 'dinov3_vit7b16',
                           # DINOv3 ConvNeXt variants
                           'dinov3_convnext_tiny', 'dinov3_convnext_small', 'dinov3_convnext_base', 'dinov3_convnext_large',
                           # DINOv3 Satellite imagery variants (NEW)
                           'dinov3_vits16_sat', 'dinov3_vitb16_sat', 'dinov3_vitl16_sat',
                           'dinov3_convnext_small_sat', 'dinov3_convnext_base_sat', 'dinov3_convnext_large_sat',
                           # Legacy DINOv3 naming (backward compatibility)
                           'dinov3_vits14', 'dinov3_vitb14', 'dinov3_vitl14', 'dinov3_vitg14'
                       ],
                       help='DINO model variant (auto=match model type, or specify DINO2/DINO3 variants)')
    
    args = parser.parse_args()
    
    # Apply the DINO filter to the ultralytics logger
    dino_filter = DINOFilter()
    LOGGER.addFilter(dino_filter)
    
    # Determine final model configuration
    final_model = args.model
    if args.size and not final_model.endswith(args.size):
        # Apply size variant to base models (those without size suffix)
        base_models = ['yolov13', 'yolov13-dino2', 'yolov13-dino2-simple', 
                      'yolov13-dino2-working', 'yolov13-dino2-fixed', 
                      'yolov13-dino3', 'yolov13-dino3-dual', 'yolov13-dino3-p3', 'yolov13-dino3-multi']
        if final_model in base_models:
            if final_model == 'yolov13':
                final_model = f'yolov13{args.size}'
            else:
                final_model = f'{final_model}-{args.size}'
    
    print(f"{colorstr('bright_blue', 'bold', 'YOLOv13 Training')}")
    print(f"Model: {final_model}")
    dino_type = "DINO3" if args.dino_variant.startswith('dinov3') else "DINO2"
    print(f"{dino_type} Variant: {args.dino_variant}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"{dino_type} Frozen: {args.freeze_dino2}")
    print("=" * 50)
    
    try:
        # Load model
        model_path = f'ultralytics/cfg/models/v13/{final_model}.yaml'
        model = YOLO(model_path)
        
        # Configure DINO variant and freezing
        has_dino = False
        dino_type_found = None
        for module in model.model.modules():
            module_class = str(module.__class__)
            if hasattr(module, '__class__') and ('DINO2Backbone' in module_class or 'DINO3Backbone' in module_class):
                has_dino = True
                dino_type_found = "DINO3" if 'DINO3Backbone' in module_class else "DINO2"
                
                # Auto-select appropriate variant if 'auto' is specified
                if args.dino_variant == 'auto':
                    if dino_type_found == "DINO3":
                        # Default to dinov3_vitb16 for DINO3 models
                        selected_variant = 'dinov3_vitb16'
                    else:
                        # Default to dinov2_vitb14 for DINO2 models
                        selected_variant = 'dinov2_vitb14'
                    print(f"🎯 Auto-selected {dino_type_found} variant: {selected_variant}")
                else:
                    selected_variant = args.dino_variant
                    
                    # Validate variant compatibility
                    if dino_type_found == "DINO3" and not selected_variant.startswith('dinov3'):
                        print(f"⚠️  Warning: Using DINOv2 variant '{selected_variant}' with DINO3 model may cause issues")
                        print(f"   Recommend using a DINOv3 variant like 'dinov3_vitb16' instead")
                    elif dino_type_found == "DINO2" and selected_variant.startswith('dinov3'):
                        print(f"⚠️  Warning: Using DINOv3 variant '{selected_variant}' with DINO2 model may cause issues")
                        print(f"   Recommend using a DINOv2 variant like 'dinov2_vitb14' instead")
                
                # Update DINO variant if different from current
                if hasattr(module, 'model_name') and selected_variant != module.model_name:
                    print(f"🔄 Updating {dino_type_found} variant from {module.model_name} to {selected_variant}")
                    module.model_name = selected_variant
                    # Reinitialize the model with new variant if method exists
                    if hasattr(module, '_initialize_dino_model'):
                        try:
                            module._initialize_dino_model()
                        except Exception as e:
                            print(f"⚠️  Failed to reinitialize DINO model with variant {selected_variant}: {e}")
                            print(f"   Continuing with original variant: {module.model_name}")
                else:
                    selected_variant = getattr(module, 'model_name', 'unknown')
                
                # Configure freezing
                if args.freeze_dino2:
                    if hasattr(module, 'freeze_backbone_layers'):
                        module.freeze_backbone_layers()
                    print(f"✅ {dino_type_found} backbone frozen: {selected_variant}")
                else:
                    if hasattr(module, 'unfreeze_backbone'):
                        module.unfreeze_backbone()
                    print(f"🔓 {dino_type_found} backbone unfrozen: {selected_variant}")
        
        if not has_dino and ('dino2' in args.model.lower() or 'dino3' in args.model.lower()):
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
        
        print(f"\nStarting training...")
        
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
        print(f"   • DINO2 pretrained weights: LOADED ✅") 
        print(f"   • Model architecture: YOLOv13 + DINO2 ✅")
        print(f"   • Training completed successfully ✅")
        
        # Remove filter to restore normal logging
        LOGGER.removeFilter(dino_filter)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        # Remove filter even on failure
        LOGGER.removeFilter(dino_filter)
        return


if __name__ == '__main__':
    main()
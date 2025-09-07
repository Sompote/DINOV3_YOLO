#!/usr/bin/env python3
"""
YOLOv13 with DINO Vision Transformer Training Script - Systematic Architecture
  
This script provides systematic training for YOLOv13 with DINO vision transformers using:
- Consistent naming: yolov13{n/s/m/l/x}-dino{2/3}-{variant}-{integration}
- All YOLOv13 sizes: Nano, Small, Medium, Large, Extra Large  
- DINO versions: DINO2 and DINO3 with all variants
- Integration types: Single-scale and Dual-scale

New Systematic Usage:
    # Base YOLOv13 models (no DINO)
    python train_yolo_dino.py --data data.yaml --yolo-size n --epochs 100                    # yolov13n
    python train_yolo_dino.py --data data.yaml --yolo-size s --epochs 100                    # yolov13s
    python train_yolo_dino.py --data data.yaml --yolo-size m --epochs 100                    # yolov13m
    python train_yolo_dino.py --data data.yaml --yolo-size l --epochs 100                    # yolov13l  
    python train_yolo_dino.py --data data.yaml --yolo-size x --epochs 100                    # yolov13x

    # DINO2 enhanced models  
    python train_yolo_dino.py --data data.yaml --yolo-size s --dino-version 2 --dino-variant vitb14 --integration single --epochs 100
    python train_yolo_dino.py --data data.yaml --yolo-size l --dino-version 2 --dino-variant vitl14 --integration dual --epochs 100

    # DINO3 enhanced models (latest)
    python train_yolo_dino.py --data data.yaml --yolo-size s --dino-version 3 --dino-variant vitb16 --integration single --epochs 100  
    python train_yolo_dino.py --data data.yaml --yolo-size l --dino-version 3 --dino-variant vitl16 --integration dual --epochs 100

    # ConvNeXt hybrid models
    python train_yolo_dino.py --data data.yaml --yolo-size m --dino-version 3 --dino-variant convnext_base --integration single --epochs 100

    # Satellite specialized models
    python train_yolo_dino.py --data satellite.yaml --yolo-size l --dino-version 3 --dino-variant vitb16_sat --integration dual --epochs 150
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


def get_model_path(yolo_size, dino_version=None, dino_variant=None, integration=None):
    """
    Generate systematic model path based on components.
    
    Args:
        yolo_size: YOLOv13 size (n/s/m/l/x)
        dino_version: DINO version (2/3) or None for base models
        dino_variant: DINO variant (vitb16, convnext_base, etc.)
        integration: Integration type (single/dual) or None
        
    Returns:
        Path to model YAML file
    """
    base_dir = Path("ultralytics/cfg/models/v13")
    
    if dino_version is None:
        # Base YOLOv13 model
        return base_dir / f"yolov13{yolo_size}.yaml"
    
    # DINO-enhanced model
    if integration is None:
        integration = "single"  # default
        
    model_name = f"yolov13{yolo_size}-dino{dino_version}-{dino_variant}-{integration}.yaml"
    model_path = base_dir / model_name
    
    # Check if systematic model exists, fallback to legacy naming if needed
    if not model_path.exists():
        # Try legacy naming patterns for backward compatibility
        legacy_patterns = [
            f"yolov13-dino{dino_version}-{yolo_size}.yaml",
            f"yolov13-dino{dino_version}-working-{yolo_size}.yaml", 
            f"yolov13-dino{dino_version}.yaml",
            f"yolov13-dino{dino_version}-working.yaml"
        ]
        
        for pattern in legacy_patterns:
            legacy_path = base_dir / pattern
            if legacy_path.exists():
                print(f"⚠️  Using legacy model: {pattern}")
                return legacy_path
    
    return model_path


def get_dino_variant_name(version, variant):
    """Convert simplified variant names to full DINO variant names."""
    
    if version == 2:
        variant_map = {
            'vits14': 'dinov2_vits14',
            'vitb14': 'dinov2_vitb14', 
            'vitl14': 'dinov2_vitl14',
            'vitg14': 'dinov2_vitg14'
        }
    elif version == 3:
        variant_map = {
            'vits16': 'dinov3_vits16',
            'vits16_plus': 'dinov3_vits16_plus',
            'vitb16': 'dinov3_vitb16',
            'vitl16': 'dinov3_vitl16', 
            'vith16_plus': 'dinov3_vith16_plus',
            'vit7b16': 'dinov3_vit7b16',
            'convnext_tiny': 'dinov3_convnext_tiny',
            'convnext_small': 'dinov3_convnext_small',
            'convnext_base': 'dinov3_convnext_base',
            'convnext_large': 'dinov3_convnext_large',
            'vits16_sat': 'dinov3_vits16_sat',
            'vitb16_sat': 'dinov3_vitb16_sat',
            'vitl16_sat': 'dinov3_vitl16_sat',
            'convnext_small_sat': 'dinov3_convnext_small_sat',
            'convnext_base_sat': 'dinov3_convnext_base_sat', 
            'convnext_large_sat': 'dinov3_convnext_large_sat'
        }
    else:
        return variant
        
    return variant_map.get(variant, f"dinov{version}_{variant}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv13 + DINO Vision Transformer Training - Systematic Architecture')
    
    # Core arguments
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--name', type=str, default=None, help='Experiment name (auto-generated if not specified)')
    parser.add_argument('--device', type=str, default=None, help='Device to run on, e.g., 0 or 0,1,2,3 for multi-GPU')
    
    # Systematic model selection
    parser.add_argument('--yolo-size', type=str, required=True,
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv13 size: n(ano), s(mall), m(edium), l(arge), x(tra large)')
    
    parser.add_argument('--dino-version', type=int, default=None,
                       choices=[2, 3],
                       help='DINO version: 2 (DINO2) or 3 (DINO3). Omit for base YOLOv13.')
    
    parser.add_argument('--dino-variant', type=str, default='vitb16',
                       choices=[
                           # DINO2 variants (simplified names)
                           'vits14', 'vitb14', 'vitl14', 'vitg14',
                           # DINO3 ViT variants  
                           'vits16', 'vits16_plus', 'vitb16', 'vitl16', 'vith16_plus', 'vit7b16',
                           # DINO3 ConvNeXt variants
                           'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
                           # DINO3 Satellite variants
                           'vits16_sat', 'vitb16_sat', 'vitl16_sat', 
                           'convnext_small_sat', 'convnext_base_sat', 'convnext_large_sat'
                       ],
                       help='DINO variant (simplified name, auto-prefixed with dinov2_/dinov3_)')
    
    parser.add_argument('--integration', type=str, default='single',
                       choices=['single', 'dual'],
                       help='DINO integration type: single-scale (P4 only) or dual-scale (P3+P4)')
    
    # Training options
    parser.add_argument('--freeze-dino', action='store_true', 
                       help='Freeze DINO backbone weights (works for both DINO2/DINO3)')
    
    # Legacy compatibility (hidden from help)
    parser.add_argument('--legacy-model', type=str, default=None,
                       help=argparse.SUPPRESS)  # Hidden legacy option
    
    args = parser.parse_args()
    
    # Apply the DINO filter to the ultralytics logger
    dino_filter = DINOFilter()
    LOGGER.addFilter(dino_filter)
    
    # Generate model configuration
    if args.legacy_model:
        # Support for legacy model names
        model_path = f"ultralytics/cfg/models/v13/{args.legacy_model}.yaml"
        model_name = args.legacy_model
    else:
        # New systematic naming
        model_path = get_model_path(args.yolo_size, args.dino_version, 
                                   args.dino_variant, args.integration)
        
        # Generate systematic model name for display
        if args.dino_version is None:
            model_name = f"yolov13{args.yolo_size}"
        else:
            model_name = f"yolov13{args.yolo_size}-dino{args.dino_version}-{args.dino_variant}-{args.integration}"
    
    # Auto-generate experiment name if not provided
    if args.name is None:
        args.name = model_name.replace('.yaml', '')
    
    # Display configuration
    print(f"{colorstr('bright_blue', 'bold', '🚀 YOLOv13 + DINO Systematic Training')}")
    print(f"Model: {model_name}")
    print(f"YOLOv13 Size: {args.yolo_size.upper()} ({'Nano' if args.yolo_size=='n' else 'Small' if args.yolo_size=='s' else 'Medium' if args.yolo_size=='m' else 'Large' if args.yolo_size=='l' else 'Extra Large'})")
    
    if args.dino_version:
        full_variant = get_dino_variant_name(args.dino_version, args.dino_variant)
        print(f"DINO Version: DINO{args.dino_version}")
        print(f"DINO Variant: {full_variant}")
        print(f"Integration: {args.integration.title()}-scale ({'P4 only' if args.integration=='single' else 'P3+P4'})")
        print(f"DINO Frozen: {args.freeze_dino}")
    else:
        print(f"DINO Enhancement: None (Base YOLOv13)")
        
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Model Path: {model_path}")
    print("=" * 80)
    
    try:
        # Check if model file exists
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            print(f"   Please ensure the systematic model file exists or create it.")
            print(f"   For legacy models, use --legacy-model option.")
            sys.exit(1)
            
        # Load model
        print(f"📥 Loading model: {model_name}")
        model = YOLO(str(model_path))
        print(f"✅ Model loaded successfully")
        
        # Configure DINO variant and freezing (if applicable)
        if args.dino_version:
            has_dino = False
            dino_type_found = None
            full_dino_variant = get_dino_variant_name(args.dino_version, args.dino_variant)
            
            for module in model.model.modules():
                module_class = str(module.__class__)
                if hasattr(module, '__class__') and ('DINO2Backbone' in module_class or 'DINO3Backbone' in module_class):
                    has_dino = True
                    dino_type_found = "DINO3" if 'DINO3Backbone' in module_class else "DINO2"
                    
                    # Update DINO variant if different from current
                    current_variant = getattr(module, 'model_name', 'unknown')
                    if full_dino_variant != current_variant:
                        print(f"🔄 Updating {dino_type_found} variant from {current_variant} to {full_dino_variant}")
                        module.model_name = full_dino_variant
                        # Reinitialize if method exists
                        if hasattr(module, '_initialize_dino_model'):
                            try:
                                module._initialize_dino_model()
                            except Exception as e:
                                print(f"⚠️  Failed to reinitialize DINO model: {e}")
                    
                    # Configure freezing
                    if args.freeze_dino:
                        if hasattr(module, 'freeze_backbone_layers'):
                            module.freeze_backbone_layers()
                        print(f"✅ {dino_type_found} backbone frozen: {full_dino_variant}")
                    else:
                        if hasattr(module, 'unfreeze_backbone'):
                            module.unfreeze_backbone()
                        print(f"🔓 {dino_type_found} backbone unfrozen: {full_dino_variant}")
            
            if not has_dino:
                print(f"⚠️  Warning: No DINO modules found in model, but DINO version specified")
        
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
        
        print(f"\\n🏁 Starting training...")
        
        # Train with filtered logging
        results = model.train(**train_args)
        
        print(f"\\n{colorstr('bright_green', 'bold', '✅ Training Completed!')}")
        print(f"Best weights: {results.save_dir}/weights/best.pt")
        
        # Show final metrics
        if hasattr(results, 'metrics') and hasattr(results.metrics, 'box'):
            metrics = results.metrics.box
            if hasattr(metrics, 'map50'):
                print(f"Final mAP50: {metrics.map50:.4f}")
            if hasattr(metrics, 'map'):
                print(f"Final mAP50-95: {metrics.map:.4f}")
        
        print(f"\\nTraining Summary:")
        if args.dino_version:
            print(f"   • DINO{args.dino_version} enhanced YOLOv13{args.yolo_size.upper()}: LOADED ✅") 
            print(f"   • Integration: {args.integration}-scale ✅")
            print(f"   • Variant: {full_dino_variant} ✅")
        else:
            print(f"   • Base YOLOv13{args.yolo_size.upper()}: LOADED ✅")
        print(f"   • Training completed successfully ✅")
        
        # Remove filter to restore normal logging
        LOGGER.removeFilter(dino_filter)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        # Remove filter even on failure
        LOGGER.removeFilter(dino_filter)
        sys.exit(1)


if __name__ == '__main__':
    main()
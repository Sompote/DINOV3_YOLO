#!/usr/bin/env python3
"""
YOLOv13 with DINO Vision Transformer Backbone Inference Script

This script performs inference using trained YOLOv13-DINO models (DINO2/DINO3) for object detection.
Supports single images, batch processing, video files, and webcam inference.
Now supports systematic architecture with 125+ model combinations!

Developed by: Artificial Intelligence Research Group
Department: Civil Engineering
Institution: King Mongkut's University of Technology Thonburi (KMUTT)

Usage Examples:
    # Systematic architecture models (RECOMMENDED)
    python dino_inference.py --weights yolov13s-dino3-vitb16-single-best.pt --source image.jpg
    python dino_inference.py --weights yolov13l-dino3-vitl16-dual-best.pt --source images/
    python dino_inference.py --weights yolov13m-dino3-vitb16_sat-single-best.pt --source satellite_images/
    python dino_inference.py --weights yolov13l-dino3-convnext_base-single-best.pt --source video.mp4
    
    # Legacy DINO3 model inference (still supported)
    python dino_inference.py --weights yolov13-dino3-best.pt --source image.jpg
    python dino_inference.py --weights yolov13-dino3-dual-best.pt --source images/
    python dino_inference.py --weights yolov13-dino3-multi-best.pt --source video.mp4
    
    # Legacy DINO2 model inference (still supported)
    python dino_inference.py --weights yolov13-dino2-best.pt --source image.jpg
    
    # Batch image processing
    python dino_inference.py --weights best.pt --source images/
    
    # Video inference
    python dino_inference.py --weights best.pt --source video.mp4
    
    # Webcam inference
    python dino_inference.py --weights best.pt --source 0
    
    # High confidence threshold for precision
    python dino_inference.py --weights best.pt --source test.jpg --conf 0.7
    
    # Save results with custom name
    python dino_inference.py --weights best.pt --source test.jpg --name custom_results
    
    # Satellite imagery specialist with advanced settings
    python dino_inference.py --weights yolov13m-dino3-vitb16_sat-single-best.pt --source satellite.jpg --conf 0.6 --save --save-txt
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr
import cv2
import numpy as np


class DINOFilter(logging.Filter):
    """Custom logging filter to suppress DINO warnings during inference."""
    
    def filter(self, record):
        """Filter out DINO-specific warnings."""
        if hasattr(record, 'getMessage'):
            message = record.getMessage()
            
            # Filter out DINO2/DINO3 warnings during inference
            if ("setting 'requires_grad=True' for frozen layer" in message and 
                ("dino_model" in message)):
                return False
                
        return True


def validate_source(source):
    """Validate and process the input source."""
    if source.isdigit():
        # Webcam source
        return int(source)
    
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    
    return str(source_path)


def print_results_summary(results, source_type, total_time, num_items):
    """Print a summary of inference results."""
    print(f"\n{colorstr('bright_green', 'bold', '🎯 Inference Summary')}")
    print(f"Source Type: {source_type}")
    print(f"Total Items Processed: {num_items}")
    print(f"Total Time: {total_time:.2f}s")
    if num_items > 0:
        print(f"Average Time per Item: {total_time/num_items:.3f}s")
    
    if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
        total_detections = sum(len(r.boxes) for r in results if r.boxes is not None)
        print(f"Total Detections: {total_detections}")
        
        # Show class distribution
        if total_detections > 0:
            class_counts = {}
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        if hasattr(box, 'cls'):
                            cls_id = int(box.cls.item())
                            class_name = result.names.get(cls_id, f"class_{cls_id}")
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print(f"\n{colorstr('bright_blue', 'Class Distribution:')}")
            for class_name, count in sorted(class_counts.items()):
                print(f"  • {class_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv13 + DINO Vision Transformer Inference')
    
    # Core arguments
    parser.add_argument('--weights', type=str, required=True, 
                       help='Path to trained model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Source: image file, directory, video file, or webcam (0)')
    
    # Inference settings
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (0.0-1.0)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--max-det', type=int, default=1000,
                       help='Maximum number of detections per image')
    
    # Output settings
    parser.add_argument('--name', type=str, default='dino_inference',
                       help='Experiment name for saving results')
    parser.add_argument('--save', action='store_true',
                       help='Save inference results')
    parser.add_argument('--save-txt', action='store_true',
                       help='Save results in YOLO format (.txt)')
    parser.add_argument('--save-conf', action='store_true',
                       help='Save confidence scores in labels')
    parser.add_argument('--save-crop', action='store_true',
                       help='Save cropped detection images')
    parser.add_argument('--nosave', action='store_true',
                       help='Do not save images/videos')
    parser.add_argument('--show', action='store_true',
                       help='Display results')
    
    # Device settings
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on: cpu, 0, 1, 2, 3, etc.')
    parser.add_argument('--half', action='store_true',
                       help='Use FP16 half-precision inference')
    
    # Advanced settings
    parser.add_argument('--agnostic-nms', action='store_true',
                       help='Class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                       help='Augmented inference (TTA)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize model features')
    parser.add_argument('--line-thickness', type=int, default=3,
                       help='Bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', action='store_true',
                       help='Hide labels in output')
    parser.add_argument('--hide-conf', action='store_true',
                       help='Hide confidence scores in output')
    
    args = parser.parse_args()
    
    # Apply DINO filter to suppress warnings
    dino_filter = DINOFilter()
    LOGGER.addFilter(dino_filter)
    
    print(f"{colorstr('bright_blue', 'bold', '🔍 YOLOv13 + DINO Vision Transformer Inference')}")
    print(f"Weights: {args.weights}")
    print(f"Source: {args.source}")
    print(f"Confidence Threshold: {args.conf}")
    print(f"IoU Threshold: {args.iou}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print("=" * 60)
    
    try:
        # Validate inputs
        if not Path(args.weights).exists():
            raise FileNotFoundError(f"Weights file not found: {args.weights}")
        
        source = validate_source(args.source)
        
        # Load model
        print(f"📥 Loading model from: {args.weights}")
        model = YOLO(args.weights)
        print(f"✅ Model loaded successfully")
        
        # Check if model has DINO components (DINO2 or DINO3)
        has_dino = False
        dino_type_found = None
        dino_variant = 'unknown'
        dino_count = 0
        model_architecture = 'Standard YOLOv13'
        
        for module in model.model.modules():
            module_class = str(module.__class__)
            if hasattr(module, '__class__') and ('DINO2Backbone' in module_class or 'DINO3Backbone' in module_class):
                has_dino = True
                dino_count += 1
                dino_type_found = "DINO3" if 'DINO3Backbone' in module_class else "DINO2"
                dino_variant = getattr(module, 'model_name', 'unknown')
        
        # Detect systematic model architecture from weight file name
        weight_file = Path(args.weights).stem
        if 'dino' in weight_file.lower():
            # Try to parse systematic naming
            if '-dino2-' in weight_file or '-dino3-' in weight_file:
                parts = weight_file.split('-')
                if len(parts) >= 4:
                    try:
                        yolo_size = parts[0].replace('yolov13', '')
                        dino_version = parts[1].replace('dino', '')
                        variant = parts[2]
                        integration = parts[3] if len(parts) > 3 else 'unknown'
                        
                        model_architecture = f"Systematic YOLOv13{yolo_size.upper()}-DINO{dino_version}"
                        if 'sat' in variant:
                            model_architecture += f"-Satellite ({variant})"
                        elif 'convnext' in variant:
                            model_architecture += f"-ConvNeXt ({variant})"
                        else:
                            model_architecture += f"-ViT ({variant})"
                        model_architecture += f" [{integration}-scale]"
                    except:
                        model_architecture = f"Legacy {dino_type_found} Enhanced"
            else:
                model_architecture = f"Legacy {dino_type_found} Enhanced"
        
        if has_dino:
            print(f"🔬 {model_architecture}")
            if dino_count > 1:
                print(f"   Multi-scale architecture: {dino_count} DINO backbones detected")
                print(f"   DINO variant: {dino_variant}")
                
                # Determine architecture type based on DINO count
                if dino_count == 2:
                    print(f"   Integration: Dual-scale enhancement (P3+P4)")
                elif dino_count == 3:
                    print(f"   Integration: Triple-scale enhancement (P3+P4+P5)")
                else:
                    print(f"   Integration: Custom multi-scale configuration")
            else:
                print(f"   DINO variant: {dino_variant}")
                print(f"   Integration: Single-scale enhancement (P4)")
                
            # Special architecture detection
            if 'sat' in dino_variant.lower():
                print(f"   🛰️ Satellite imagery specialist detected")
            elif 'convnext' in dino_variant.lower():
                print(f"   🧠 ConvNeXt hybrid architecture detected")
        else:
            print(f"ℹ️  {model_architecture} (no DINO enhancement detected)")
        
        # Determine source type
        if isinstance(source, int):
            source_type = "Webcam"
            num_items = "N/A (live)"
        elif Path(source).is_file():
            if Path(source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                source_type = "Single Image"
                num_items = 1
            else:
                source_type = "Video File"
                # Try to get video frame count
                try:
                    cap = cv2.VideoCapture(source)
                    num_items = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                except:
                    num_items = "Unknown"
        else:
            source_type = "Image Directory"
            # Count images in directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            num_items = len([f for f in Path(source).glob('*') 
                           if f.suffix.lower() in image_extensions])
        
        print(f"🎯 Processing {source_type}: {num_items} items")
        
        # Configure inference parameters
        inference_args = {
            'source': source,
            'conf': args.conf,
            'iou': args.iou,
            'imgsz': args.imgsz,
            'max_det': args.max_det,
            'save': not args.nosave and (args.save or not args.show),
            'save_txt': args.save_txt,
            'save_conf': args.save_conf,
            'save_crop': args.save_crop,
            'show': args.show,
            'name': args.name,
            'half': args.half,
            'agnostic_nms': args.agnostic_nms,
            'augment': args.augment,
            'visualize': args.visualize,
            'line_width': args.line_thickness,
            'hide_labels': args.hide_labels,
            'hide_conf': args.hide_conf,
            'verbose': True
        }
        
        # Add device if specified
        if args.device is not None:
            inference_args['device'] = args.device
        
        # Run inference
        print(f"\n🚀 Starting inference...")
        start_time = time.time()
        
        results = model.predict(**inference_args)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print results summary
        actual_num_items = len(results) if isinstance(results, list) else 1
        print_results_summary(results, source_type, total_time, actual_num_items)
        
        # Save location info
        if not args.nosave and (args.save or not args.show):
            save_dir = Path(f"runs/detect/{args.name}")
            print(f"\n💾 Results saved to: {save_dir}")
            
            if args.save_txt:
                print(f"📄 Labels saved in YOLO format")
            if args.save_crop:
                print(f"✂️  Cropped detections saved")
        
        print(f"\n{colorstr('bright_green', 'bold', '✅ Inference completed successfully!')}")
        
        # Remove filter
        LOGGER.removeFilter(dino_filter)
        
    except KeyboardInterrupt:
        print(f"\n{colorstr('yellow', '⚠️  Inference interrupted by user')}")
        LOGGER.removeFilter(dino_filter)
        
    except Exception as e:
        print(f"\n{colorstr('red', 'bold', '❌ Inference failed:')}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        LOGGER.removeFilter(dino_filter)
        sys.exit(1)


if __name__ == '__main__':
    main()

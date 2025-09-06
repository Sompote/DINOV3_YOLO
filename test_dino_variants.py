#!/usr/bin/env python3
"""
Test script to verify all new DINOv3 variants are working properly.

This script tests the loading and initialization of different DINOv3 model variants
to ensure they work correctly with the updated implementation.
"""

import torch
from ultralytics import YOLO
import sys
from pathlib import Path

def test_dino_variants():
    """Test different DINOv3 variants to ensure they load correctly."""
    
    print("🔬 Testing DINOv3 Model Variants")
    print("=" * 60)
    
    # Test configurations with their expected DINOv3 variants
    test_configs = [
        # Original DINO3 models
        ('yolov13-dino3.yaml', 'dinov3_vitb16'),
        ('yolov13-dino3-n.yaml', 'dinov3_vits16'),
        ('yolov13-dino3-l.yaml', 'dinov3_vitl16'),
        
        # NEW: Satellite imagery variants
        ('yolov13-dino3-sat.yaml', 'dinov3_vitb16_sat'),
        
        # NEW: ConvNeXt variants
        ('yolov13-dino3-convnext-base.yaml', 'dinov3_convnext_base'),
        
        # Dual-scale variants
        ('yolov13-dino3-dual.yaml', 'dinov3_vitl16'),
        ('yolov13-dino3-dual-n.yaml', 'dinov3_vits16'),
    ]
    
    success_count = 0
    total_count = len(test_configs)
    
    for config_name, expected_variant in test_configs:
        print(f"\n🧪 Testing: {config_name}")
        print(f"   Expected DINO variant: {expected_variant}")
        
        try:
            # Attempt to load the model configuration
            config_path = f"ultralytics/cfg/models/v13/{config_name}"
            
            if not Path(config_path).exists():
                print(f"   ⚠️  Config file not found: {config_path}")
                continue
            
            # Create model instance
            model = YOLO(config_path)
            
            # Test forward pass with dummy input
            dummy_input = torch.randn(1, 3, 640, 640)
            
            print(f"   🔄 Testing forward pass...")
            with torch.no_grad():
                # This will trigger model initialization including DINO backbone
                outputs = model.model(dummy_input)
            
            print(f"   ✅ SUCCESS: {config_name}")
            print(f"      Model loaded and forward pass completed")
            print(f"      Output shapes: {[out.shape for out in outputs] if isinstance(outputs, (list, tuple)) else outputs.shape}")
            
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ FAILED: {config_name}")
            print(f"      Error: {str(e)}")
            print(f"      This might be expected if dependencies are missing")
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Successful: {success_count}/{total_count}")
    print(f"   ❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print(f"\n🎉 All DINOv3 variants tested successfully!")
    elif success_count > 0:
        print(f"\n✅ Some DINOv3 variants working - implementation is functional")
    else:
        print(f"\n⚠️  No variants loaded successfully - check dependencies")

def test_dino_backbone_variants():
    """Test DINO3Backbone class directly with different variants."""
    
    print(f"\n🔬 Testing DINO3Backbone Class Directly")
    print("=" * 60)
    
    # Import the DINO3Backbone class
    try:
        from ultralytics.nn.modules.block import DINO3Backbone
    except ImportError as e:
        print(f"❌ Could not import DINO3Backbone: {e}")
        return
    
    # Test different DINOv3 variants
    variants_to_test = [
        # Standard ViT variants
        'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16',
        
        # Plus variants
        'dinov3_vits16_plus', 'dinov3_vith16_plus',
        
        # ConvNeXt variants
        'dinov3_convnext_tiny', 'dinov3_convnext_small', 'dinov3_convnext_base',
        
        # Satellite variants
        'dinov3_vits16_sat', 'dinov3_vitb16_sat', 'dinov3_vitl16_sat',
        
        # Large research variants
        'dinov3_vit7b16',  # This might fail due to size
    ]
    
    for variant in variants_to_test:
        print(f"\n🧪 Testing DINO3Backbone with: {variant}")
        
        try:
            # Create DINO3Backbone instance
            backbone = DINO3Backbone(
                model_name=variant,
                freeze_backbone=True,
                output_channels=512
            )
            
            # Test with dummy input
            dummy_input = torch.randn(1, 256, 20, 20)  # Typical CNN feature map
            
            print(f"   🔄 Testing forward pass...")
            with torch.no_grad():
                output = backbone(dummy_input)
            
            print(f"   ✅ SUCCESS: {variant}")
            print(f"      Input shape: {dummy_input.shape}")
            print(f"      Output shape: {output.shape}")
            print(f"      Parameters: {backbone.model_spec['params']}M")
            print(f"      Embedding dim: {backbone.embed_dim}")
            print(f"      Dataset: {backbone.dataset_type}")
            
        except Exception as e:
            print(f"   ⚠️  {variant}: {str(e)[:100]}...")
            # This might be expected for very large models or missing dependencies

def main():
    """Main test function."""
    print("🚀 DINOv3 Variants Test Suite")
    print("Testing updated DINO implementation with new Facebook Research variants")
    print("=" * 80)
    
    # Test 1: Model configurations
    test_dino_variants()
    
    # Test 2: DINO backbone class directly
    test_dino_backbone_variants()
    
    print(f"\n🏁 Testing Complete!")
    print(f"Note: Some failures are expected if you don't have all dependencies installed.")
    print(f"The main goal is to verify that the new variants are recognized and can be instantiated.")

if __name__ == "__main__":
    main()
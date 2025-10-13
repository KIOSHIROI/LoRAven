#!/usr/bin/env python3
"""
Core functionality test without any visualization dependencies
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

def test_basic_imports():
    """Test basic imports"""
    print("=== Testing Basic Imports ===")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except Exception as e:
        print(f"✗ PyTorch failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except Exception as e:
        print(f"✗ NumPy failed: {e}")
        return False
    
    return True

def test_method_imports():
    """Test method class imports"""
    print("\n=== Testing Method Imports ===")
    
    try:
        from standard_task.methods.base_method import BaseMethod
        print("✓ BaseMethod imported")
    except Exception as e:
        print(f"✗ BaseMethod failed: {e}")
        return False
    
    try:
        from standard_task.methods.full_rank import FullRankMethod
        print("✓ FullRankMethod imported")
    except Exception as e:
        print(f"✗ FullRankMethod failed: {e}")
        return False
    
    try:
        from standard_task.methods.lora import LoRAMethod
        print("✓ LoRAMethod imported")
    except Exception as e:
        print(f"✗ LoRAMethod failed: {e}")
        return False
    
    return True

def test_method_creation():
    """Test method instantiation"""
    print("\n=== Testing Method Creation ===")
    
    try:
        from standard_task.methods.full_rank import FullRankMethod
        method = FullRankMethod()
        print("✓ FullRankMethod created")
    except Exception as e:
        print(f"✗ FullRankMethod creation failed: {e}")
        return False
    
    try:
        from standard_task.methods.lora import LoRAMethod
        method = LoRAMethod(rank=8, alpha=16, dropout=0.1)
        print("✓ LoRAMethod created")
    except Exception as e:
        print(f"✗ LoRAMethod creation failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation"""
    print("\n=== Testing Model Creation ===")
    
    try:
        from standard_task.methods.full_rank import FullRankMethod
        
        method = FullRankMethod()
        
        # Test vision model
        task_config = {
            'type': 'image_classification',
            'num_classes': 10,
            'input_shape': [3, 32, 32]
        }
        
        model = method.create_model('image_classification', task_config)
        print(f"✓ Vision model created: {type(model).__name__}")
        
        # Test with sample input
        sample_input = torch.randn(1, 3, 32, 32)
        output = model(sample_input)
        print(f"✓ Model forward pass: {output.shape}")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    return True

def test_synthetic_data():
    """Test synthetic data generation"""
    print("\n=== Testing Synthetic Data ===")
    
    try:
        from standard_task.data.dataset_loader import get_dataloader
        
        dataloader = get_dataloader(
            'CIFAR-10',
            'image_classification',
            batch_size=4,
            synthetic=True
        )
        
        batch = next(iter(dataloader))
        inputs, targets = batch
        print(f"✓ Synthetic data: inputs {inputs.shape}, targets {targets.shape}")
        
    except Exception as e:
        print(f"✗ Synthetic data failed: {e}")
        return False
    
    return True

def test_performance_metrics():
    """Test performance metrics"""
    print("\n=== Testing Performance Metrics ===")
    
    try:
        from standard_task.utils.metrics import PerformanceMetrics
        
        metrics = PerformanceMetrics()
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # Test metrics
        model_size = metrics.get_model_size(model)
        print(f"✓ Model size: {model_size:.2f} MB")
        
        sample_input = torch.randn(1, 100)
        inference_time = metrics.measure_inference_time(model, sample_input)
        print(f"✓ Inference time: {inference_time*1000:.2f} ms")
        
    except Exception as e:
        print(f"✗ Performance metrics failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("Core Functionality Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_method_imports,
        test_method_creation,
        test_model_creation,
        test_synthetic_data,
        test_performance_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print(f"\n=== Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Core functionality is working.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
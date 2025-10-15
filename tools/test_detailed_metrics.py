"""
详细指标系统测试文件
测试新增的详细指标收集器的功能和准确性

测试内容：
1. 测试MetricsCollector的基本功能
2. 测试各个子模块的指标收集
3. 测试FLOPs计算的准确性
4. 测试动态性分析功能
5. 测试收敛性分析功能
6. 测试可解释性分析功能
7. 测试可视化功能
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import json
from typing import Dict, Any

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入测试模块
from tools.metrics_collector import MetricsCollector, DetailedMetrics
from tools.flops_calculator import FLOPsCalculator
from tools.dynamic_analyzer import DynamicAnalyzer
from tools.convergence_analyzer import ConvergenceAnalyzer
from tools.interpretability_analyzer import InterpretabilityAnalyzer


class SimpleTestModel(nn.Module):
    """简单的测试模型"""
    def __init__(self, input_size=768, hidden_size=256, num_classes=2):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class LoRATestModel(nn.Module):
    """带有LoRA层的测试模型"""
    def __init__(self, input_size=768, hidden_size=256, num_classes=2, rank=8):
        super().__init__()
        self.base_linear = nn.Linear(input_size, hidden_size)
        
        # 模拟LoRA层，添加r属性以便InterpretabilityAnalyzer识别
        self.lora_A = nn.Linear(input_size, rank, bias=False)
        self.lora_B = nn.Linear(rank, hidden_size, bias=False)
        self.lora_A.r = rank  # 添加rank属性
        self.lora_B.r = rank  # 添加rank属性
        
        # 模拟PEFT结构，避免循环引用
        class DefaultWrapper:
            def __init__(self, module):
                self.weight = module.weight
                
        self.lora_A.default = DefaultWrapper(self.lora_A)
        self.lora_B.default = DefaultWrapper(self.lora_B)
        
        self.scaling = 1.0
        
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # 初始化LoRA权重
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        # 基础变换
        base_out = self.base_linear(x)
        
        # LoRA变换
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        
        # 合并输出
        x = base_out + lora_out
        x = self.relu(x)
        x = self.classifier(x)
        return x


def test_flops_calculator():
    """测试FLOPs计算器"""
    print("Testing FLOPs Calculator...")
    
    calculator = FLOPsCalculator()
    model = SimpleTestModel()
    
    # 测试基本FLOPs计算
    input_shape = (32, 768)  # batch_size=32, input_size=768
    flops_info = calculator.calculate_model_flops(model, input_shape)
    
    print(f"Total FLOPs: {flops_info['total_flops']:,}")
    
    # 验证FLOPs计算结果
    assert 'total_flops' in flops_info
    assert 'total_flops_m' in flops_info
    assert 'total_flops_g' in flops_info
    assert 'flops_per_param' in flops_info
    assert flops_info['total_flops'] > 0
    
    print("✓ FLOPs Calculator test passed")
    return True


def test_dynamic_analyzer():
    """测试动态性分析器"""
    print("Testing Dynamic Analyzer...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = DynamicAnalyzer()
        model = LoRATestModel()
        
        # 测试rank动态分析
        dynamics = analyzer.analyze_rank_dynamics(model, "lora")
        
        print(f"Rank statistics: {dynamics.get('rank_statistics', {})}")
        print(f"Rank changes: {dynamics.get('rank_changes', {})}")
        
        # 验证结果
        assert 'rank_statistics' in dynamics, "Should have rank statistics"
        assert 'layer_wise_ranks' in dynamics
        assert 'method' in dynamics
        
        print("✓ Dynamic Analyzer test passed")
    return True


def test_convergence_analyzer():
    """测试收敛性分析器"""
    print("Testing Convergence Analyzer...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = ConvergenceAnalyzer()
        
        # 创建模拟训练历史
        training_history = [
            {'eval_accuracy': 0.5, 'train_loss': 1.0},
            {'eval_accuracy': 0.6, 'train_loss': 0.8},
            {'eval_accuracy': 0.7, 'train_loss': 0.6},
            {'eval_accuracy': 0.75, 'train_loss': 0.5},
            {'eval_accuracy': 0.8, 'train_loss': 0.4},
            {'eval_accuracy': 0.82, 'train_loss': 0.35},
            {'eval_accuracy': 0.83, 'train_loss': 0.32},
            {'eval_accuracy': 0.84, 'train_loss': 0.30},
            {'eval_accuracy': 0.84, 'train_loss': 0.29},
            {'eval_accuracy': 0.85, 'train_loss': 0.28}
        ]
        
        # 测试收敛分析
        convergence = analyzer.analyze_convergence_stability(training_history, 'eval_accuracy')
        
        print(f"Convergence metrics: {convergence.get('convergence_metrics', {})}")
        print(f"Stability metrics: {convergence.get('stability_metrics', {})}")
        
        # 验证结果
        assert 'convergence_metrics' in convergence, "Should have convergence metrics"
        assert 'stability_metrics' in convergence, "Should have stability metrics"
        
        print("✓ Convergence Analyzer test passed")
    return True


def test_interpretability_analyzer():
    """测试可解释性分析器"""
    print("Testing Interpretability Analyzer...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = InterpretabilityAnalyzer(temp_dir)
        model = LoRATestModel()
        
        # 测试可解释性分析
        interpretability = analyzer.analyze_interpretability(model, "lora", "test_task")
        
        print(f"Rank distribution: {interpretability.get('rank_distribution', {})}")
        print(f"Gate activations: {interpretability.get('gate_activations', {})}")
        
        # 验证结果
        assert 'rank_distribution' in interpretability, "Should have rank distribution"
        
        print("✓ Interpretability Analyzer test passed")
    return True


def test_metrics_collector():
    """测试指标收集器"""
    print("Testing Metrics Collector...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        collector = MetricsCollector(temp_dir)
        model = LoRATestModel()
        
        # 创建模拟数据
        sample_input = torch.randn(32, 768)
        training_history = [
            {'eval_accuracy': 0.5, 'train_loss': 1.0},
            {'eval_accuracy': 0.6, 'train_loss': 0.8},
            {'eval_accuracy': 0.7, 'train_loss': 0.6},
            {'eval_accuracy': 0.8, 'train_loss': 0.4},
            {'eval_accuracy': 0.85, 'train_loss': 0.3}
        ]
        eval_results = {'eval_accuracy': 0.85, 'train_accuracy': 0.90}
        
        # 测试综合指标收集
        detailed_metrics = collector.generate_comprehensive_report(
            model=model,
            method="lora",
            task="test_task",
            sample_input=sample_input,
            training_history=training_history,
            eval_results=eval_results
        )
        
        # 验证结果结构
        assert isinstance(detailed_metrics, DetailedMetrics), "Should return DetailedMetrics object"
        assert detailed_metrics.task_performance is not None, "Should have task performance"
        assert detailed_metrics.parameter_efficiency is not None, "Should have parameter efficiency"
        assert detailed_metrics.resource_consumption is not None, "Should have resource consumption"
        
        print(f"Task performance: {detailed_metrics.task_performance}")
        print(f"Parameter efficiency: {detailed_metrics.parameter_efficiency}")
        print(f"Resource consumption keys: {list(detailed_metrics.resource_consumption.keys())}")
        
        print("✓ Metrics Collector test passed")
    return True


def test_visualization_integration():
    """测试可视化集成"""
    print("Testing Visualization Integration...")
    
    try:
        from tools.benchmarks.visualization import BenchmarkVisualizer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = BenchmarkVisualizer(Path(temp_dir))
            
            # 创建模拟详细指标数据
            detailed_metrics = {
                'task_performance': {'eval_accuracy': 0.85, 'train_accuracy': 0.90},
                'parameter_efficiency': {'total_parameters': 1000000, 'trainable_parameters': 50000},
                'resource_consumption': {'flops_m': 100.5, 'memory_usage_mb': 512, 'inference_latency_ms': 25.3},
                'dynamic_metrics': {'rank_statistics': {'mean_rank': 8.5, 'std_rank': 1.2, 'max_rank': 16, 'min_rank': 4}},
                'convergence_stability': {'convergence_step_90': 50, 'convergence_step_95': 80, 'convergence_speed': 0.02},
                'interpretability': {'rank_distribution': {'nominal_rank': 16, 'effective_rank': 12.3}},
                'generalization': {'model_complexity': 0.05, 'parameter_efficiency': 17.0, 'generalization_score': 0.81}
            }
            
            # 测试详细指标可视化
            viz_path = visualizer.visualize_detailed_metrics(detailed_metrics)
            
            # 验证文件是否创建
            assert viz_path.exists(), f"Visualization file should be created at {viz_path}"
            
            print(f"✓ Visualization created at: {viz_path}")
            print("✓ Visualization Integration test passed")
        
        return True
        
    except ImportError as e:
        print(f"⚠ Visualization test skipped due to missing dependencies: {e}")
        return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("Running Detailed Metrics System Tests")
    print("=" * 50)
    
    tests = [
        test_flops_calculator,
        test_dynamic_analyzer,
        test_convergence_analyzer,
        test_interpretability_analyzer,
        test_metrics_collector,
        test_visualization_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n{'-' * 30}")
            if test():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
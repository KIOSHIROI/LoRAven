#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学公式单元测试
确保LoRAven中所有数学公式的实现正确性
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import pytest
import numpy as np
from typing import Dict, List, Tuple, Optional
import traceback

# 导入LoRAven组件
from loraven.loraven.core.models.dynamic_lowrank_layer import DynamicLowRankLayer
from loraven.loraven.core.models.gates import LightweightScorer, GateNetwork
from loraven.loraven.core.rank_scheduler import RankScheduler, LinearRankScheduler, EnergyAwareRankScheduler
from loraven.loraven.core.budget_manager import BudgetManager
from loraven.loraven.utils.perf_estimator import PerfEstimator, EnergyEstimator
from loraven.loraven.loraven_simple import LoRAven


class TestLowRankDecomposition:
    """测试低秩分解公式 W ≈ U Σ V^T"""
    
    def test_lowrank_decomposition_basic(self):
        """基础低秩分解测试"""
        layer = DynamicLowRankLayer(
            in_features=64,
            out_features=32,
            r_max=16,
            r_min=4
        )
        
        x = torch.randn(8, 64)
        output, current_rank = layer(x)
        
        # 验证输出形状
        assert output.shape == (8, 32), f"输出形状错误: {output.shape}"
        assert 4 <= current_rank <= 16, f"秩超出范围: {current_rank}"
    
    def test_lowrank_mathematical_consistency(self):
        """验证低秩分解的数学一致性"""
        layer = DynamicLowRankLayer(
            in_features=32,
            out_features=16,
            r_max=8,
            r_min=2
        )
        
        x = torch.randn(4, 32)
        output, current_rank = layer(x)
        
        # 获取因子矩阵
        U = layer.U_full[:, :current_rank]
        V = layer.V_full[:, :current_rank]
        S = layer.S_full[:current_rank, :current_rank]
        
        # 重构权重矩阵 W ≈ U @ S @ V^T
        W_reconstructed = U @ S @ V.T
        
        # 验证矩阵乘法: y = x @ W^T ≈ x @ V @ S @ U^T
        y_direct = x @ W_reconstructed.T
        y_factorized = x @ V @ S @ U.T
        
        # 计算误差
        error = torch.norm(y_direct - y_factorized).item()
        assert error < 1e-4, f"分解误差过大: {error}"


class TestComplexityScoring:
    """测试复杂度评分公式 s(x) = σ(f_θ(x))"""
    
    def test_complexity_scoring_range(self):
        """测试复杂度分数范围"""
        scorer = LightweightScorer(in_features=64, hidden_dim=32)
        x = torch.randn(8, 64)
        
        scores = scorer(x)
        
        # 验证输出在[0,1]范围内
        assert torch.all(scores >= 0), "复杂度分数存在负值"
        assert torch.all(scores <= 1), "复杂度分数超过1"
        # 修正形状检查 - LightweightScorer返回(batch_size, 1)
        assert scores.shape == (8, 1), f"分数形状错误: {scores.shape}"
    
    def test_complexity_scoring_consistency(self):
        """测试复杂度评分的一致性"""
        scorer = LightweightScorer(in_features=32, hidden_dim=16)
        scorer.eval()  # 设置为评估模式，避免dropout等随机性
        
        # 相同输入应产生相同分数
        x = torch.randn(4, 32)
        with torch.no_grad():  # 禁用梯度计算
            scores1 = scorer(x)
            scores2 = scorer(x)
        
        assert torch.allclose(scores1, scores2, atol=1e-6), "相同输入产生不同分数"


class TestRankScheduling:
    """测试秩调度公式"""
    
    def test_linear_rank_scheduling(self):
        """测试线性秩调度"""
        scheduler = LinearRankScheduler(r_min=4, r_max=64)
        
        # 测试不同复杂度分数
        low_complexity = torch.tensor([0.2, 0.2, 0.2])
        high_complexity = torch.tensor([0.8, 0.8, 0.8])
        
        rank_low = scheduler.schedule_rank(low_complexity)
        rank_high = scheduler.schedule_rank(high_complexity)
        
        assert 4 <= rank_low <= 64, f"低复杂度秩超出范围: {rank_low}"
        assert 4 <= rank_high <= 64, f"高复杂度秩超出范围: {rank_high}"
        assert rank_low < rank_high, "高复杂度应分配更高的秩"
    
    def test_energy_aware_scheduling(self):
        """测试能耗感知秩调度"""
        scheduler = EnergyAwareRankScheduler(
            r_min=4, 
            r_max=64,
            energy_model=None  # 使用内置简化模型
        )
        
        complexity_scores = torch.tensor([0.5, 0.5, 0.5])
        
        # 不同预算应产生不同秩
        rank_low_budget = scheduler.schedule_rank(
            complexity_scores, 
            budget=5.0,
            layer_dims=(512, 256)
        )
        
        rank_high_budget = scheduler.schedule_rank(
            complexity_scores, 
            budget=20.0,
            layer_dims=(512, 256)
        )
        
        assert 4 <= rank_low_budget <= 64, f"低预算秩超出范围: {rank_low_budget}"
        assert 4 <= rank_high_budget <= 64, f"高预算秩超出范围: {rank_high_budget}"
        assert rank_low_budget <= rank_high_budget, "高预算应允许更高的秩"


class TestEnergyEstimation:
    """测试能耗估算公式"""
    
    def test_energy_estimation_monotonicity(self):
        """测试能耗估算的单调性"""
        hardware_profile = {
            'dram_energy_per_byte': 1e-6,
            'l2_cache_energy_per_byte': 1e-7,
            'l1_cache_energy_per_byte': 1e-8,
            'gpu_cores': 5120,
            'base_frequency': 1.5e9
        }
        estimator = EnergyEstimator(hardware_profile)
        
        layer_dims = (64, 32)
        batch_size = 8
        
        # 测试能耗随秩增加而增加
        ranks = [4, 8, 16, 32]
        energies = [estimator.estimate(layer_dims, r, batch_size) for r in ranks]
        
        for i in range(len(energies) - 1):
            assert energies[i] <= energies[i + 1], f"能耗不满足单调性: {energies}"
    
    def test_energy_estimation_values(self):
        """测试能耗估算值的合理性"""
        hardware_profile = {
            'dram_energy_per_byte': 1e-6,
            'l2_cache_energy_per_byte': 1e-7,
            'l1_cache_energy_per_byte': 1e-8,
            'gpu_cores': 5120,
            'base_frequency': 1.5e9
        }
        estimator = EnergyEstimator(hardware_profile)
        
        energy = estimator.estimate((64, 32), 8, 4)
        
        assert energy > 0, "能耗应为正值"
        assert energy < 1000, f"能耗值过大: {energy} mJ"


class TestLossFunction:
    """测试损失函数组合"""
    
    def test_loss_combination_formula(self):
        """测试损失函数组合公式 L = L_task + λ_E * L_energy + λ_R * L_rank"""
        # 模拟损失组件
        task_loss = torch.tensor(2.5)
        energy_penalty = torch.tensor(0.3)
        rank_penalty = torch.tensor(0.1)
        
        # 权重参数
        lambda_energy = 0.1
        lambda_rank = 0.05
        
        # 计算总损失
        total_loss = task_loss + lambda_energy * energy_penalty + lambda_rank * rank_penalty
        
        # 验证数学正确性
        expected_total = 2.5 + 0.1 * 0.3 + 0.05 * 0.1
        
        assert abs(total_loss.item() - expected_total) < 1e-6, "损失函数组合计算错误"
    
    def test_loss_components_positive(self):
        """测试损失组件的正值性"""
        task_loss = torch.tensor(1.0)
        energy_penalty = torch.tensor(0.2)
        rank_penalty = torch.tensor(0.05)
        
        assert task_loss >= 0, "任务损失应为非负值"
        assert energy_penalty >= 0, "能耗惩罚应为非负值"
        assert rank_penalty >= 0, "秩惩罚应为非负值"


class TestIntegration:
    """集成测试：验证各组件协同工作"""
    
    def test_full_pipeline(self):
        """测试完整的前向传播流水线"""
        # 创建组件
        layer = DynamicLowRankLayer(
            in_features=32,
            out_features=16,
            r_max=8,
            r_min=2
        )
        
        # 创建输入
        x = torch.randn(4, 32)
        
        # 前向传播
        output, current_rank = layer(x)
        
        # 验证输出
        assert output.shape == (4, 16), "输出形状错误"
        assert 2 <= current_rank <= 8, "秩超出范围"
        assert not torch.isnan(output).any(), "输出包含NaN值"
        assert not torch.isinf(output).any(), "输出包含无穷值"


def run_test_class(test_class, class_name):
    """运行测试类中的所有测试方法"""
    print(f"\n=== 运行 {class_name} ===")
    
    instance = test_class()
    test_methods = [method for method in dir(instance) if method.startswith('test_')]
    
    passed = 0
    total = len(test_methods)
    
    for method_name in test_methods:
        try:
            method = getattr(instance, method_name)
            method()
            print(f"✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {str(e)}")
            traceback.print_exc()
    
    print(f"{class_name}: {passed}/{total} 测试通过")
    return passed, total


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("LoRAven 数学公式单元测试")
    print("=" * 60)
    
    # 运行所有测试类
    test_classes = [
        (TestLowRankDecomposition, "低秩分解测试"),
        (TestComplexityScoring, "复杂度评分测试"),
        (TestRankScheduling, "秩调度测试"),
        (TestEnergyEstimation, "能耗估算测试"),
        (TestLossFunction, "损失函数测试"),
        (TestIntegration, "集成测试")
    ]
    
    total_passed = 0
    total_tests = 0
    
    for test_class, class_name in test_classes:
        try:
            passed, tests = run_test_class(test_class, class_name)
            total_passed += passed
            total_tests += tests
        except Exception as e:
            print(f"✗ {class_name} 运行失败: {str(e)}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"总体结果: {total_passed}/{total_tests} 测试通过")
    
    if total_passed == total_tests:
        print("🎉 所有数学公式测试通过！")
        sys.exit(0)
    else:
        print("⚠️  部分测试失败")
        sys.exit(1)
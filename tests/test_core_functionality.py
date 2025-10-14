#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心功能单元测试
测试LoRAven的核心组件和功能
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import pytest
import numpy as np
from typing import Dict, List, Tuple, Optional

# 导入LoRAven组件
from loraven.loraven.core.models.dynamic_lowrank_layer import DynamicLowRankLayer
from loraven.loraven.core.models.gates import LightweightScorer, GateNetwork
from loraven.loraven.core.rank_scheduler import RankScheduler, LinearRankScheduler, EnergyAwareRankScheduler
from loraven.loraven.core.budget_manager import BudgetManager
from loraven.loraven.utils.perf_estimator import PerfEstimator, EnergyEstimator
from loraven.loraven.loraven_simple import LoRAven


class TestDynamicLowRankLayer:
    """测试动态低秩层"""
    
    def test_layer_initialization(self):
        """测试层初始化"""
        layer = DynamicLowRankLayer(
            in_features=64,
            out_features=32,
            r_max=16,
            r_min=4
        )
        
        assert layer.in_features == 64
        assert layer.out_features == 32
        assert layer.r_max == 16
        assert layer.r_min == 4
        assert layer.r_curr >= 4
        assert layer.r_curr <= 16
    
    def test_forward_pass(self):
        """测试前向传播"""
        layer = DynamicLowRankLayer(
            in_features=32,
            out_features=16,
            r_max=8,
            r_min=2
        )
        
        x = torch.randn(4, 32)
        output, rank = layer(x)
        
        assert output.shape == (4, 16)
        assert isinstance(rank, int)
        assert 2 <= rank <= 8
    
    def test_rank_adaptation(self):
        """测试秩自适应"""
        layer = DynamicLowRankLayer(
            in_features=64,
            out_features=32,
            r_max=16,
            r_min=4
        )
        
        # 多次前向传播，观察秩变化
        ranks = []
        for _ in range(10):
            x = torch.randn(8, 64)
            _, rank = layer(x)
            ranks.append(rank)
        
        # 秩应在合理范围内
        assert all(4 <= r <= 16 for r in ranks)


class TestLightweightScorer:
    """测试轻量级评分器"""
    
    def test_scorer_initialization(self):
        """测试评分器初始化"""
        scorer = LightweightScorer(in_features=64, hidden_dim=32)
        
        assert scorer.in_features == 64
        assert scorer.hidden_dim == 32
    
    def test_scoring_output(self):
        """测试评分输出"""
        scorer = LightweightScorer(in_features=32, hidden_dim=16)
        
        x = torch.randn(8, 32)
        scores = scorer(x)
        
        assert scores.shape == (8, 1)  # LightweightScorer输出形状为(batch_size, 1)
        assert torch.all(scores >= 0)
        assert torch.all(scores <= 1)
    
    def test_batch_consistency(self):
        """测试批次一致性"""
        scorer = LightweightScorer(in_features=16, hidden_dim=8)
        scorer.eval()  # 设置为评估模式避免BatchNorm问题
        
        # 单个样本
        x_single = torch.randn(1, 16)
        score_single = scorer(x_single)
        
        # 批次样本
        x_batch = x_single.repeat(4, 1)
        scores_batch = scorer(x_batch)
        
        assert torch.allclose(scores_batch, score_single.repeat(4, 1), atol=1e-6)  # 修复维度问题


class TestRankSchedulers:
    """测试秩调度器"""
    
    def test_linear_scheduler(self):
        """测试线性调度器"""
        scheduler = LinearRankScheduler(r_min=4, r_max=32)
        
        # 测试不同复杂度
        low_scores = torch.tensor([0.1, 0.1, 0.1])
        high_scores = torch.tensor([0.9, 0.9, 0.9])
        
        rank_low = scheduler.schedule_rank(low_scores)
        rank_high = scheduler.schedule_rank(high_scores)
        
        assert 4 <= rank_low <= 32
        assert 4 <= rank_high <= 32
        assert rank_low < rank_high
    
    def test_energy_aware_scheduler(self):
        """测试能耗感知调度器"""
        scheduler = EnergyAwareRankScheduler(
            r_min=4,
            r_max=32,
            energy_model=None
        )
        
        scores = torch.tensor([0.5, 0.5, 0.5])
        
        rank = scheduler.schedule_rank(
            scores,
            budget=10.0,
            layer_dims=(64, 32)
        )
        
        assert 4 <= rank <= 32


class TestBudgetManager:
    """测试预算管理器"""
    
    def test_budget_allocation(self):
        """测试预算分配"""
        manager = BudgetManager(total_budget=100.0)
        
        # 分配预算
        budget = manager.allocate_budget(
            "layer1",
            complexity_score=0.5,
            layer_dims=(64, 32)
        )
        
        assert budget > 0
        assert budget <= 100.0
    
    def test_budget_tracking(self):
        """测试预算跟踪"""
        manager = BudgetManager(total_budget=50.0)
        
        # 多次分配
        budget1 = manager.allocate_budget("layer1", 0.3, (32, 16))
        budget2 = manager.allocate_budget("layer2", 0.7, (64, 32))
        
        # 检查剩余预算
        remaining = manager.get_remaining_budget()
        assert remaining >= 0
        assert remaining <= 50.0


class TestPerformanceEstimators:
    """测试性能估算器"""
    
    def test_energy_estimator(self):
        """测试能耗估算器"""
        hardware_profile = {
            'dram_energy_per_byte': 1e-6,
            'l2_cache_energy_per_byte': 1e-7,
            'l1_cache_energy_per_byte': 1e-8,
            'gpu_cores': 5120,
            'base_frequency': 1.5e9
        }
        
        estimator = EnergyEstimator(hardware_profile)
        
        energy = estimator.estimate((64, 32), 8, 4)
        
        assert energy > 0
        assert isinstance(energy, float)
    
    def test_perf_estimator(self):
        """测试性能估算器"""
        # 创建硬件配置文件
        hardware_profile = {
            'dram_energy_per_byte': 1e-6,
            'l2_cache_energy_per_byte': 1e-7,
            'l1_cache_energy_per_byte': 1e-8,
            'compute_energy_per_flop': 1e-6
        }
        # 使用具体的EnergyEstimator而不是抽象的PerfEstimator
        estimator = EnergyEstimator(hardware_profile)
        
        # 创建简单模型
        model = nn.Linear(64, 32)
        x = torch.randn(1, 64)
        
        # 估算延迟 - 使用正确的方法名和参数
        layer_dims = (64, 32)
        rank = 16
        latency = estimator.estimate(layer_dims, rank)
        
        assert latency > 0
        assert isinstance(latency, float)


class TestErrorHandling:
    """测试错误处理"""
    
    def test_invalid_dimensions(self):
        """测试无效维度处理"""
        with pytest.raises((ValueError, AssertionError)):
            DynamicLowRankLayer(
                in_features=0,  # 无效维度
                out_features=32,
                r_max=16,
                r_min=4
            )
    
    def test_invalid_rank_range(self):
        """测试无效秩范围处理"""
        with pytest.raises((ValueError, AssertionError)):
            DynamicLowRankLayer(
                in_features=64,
                out_features=32,
                r_max=4,   # r_max < r_min
                r_min=16
            )
    
    def test_nan_input_handling(self):
        """测试NaN输入处理"""
        layer = DynamicLowRankLayer(
            in_features=32,
            out_features=16,
            r_max=8,
            r_min=2
        )
        
        # 包含NaN的输入
        x = torch.randn(4, 32)
        x[0, 0] = float('nan')
        
        output, rank = layer(x)
        
        # 输出不应包含NaN
        assert not torch.isnan(output).any()


class TestMemoryEfficiency:
    """测试内存效率"""
    
    def test_memory_usage(self):
        """测试内存使用"""
        layer = DynamicLowRankLayer(
            in_features=1024,
            out_features=512,
            r_max=64,
            r_min=8
        )
        
        # 获取参数数量
        full_params = 1024 * 512  # 全连接层参数
        lowrank_params = sum(p.numel() for p in layer.parameters())
        
        # 低秩层应使用更少参数
        assert lowrank_params < full_params
    
    def test_gradient_flow(self):
        """测试梯度流"""
        layer = DynamicLowRankLayer(
            in_features=32,
            out_features=16,
            r_max=8,
            r_min=2
        )
        
        x = torch.randn(4, 32, requires_grad=True)
        output, _ = layer(x)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    pytest.main([__file__, "-v"])
"""
预算管理器：管理能耗预算和资源分配
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
from collections import deque
import time


class BudgetManager:
    """
    预算管理器：管理能耗预算和资源分配
    """
    
    def __init__(
        self,
        total_budget: float = 10.0,  # 总预算 (mJ/sample)
        window_size: int = 100,
        safety_margin: float = 0.1,  # 安全边距
        adaptation_rate: float = 0.01
    ):
        self.total_budget = total_budget
        self.window_size = window_size
        self.safety_margin = safety_margin
        self.adaptation_rate = adaptation_rate
        
        # 历史记录
        self.energy_history = deque(maxlen=window_size)
        self.performance_history = deque(maxlen=window_size)
        self.rank_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)
        
        # 当前状态
        self.current_budget = total_budget
        self.remaining_budget = total_budget
        self.budget_utilization = 0.0
        
        # 自适应参数
        self.budget_scaling_factor = 1.0
        self.performance_threshold = 0.95
        
    def allocate_budget(
        self, 
        layer_id: str, 
        complexity_score: float,
        layer_dims: Tuple[int, int],
        performance_priority: float = 1.0
    ) -> float:
        """
        为特定层分配预算
        
        Args:
            layer_id: 层标识符
            complexity_score: 复杂度分数
            layer_dims: 层维度 (in_features, out_features)
            performance_priority: 性能优先级 (0-1)
            
        Returns:
            allocated_budget: 分配的预算 (mJ/sample)
        """
        # 基于复杂度计算基础预算
        base_budget = self._calculate_base_budget(complexity_score, layer_dims)
        
        # 基于性能优先级调整
        priority_factor = 0.5 + 0.5 * performance_priority
        adjusted_budget = base_budget * priority_factor
        
        # 应用自适应缩放
        final_budget = adjusted_budget * self.budget_scaling_factor
        
        # 确保不超过剩余预算
        final_budget = min(final_budget, self.remaining_budget)
        
        # 更新剩余预算
        self.remaining_budget -= final_budget
        
        return max(0.0, final_budget)
    
    def _calculate_base_budget(
        self, 
        complexity_score: float, 
        layer_dims: Tuple[int, int]
    ) -> float:
        """
        计算基础预算
        
        Args:
            complexity_score: 复杂度分数
            layer_dims: 层维度
            
        Returns:
            base_budget: 基础预算
        """
        in_features, out_features = layer_dims
        
        # 基于层大小的基础预算
        layer_size_factor = (in_features * out_features) / (512 * 512)  # 归一化到标准层
        
        # 基于复杂度的预算
        complexity_factor = 0.5 + 0.5 * complexity_score
        
        # 基础预算计算
        base_budget = self.total_budget * layer_size_factor * complexity_factor * 0.1
        
        return base_budget
    
    def update_energy_consumption(
        self, 
        layer_id: str, 
        actual_energy: float,
        performance_metric: Optional[float] = None
    ):
        """
        更新能耗消耗记录
        
        Args:
            layer_id: 层标识符
            actual_energy: 实际能耗
            performance_metric: 性能指标
        """
        current_time = time.time()
        
        # 记录历史
        self.energy_history.append(actual_energy)
        self.timestamp_history.append(current_time)
        
        if performance_metric is not None:
            self.performance_history.append(performance_metric)
        
        # 更新预算利用率
        self.budget_utilization = sum(self.energy_history) / (len(self.energy_history) * self.total_budget)
        
        # 自适应调整
        self._adaptive_adjustment()
    
    def _adaptive_adjustment(self):
        """自适应调整预算分配策略"""
        if len(self.energy_history) < 10:
            return
        
        # 计算能耗趋势
        recent_energy = np.mean(list(self.energy_history)[-5:])
        older_energy = np.mean(list(self.energy_history)[-10:-5])
        energy_trend = recent_energy - older_energy
        
        # 计算性能趋势
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(list(self.performance_history)[-5:])
            older_performance = np.mean(list(self.performance_history)[-10:-5])
            performance_trend = recent_performance - older_performance
        else:
            performance_trend = 0.0
        
        # 根据趋势调整预算缩放因子
        if energy_trend > 0 and performance_trend < 0:
            # 能耗增加但性能下降，减少预算
            self.budget_scaling_factor = max(0.5, self.budget_scaling_factor - self.adaptation_rate)
        elif energy_trend < 0 and performance_trend > 0:
            # 能耗减少但性能提升，可以增加预算
            self.budget_scaling_factor = min(1.5, self.budget_scaling_factor + self.adaptation_rate)
    
    def get_budget_status(self) -> Dict[str, float]:
        """
        获取预算状态
        
        Returns:
            status: 预算状态字典
        """
        return {
            'total_budget': self.total_budget,
            'remaining_budget': self.remaining_budget,
            'budget_utilization': self.budget_utilization,
            'budget_scaling_factor': self.budget_scaling_factor,
            'average_energy': np.mean(self.energy_history) if self.energy_history else 0.0,
            'energy_std': np.std(self.energy_history) if self.energy_history else 0.0,
            'average_performance': np.mean(self.performance_history) if self.performance_history else 0.0
        }
    
    def reset_budget(self):
        """重置预算"""
        self.remaining_budget = self.total_budget
        self.budget_utilization = 0.0
        self.energy_history.clear()
        self.performance_history.clear()
        self.rank_history.clear()
        self.timestamp_history.clear()
    
    def set_total_budget(self, new_budget: float):
        """设置新的总预算"""
        self.total_budget = new_budget
        self.remaining_budget = new_budget
    
    def get_remaining_budget(self) -> float:
        """获取剩余预算"""
        return self.remaining_budget
    
    def get_energy_prediction(self, layer_dims: Tuple[int, int], rank: int) -> float:
        """
        预测给定层和秩的能耗
        
        Args:
            layer_dims: 层维度
            rank: 秩
            
        Returns:
            predicted_energy: 预测能耗
        """
        in_features, out_features = layer_dims
        
        # 简化的能耗预测模型
        flops = 2 * in_features * rank + 2 * rank * out_features
        memory_access = (in_features + out_features) * rank
        
        # 基于历史数据调整预测
        if self.energy_history:
            historical_factor = np.mean(self.energy_history)
            predicted_energy = (flops * 1e-6 + memory_access * 1e-7) * historical_factor
        else:
            predicted_energy = flops * 1e-6 + memory_access * 1e-7
        
        return predicted_energy


class LayerBudgetAllocator:
    """
    层预算分配器：为不同层分配预算
    """
    
    def __init__(self, budget_manager: BudgetManager):
        self.budget_manager = budget_manager
        self.layer_priorities = {}
        self.layer_allocations = {}
    
    def set_layer_priority(self, layer_id: str, priority: float):
        """
        设置层优先级
        
        Args:
            layer_id: 层标识符
            priority: 优先级 (0-1)
        """
        self.layer_priorities[layer_id] = max(0.0, min(1.0, priority))
    
    def allocate_layer_budget(
        self, 
        layer_id: str, 
        complexity_score: float,
        layer_dims: Tuple[int, int]
    ) -> float:
        """
        为层分配预算
        
        Args:
            layer_id: 层标识符
            complexity_score: 复杂度分数
            layer_dims: 层维度
            
        Returns:
            allocated_budget: 分配的预算
        """
        priority = self.layer_priorities.get(layer_id, 0.5)
        
        allocated_budget = self.budget_manager.allocate_budget(
            layer_id, complexity_score, layer_dims, priority
        )
        
        self.layer_allocations[layer_id] = allocated_budget
        return allocated_budget
    
    def get_allocation_summary(self) -> Dict[str, float]:
        """获取分配摘要"""
        return {
            'layer_allocations': dict(self.layer_allocations),
            'total_allocated': sum(self.layer_allocations.values()),
            'remaining_budget': self.budget_manager.remaining_budget,
            'layer_priorities': dict(self.layer_priorities)
        }

"""
秩调度器实现：能耗感知的秩调度策略
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod


class RankScheduler(ABC):
    """
    秩调度器基类
    """
    
    def __init__(self, r_min: int = 4, r_max: int = 128):
        self.r_min = r_min
        self.r_max = r_max
    
    @abstractmethod
    def schedule_rank(
        self, 
        complexity_scores: torch.Tensor, 
        budget: Optional[float] = None,
        **kwargs
    ) -> int:
        """
        调度秩
        
        Args:
            complexity_scores: 复杂度分数 (batch_size,)
            budget: 能耗预算 (mJ/sample)
            **kwargs: 其他参数
            
        Returns:
            target_rank: 目标秩
        """
        pass


class LinearRankScheduler(RankScheduler):
    """
    改进的线性秩调度器：基于复杂度分数的线性映射，增加多样性和自适应机制
    """
    
    def __init__(self, r_min: int = 4, r_max: int = 128, diversity_factor: float = 0.1):
        super().__init__(r_min, r_max)
        self.diversity_factor = diversity_factor
        self.rank_history = []
        self.complexity_history = []
    
    def schedule_rank(
        self, 
        complexity_scores: torch.Tensor, 
        budget: Optional[float] = None,
        current_loss: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        **kwargs
    ) -> int:
        """
        改进的线性映射：考虑复杂度分布、梯度信息和历史性能
        """
        # 计算复杂度统计
        s_avg = complexity_scores.mean().item()
        s_std = complexity_scores.std().item()
        
        # 基础线性映射
        r_base = int(round(self.r_min + s_avg * (self.r_max - self.r_min)))
        
        # 考虑复杂度分布的多样性
        diversity_bonus = int(s_std * self.diversity_factor * (self.r_max - self.r_min))
        r_base += diversity_bonus
        
        # 梯度信息调整
        if gradient_norm is not None:
            gradient_factor = min(gradient_norm / 10.0, 1.0)
            gradient_bonus = int(gradient_factor * 0.2 * (self.r_max - self.r_min))
            r_base += gradient_bonus
        
        # 强制多样性机制
        r_final = self._enforce_diversity(r_base)
        
        # 更新历史记录
        self.rank_history.append(r_final)
        self.complexity_history.append(s_avg)
        if len(self.rank_history) > 50:
            self.rank_history.pop(0)
            self.complexity_history.pop(0)
        
        return max(self.r_min, min(r_final, self.r_max))
    
    def _enforce_diversity(self, r_candidate: int) -> int:
        """
        强制多样性机制：避免秩值过于集中
        """
        if len(self.rank_history) < 10:
            return r_candidate
        
        recent_ranks = self.rank_history[-10:]
        rank_std = np.std(recent_ranks)
        
        min_diversity = (self.r_max - self.r_min) * 0.05
        if rank_std < min_diversity:
            perturbation = np.random.randint(-int(min_diversity), int(min_diversity) + 1)
            return r_candidate + perturbation
        
        return r_candidate


class EnergyAwareRankScheduler(RankScheduler):
    """
    能耗感知秩调度器：考虑能耗预算的秩调度
    """
    
    def __init__(
        self, 
        r_min: int = 4, 
        r_max: int = 128,
        energy_model: Optional[Any] = None,
        alpha: float = 1.0,
        beta: float = 0.1
    ):
        super().__init__(r_min, r_max)
        self.energy_model = energy_model
        self.alpha = alpha  # 能耗权重
        self.beta = beta    # 性能权重
    
    def schedule_rank(
        self, 
        complexity_scores: torch.Tensor, 
        budget: Optional[float] = None,
        layer_dims: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> int:
        """
        能耗感知的秩调度
        
        Args:
            complexity_scores: 复杂度分数 (batch_size,)
            budget: 能耗预算 (mJ/sample)
            layer_dims: 层维度 (in_features, out_features)
            **kwargs: 其他参数
            
        Returns:
            target_rank: 目标秩
        """
        # 1. 基于复杂度的初始秩
        s_avg = complexity_scores.mean().item()
        r_complexity = int(round(self.r_min + s_avg * (self.r_max - self.r_min)))
        
        # 2. 如果没有能耗预算，直接返回基于复杂度的秩
        if budget is None:
            return max(self.r_min, min(r_complexity, self.r_max))
        
        # 3. 能耗约束调整
        r_energy = self._energy_constrained_rank(r_complexity, budget, layer_dims)
        
        # 4. 平衡复杂度和能耗
        r_final = self._balance_complexity_energy(
            r_complexity, r_energy, s_avg, budget, layer_dims
        )
        
        return max(self.r_min, min(r_final, self.r_max))
    
    def _energy_constrained_rank(
        self, 
        r_initial: int, 
        budget: float, 
        layer_dims: Optional[Tuple[int, int]] = None
    ) -> int:
        """
        能耗约束的秩调整
        
        Args:
            r_initial: 初始秩
            budget: 能耗预算
            layer_dims: 层维度
            
        Returns:
            energy_constrained_rank: 能耗约束下的秩
        """
        r = r_initial
        
        # 从高到低搜索满足能耗预算的最大秩
        while r >= self.r_min:
            estimated_energy = self._estimate_energy(r, layer_dims)
            if estimated_energy <= budget:
                break
            r -= 1
        
        return max(r, self.r_min)
    
    def _balance_complexity_energy(
        self,
        r_complexity: int,
        r_energy: int,
        complexity_score: float,
        budget: float,
        layer_dims: Optional[Tuple[int, int]] = None
    ) -> int:
        """
        平衡复杂度和能耗的秩选择
        
        Args:
            r_complexity: 基于复杂度的秩
            r_energy: 基于能耗的秩
            complexity_score: 复杂度分数
            budget: 能耗预算
            layer_dims: 层维度
            
        Returns:
            balanced_rank: 平衡后的秩
        """
        # 如果能耗约束很严格，优先满足能耗
        if r_energy < r_complexity * 0.5:
            return r_energy
        
        # 如果复杂度很高，优先满足性能
        if complexity_score > 0.8:
            return min(r_complexity, r_energy)
        
        # 否则在两者之间平衡
        return int(round(self.alpha * r_energy + self.beta * r_complexity))
    
    def _estimate_energy(
        self, 
        r: int, 
        layer_dims: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        估算给定秩的能耗
        
        Args:
            r: 秩
            layer_dims: 层维度 (in_features, out_features)
            
        Returns:
            energy: 估算能耗 (mJ/sample)
        """
        if layer_dims is None:
            # 使用默认维度估算
            in_features, out_features = 512, 512
        else:
            in_features, out_features = layer_dims
        
        # 计算 FLOPs
        flops = 2 * in_features * r + 2 * r * out_features
        
        # 计算内存访问
        memory_access = (in_features + out_features) * r
        
        # 简化的能耗模型
        # 每 FLOP 消耗 1e-6 mJ，每内存访问消耗 1e-7 mJ
        energy = flops * 1e-6 + memory_access * 1e-7
        
        # 如果有自定义能耗模型，使用它
        if self.energy_model is not None:
            energy = self.energy_model.estimate_energy(r, in_features, out_features)
        
        return energy


class AdaptiveRankScheduler(RankScheduler):
    """
    自适应秩调度器：根据历史性能动态调整调度策略
    """
    
    def __init__(
        self, 
        r_min: int = 4, 
        r_max: int = 128,
        learning_rate: float = 0.01,
        window_size: int = 100
    ):
        super().__init__(r_min, r_max)
        self.learning_rate = learning_rate
        self.window_size = window_size
        
        # 历史记录
        self.performance_history = []
        self.rank_history = []
        self.energy_history = []
        
        # 自适应参数
        self.alpha = 1.0  # 复杂度权重
        self.beta = 0.1   # 能耗权重
        self.gamma = 0.5  # 历史性能权重
    
    def schedule_rank(
        self, 
        complexity_scores: torch.Tensor, 
        budget: Optional[float] = None,
        current_performance: Optional[float] = None,
        **kwargs
    ) -> int:
        """
        自适应秩调度
        
        Args:
            complexity_scores: 复杂度分数
            budget: 能耗预算
            current_performance: 当前性能指标
            **kwargs: 其他参数
            
        Returns:
            target_rank: 目标秩
        """
        # 更新历史记录
        if current_performance is not None:
            self._update_history(current_performance, budget)
        
        # 基于复杂度的秩
        s_avg = complexity_scores.mean().item()
        r_complexity = int(round(self.r_min + s_avg * (self.r_max - self.r_min)))
        
        # 能耗约束
        if budget is not None:
            r_energy = self._energy_constrained_rank(r_complexity, budget)
        else:
            r_energy = r_complexity
        
        # 自适应调整
        r_adaptive = self._adaptive_adjustment(r_complexity, r_energy, s_avg)
        
        return max(self.r_min, min(r_adaptive, self.r_max))
    
    def _update_history(
        self, 
        performance: float, 
        energy: Optional[float]
    ):
        """更新历史记录"""
        self.performance_history.append(performance)
        if energy is not None:
            self.energy_history.append(energy)
        
        # 保持窗口大小
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
        if len(self.energy_history) > self.window_size:
            self.energy_history.pop(0)
    
    def _adaptive_adjustment(
        self, 
        r_complexity: int, 
        r_energy: int, 
        complexity_score: float
    ) -> int:
        """自适应调整策略"""
        if len(self.performance_history) < 10:
            # 历史数据不足，使用简单策略
            return min(r_complexity, r_energy)
        
        # 分析历史性能趋势
        recent_performance = np.mean(self.performance_history[-10:])
        older_performance = np.mean(self.performance_history[-20:-10]) if len(self.performance_history) >= 20 else recent_performance
        
        performance_trend = recent_performance - older_performance
        
        # 根据性能趋势调整权重
        if performance_trend > 0:
            # 性能提升，可以增加复杂度权重
            self.alpha = min(1.5, self.alpha + self.learning_rate)
        else:
            # 性能下降，减少复杂度权重
            self.alpha = max(0.5, self.alpha - self.learning_rate)
        
        # 平衡复杂度和能耗
        r_balanced = int(round(
            self.alpha * r_complexity + self.beta * r_energy
        ))
        
        return r_balanced
    
    def _energy_constrained_rank(self, r_initial: int, budget: float) -> int:
        """能耗约束的秩调整"""
        r = r_initial
        while r >= self.r_min:
            # 简化的能耗估算
            estimated_energy = r * 0.01  # 简化模型
            if estimated_energy <= budget:
                break
            r -= 1
        return max(r, self.r_min)
    
    def get_adaptive_params(self) -> Dict[str, float]:
        """获取当前自适应参数"""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'performance_trend': np.mean(self.performance_history[-10:]) - np.mean(self.performance_history[-20:-10]) if len(self.performance_history) >= 20 else 0.0
        }


def create_rank_scheduler(
    scheduler_type: str = 'linear',
    **kwargs
) -> RankScheduler:
    """
    创建秩调度器工厂函数
    
    Args:
        scheduler_type: 调度器类型 ('linear', 'energy_aware', 'adaptive')
        **kwargs: 调度器参数
        
    Returns:
        scheduler: 秩调度器实例
    """
    if scheduler_type == 'linear':
        return LinearRankScheduler(**kwargs)
    elif scheduler_type == 'energy_aware':
        return EnergyAwareRankScheduler(**kwargs)
    elif scheduler_type == 'adaptive':
        return AdaptiveRankScheduler(**kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
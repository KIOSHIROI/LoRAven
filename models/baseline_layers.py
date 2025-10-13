"""
对照组基线层实现
用于消融实验的LoRAven变体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Union, List
from .gates import LightweightScorer, GateNetwork
from .dynamic_lowrank_layer import DynamicLowRankLayer


class LoRAvenFixedLayer(DynamicLowRankLayer):
    """
    LoRAven-fixed: 禁用动态更新的固定秩版本
    用于验证动态秩机制的有效性
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r_max: int,
        r_min: int = 4,
        init_rank: Optional[int] = None,
        bias: bool = True,
        scorer_hidden: int = 32,
        num_gate_blocks: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            r_max=r_max,
            r_min=r_min,
            init_rank=init_rank,
            bias=bias,
            scorer_hidden=scorer_hidden,
            num_gate_blocks=num_gate_blocks,
            device=device,
            dtype=dtype
        )
        
        # 固定秩为初始值
        self.fixed_rank = init_rank or r_min
        self.r_curr = self.fixed_rank
        
        # 禁用动态更新
        self.enable_dynamic_update = False
        
    def forward(
        self, 
        x: torch.Tensor, 
        budget: Optional[float] = None,
        mode: str = 'inference'
    ) -> Tuple[torch.Tensor, int]:
        """
        前向传播 - 固定秩版本
        """
        batch_size = x.size(0)
        
        # 0. 强化输入NaN处理
        x_processed = self._preprocess_input(x)
        
        # 1. 使用固定秩，不进行动态调度
        r_target = self.fixed_rank
        
        # 2. 门控决策（可选）
        gate_mask = self.gate(x_processed)  # (batch_size, num_blocks)
        
        # 3. 切片因子矩阵
        U = self.U_full[:, :r_target]  # (out_features, r_target)
        V = self.V_full[:, :r_target]  # (in_features, r_target)
        S = self.S_full[:r_target, :r_target]  # (r_target, r_target)
        
        # 4. 奇异值监控（保持监控功能）
        self.batch_counter += 1
        if self.batch_counter % self.monitor_frequency == 0:
            self._monitor_singular_values(U, S, V)
        
        # 5. 计算低秩矩阵乘法
        try:
            z = torch.matmul(x_processed, V)  # (batch_size, r_target)
            if torch.isnan(z).any() or not torch.isfinite(z).all():
                z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
            
            z = torch.matmul(z, S.T)  # (batch_size, r_target)
            if torch.isnan(z).any() or not torch.isfinite(z).all():
                z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
            
            y = torch.matmul(z, U.T)  # (batch_size, out_features)
            if torch.isnan(y).any() or not torch.isfinite(y).all():
                y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
                
        except Exception as e:
            print(f"矩阵乘法出错: {e}，使用零输出")
            y = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        
        # 6. 添加偏置
        if self.bias is not None:
            y = y + self.bias
            if torch.isnan(y).any() or not torch.isfinite(y).all():
                y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
        
        # 7. 最终输出验证和修复
        y = self._postprocess_output(y)
        
        # 8. 保持固定秩
        self.r_curr = self.fixed_rank
        
        # 9. Hebbian更新（如果启用）
        if mode == 'training' and self.enable_hebbian:
            self._hebbian_update(x_processed, y)
        
        # 10. 不进行事件触发更新（固定秩）
        
        return y, r_target
    
    def extra_repr(self) -> str:
        """返回层的额外表示信息"""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'fixed_rank={self.fixed_rank}, '
                f'bias={self.bias is not None}, mode=fixed')


class LoRAvenNoHebbLayer(DynamicLowRankLayer):
    """
    LoRAven-noHebb: 禁用Hebbian更新，只保留能耗约束的版本
    用于验证Hebbian更新机制的有效性
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r_max: int,
        r_min: int = 4,
        init_rank: Optional[int] = None,
        bias: bool = True,
        scorer_hidden: int = 32,
        num_gate_blocks: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            r_max=r_max,
            r_min=r_min,
            init_rank=init_rank,
            bias=bias,
            scorer_hidden=scorer_hidden,
            num_gate_blocks=num_gate_blocks,
            device=device,
            dtype=dtype
        )
        
        # 禁用Hebbian更新
        self.enable_hebbian = False
        
    def forward(
        self, 
        x: torch.Tensor, 
        budget: Optional[float] = None,
        mode: str = 'inference'
    ) -> Tuple[torch.Tensor, int]:
        """
        前向传播 - 无Hebbian更新版本
        """
        batch_size = x.size(0)
        
        # 0. 强化输入NaN处理
        x_processed = self._preprocess_input(x)
        
        # 1. 计算样本复杂度分数
        s = self.scorer(x_processed)  # (batch_size, 1) -> (batch_size,)
        s = s.squeeze(-1) if s.dim() > 1 else s
        
        # 2. 秩调度（保持动态调度）
        r_target = self._rank_scheduler(s, budget)
        
        # 确保r_target是整数类型
        if isinstance(r_target, torch.Tensor):
            r_target = int(r_target.item())
        else:
            r_target = int(r_target)
        
        # 3. 门控决策（可选）
        gate_mask = self.gate(x_processed)  # (batch_size, num_blocks)
        
        # 4. 切片因子矩阵
        U = self.U_full[:, :r_target]  # (out_features, r_target)
        V = self.V_full[:, :r_target]  # (in_features, r_target)
        S = self.S_full[:r_target, :r_target]  # (r_target, r_target)
        
        # 5. 奇异值监控
        self.batch_counter += 1
        if self.batch_counter % self.monitor_frequency == 0:
            self._monitor_singular_values(U, S, V)
        
        # 6. 计算低秩矩阵乘法
        try:
            z = torch.matmul(x_processed, V)  # (batch_size, r_target)
            if torch.isnan(z).any() or not torch.isfinite(z).all():
                z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
            
            z = torch.matmul(z, S.T)  # (batch_size, r_target)
            if torch.isnan(z).any() or not torch.isfinite(z).all():
                z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
            
            y = torch.matmul(z, U.T)  # (batch_size, out_features)
            if torch.isnan(y).any() or not torch.isfinite(y).all():
                y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
                
        except Exception as e:
            print(f"矩阵乘法出错: {e}，使用零输出")
            y = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        
        # 7. 添加偏置
        if self.bias is not None:
            y = y + self.bias
            if torch.isnan(y).any() or not torch.isfinite(y).all():
                y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
        
        # 8. 最终输出验证和修复
        y = self._postprocess_output(y)
        
        # 9. 更新当前秩
        self.r_curr = r_target
        
        # 10. 跳过Hebbian更新（已禁用）
        
        # 11. 事件触发更新（保持动态更新）
        if mode == 'training':
            triggered = self._event_triggered_update(x_processed, y)
        
        return y, r_target
    
    def extra_repr(self) -> str:
        """返回层的额外表示信息"""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'r_min={self.r_min}, r_max={self.r_max}, r_curr={self.r_curr}, '
                f'bias={self.bias is not None}, mode=no_hebbian')


def create_baseline_layer(
    layer_type: str,
    in_features: int,
    out_features: int,
    r_max: int,
    r_min: int = 4,
    init_rank: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    创建基线层的工厂函数
    
    Args:
        layer_type: 层类型 ('fixed', 'no_hebb', 'full')
        in_features: 输入特征数
        out_features: 输出特征数
        r_max: 最大秩
        r_min: 最小秩
        init_rank: 初始秩
        **kwargs: 其他参数
        
    Returns:
        对应的层实例
    """
    if layer_type == 'fixed':
        return LoRAvenFixedLayer(
            in_features=in_features,
            out_features=out_features,
            r_max=r_max,
            r_min=r_min,
            init_rank=init_rank,
            **kwargs
        )
    elif layer_type == 'no_hebb':
        return LoRAvenNoHebbLayer(
            in_features=in_features,
            out_features=out_features,
            r_max=r_max,
            r_min=r_min,
            init_rank=init_rank,
            **kwargs
        )
    elif layer_type == 'full':
        return DynamicLowRankLayer(
            in_features=in_features,
            out_features=out_features,
            r_max=r_max,
            r_min=r_min,
            init_rank=init_rank,
            **kwargs
        )
    else:
        raise ValueError(f"未知的层类型: {layer_type}")
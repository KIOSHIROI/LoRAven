"""
基础层实现：普通全秩层基类
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class BaseLayer(nn.Module):
    """
    基础全秩层，作为 LoRAven 的基线实现
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 权重矩阵
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        # 偏置项
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)
            
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return nn.functional.linear(x, self.weight, self.bias)
    
    def extra_repr(self) -> str:
        """额外的字符串表示"""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


import math

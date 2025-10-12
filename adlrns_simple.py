"""ADLRNS 简化接口

这个模块提供了一个简化的ADLRNS接口，让用户能够像使用标准PyTorch模块一样直接使用ADLRNS。

使用示例:
    import torch
    import torch.nn as nn
    from adlrns_simple import ADLRNS
    
    # 创建ADLRNS层
    layer = ADLRNS(512, 256, mode='balanced')
    
    # 在模型中使用
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.adlrns1 = ADLRNS(784, 512, mode='high_performance')
            self.adlrns2 = ADLRNS(512, 256, mode='low_power')
            self.classifier = nn.Linear(256, 10)
        
        def forward(self, x):
            x = self.adlrns1(x)
            x = torch.relu(x)
            x = self.adlrns2(x)
            x = torch.relu(x)
            return self.classifier(x)
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any, Tuple
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.dynamic_lowrank_layer import DynamicLowRankLayer
from schedulers.budget_manager import BudgetManager
from utils.perf_estimator import PerfEstimator


class ADLRNS(nn.Module):
    """ADLRNS 简化接口
    
    这是一个简化的ADLRNS包装类，提供了类似标准PyTorch模块的接口。
    用户可以通过预设模式或简单参数快速创建和使用ADLRNS层。
    
    Args:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        mode (str): 预设模式，可选 'high_performance', 'balanced', 'low_power', 'custom'
        max_rank (int, optional): 最大秩，仅在mode='custom'时使用
        min_rank (int, optional): 最小秩，仅在mode='custom'时使用
        energy_budget (float, optional): 能耗预算(mJ)，默认自动计算
        bias (bool): 是否使用偏置，默认True
        device (torch.device, optional): 设备，默认自动选择
        
    预设模式说明:
        - 'high_performance': 高性能模式，优先准确率，能耗较高
        - 'balanced': 平衡模式，准确率和能耗的平衡
        - 'low_power': 低功耗模式，优先节能，准确率略低
        - 'custom': 自定义模式，需要手动指定参数
    """
    
    # 预设配置
    PRESETS = {
        'high_performance': {
            'rank_ratio': 0.8,  # 最大秩比例
            'min_rank_ratio': 0.3,  # 最小秩比例
            'energy_multiplier': 1.5,  # 能耗预算倍数
            'scorer_hidden': 64,  # 复杂度评分器隐藏层维度
            'gate_blocks': 16,  # 门控网络块数
        },
        'balanced': {
            'rank_ratio': 0.5,
            'min_rank_ratio': 0.2,
            'energy_multiplier': 1.0,
            'scorer_hidden': 32,
            'gate_blocks': 8,
        },
        'low_power': {
            'rank_ratio': 0.3,
            'min_rank_ratio': 0.1,
            'energy_multiplier': 0.7,
            'scorer_hidden': 16,
            'gate_blocks': 4,
        }
    }
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str = 'balanced',
        max_rank: Optional[int] = None,
        min_rank: Optional[int] = None,
        energy_budget: Optional[float] = None,
        bias: bool = True,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        # 设备设置
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # 参数验证
        if mode not in self.PRESETS and mode != 'custom':
            raise ValueError(f"模式 '{mode}' 不支持。支持的模式: {list(self.PRESETS.keys()) + ['custom']}")
        
        # 获取配置
        if mode == 'custom':
            if max_rank is None or min_rank is None:
                raise ValueError("自定义模式需要指定 max_rank 和 min_rank")
            config = {
                'scorer_hidden': 32,
                'gate_blocks': 8,
            }
        else:
            config = self.PRESETS[mode].copy()
            # 自动计算秩范围
            feature_dim = min(in_features, out_features)
            if max_rank is None:
                max_rank = max(4, int(feature_dim * config['rank_ratio']))
            if min_rank is None:
                min_rank = max(2, int(feature_dim * config['min_rank_ratio']))
        
        # 确保秩范围合理
        max_rank = min(max_rank, min(in_features, out_features))
        min_rank = max(min_rank, 2)
        if min_rank >= max_rank:
            min_rank = max(2, max_rank // 2)
        
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.min_rank = min_rank
        self.mode = mode
        
        # 创建核心ADLRNS层
        self.adlrns_layer = DynamicLowRankLayer(
            in_features=in_features,
            out_features=out_features,
            r_max=max_rank,
            r_min=min_rank,
            bias=bias,
            scorer_hidden=config['scorer_hidden'],
            num_gate_blocks=config['gate_blocks'],
            device=device
        )
        
        # 能耗预算管理
        if energy_budget is None and mode != 'custom':
            # 自动估算能耗预算
            hardware_profile = self._get_default_hardware_profile()
            perf_estimator = PerfEstimator(hardware_profile)
            baseline_perf = perf_estimator.estimate_all(
                (in_features, out_features), 
                max_rank, 
                batch_size=32
            )
            energy_budget = baseline_perf['energy_mj'] * config.get('energy_multiplier', 1.0)
        
        self.energy_budget = energy_budget
        
        # 预算管理器
        if energy_budget is not None:
            self.budget_manager = BudgetManager(total_budget=energy_budget)
        else:
            self.budget_manager = None
        
        # 移动到指定设备
        self.to(device)
    
    def _get_default_hardware_profile(self) -> Dict[str, float]:
        """获取默认硬件配置"""
        if self.device.type == 'cuda':
            # GPU配置 (基于RTX 4090)
            return {
                'gpu_cores': 16384,
                'memory_bandwidth': 1e12,  # 1TB/s
                'compute_bandwidth': 1.5e12,  # 1.5 TFLOPS
            }
        else:
            # CPU配置 (基于现代CPU)
            return {
                'gpu_cores': 16,  # CPU核心数
                'memory_bandwidth': 1e11,  # 100GB/s
                'compute_bandwidth': 1e11,  # 100 GFLOPS
            }
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """前向传播
        
        Args:
            x (torch.Tensor): 输入张量
            return_info (bool): 是否返回额外信息
            
        Returns:
            torch.Tensor: 输出张量
            或 Tuple[torch.Tensor, Dict]: 输出张量和信息字典
        """
        # 获取当前预算
        current_budget = None
        if self.budget_manager is not None:
            current_budget = self.budget_manager.get_remaining_budget()
        
        # ADLRNS前向传播
        output, current_rank = self.adlrns_layer(
            x, 
            budget=current_budget,
            mode='inference'
        )
        
        if return_info:
            # 收集信息
            info = {
                'current_rank': current_rank,
                'max_rank': self.max_rank,
                'min_rank': self.min_rank,
                'rank_utilization': current_rank / self.max_rank,
                'mode': self.mode,
            }
            
            # 添加预算信息
            if self.budget_manager is not None:
                budget_status = self.budget_manager.get_budget_status()
                info.update({
                    'energy_budget': self.energy_budget,
                    'remaining_budget': budget_status.get('remaining_budget', 0),
                    'budget_utilization': budget_status.get('utilization', 0),
                })
            
            # 添加层信息
            rank_info = self.adlrns_layer.get_rank_info()
            info.update(rank_info)
            
            return output, info
        
        return output
    
    def get_info(self) -> Dict[str, Any]:
        """获取层信息"""
        info = {
            'in_features': self.in_features,
            'out_features': self.out_features,
            'max_rank': self.max_rank,
            'min_rank': self.min_rank,
            'mode': self.mode,
            'device': str(self.device),
            'energy_budget': self.energy_budget,
        }
        
        # 添加ADLRNS层信息
        rank_info = self.adlrns_layer.get_rank_info()
        info.update(rank_info)
        
        # 添加预算信息
        if self.budget_manager is not None:
            budget_status = self.budget_manager.get_budget_status()
            info.update(budget_status)
        
        return info
    
    def set_mode(self, mode: str):
        """设置运行模式"""
        if mode not in self.PRESETS:
            raise ValueError(f"模式 '{mode}' 不支持。支持的模式: {list(self.PRESETS.keys())}")
        
        self.mode = mode
        config = self.PRESETS[mode]
        
        # 更新能耗预算
        if self.energy_budget is not None:
            hardware_profile = self._get_default_hardware_profile()
            perf_estimator = PerfEstimator(hardware_profile)
            baseline_perf = perf_estimator.estimate_all(
                (self.in_features, self.out_features), 
                self.max_rank, 
                batch_size=32
            )
            new_budget = baseline_perf['energy_mj'] * config.get('energy_multiplier', 1.0)
            
            if self.budget_manager is not None:
                self.budget_manager.total_budget = new_budget
            self.energy_budget = new_budget
    
    def reset_budget(self):
        """重置能耗预算"""
        if self.budget_manager is not None:
            self.budget_manager.reset()
    
    def get_current_rank(self) -> int:
        """获取当前秩值"""
        return self.adlrns_layer.r_curr
    
    def get_budget_usage(self) -> float:
        """获取预算使用率"""
        if self.budget_manager is not None:
            budget_status = self.budget_manager.get_budget_status()
            return budget_status.get('utilization', 0.0)
        return 0.0
    
    def get_compression_ratio(self) -> float:
        """获取压缩比率"""
        current_rank = self.get_current_rank()
        full_rank = min(self.in_features, self.out_features)
        return current_rank / full_rank if full_rank > 0 else 0.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        total_params = sum(p.numel() for p in self.parameters())
        param_memory = total_params * 4 / (1024 * 1024)  # 假设float32，转换为MB
        
        return {
            'parameter_memory_mb': param_memory,
            'total_parameters': total_params,
            'estimated_forward_memory_mb': param_memory * 2,  # 粗略估计
        }
    
    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数数量统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 计算不同秩下的有效参数数量
        current_rank = self.adlrns_layer.r_curr
        effective_params = (
            self.in_features * current_rank +  # V矩阵
            current_rank * current_rank +      # S矩阵
            self.out_features * current_rank   # U矩阵
        )
        if self.adlrns_layer.bias is not None:
            effective_params += self.out_features
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'effective_parameters': effective_params,
            'parameter_efficiency': effective_params / total_params,
            'current_rank': current_rank,
        }
    
    def __repr__(self) -> str:
        return (
            f"ADLRNS("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"mode='{self.mode}', "
            f"rank_range=({self.min_rank}, {self.max_rank}), "
            f"device='{self.device}'"
            f")"
        )


# 便捷函数
def create_adlrns_layer(
    in_features: int,
    out_features: int,
    mode: str = 'balanced',
    **kwargs
) -> ADLRNS:
    """创建ADLRNS层的便捷函数
    
    Args:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        mode (str): 预设模式
        **kwargs: 其他参数
        
    Returns:
        ADLRNS: ADLRNS层实例
    """
    return ADLRNS(in_features, out_features, mode=mode, **kwargs)


def replace_linear_with_adlrns(
    model: nn.Module,
    mode: str = 'balanced',
    exclude_layers: Optional[list] = None
) -> nn.Module:
    """将模型中的Linear层替换为ADLRNS层
    
    Args:
        model (nn.Module): 要修改的模型
        mode (str): ADLRNS模式
        exclude_layers (list, optional): 要排除的层名称列表
        
    Returns:
        nn.Module: 修改后的模型
    """
    if exclude_layers is None:
        exclude_layers = []
    
    for name, module in model.named_children():
        if name in exclude_layers:
            continue
            
        if isinstance(module, nn.Linear):
            # 替换Linear层为ADLRNS层
            adlrns_layer = ADLRNS(
                in_features=module.in_features,
                out_features=module.out_features,
                mode=mode,
                bias=module.bias is not None,
                device=next(module.parameters()).device
            )
            setattr(model, name, adlrns_layer)
        else:
            # 递归处理子模块
            replace_linear_with_adlrns(module, mode, exclude_layers)
    
    return model


# 导出的主要接口
__all__ = [
    'ADLRNS',
    'create_adlrns_layer',
    'replace_linear_with_adlrns',
]
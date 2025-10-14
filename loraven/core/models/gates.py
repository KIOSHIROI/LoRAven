"""
门控网络和复杂度评分器实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any


class LightweightScorer(nn.Module):
    """
    轻量级复杂度评分器
    计算输入样本的复杂度分数 s(x) ∈ [0, 1]
    增强的NaN处理和输入验证机制
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        gradient_clip_val: float = 1.0,
        stability_eps: float = 1e-8
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.gradient_clip_val = gradient_clip_val
        self.stability_eps = stability_eps
        
        # 轻量级网络结构 - 添加BatchNorm提高稳定性
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim, device=device, dtype=dtype),
            nn.BatchNorm1d(hidden_dim, device=device, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device, dtype=dtype),
            nn.BatchNorm1d(hidden_dim // 2, device=device, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1, device=device, dtype=dtype),
            nn.Sigmoid()
        )
        
        # 运行统计用于智能默认值
        self.register_buffer('running_mean', torch.tensor(0.5))
        self.register_buffer('running_std', torch.tensor(0.1))
        self.register_buffer('sample_count', torch.tensor(0))
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _validate_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入验证和预处理
        
        Args:
            x: 输入张量
            
        Returns:
            validated_x: 验证后的输入张量
        """
        # 更强的NaN和无穷值处理
        # 1. 检测并替换NaN值
        nan_mask = torch.isnan(x)
        if nan_mask.any():
            # 使用局部均值替换NaN，如果全是NaN则用0
            finite_mask = torch.isfinite(x)
            if finite_mask.any():
                finite_mean = torch.mean(x[finite_mask])
                x = torch.where(nan_mask, finite_mean, x)
            else:
                x = torch.where(nan_mask, torch.zeros_like(x), x)
        
        # 2. 处理无穷值
        inf_mask = torch.isinf(x)
        if inf_mask.any():
            # 用有限值的最大/最小值替换无穷值
            finite_mask = torch.isfinite(x)
            if finite_mask.any():
                finite_values = x[finite_mask]
                max_finite = torch.max(finite_values)
                min_finite = torch.min(finite_values)
                x = torch.where(torch.isposinf(x), max_finite * 1.1, x)
                x = torch.where(torch.isneginf(x), min_finite * 1.1, x)
            else:
                x = torch.where(torch.isposinf(x), torch.ones_like(x) * 1e6, x)
                x = torch.where(torch.isneginf(x), torch.ones_like(x) * -1e6, x)
        
        # 3. 再次检查是否还有异常值
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        # 4. 稳健的归一化
        x_flat = x.view(-1)
        if x_flat.numel() > 0:
            # 使用分位数进行稳健归一化
            q25 = torch.quantile(x_flat, 0.25)
            q75 = torch.quantile(x_flat, 0.75)
            iqr = q75 - q25
            
            if iqr > 1e-8:  # 避免除零
                # 使用IQR进行归一化
                x = (x - q25) / iqr
            else:
                # 如果IQR太小，使用标准归一化
                x_mean = torch.mean(x_flat)
                x_std = torch.std(x_flat)
                if x_std > 1e-8:
                    x = (x - x_mean) / x_std
                else:
                    x = x - x_mean  # 只减去均值
        
        # 5. 最终的梯度裁剪和范围限制
        x = torch.clamp(x, -10, 10)
        
        # 6. 最后一次NaN检查
        if torch.isnan(x).any():
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        return x
    
    def _get_adaptive_default(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        获取自适应默认值
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            default_scores: 默认复杂度分数
        """
        if self.sample_count > 100:  # 有足够样本时使用统计信息
            # 基于历史统计的智能默认值
            default_val = torch.clamp(self.running_mean, 0.3, 0.7)  # 限制在合理范围
            noise = torch.randn(batch_size, 1, device=device) * self.running_std * 0.1
            return torch.clamp(default_val + noise, 0.0, 1.0)
        else:
            # 使用保守的默认值
            return torch.full((batch_size, 1), 0.5, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 计算输入复杂度分数
        
        数学公式: s(x) = σ(MLP(x)) ∈ [0, 1]
        其中 σ 是 sigmoid 激活函数，MLP 是多层感知器
        
        Args:
            x: 输入张量 (batch_size, in_features) 或 (batch_size, seq_len, in_features)
            
        Returns:
            complexity_scores: 复杂度分数 (batch_size, 1)，范围 [0, 1]
        """
        batch_size = x.size(0)
        device = x.device
        
        try:
            # 处理不同维度的输入
            if x.dim() == 3:  # (batch_size, seq_len, in_features)
                x = x.mean(dim=1)  # 平均池化到 (batch_size, in_features)
            elif x.dim() == 4:  # (batch_size, channels, height, width)
                x = x.view(x.size(0), -1)  # 展平到 (batch_size, in_features)
            
            # 输入验证和预处理
            x = self._validate_input(x)
            
            # 前向传播
            output = self.net(x)
            
            # 强化的输出有效性检查和修复
            if torch.isnan(output).any() or not torch.isfinite(output).all():
                print(f"警告: 检测到NaN/Inf输出，使用默认值替换")
                # 使用自适应默认值替换异常输出
                default_val = self._get_adaptive_default(batch_size, device)
                output = torch.where(torch.isnan(output) | torch.isinf(output), 
                                   default_val, output)
            else:
                # 更新运行统计
                if self.training:
                    with torch.no_grad():
                        current_mean = output.mean()
                        current_std = output.std()
                        
                        # 指数移动平均更新
                        momentum = 0.1
                        self.running_mean = (1 - momentum) * self.running_mean + momentum * current_mean
                        self.running_std = (1 - momentum) * self.running_std + momentum * current_std
                        self.sample_count += batch_size
            
            # 确保输出在合理范围内
            output = torch.clamp(output, 0.0, 1.0)
            
            # 最终NaN检查
            if torch.isnan(output).any():
                print(f"警告: 最终输出仍包含NaN，强制设为默认值")
                output = torch.full_like(output, 0.5)  # 使用中性默认值
            
            return output
            
        except Exception as e:
            # 异常情况下的fallback
            print(f"Warning: LightweightScorer forward failed with error: {e}")
            return self._get_adaptive_default(batch_size, device)
    
    def get_health_stats(self) -> dict:
        """
        获取模型健康状态统计
        
        Returns:
            stats: 健康状态字典
        """
        return {
            'running_mean': self.running_mean.item(),
            'running_std': self.running_std.item(),
            'sample_count': self.sample_count.item(),
            'gradient_clip_val': self.gradient_clip_val,
            'stability_eps': self.stability_eps
        }
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, hidden_dim={self.hidden_dim}, gradient_clip_val={self.gradient_clip_val}'


class GateNetwork(nn.Module):
    """
    门控网络
    决定哪些权重子空间被激活/更新
    """
    
    def __init__(
        self,
        in_features: int,
        num_blocks: int = 8,
        hidden_dim: int = 64,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim, device=device, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_blocks, device=device, dtype=dtype),
            nn.Sigmoid()
        )
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        for module in self.gate_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mode: str = 'soft'
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, in_features) 或 (batch_size, seq_len, in_features)
            mode: 门控模式 ('soft', 'hard', 'gumbel')
            
        Returns:
            gate_mask: 门控掩码 (batch_size, num_blocks)
        """
        # 处理不同维度的输入
        if x.dim() == 3:  # (batch_size, seq_len, in_features)
            x = x.mean(dim=1)  # 平均池化到 (batch_size, in_features)
        elif x.dim() == 4:  # (batch_size, channels, height, width)
            x = x.view(x.size(0), -1)  # 展平到 (batch_size, in_features)
        
        # 计算门控分数
        gate_scores = self.gate_net(x)  # (batch_size, num_blocks)
        
        if mode == 'soft':
            return gate_scores
        elif mode == 'hard':
            return (gate_scores > 0.5).float()
        elif mode == 'gumbel':
            # Gumbel-Softmax 用于可微分的离散采样
            if self.training:
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(gate_scores) + 1e-8) + 1e-8)
                logits = torch.log(gate_scores + 1e-8) + gumbel_noise
                return F.softmax(logits, dim=-1)
            else:
                return (gate_scores > 0.5).float()
        else:
            raise ValueError(f"Unknown gate mode: {mode}")
    
    def get_activation_stats(self, x: torch.Tensor) -> dict:
        """
        获取门控激活统计信息
        
        Args:
            x: 输入张量
            
        Returns:
            stats: 统计信息字典
        """
        with torch.no_grad():
            gate_scores = self.forward(x, mode='soft')
            
            return {
                'mean_activation': gate_scores.mean().item(),
                'std_activation': gate_scores.std().item(),
                'active_blocks': (gate_scores > 0.5).sum(dim=1).float().mean().item(),
                'max_activation': gate_scores.max().item(),
                'min_activation': gate_scores.min().item()
            }
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, num_blocks={self.num_blocks}, hidden_dim={self.hidden_dim}'


class EventTriggeredUpdater(nn.Module):
    """
    事件触发更新器
    基于多种指标决定是否触发权重更新
    """
    
    def __init__(
        self,
        update_threshold: float = 0.02,
        decay_factor: float = 0.9,
        min_samples: int = 10,
        loss_change_threshold: float = 0.001,
        stability_window: int = 5,
        gradient_threshold: float = 1e-6,
        adaptive_threshold: bool = True
    ):
        super().__init__()
        self.update_threshold = update_threshold
        self.decay_factor = decay_factor
        self.min_samples = min_samples
        self.loss_change_threshold = loss_change_threshold
        self.stability_window = stability_window
        self.gradient_threshold = gradient_threshold
        self.adaptive_threshold = adaptive_threshold
        
        # 运行统计
        self.register_buffer('error_running', torch.tensor(0.0))
        self.register_buffer('sample_count', torch.tensor(0))
        self.register_buffer('last_error', torch.tensor(0.0))
        
        # 损失变化率统计
        self.register_buffer('loss_history', torch.zeros(stability_window))
        self.register_buffer('loss_ptr', torch.tensor(0))
        self.register_buffer('last_loss', torch.tensor(float('inf')))
        
        # 梯度统计
        self.register_buffer('gradient_norm_running', torch.tensor(0.0))
        self.register_buffer('gradient_variance', torch.tensor(0.0))
        
        # 自适应阈值
        self.register_buffer('adaptive_factor', torch.tensor(1.0))
        self.register_buffer('update_count', torch.tensor(0))
        self.register_buffer('false_positive_count', torch.tensor(0))
    
    def update_error(
        self, 
        current_error: torch.Tensor, 
        current_loss: Optional[torch.Tensor] = None,
        gradient_norm: Optional[torch.Tensor] = None
    ) -> bool:
        """
        更新误差并决定是否触发更新
        
        Args:
            current_error: 当前误差
            current_loss: 当前损失值
            gradient_norm: 梯度范数
            
        Returns:
            should_update: 是否应该触发更新
        """
        current_error = current_error.detach()
        
        # 更新运行误差
        if self.sample_count < self.min_samples:
            self.error_running = current_error
        else:
            self.error_running = self.decay_factor * self.error_running + (1 - self.decay_factor) * current_error
        
        self.sample_count += 1
        self.last_error = current_error
        
        # 更新损失历史
        loss_change_trigger = False
        if current_loss is not None:
            current_loss = current_loss.detach()
            
            # 计算损失变化率
            if not torch.isinf(self.last_loss):
                loss_change = abs(current_loss - self.last_loss) / (abs(self.last_loss) + 1e-8)
                loss_change_trigger = loss_change > self.loss_change_threshold
            
            # 更新损失历史
            self.loss_history[self.loss_ptr] = current_loss
            self.loss_ptr = (self.loss_ptr + 1) % self.stability_window
            self.last_loss = current_loss
        
        # 更新梯度统计
        gradient_trigger = False
        if gradient_norm is not None:
            gradient_norm = gradient_norm.detach()
            
            # 更新梯度统计
            if self.sample_count == 1:
                self.gradient_norm_running = gradient_norm
                self.gradient_variance = torch.tensor(0.0)
            else:
                # 指数移动平均
                old_mean = self.gradient_norm_running
                self.gradient_norm_running = self.decay_factor * self.gradient_norm_running + (1 - self.decay_factor) * gradient_norm
                
                # 更新方差
                self.gradient_variance = self.decay_factor * self.gradient_variance + (1 - self.decay_factor) * (gradient_norm - old_mean) ** 2
            
            # 梯度触发条件
            gradient_trigger = gradient_norm > self.gradient_threshold
        
        # 稳定性检查
        stability_trigger = self._check_stability()
        
        # 自适应阈值调整
        effective_threshold = self.update_threshold
        if self.adaptive_threshold:
            effective_threshold = self._get_adaptive_threshold()
        
        # 综合判断是否触发更新
        error_trigger = (self.error_running > effective_threshold) and (self.sample_count >= self.min_samples)
        
        should_update = error_trigger or loss_change_trigger or gradient_trigger or stability_trigger
        
        # 更新统计
        if should_update:
            self.update_count += 1
        
        return should_update
    
    def _check_stability(self) -> bool:
        """
        检查损失稳定性
        
        Returns:
            is_unstable: 是否不稳定需要更新
        """
        if self.sample_count < self.stability_window:
            return False
        
        # 计算损失方差
        loss_var = torch.var(self.loss_history)
        loss_mean = torch.mean(self.loss_history)
        
        # 如果损失方差过大，认为不稳定
        cv = loss_var.sqrt() / (abs(loss_mean) + 1e-8)  # 变异系数
        
        return cv > 0.1  # 变异系数阈值
    
    def _get_adaptive_threshold(self) -> float:
        """
        获取自适应阈值
        
        Returns:
            adaptive_threshold: 自适应阈值
        """
        # 基于历史更新成功率调整阈值
        if self.update_count > 0:
            success_rate = 1.0 - (self.false_positive_count.float() / self.update_count.float())
            
            # 如果成功率低，降低阈值；如果成功率高，提高阈值
            if success_rate < 0.5:
                self.adaptive_factor = torch.clamp(self.adaptive_factor * 0.95, 0.5, 2.0)
            elif success_rate > 0.8:
                self.adaptive_factor = torch.clamp(self.adaptive_factor * 1.05, 0.5, 2.0)
        
        return self.update_threshold * self.adaptive_factor.item()
    
    def report_false_positive(self):
        """
        报告假阳性更新
        """
        self.false_positive_count += 1
    
    def reset(self):
        """重置统计信息"""
        self.error_running.zero_()
        self.sample_count.zero_()
        self.last_error.zero_()
        self.loss_history.zero_()
        self.loss_ptr.zero_()
        self.last_loss.fill_(float('inf'))
        self.gradient_norm_running.zero_()
        self.gradient_variance.zero_()
        self.adaptive_factor.fill_(1.0)
        self.update_count.zero_()
        self.false_positive_count.zero_()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        effective_threshold = self.update_threshold
        if self.adaptive_threshold:
            effective_threshold = self._get_adaptive_threshold()
            
        stats = {
            'error_running': self.error_running.item(),
            'sample_count': self.sample_count.item(),
            'last_error': self.last_error.item(),
            'should_update': self.error_running.item() > effective_threshold,
            'effective_threshold': effective_threshold,
            'adaptive_factor': self.adaptive_factor.item(),
            'update_count': self.update_count.item(),
            'false_positive_count': self.false_positive_count.item()
        }
        
        # 添加损失统计
        if not torch.isinf(self.last_loss):
            stats.update({
                'last_loss': self.last_loss.item(),
                'loss_variance': torch.var(self.loss_history).item() if self.sample_count >= self.stability_window else 0.0,
                'loss_mean': torch.mean(self.loss_history).item() if self.sample_count >= self.stability_window else 0.0
            })
        
        # 添加梯度统计
        if self.gradient_norm_running > 0:
            stats.update({
                'gradient_norm_running': self.gradient_norm_running.item(),
                'gradient_variance': self.gradient_variance.item(),
                'gradient_std': self.gradient_variance.sqrt().item()
            })
        
        return stats

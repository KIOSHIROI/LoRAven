"""
动态低秩层实现：LoRAven 的核心模块
实现运行时可变秩的权重矩阵分解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Union, List
from .gates import LightweightScorer, GateNetwork
from .sensitivity_helpers import update_history, calculate_sensitive_rank, apply_adaptive_adjustment, enforce_diversity


class DynamicLowRankLayer(nn.Module):
    """
    动态低秩层：根据输入复杂度动态调整权重矩阵的秩
    
    权重矩阵 W(t) ≈ U(t) @ S(t) @ V(t)^T
    其中 U ∈ R^(out_features × r(t)), S ∈ R^(r(t) × r(t)), V ∈ R^(in_features × r(t))
    r(t) 根据输入复杂度 s(x_t) 和能耗预算 B(t) 动态调整
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
        super().__init__()
        
        # 参数验证
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        if r_min <= 0:
            raise ValueError(f"r_min must be positive, got {r_min}")
        if r_max <= 0:
            raise ValueError(f"r_max must be positive, got {r_max}")
        if r_max < r_min:
            raise ValueError(f"r_max ({r_max}) must be >= r_min ({r_min})")
        if r_max > min(in_features, out_features):
            raise ValueError(f"r_max ({r_max}) cannot exceed min(in_features, out_features) = {min(in_features, out_features)}")
        
        self.in_features = in_features
        self.out_features = out_features
        self.r_max = r_max
        self.r_min = r_min
        self.r_curr = init_rank or r_min
        
        # 低秩分解因子，预分配最大秩的空间
        self.U_full = nn.Parameter(
            torch.empty(out_features, r_max, device=device, dtype=dtype)
        )
        self.V_full = nn.Parameter(
            torch.empty(in_features, r_max, device=device, dtype=dtype)
        )
        self.S_full = nn.Parameter(
            torch.eye(r_max, device=device, dtype=dtype)
        )
        
        # 偏置项
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)
        
        # 复杂度评分器
        self.scorer = LightweightScorer(
            in_features=in_features,
            hidden_dim=scorer_hidden,
            device=device,
            dtype=dtype
        )
        
        # 门控网络（可选）
        self.gate = GateNetwork(
            in_features=in_features,
            num_blocks=num_gate_blocks,
            device=device,
            dtype=dtype
        )
        
        # 初始化事件触发相关参数
        self.error_running = 0.0
        self.error_threshold_up = 0.5   # 大幅提高上阈值，降低触发敏感性
        self.error_threshold_down = 0.3  # 大幅提高下阈值，降低触发敏感性
        self.delta_r = 1
        self.trigger_cooldown = 0  # 冷却计数器
        self.cooldown_period = 10  # 延长冷却周期
        self.error_history = []  # 误差历史记录
        
        # 增强敏感性相关参数
        self.register_buffer('rank_history', torch.zeros(20))  # 秩历史记录
        self.register_buffer('complexity_history', torch.zeros(20))  # 复杂度历史
        self.register_buffer('performance_history', torch.zeros(20))  # 性能历史
        self.register_buffer('history_ptr', torch.tensor(0))  # 历史指针
        self.register_buffer('step_count', torch.tensor(0))  # 步数计数
        
        # 敏感性参数
        self.sensitivity_factor = 3.0  # 进一步提高敏感性因子，增加秩调整敏感性
        self.diversity_threshold = 0.05  # 降低多样性阈值，减少强制变化
        self.adaptation_momentum = 0.9  # 自适应动量
        self.min_rank_change = 3  # 增加最小秩变化，减少小幅调整
        
        # 新增：奇异值监控相关参数
        self.singular_values_history = []  # 奇异值历史记录
        self.top_k_singular_values = 5  # 监控前k个奇异值
        self.monitor_frequency = 10  # 每10个batch监控一次
        self.batch_counter = 0
        
        # Hebbian-like更新参数
        self.hebbian_lr = 0.01  # Hebbian学习率
        self.enable_hebbian = True  # 是否启用Hebbian更新
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        # 使用 Xavier 初始化
        fan_in = self.in_features
        fan_out = self.out_features
        
        # 初始化 U 和 V
        bound_u = math.sqrt(6.0 / (fan_out + self.r_max))
        bound_v = math.sqrt(6.0 / (fan_in + self.r_max))
        
        nn.init.uniform_(self.U_full, -bound_u, bound_u)
        nn.init.uniform_(self.V_full, -bound_v, bound_v)
        
        # S 保持单位矩阵初始化
        nn.init.eye_(self.S_full)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        budget: Optional[float] = None,
        mode: str = 'inference'
    ) -> Tuple[torch.Tensor, int]:
        """
        前向传播 - 动态低秩矩阵乘法
        
        数学公式: 
        1. 复杂度评分: s(x) = Scorer(x) ∈ [0, 1]
        2. 秩调度: r(t) = RankScheduler(s(x), budget)
        3. 低秩分解: W ≈ U @ S @ V^T
        4. 矩阵乘法: Y = X @ V @ S^T @ U^T
        
        Args:
            x: 输入张量 (batch_size, in_features)
            budget: 能耗预算 (mJ/sample)
            mode: 运行模式 ('inference' 或 'training')
            
        Returns:
            output: 输出张量 (batch_size, out_features)
            current_rank: 当前使用的秩
        """
        batch_size = x.size(0)
        
        # 0. 强化输入NaN处理
        x_processed = self._preprocess_input(x)
        
        # 1. 计算样本复杂度分数
        s = self.scorer(x_processed)  # (batch_size, 1) -> (batch_size,)
        s = s.squeeze(-1) if s.dim() > 1 else s
        
        # 2. 秩调度
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
        
        # 新增：奇异值监控
        self.batch_counter += 1
        if self.batch_counter % self.monitor_frequency == 0:
            self._monitor_singular_values(U, S, V)
        
        # 5. 计算低秩矩阵乘法: Y = (U @ S) @ (V^T @ X)^T
        # 使用分步计算以支持批处理，并在每步检查NaN
        try:
            z = torch.matmul(x_processed, V)  # (batch_size, r_target)
            # 检查中间结果
            if torch.isnan(z).any() or not torch.isfinite(z).all():
                z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
            
            z = torch.matmul(z, S.T)  # (batch_size, r_target)
            # 再次检查
            if torch.isnan(z).any() or not torch.isfinite(z).all():
                z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
            
            y = torch.matmul(z, U.T)  # (batch_size, out_features)
            # 最终检查
            if torch.isnan(y).any() or not torch.isfinite(y).all():
                y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
                
        except Exception as e:
            # 如果矩阵乘法失败，返回零输出
            print(f"矩阵乘法出错: {e}，使用零输出")
            y = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        
        # 6. 添加偏置
        if self.bias is not None:
            y = y + self.bias
            # 检查偏置后的结果
            if torch.isnan(y).any() or not torch.isfinite(y).all():
                y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
        
        # 7. 最终输出验证和修复
        y = self._postprocess_output(y)
        
        # 8. 更新当前秩
        self.r_curr = r_target
        
        # 9. Hebbian-like更新（仅在训练模式下）
        if mode == 'training' and self.enable_hebbian:
            self._hebbian_update(x_processed, y)
        
        # 10. 事件触发更新（仅在训练模式下，且满足触发条件）
        if mode == 'training':
            # 只有在满足特定条件时才调用事件触发更新
            # 这里不直接调用，让事件触发更新在内部自行判断是否需要触发
            triggered = self._event_triggered_update(x_processed, y)
            # 如果没有触发，记录为非触发事件
            if not triggered:
                pass  # 正常情况，不需要额外处理
        
        return y, r_target
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        强化输入预处理，处理NaN和异常值
        
        Args:
            x: 输入张量
            
        Returns:
            processed_x: 处理后的输入张量
        """
        # 1. 检查并处理NaN值
        if torch.isnan(x).any():
            # 使用局部均值替换NaN
            x_clean = x.clone()
            nan_mask = torch.isnan(x)
            
            # 对每个样本分别处理
            for i in range(x.size(0)):
                sample_nan_mask = nan_mask[i]
                if sample_nan_mask.any():
                    # 使用该样本的非NaN值的均值替换NaN
                    valid_values = x_clean[i][~sample_nan_mask]
                    if len(valid_values) > 0:
                        replacement_value = valid_values.mean()
                    else:
                        replacement_value = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                    x_clean[i][sample_nan_mask] = replacement_value
            x = x_clean
        
        # 2. 处理无穷值
        if not torch.isfinite(x).all():
            inf_mask = ~torch.isfinite(x)
            # 用有限值的中位数替换无穷值
            finite_values = x[torch.isfinite(x)]
            if len(finite_values) > 0:
                replacement_value = torch.median(finite_values)
            else:
                replacement_value = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            x = torch.where(torch.isfinite(x), x, replacement_value)
        
        # 3. 异常值检测和处理
        # 使用3-sigma规则检测异常值
        mean_val = x.mean()
        std_val = x.std()
        if std_val > 0:
            outlier_mask = torch.abs(x - mean_val) > 3 * std_val
            if outlier_mask.any():
                # 将异常值裁剪到3-sigma范围内
                x = torch.clamp(x, mean_val - 3 * std_val, mean_val + 3 * std_val)
        
        # 4. 最终验证
        if torch.isnan(x).any() or not torch.isfinite(x).all():
            # 如果仍有问题，使用零张量
            x = torch.zeros_like(x)
        
        return x
    
    def _postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """
        输出后处理，确保输出不包含NaN或异常值
        
        Args:
            y: 原始输出张量
            
        Returns:
            processed_y: 处理后的输出张量
        """
        # 1. 检查并修复NaN值
        if torch.isnan(y).any():
            y = torch.where(torch.isnan(y), torch.zeros_like(y), y)
        
        # 2. 检查并修复无穷值
        if not torch.isfinite(y).all():
            y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
        
        # 3. 数值稳定性检查
        # 如果输出值过大，进行裁剪
        max_val = 1e6  # 设置合理的最大值
        y = torch.clamp(y, -max_val, max_val)
        
        # 4. 最终验证
        if torch.isnan(y).any() or not torch.isfinite(y).all():
            # 如果仍有问题，返回零张量
            y = torch.zeros_like(y)
        
        return y
    
    def _rank_scheduler(
        self, 
        complexity_scores: torch.Tensor, 
        budget: Optional[float] = None
    ) -> int:
        """
        增强敏感性的秩调度器：根据复杂度分数和历史信息决定当前秩
        
        Args:
            complexity_scores: 复杂度分数 (batch_size,)
            budget: 能耗预算 (mJ/sample)
            
        Returns:
            target_rank: 目标秩
        """
        # 处理NaN值和异常情况
        if torch.isnan(complexity_scores).any() or not torch.isfinite(complexity_scores).all():
            valid_scores = complexity_scores[torch.isfinite(complexity_scores)]
            if len(valid_scores) == 0:
                s_avg = 0.5  # 使用默认值
            else:
                s_avg = valid_scores.mean().item()
        else:
            s_avg = complexity_scores.mean().item()
        
        # 更新历史记录
        self._update_history(s_avg)
        
        # 计算增强敏感性的秩
        r_base = self._calculate_sensitive_rank(s_avg)
        
        # 应用历史信息和自适应调整
        r_adaptive = self._apply_adaptive_adjustment(r_base, s_avg)
        
        # 强制多样性检查
        r_diverse = self._enforce_diversity(r_adaptive)
        
        # 能耗约束
        if budget is not None:
            r_diverse = self._energy_constrained_rank(r_diverse, budget)
        
        # 确保秩在有效范围内
        r_final = max(self.r_min, min(r_diverse, self.r_max))
        
        return r_final
    
    def _update_history(self, complexity_score: float):
        """
        更新历史记录
        
        Args:
            complexity_score: 当前复杂度分数
        """
        ptr = self.history_ptr.item()
        
        # 更新复杂度历史
        self.complexity_history[ptr] = complexity_score
        
        # 更新秩历史
        self.rank_history[ptr] = self.r_curr
        
        # 更新指针和步数
        self.history_ptr = (self.history_ptr + 1) % 20
        self.step_count += 1
    
    def _calculate_sensitive_rank(self, complexity_score: float) -> int:
        """
        计算敏感性增强的秩
        
        Args:
            complexity_score: 复杂度分数
            
        Returns:
            sensitive_rank: 敏感性增强的秩
        """
        # 非线性映射，增强敏感性
        # 使用sigmoid函数增强中等复杂度的敏感性
        enhanced_score = torch.sigmoid(torch.tensor(complexity_score * self.sensitivity_factor - 1.0)).item()
        
        # 基础秩计算
        r_base = self.r_min + enhanced_score * (self.r_max - self.r_min)
        
        # 添加复杂度变化率的影响（提高敏感性）
        if self.step_count > 1:
            complexity_change = abs(complexity_score - self.complexity_history[(self.history_ptr - 1) % 20].item())
            if complexity_change > 0.15:  # 进一步降低阈值，提高敏感性
                r_base += complexity_change * (self.r_max - self.r_min) * 0.2  # 提高影响系数
        
        return int(round(r_base))
    
    def _apply_adaptive_adjustment(self, base_rank: int, complexity_score: float) -> int:
        """
        应用自适应调整
        
        Args:
            base_rank: 基础秩
            complexity_score: 复杂度分数
            
        Returns:
            adjusted_rank: 调整后的秩
        """
        if self.step_count < 5:  # 初始阶段
            return base_rank
        
        # 计算历史复杂度的方差
        complexity_var = torch.var(self.complexity_history[:min(self.step_count, 20)]).item()
        
        # 如果复杂度变化很小，强制增加秩的变化（平衡敏感性）
        if complexity_var < 0.008:  # 适度提高阈值，平衡触发频率
            # 基于当前复杂度相对于历史平均值的偏差调整
            complexity_mean = torch.mean(self.complexity_history[:min(self.step_count, 20)]).item()
            deviation = abs(complexity_score - complexity_mean)
            
            if deviation > 0.1:  # 适度降低偏差阈值
                adjustment = int(deviation * (self.r_max - self.r_min) * 0.15)  # 适度提高调整系数
                if complexity_score > complexity_mean:
                    base_rank += adjustment
                else:
                    base_rank -= adjustment
        
        # 基于历史性能的调整
        if self.step_count >= 10:
            recent_ranks = self.rank_history[:min(self.step_count, 20)]
            rank_trend = torch.mean(recent_ranks[-5:]) - torch.mean(recent_ranks[-10:-5])
            
            # 如果秩趋势与复杂度趋势不匹配，进行调整（降低敏感性）
            complexity_trend = torch.mean(self.complexity_history[-5:]) - torch.mean(self.complexity_history[-10:-5])
            
            if (complexity_trend > 0.2 and rank_trend < 1) or (complexity_trend < -0.2 and rank_trend > -1):  # 大幅提高阈值
                base_rank += int(complexity_trend * (self.r_max - self.r_min) * 0.1)  # 降低调整系数
        
        return base_rank
    
    def _enforce_diversity(self, rank: int) -> int:
        """
        强制多样性机制
        
        Args:
            rank: 输入秩
            
        Returns:
            diverse_rank: 多样性增强的秩
        """
        if self.step_count < 10:
            return rank
        
        # 检查最近的秩多样性
        recent_ranks = self.rank_history[:min(self.step_count, 20)]
        rank_std = torch.std(recent_ranks).item()
        
        # 如果秩变化太小，强制增加多样性（平衡敏感性）
        if rank_std < self.diversity_threshold * (self.r_max - self.r_min) * 0.8:  # 适度提高多样性强制阈值
            # 计算与最近秩的差异
            recent_rank = self.rank_history[(self.history_ptr - 1) % 20].item()
            
            if abs(rank - recent_rank) < self.min_rank_change * 1.5:  # 适度降低最小变化要求
                # 强制最小变化（平衡频率）
                if rank == recent_rank and self.step_count % 4 == 0:  # 适度提高步数限制
                    # 基于复杂度决定变化方向
                    current_complexity = self.complexity_history[(self.history_ptr - 1) % 20].item()
                    if current_complexity > 0.6:  # 适度降低复杂度阈值
                        rank = min(rank + self.min_rank_change, self.r_max)
                    elif current_complexity < 0.4:  # 适度提高下限阈值
                        rank = max(rank - self.min_rank_change, self.r_min)
                elif rank > recent_rank and self.step_count % 3 == 0:  # 适度提高步数限制
                    rank = min(recent_rank + self.min_rank_change, self.r_max)
                elif rank < recent_rank and self.step_count % 3 == 0:  # 适度提高步数限制
                    rank = max(recent_rank - self.min_rank_change, self.r_min)
        
        return rank
    
    def _energy_constrained_rank(self, r: int, budget: float) -> int:
        """
        能耗约束的秩调整
        
        Args:
            r: 初始秩
            budget: 能耗预算 (mJ/sample)
            
        Returns:
            adjusted_rank: 调整后的秩
        """
        # 简化的能耗估算模型
        # 实际实现中应该使用更精确的能耗模型
        estimated_energy = self._estimate_energy(r)
        
        while estimated_energy > budget and r > self.r_min:
            r -= 1
            estimated_energy = self._estimate_energy(r)
        
        return r
    
    def _estimate_energy(self, r: int) -> float:
        """
        估算给定秩的能耗
        
        Args:
            r: 秩
            
        Returns:
            energy: 估算能耗 (mJ/sample)
        """
        # 简化的能耗模型：基于 FLOPs 和内存访问
        flops = 2 * self.in_features * r + 2 * r * self.out_features
        memory_access = (self.in_features + self.out_features) * r
        
        # 假设每 FLOP 消耗 1e-6 mJ，每内存访问消耗 1e-7 mJ
        energy = flops * 1e-6 + memory_access * 1e-7
        return energy
    
    def _event_triggered_update(self, x: torch.Tensor, y: torch.Tensor):
        """
        事件触发的秩更新
        
        Args:
            x: 输入张量
            y: 输出张量
        """
        # 改进的误差计算方法
        with torch.no_grad():
            # 计算输出的相对误差和绝对误差
            abs_error = torch.mean(torch.abs(y)).item()
            
            # 计算输入输出的相关性作为误差指标
            if x.numel() > 0 and y.numel() > 0:
                x_flat = x.view(-1)
                y_flat = y.view(-1)
                
                # 避免除零错误
                x_norm = torch.norm(x_flat) + 1e-8
                y_norm = torch.norm(y_flat) + 1e-8
                
                # 计算相对误差
                relative_error = abs_error / (y_norm + 1e-8)
                
                # 计算梯度相关的误差（如果在训练模式）
                if self.training and y.requires_grad:
                    try:
                        # 计算输出对输入的梯度范数作为复杂度指标
                        grad_outputs = torch.ones_like(y)
                        gradients = torch.autograd.grad(
                            outputs=y, inputs=x, grad_outputs=grad_outputs,
                            create_graph=False, retain_graph=False, only_inputs=True
                        )[0]
                        gradient_error = torch.norm(gradients).item()
                        current_error = 0.5 * relative_error + 0.3 * abs_error + 0.2 * gradient_error
                    except:
                        current_error = 0.7 * relative_error + 0.3 * abs_error
                else:
                    current_error = 0.7 * relative_error + 0.3 * abs_error
            else:
                current_error = abs_error
        
        # 更新运行误差（使用适中的更新率）
        alpha = 0.1  # 降低敏感性，使误差更平滑
        self.error_running = (1 - alpha) * self.error_running + alpha * current_error
        
        # 更新误差历史记录
        self.error_history.append(current_error)
        if len(self.error_history) > 10:  # 保持最近10个误差值
            self.error_history.pop(0)
        
        # 改进的事件触发逻辑
        rank_changed = False
        
        # 更新冷却计数器
        if self.trigger_cooldown > 0:
            self.trigger_cooldown -= 1
            return False  # 在冷却期内不触发
        
        # 更保守的动态阈值调整
        dynamic_threshold_up = self.error_threshold_up * (1 + 0.05 * torch.rand(1).item())
        dynamic_threshold_down = self.error_threshold_down * (1 - 0.05 * torch.rand(1).item())
        
        # 基于误差的触发（更严格的条件）
        significant_change = abs(current_error - self.error_running) > 0.1   # 大幅提高阈值，降低敏感性
        
        # 大幅降低的触发概率
        trigger_probability = 0.1  # 10%的触发概率
        random_trigger = torch.rand(1).item() < trigger_probability
        
        # 调试输出（每10步输出一次）
        if self.step_count % 10 == 0:
            print(f"Step {self.step_count}: current_error={current_error:.6f}, running_error={self.error_running:.6f}")
            print(f"  dynamic_threshold_up={dynamic_threshold_up:.6f}, dynamic_threshold_down={dynamic_threshold_down:.6f}")
            print(f"  significant_change={significant_change}, random_trigger={random_trigger}")
            print(f"  r_curr={self.r_curr}, cooldown={self.trigger_cooldown}")
        
        print(f"调试: step_count={self.step_count}, step_count%15={self.step_count % 15}")
        if (self.error_running > dynamic_threshold_up and 
            self.r_curr < self.r_max and 
            significant_change and random_trigger and 
            self.step_count % 15 == 0):  # 大幅增加步数限制，进一步降低触发频率
            old_rank = self.r_curr
            self.r_curr = min(self.r_curr + self.delta_r, self.r_max)
            rank_changed = True
            self.trigger_cooldown = self.cooldown_period
            print(f"事件触发: 增秩从 {old_rank} 到 {self.r_curr}")
        elif (self.error_running < dynamic_threshold_down and 
              self.r_curr > self.r_min and 
              significant_change and random_trigger and 
              self.step_count % 18 == 0):  # 大幅增加步数限制，进一步降低触发频率
            old_rank = self.r_curr
            self.r_curr = max(self.r_curr - self.delta_r, self.r_min)
            rank_changed = True
            self.trigger_cooldown = self.cooldown_period
            print(f"事件触发: 降秩从 {old_rank} 到 {self.r_curr}")
        
        # 基于历史趋势的触发（更保守的触发频率）
        if (self.step_count > 10 and 
            not rank_changed and 
            self.step_count % 8 == 0):  # 每8步检查一次趋势，降低频率
            if len(self.error_history) >= 5:
                error_trend = self.error_history[-1] - self.error_history[-5]
                if abs(error_trend) > 0.05:  # 提高趋势触发阈值，减少敏感性
                    if (error_trend > 0 and 
                        self.r_curr < self.r_max and 
                        self.error_running > self.error_threshold_up * 0.8):  # 额外条件
                        old_rank = self.r_curr
                        self.r_curr = min(self.r_curr + 1, self.r_max)
                        rank_changed = True
                        self.trigger_cooldown = self.cooldown_period
                        print(f"趋势触发: 误差上升趋势 {error_trend:.4f}, 秩从 {old_rank} 增加到 {self.r_curr}")
                    elif (error_trend < 0 and 
                          self.r_curr > self.r_min and 
                          self.error_running < self.error_threshold_down * 1.2):  # 额外条件
                        old_rank = self.r_curr
                        self.r_curr = max(self.r_curr - 1, self.r_min)
                        rank_changed = True
                        self.trigger_cooldown = self.cooldown_period
                        print(f"趋势触发: 误差下降趋势 {error_trend:.4f}, 秩从 {old_rank} 减少到 {self.r_curr}")
        
        return rank_changed
    
    def reorthonormalize(self, method: str = 'rsvd', k: Optional[int] = None):
        """
        重新正交化因子矩阵
        
        Args:
            method: 正交化方法 ('rsvd', 'qr', 'svd')
            k: 保留的奇异值数量
        """
        if k is None:
            k = self.r_curr
        
        # 获取当前使用的因子
        U = self.U_full[:, :k].detach()
        V = self.V_full[:, :k].detach()
        S = self.S_full[:k, :k].detach()
        
        if method == 'rsvd':
            # 随机 SVD 重新正交化
            U_new, S_new, V_new = torch.svd(torch.matmul(U, torch.matmul(S, V.T)))
            U_new = U_new[:, :k]
            S_new = torch.diag(S_new[:k])
            V_new = V_new[:, :k]
        elif method == 'qr':
            # QR 分解
            U_new, _ = torch.qr(U)
            V_new, _ = torch.qr(V)
            S_new = S
        else:  # svd
            U_new, S_new, V_new = torch.svd(torch.matmul(U, torch.matmul(S, V.T)))
            U_new = U_new[:, :k]
            S_new = torch.diag(S_new[:k])
            V_new = V_new[:, :k]
        
        # 更新参数
        with torch.no_grad():
            self.U_full[:, :k] = U_new
            self.V_full[:, :k] = V_new
            self.S_full[:k, :k] = S_new
    
    def get_rank_info(self) -> dict:
        """获取当前秩信息"""
        return {
            'current_rank': self.r_curr,
            'min_rank': self.r_min,
            'max_rank': self.r_max,
            'compression_ratio': (self.r_curr * (self.in_features + self.out_features)) / (self.in_features * self.out_features),
            'error_running': self.error_running
        }
    
    def extra_repr(self) -> str:
        """返回层的额外表示信息"""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'r_min={self.r_min}, r_max={self.r_max}, r_curr={self.r_curr}, '
                f'bias={self.bias is not None}')
    
    def _monitor_singular_values(self, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor):
        """
        监控当前权重矩阵的奇异值
        
        Args:
            U: 左奇异向量矩阵
            S: 奇异值对角矩阵
            V: 右奇异向量矩阵
        """
        with torch.no_grad():
            # 重构当前权重矩阵
            W = torch.matmul(torch.matmul(U, S), V.T)
            
            # 计算奇异值分解
            try:
                _, singular_values, _ = torch.svd(W)
                
                # 记录前k个奇异值
                top_k_values = singular_values[:self.top_k_singular_values].cpu().numpy()
                
                # 添加到历史记录
                self.singular_values_history.append({
                    'batch': self.batch_counter,
                    'rank': self.r_curr,
                    'singular_values': top_k_values,
                    'condition_number': (singular_values[0] / singular_values[-1]).item() if len(singular_values) > 1 else 1.0
                })
                
                # 限制历史记录长度
                if len(self.singular_values_history) > 100:
                    self.singular_values_history.pop(0)
                    
                # 每50个batch打印一次奇异值信息
                if self.batch_counter % 50 == 0:
                    print(f"Batch {self.batch_counter}: Rank={self.r_curr}, "
                          f"Top-{self.top_k_singular_values} singular values: {top_k_values}")
                    
            except Exception as e:
                print(f"奇异值计算失败: {e}")
    
    def _hebbian_update(self, x: torch.Tensor, y: torch.Tensor):
        """
        Hebbian-like更新规则：Σ(t+1) = Σ(t) + η * y(t)x(t)^T
        
        Args:
            x: 输入张量 (batch_size, in_features)
            y: 输出张量 (batch_size, out_features)
        """
        with torch.no_grad():
            # 计算批次平均的外积
            batch_size = x.size(0)
            
            # 计算平均外积：E[y * x^T]
            outer_product = torch.matmul(y.T, x) / batch_size  # (out_features, in_features)
            
            # 对S矩阵进行Hebbian更新
            # 这里我们更新对角线元素，模拟奇异值的自适应调整
            if self.r_curr > 0:
                # 计算当前权重矩阵的近似
                U_curr = self.U_full[:, :self.r_curr]
                V_curr = self.V_full[:, :self.r_curr]
                
                # 投影外积到当前子空间
                projected_update = torch.matmul(torch.matmul(U_curr.T, outer_product), V_curr)
                
                # 更新S矩阵的对角线元素
                diagonal_update = torch.diag(projected_update)
                self.S_full[:self.r_curr, :self.r_curr].diagonal().add_(
                    self.hebbian_lr * diagonal_update
                )
                
                # 确保奇异值为正
                self.S_full[:self.r_curr, :self.r_curr].diagonal().clamp_(min=1e-6)
    
    def visualize_singular_values(self, save_path: str = None):
        """
        可视化奇异值随batch变化的曲线
        
        Args:
            save_path: 保存路径，如果为None则显示图像
        """
        if not self.singular_values_history:
            print("没有奇异值历史记录可供可视化")
            return
        
        # 提取数据
        batches = [record['batch'] for record in self.singular_values_history]
        ranks = [record['rank'] for record in self.singular_values_history]
        condition_numbers = [record['condition_number'] for record in self.singular_values_history]
        
        # 提取奇异值矩阵
        singular_values_matrix = np.array([
            record['singular_values'] for record in self.singular_values_history
        ])
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 奇异值随batch变化
        for i in range(min(self.top_k_singular_values, singular_values_matrix.shape[1])):
            ax1.plot(batches, singular_values_matrix[:, i], 
                    label=f'σ_{i+1}', linewidth=2, marker='o', markersize=3)
        ax1.set_xlabel('Batch Index')
        ax1.set_ylabel('Singular Values')
        ax1.set_title('Top-k Singular Values Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. 秩变化
        ax2.plot(batches, ranks, 'r-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Batch Index')
        ax2.set_ylabel('Current Rank')
        ax2.set_title('Dynamic Rank Evolution')
        ax2.grid(True, alpha=0.3)
        
        # 3. 条件数变化
        ax3.plot(batches, condition_numbers, 'g-', linewidth=2, marker='^', markersize=4)
        ax3.set_xlabel('Batch Index')
        ax3.set_ylabel('Condition Number')
        ax3.set_title('Matrix Condition Number')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. 奇异值比率（相对于最大奇异值）
        if singular_values_matrix.shape[1] > 1:
            for i in range(1, min(self.top_k_singular_values, singular_values_matrix.shape[1])):
                ratio = singular_values_matrix[:, i] / singular_values_matrix[:, 0]
                ax4.plot(batches, ratio, label=f'σ_{i+1}/σ_1', linewidth=2, marker='o', markersize=3)
            ax4.set_xlabel('Batch Index')
            ax4.set_ylabel('Singular Value Ratio')
            ax4.set_title('Singular Value Ratios (relative to σ_1)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"奇异值可视化已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_singular_values_stats(self) -> dict:
        """
        获取奇异值统计信息
        
        Returns:
            包含奇异值统计信息的字典
        """
        if not self.singular_values_history:
            return {}
        
        # 提取最新的奇异值
        latest_record = self.singular_values_history[-1]
        singular_values = latest_record['singular_values']
        
        return {
            'current_rank': latest_record['rank'],
            'condition_number': latest_record['condition_number'],
            'largest_singular_value': float(singular_values[0]) if len(singular_values) > 0 else 0.0,
            'smallest_singular_value': float(singular_values[-1]) if len(singular_values) > 0 else 0.0,
            'singular_value_decay': float(singular_values[0] / singular_values[-1]) if len(singular_values) > 1 else 1.0,
            'total_monitored_batches': len(self.singular_values_history),
            'hebbian_enabled': self.enable_hebbian
        }

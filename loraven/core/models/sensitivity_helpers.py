import torch
from typing import Optional

def update_history(layer, complexity_score: float):
    """
    更新历史记录
    
    Args:
        layer: DynamicLowRankLayer实例
        complexity_score: 当前复杂度分数
    """
    ptr = layer.history_ptr.item()
    
    # 更新复杂度历史
    layer.complexity_history[ptr] = complexity_score
    
    # 更新秩历史
    layer.rank_history[ptr] = layer.r_curr
    
    # 更新指针和步数
    layer.history_ptr = (layer.history_ptr + 1) % 20
    layer.step_count += 1

def calculate_sensitive_rank(layer, complexity_score: float) -> int:
    """
    计算敏感性增强的秩
    
    Args:
        layer: DynamicLowRankLayer实例
        complexity_score: 复杂度分数
        
    Returns:
        sensitive_rank: 敏感性增强的秩
    """
    # 非线性映射，增强敏感性
    # 使用sigmoid函数增强中等复杂度的敏感性
    enhanced_score = torch.sigmoid(torch.tensor(complexity_score * layer.sensitivity_factor - 1.0)).item()
    
    # 基础秩计算
    r_base = layer.r_min + enhanced_score * (layer.r_max - layer.r_min)
    
    # 添加复杂度变化率的影响
    if layer.step_count > 1:
        complexity_change = abs(complexity_score - layer.complexity_history[(layer.history_ptr - 1) % 20].item())
        if complexity_change > 0.1:  # 复杂度变化较大时增加秩
            r_base += complexity_change * (layer.r_max - layer.r_min) * 0.5
    
    return int(round(r_base))

def apply_adaptive_adjustment(layer, base_rank: int, complexity_score: float) -> int:
    """
    应用自适应调整
    
    Args:
        layer: DynamicLowRankLayer实例
        base_rank: 基础秩
        complexity_score: 复杂度分数
        
    Returns:
        adjusted_rank: 调整后的秩
    """
    if layer.step_count < 5:  # 初始阶段
        return base_rank
    
    # 计算历史复杂度的方差
    complexity_var = torch.var(layer.complexity_history[:min(layer.step_count, 20)]).item()
    
    # 如果复杂度变化很小，强制增加秩的变化
    if complexity_var < 0.01:  # 复杂度变化很小
        # 基于当前复杂度相对于历史平均值的偏差调整
        complexity_mean = torch.mean(layer.complexity_history[:min(layer.step_count, 20)]).item()
        deviation = abs(complexity_score - complexity_mean)
        
        if deviation > 0.05:  # 有一定偏差时
            adjustment = int(deviation * (layer.r_max - layer.r_min) * 0.3)
            if complexity_score > complexity_mean:
                base_rank += adjustment
            else:
                base_rank -= adjustment
    
    # 基于历史性能的调整
    if layer.step_count >= 10:
        recent_ranks = layer.rank_history[:min(layer.step_count, 20)]
        rank_trend = torch.mean(recent_ranks[-5:]) - torch.mean(recent_ranks[-10:-5])
        
        # 如果秩趋势与复杂度趋势不匹配，进行调整
        complexity_trend = torch.mean(layer.complexity_history[-5:]) - torch.mean(layer.complexity_history[-10:-5])
        
        if (complexity_trend > 0.05 and rank_trend < 1) or (complexity_trend < -0.05 and rank_trend > -1):
            base_rank += int(complexity_trend * (layer.r_max - layer.r_min) * 0.4)
    
    return base_rank

def enforce_diversity(layer, rank: int) -> int:
    """
    强制多样性机制
    
    Args:
        layer: DynamicLowRankLayer实例
        rank: 输入秩
        
    Returns:
        diverse_rank: 多样性增强的秩
    """
    if layer.step_count < 10:
        return rank
    
    # 检查最近的秩多样性
    recent_ranks = layer.rank_history[:min(layer.step_count, 20)]
    rank_std = torch.std(recent_ranks).item()
    
    # 如果秩变化太小，强制增加多样性
    if rank_std < layer.diversity_threshold * (layer.r_max - layer.r_min):
        # 计算与最近秩的差异
        recent_rank = layer.rank_history[(layer.history_ptr - 1) % 20].item()
        
        if abs(rank - recent_rank) < layer.min_rank_change:
            # 强制最小变化
            if rank == recent_rank:
                # 基于复杂度决定变化方向
                current_complexity = layer.complexity_history[(layer.history_ptr - 1) % 20].item()
                if current_complexity > 0.5:
                    rank = min(rank + layer.min_rank_change, layer.r_max)
                else:
                    rank = max(rank - layer.min_rank_change, layer.r_min)
            elif rank > recent_rank:
                rank = min(recent_rank + layer.min_rank_change, layer.r_max)
            else:
                rank = max(recent_rank - layer.min_rank_change, layer.r_min)
    
    return rank
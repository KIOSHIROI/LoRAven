#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRAven基本使用示例
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from loraven import LoRAven, DynamicLowRankLayer, BudgetManager

def basic_usage_example():
    """基本使用示例"""
    print("=== LoRAven 基本使用示例 ===")
    
    # 设置设备
    device = torch.device('cpu')  # 强制使用CPU避免设备不匹配
    
    # 创建LoRAven层
    loraven_layer = LoRAven(
        in_features=512,
        out_features=256,
        mode='balanced',  # 使用平衡模式
        device=device
    )
    
    # 创建输入数据
    batch_size = 32
    input_data = torch.randn(batch_size, 512, device=device)
    
    # 前向传播
    output = loraven_layer(input_data)
    print(f"输入形状: {input_data.shape}")
    print(f"输出形状: {output.shape}")
    
    return loraven_layer

def advanced_usage_example():
    """高级使用示例"""
    print("=== Advanced Usage ===")
    
    # 设置设备
    device = torch.device('cpu')  # 强制使用CPU避免设备不匹配
    
    # 创建预算管理器
    budget_manager = BudgetManager(
        total_budget=1000.0
    )
    
    # 创建自定义LoRAven层
    loraven_layer = LoRAven(
        in_features=1024,
        out_features=512,
        mode='custom',
        max_rank=128,
        min_rank=8,
        energy_budget=500.0,
        device=device
    )
    
    # 模拟训练过程
    for step in range(5):
        input_data = torch.randn(16, 1024, device=device)
        output = loraven_layer(input_data)
        
        # 获取当前秩
        current_rank = loraven_layer.loraven_layer.r_curr
        print(f"Step {step}: 输出形状 {output.shape}, 当前秩: {current_rank}")
    
    return loraven_layer

if __name__ == "__main__":
    print("=== Basic Usage ===")
    basic_usage_example()
    
    print("\n=== Advanced Usage ===")
    advanced_usage_example()
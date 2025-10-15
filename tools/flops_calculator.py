# FLOPs计算器
# 功能：计算模型的浮点运算次数(FLOPs)，用于评估计算复杂度

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class FLOPsCalculator:
    """FLOPs计算器"""
    
    def __init__(self):
        self.flop_counts = defaultdict(int)
        self.hooks = []
        
    def calculate_model_flops(self, model: nn.Module, input_shape: Tuple[int, ...], 
                            device: str = 'cpu') -> Dict[str, Any]:
        """计算模型的FLOPs"""
        
        # 创建示例输入
        if isinstance(input_shape, dict):
            # 对于transformer模型，输入通常是字典格式
            sample_input = {}
            for key, shape in input_shape.items():
                if key in ['input_ids', 'attention_mask']:
                    sample_input[key] = torch.randint(0, 1000, shape).to(device)
                else:
                    sample_input[key] = torch.randn(shape).to(device)
        else:
            # 对于普通模型，输入是tensor
            sample_input = torch.randn(input_shape).to(device)
        
        # 注册hooks来计算FLOPs
        self._register_hooks(model)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            if isinstance(sample_input, dict):
                _ = model(**sample_input)
            else:
                _ = model(sample_input)
        
        # 移除hooks
        self._remove_hooks()
        
        # 计算总FLOPs
        total_flops = sum(self.flop_counts.values())
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results = {
            'total_flops': total_flops,
            'total_flops_m': total_flops / 1e6,  # Million FLOPs
            'total_flops_g': total_flops / 1e9,  # Billion FLOPs
            'total_params': total_params,
            'trainable_params': trainable_params,
            'flops_per_param': total_flops / total_params if total_params > 0 else 0,
            'layer_wise_flops': dict(self.flop_counts)
        }
        
        # 重置计数器
        self.flop_counts.clear()
        
        return results
    
    def _register_hooks(self, model: nn.Module):
        """注册hooks来计算FLOPs"""
        
        def conv_flop_count(module, input, output):
            """计算卷积层的FLOPs"""
            input_dims = input[0].shape
            output_dims = output.shape
            kernel_dims = module.kernel_size
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups
            
            filters_per_channel = out_channels // groups
            conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // groups
            
            active_elements_count = int(np.prod(output_dims))
            overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
            
            bias_flops = 0
            if module.bias is not None:
                bias_flops = out_channels * active_elements_count
            
            overall_flops = overall_conv_flops + bias_flops
            self.flop_counts[f"{module.__class__.__name__}_{id(module)}"] += overall_flops
        
        def linear_flop_count(module, input, output):
            """计算线性层的FLOPs"""
            input_dims = input[0].shape
            # 输入特征数 * 输出特征数 * batch_size
            flops = module.in_features * module.out_features
            if len(input_dims) > 2:
                # 对于多维输入，考虑所有维度
                flops *= int(np.prod(input_dims[:-1]))
            else:
                flops *= input_dims[0]  # batch_size
            
            # 如果有bias，添加bias的FLOPs
            if module.bias is not None:
                if len(input_dims) > 2:
                    flops += module.out_features * int(np.prod(input_dims[:-1]))
                else:
                    flops += module.out_features * input_dims[0]
            
            self.flop_counts[f"{module.__class__.__name__}_{id(module)}"] += flops
        
        def attention_flop_count(module, input, output):
            """计算注意力层的FLOPs (近似)"""
            if hasattr(module, 'num_attention_heads') and hasattr(module, 'attention_head_size'):
                # Transformer attention的FLOPs计算
                seq_len = input[0].shape[1] if len(input[0].shape) > 2 else input[0].shape[0]
                batch_size = input[0].shape[0] if len(input[0].shape) > 2 else 1
                
                # Q, K, V projections
                qkv_flops = 3 * seq_len * module.all_head_size * module.all_head_size * batch_size
                
                # Attention computation: Q @ K^T
                attention_flops = seq_len * seq_len * module.all_head_size * batch_size
                
                # Attention @ V
                output_flops = seq_len * seq_len * module.all_head_size * batch_size
                
                total_flops = qkv_flops + attention_flops + output_flops
                self.flop_counts[f"{module.__class__.__name__}_{id(module)}"] += total_flops
        
        def activation_flop_count(module, input, output):
            """计算激活函数的FLOPs"""
            # 激活函数通常每个元素一个操作
            flops = output.numel()
            self.flop_counts[f"{module.__class__.__name__}_{id(module)}"] += flops
        
        # 注册不同类型层的hooks
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
                hook = module.register_forward_hook(conv_flop_count)
                self.hooks.append(hook)
            elif isinstance(module, nn.Linear):
                hook = module.register_forward_hook(linear_flop_count)
                self.hooks.append(hook)
            elif 'Attention' in module.__class__.__name__:
                hook = module.register_forward_hook(attention_flop_count)
                self.hooks.append(hook)
            elif isinstance(module, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)):
                hook = module.register_forward_hook(activation_flop_count)
                self.hooks.append(hook)
    
    def _remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def estimate_inference_flops(self, model: nn.Module, input_shape: Tuple[int, ...], 
                               num_samples: int = 1000) -> Dict[str, float]:
        """估算推理时的FLOPs"""
        
        # 计算单次推理的FLOPs
        single_inference_flops = self.calculate_model_flops(model, input_shape)
        
        # 估算多次推理的总FLOPs
        total_flops = single_inference_flops['total_flops'] * num_samples
        
        return {
            'single_inference_flops': single_inference_flops['total_flops'],
            'single_inference_flops_m': single_inference_flops['total_flops_m'],
            'single_inference_flops_g': single_inference_flops['total_flops_g'],
            'total_inference_flops': total_flops,
            'total_inference_flops_m': total_flops / 1e6,
            'total_inference_flops_g': total_flops / 1e9,
            'num_samples': num_samples
        }

def calculate_transformer_flops(seq_len: int, hidden_size: int, num_layers: int, 
                              vocab_size: int, num_heads: int) -> Dict[str, float]:
    """计算Transformer模型的理论FLOPs"""
    
    # 每层的FLOPs计算
    # 1. Self-attention
    attention_flops = 4 * seq_len * hidden_size * hidden_size + 2 * seq_len * seq_len * hidden_size
    
    # 2. Feed-forward network (通常是4 * hidden_size)
    ffn_flops = 8 * seq_len * hidden_size * hidden_size
    
    # 每层总FLOPs
    layer_flops = attention_flops + ffn_flops
    
    # 所有层的FLOPs
    total_layer_flops = layer_flops * num_layers
    
    # Embedding层FLOPs (通常很小，可以忽略)
    embedding_flops = seq_len * hidden_size
    
    # 输出层FLOPs
    output_flops = seq_len * hidden_size * vocab_size
    
    # 总FLOPs
    total_flops = total_layer_flops + embedding_flops + output_flops
    
    return {
        'attention_flops_per_layer': attention_flops,
        'ffn_flops_per_layer': ffn_flops,
        'total_flops_per_layer': layer_flops,
        'total_layer_flops': total_layer_flops,
        'embedding_flops': embedding_flops,
        'output_flops': output_flops,
        'total_flops': total_flops,
        'total_flops_m': total_flops / 1e6,
        'total_flops_g': total_flops / 1e9
    }
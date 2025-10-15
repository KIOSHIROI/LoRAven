# 动态性分析器
# 功能：分析LoRAven等动态方法的rank变化，包括层级和样本级别的变化

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

class DynamicAnalyzer:
    """动态性分析器，专门用于分析LoRAven等动态方法"""
    
    def __init__(self):
        self.rank_history = defaultdict(list)
        self.gate_history = defaultdict(list)
        self.layer_activations = defaultdict(list)
        
    def analyze_rank_dynamics(self, model, method: str) -> Dict[str, Any]:
        """分析rank动态变化"""
        dynamics = {
            'layer_wise_ranks': {},
            'rank_statistics': {},
            'rank_variations': {},
            'method': method
        }
        
        if method.lower() in ['loraven', 'adalora', 'dora']:
            # 收集各层的rank信息
            layer_ranks = []
            rank_info = {}
            
            for name, module in model.named_modules():
                if self._is_lora_module(module):
                    rank = self._extract_rank(module)
                    if rank is not None:
                        rank_info[name] = rank
                        layer_ranks.append(rank)
            
            dynamics['layer_wise_ranks'] = rank_info
            
            # 计算rank统计信息
            if layer_ranks:
                dynamics['rank_statistics'] = {
                    'mean_rank': float(np.mean(layer_ranks)),
                    'std_rank': float(np.std(layer_ranks)),
                    'min_rank': int(np.min(layer_ranks)),
                    'max_rank': int(np.max(layer_ranks)),
                    'rank_variance': float(np.var(layer_ranks)),
                    'total_layers': len(layer_ranks),
                    'rank_distribution': self._calculate_rank_distribution(layer_ranks)
                }
                
                # 计算rank变化指标
                dynamics['rank_variations'] = self._calculate_rank_variations(layer_ranks)
        
        return dynamics
    
    def _is_lora_module(self, module) -> bool:
        """判断是否为LoRA相关模块"""
        module_name = module.__class__.__name__.lower()
        return any(keyword in module_name for keyword in ['lora', 'adapter', 'peft'])
    
    def _extract_rank(self, module) -> Optional[int]:
        """从模块中提取rank信息"""
        # 尝试不同的rank属性名
        rank_attrs = ['r', 'rank', 'lora_r', 'adapter_rank']
        
        for attr in rank_attrs:
            if hasattr(module, attr):
                rank = getattr(module, attr)
                if isinstance(rank, (int, torch.Tensor)):
                    return int(rank) if isinstance(rank, torch.Tensor) else rank
        
        # 如果没有直接的rank属性，尝试从权重形状推断
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            return module.lora_A.shape[0]
        
        return None
    
    def _calculate_rank_distribution(self, ranks: List[int]) -> Dict[str, int]:
        """计算rank分布"""
        unique_ranks, counts = np.unique(ranks, return_counts=True)
        return {f'rank_{rank}': int(count) for rank, count in zip(unique_ranks, counts)}
    
    def _calculate_rank_variations(self, ranks: List[int]) -> Dict[str, float]:
        """计算rank变化指标"""
        if len(ranks) < 2:
            return {}
        
        # 计算相邻层之间的rank变化
        rank_diffs = np.diff(ranks)
        
        return {
            'layer_wise_rank_variance': float(np.var(rank_diffs)),
            'max_rank_jump': float(np.max(np.abs(rank_diffs))),
            'avg_rank_change': float(np.mean(np.abs(rank_diffs))),
            'rank_stability_score': 1.0 / (1.0 + np.std(ranks))  # 稳定性评分
        }
    
    def analyze_sample_wise_dynamics(self, model, dataloader, method: str, 
                                   num_samples: int = 100) -> Dict[str, Any]:
        """分析样本级别的动态变化"""
        if method.lower() not in ['loraven']:
            return {'message': 'Sample-wise dynamics only available for LoRAven'}
        
        sample_dynamics = {
            'sample_rank_variations': [],
            'gate_activations': [],
            'adaptation_patterns': {}
        }
        
        model.eval()
        sample_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= num_samples:
                    break
                
                # 收集每个样本的动态信息
                batch_dynamics = self._collect_batch_dynamics(model, batch)
                sample_dynamics['sample_rank_variations'].extend(batch_dynamics.get('rank_vars', []))
                sample_dynamics['gate_activations'].extend(batch_dynamics.get('gate_acts', []))
                
                sample_count += len(batch['input_ids']) if isinstance(batch, dict) else batch.size(0)
        
        # 计算样本级别统计
        if sample_dynamics['sample_rank_variations']:
            sample_dynamics['statistics'] = {
                'mean_sample_variance': float(np.mean(sample_dynamics['sample_rank_variations'])),
                'std_sample_variance': float(np.std(sample_dynamics['sample_rank_variations'])),
                'sample_adaptation_diversity': self._calculate_adaptation_diversity(
                    sample_dynamics['sample_rank_variations']
                )
            }
        
        return sample_dynamics
    
    def _collect_batch_dynamics(self, model, batch) -> Dict[str, List]:
        """收集批次动态信息"""
        dynamics = {'rank_vars': [], 'gate_acts': []}
        
        # 这里需要根据具体的LoRAven实现来收集动态信息
        # 由于LoRAven的具体实现可能不同，这里提供一个通用框架
        
        # 注册hooks来收集中间激活
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if hasattr(module, 'gate') or 'gate' in name.lower():
                    activations[name] = output.detach().cpu()
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if 'gate' in name.lower() or hasattr(module, 'gate'):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # 前向传播
        if isinstance(batch, dict):
            _ = model(**batch)
        else:
            _ = model(batch)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        # 处理收集到的激活
        for name, activation in activations.items():
            if activation is not None:
                # 计算激活的变化
                var = torch.var(activation).item()
                dynamics['gate_acts'].append(var)
        
        return dynamics
    
    def _calculate_adaptation_diversity(self, sample_variations: List[float]) -> float:
        """计算适应性多样性"""
        if not sample_variations:
            return 0.0
        
        # 使用变异系数作为多样性指标
        mean_var = np.mean(sample_variations)
        std_var = np.std(sample_variations)
        
        return std_var / mean_var if mean_var > 0 else 0.0
    
    def generate_rank_heatmap(self, layer_ranks: Dict[str, int], 
                            output_path: str = None) -> Optional[str]:
        """生成rank热力图"""
        if not layer_ranks:
            return None
        
        # 准备数据
        layer_names = list(layer_ranks.keys())
        ranks = list(layer_ranks.values())
        
        # 创建热力图数据
        heatmap_data = np.array(ranks).reshape(-1, 1)
        
        # 绘制热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data.T, 
                   xticklabels=[name.split('.')[-1] for name in layer_names],
                   yticklabels=['Rank'],
                   annot=True, 
                   fmt='d',
                   cmap='viridis',
                   cbar_kws={'label': 'Rank Value'})
        
        plt.title('Layer-wise Rank Distribution')
        plt.xlabel('Layers')
        plt.ylabel('Rank')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return None
    
    def generate_gate_activation_map(self, gate_activations: Dict[str, Any], 
                                   output_path: str = None) -> Optional[str]:
        """生成门控激活图"""
        if not gate_activations:
            return None
        
        # 这里需要根据具体的门控机制实现
        # 暂时创建一个示例图
        plt.figure(figsize=(12, 8))
        
        # 示例：如果有门控激活数据
        if isinstance(gate_activations, dict) and gate_activations:
            # 创建激活模式可视化
            activation_data = []
            labels = []
            
            for name, activations in gate_activations.items():
                if isinstance(activations, (list, np.ndarray, torch.Tensor)):
                    activation_data.append(np.array(activations).flatten())
                    labels.append(name)
            
            if activation_data:
                # 绘制激活分布
                for i, (data, label) in enumerate(zip(activation_data, labels)):
                    plt.subplot(len(activation_data), 1, i+1)
                    plt.hist(data, bins=50, alpha=0.7, label=label)
                    plt.xlabel('Activation Value')
                    plt.ylabel('Frequency')
                    plt.title(f'Gate Activation Distribution - {label}')
                    plt.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return None
    
    def save_dynamics_report(self, dynamics_data: Dict[str, Any], 
                           output_path: str):
        """保存动态性分析报告"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成markdown报告
        report_lines = [
            "# Dynamic Analysis Report",
            "",
            f"## Method: {dynamics_data.get('method', 'Unknown')}",
            "",
            "### Rank Statistics",
            ""
        ]
        
        if 'rank_statistics' in dynamics_data:
            stats = dynamics_data['rank_statistics']
            report_lines.extend([
                f"- **Mean Rank**: {stats.get('mean_rank', 'N/A'):.2f}",
                f"- **Std Rank**: {stats.get('std_rank', 'N/A'):.2f}",
                f"- **Min Rank**: {stats.get('min_rank', 'N/A')}",
                f"- **Max Rank**: {stats.get('max_rank', 'N/A')}",
                f"- **Rank Variance**: {stats.get('rank_variance', 'N/A'):.4f}",
                f"- **Total Layers**: {stats.get('total_layers', 'N/A')}",
                ""
            ])
        
        if 'rank_variations' in dynamics_data:
            variations = dynamics_data['rank_variations']
            report_lines.extend([
                "### Rank Variations",
                "",
                f"- **Layer-wise Rank Variance**: {variations.get('layer_wise_rank_variance', 'N/A'):.4f}",
                f"- **Max Rank Jump**: {variations.get('max_rank_jump', 'N/A'):.2f}",
                f"- **Average Rank Change**: {variations.get('avg_rank_change', 'N/A'):.2f}",
                f"- **Rank Stability Score**: {variations.get('rank_stability_score', 'N/A'):.4f}",
                ""
            ])
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Dynamic analysis report saved to {output_path}")
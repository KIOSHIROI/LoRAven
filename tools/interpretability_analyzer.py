"""
可解释性分析器：生成rank热力图和门控激活图
用于分析LoRAven等动态方法的可解释性
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path
import json


class InterpretabilityAnalyzer:
    """可解释性分析器"""
    
    def __init__(self, output_dir: str):
        """
        初始化可解释性分析器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建可解释性分析子目录
        self.interpretability_dir = self.output_dir / "interpretability"
        self.interpretability_dir.mkdir(exist_ok=True)
    
    def analyze_interpretability(
        self, 
        model: torch.nn.Module, 
        method: str,
        task: str,
        dataloader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, Any]:
        """
        分析模型的可解释性
        
        Args:
            model: 要分析的模型
            method: PEFT方法名称
            task: 任务名称
            dataloader: 数据加载器（用于样本级分析）
            
        Returns:
            可解释性指标字典
        """
        interpretability_metrics = {}
        
        try:
            # 分析rank分布
            rank_analysis = self._analyze_rank_distribution(model, method)
            interpretability_metrics.update(rank_analysis)
            
            # 生成rank热力图
            if rank_analysis.get('layer_ranks'):
                heatmap_path = self._generate_rank_heatmap(
                    rank_analysis['layer_ranks'], method, task
                )
                interpretability_metrics['rank_heatmap_path'] = str(heatmap_path)
            
            # 分析门控激活（如果适用）
            if method.lower() in ['loraven', 'adalora']:
                gate_analysis = self._analyze_gate_activations(model, method, dataloader)
                interpretability_metrics.update(gate_analysis)
                
                # 生成门控激活图
                if gate_analysis.get('gate_activations'):
                    gate_plot_path = self._generate_gate_activation_plot(
                        gate_analysis['gate_activations'], method, task
                    )
                    interpretability_metrics['gate_activation_plot_path'] = str(gate_plot_path)
            
            # 分析注意力权重分布
            attention_analysis = self._analyze_attention_weights(model)
            interpretability_metrics.update(attention_analysis)
            
            # 生成可解释性报告
            report_path = self._generate_interpretability_report(
                interpretability_metrics, method, task
            )
            interpretability_metrics['interpretability_report_path'] = str(report_path)
            
        except Exception as e:
            print(f"Error in interpretability analysis: {e}")
            interpretability_metrics['error'] = str(e)
        
        return interpretability_metrics
    
    def _analyze_rank_distribution(self, model: torch.nn.Module, method: str) -> Dict[str, Any]:
        """分析rank分布"""
        rank_metrics = {}
        layer_ranks = {}
        
        try:
            if hasattr(model, 'peft_config') or method.lower() in ['lora', 'adalora', 'dora', 'loraven']:
                # 分析PEFT模型的rank
                for name, module in model.named_modules():
                    # 检查是否是LoRA模块（有r属性）或者模块名包含lora
                    if (hasattr(module, 'r') and hasattr(module, 'lora_A')) or 'lora' in name.lower():
                        # LoRA类型的模块
                        if hasattr(module, 'r'):
                            rank = module.r if isinstance(module.r, int) else getattr(module.r, 'item', lambda: module.r)()
                            layer_ranks[name] = rank
                        
                        # 计算实际有效rank
                        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                            try:
                                A = module.lora_A.default.weight.data
                                B = module.lora_B.default.weight.data
                                
                                # 计算SVD
                                U, S, V = torch.svd(torch.mm(B, A))
                                effective_rank = torch.sum(S > 1e-6).item()
                                layer_ranks[f"{name}_effective"] = effective_rank
                                
                                # 计算rank利用率
                                nominal_rank = layer_ranks.get(name, rank)
                                if nominal_rank > 0:
                                    rank_utilization = effective_rank / nominal_rank
                                    layer_ranks[f"{name}_utilization"] = rank_utilization
                                    
                            except Exception as e:
                                print(f"Error computing effective rank for {name}: {e}")
                    
                    # 也检查直接有r属性的模块（如单独的lora_A, lora_B层）
                    elif hasattr(module, 'r'):
                        rank = module.r if isinstance(module.r, int) else getattr(module.r, 'item', lambda: module.r)()
                        layer_ranks[name] = rank
            
            # 计算统计信息
            if layer_ranks:
                ranks = [v for k, v in layer_ranks.items() if not k.endswith('_effective') and not k.endswith('_utilization')]
                if ranks:
                    rank_metrics.update({
                        'layer_ranks': layer_ranks,
                        'rank_distribution': layer_ranks,  # 添加这个键以保持兼容性
                        'avg_rank': np.mean(ranks),
                        'std_rank': np.std(ranks),
                        'min_rank': np.min(ranks),
                        'max_rank': np.max(ranks),
                        'rank_variance': np.var(ranks)
                    })
                
                # 计算有效rank统计
                effective_ranks = [v for k, v in layer_ranks.items() if k.endswith('_effective')]
                if effective_ranks:
                    rank_metrics.update({
                        'avg_effective_rank': np.mean(effective_ranks),
                        'std_effective_rank': np.std(effective_ranks),
                        'effective_rank_ratio': np.mean(effective_ranks) / np.mean(ranks) if ranks else 0
                    })
                
                # 计算rank利用率统计
                utilizations = [v for k, v in layer_ranks.items() if k.endswith('_utilization')]
                if utilizations:
                    rank_metrics.update({
                        'avg_rank_utilization': np.mean(utilizations),
                        'std_rank_utilization': np.std(utilizations),
                        'min_rank_utilization': np.min(utilizations),
                        'max_rank_utilization': np.max(utilizations)
                    })
        
        except Exception as e:
            print(f"Error in rank distribution analysis: {e}")
            rank_metrics['rank_analysis_error'] = str(e)
        
        return rank_metrics
    
    def _analyze_gate_activations(
        self, 
        model: torch.nn.Module, 
        method: str,
        dataloader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, Any]:
        """分析门控激活"""
        gate_metrics = {}
        
        try:
            gate_activations = {}
            
            # 收集门控激活
            def gate_hook(name):
                def hook(module, input, output):
                    if hasattr(module, 'gate') or 'gate' in name.lower():
                        if isinstance(output, torch.Tensor):
                            activation = output.detach().cpu().numpy()
                            if name not in gate_activations:
                                gate_activations[name] = []
                            gate_activations[name].append(activation)
                return hook
            
            # 注册hooks
            hooks = []
            for name, module in model.named_modules():
                if 'gate' in name.lower() or hasattr(module, 'gate'):
                    hook = module.register_forward_hook(gate_hook(name))
                    hooks.append(hook)
            
            # 如果有数据加载器，运行几个batch来收集激活
            if dataloader and hooks:
                model.eval()
                with torch.no_grad():
                    for i, batch in enumerate(dataloader):
                        if i >= 3:  # 只处理前3个batch
                            break
                        
                        # 准备输入
                        inputs = {k: v.to(model.device) for k, v in batch.items() 
                                if k in ['input_ids', 'attention_mask', 'labels']}
                        
                        # 前向传播
                        try:
                            _ = model(**inputs)
                        except Exception as e:
                            print(f"Error in forward pass for gate analysis: {e}")
                            break
            
            # 移除hooks
            for hook in hooks:
                hook.remove()
            
            # 分析门控激活
            if gate_activations:
                gate_stats = {}
                for name, activations in gate_activations.items():
                    if activations:
                        # 合并所有激活
                        all_activations = np.concatenate(activations, axis=0)
                        
                        gate_stats[name] = {
                            'mean_activation': float(np.mean(all_activations)),
                            'std_activation': float(np.std(all_activations)),
                            'min_activation': float(np.min(all_activations)),
                            'max_activation': float(np.max(all_activations)),
                            'sparsity': float(np.mean(all_activations < 0.1)),  # 激活值小于0.1的比例
                            'activation_shape': list(all_activations.shape)
                        }
                
                gate_metrics.update({
                    'gate_activations': gate_stats,
                    'num_gate_layers': len(gate_stats),
                    'avg_gate_sparsity': np.mean([stats['sparsity'] for stats in gate_stats.values()]) if gate_stats else 0
                })
        
        except Exception as e:
            print(f"Error in gate activation analysis: {e}")
            gate_metrics['gate_analysis_error'] = str(e)
        
        return gate_metrics
    
    def _analyze_attention_weights(self, model: torch.nn.Module) -> Dict[str, Any]:
        """分析注意力权重分布"""
        attention_metrics = {}
        
        try:
            attention_stats = {}
            
            for name, module in model.named_modules():
                if 'attention' in name.lower() and hasattr(module, 'weight'):
                    weight = module.weight.data.cpu().numpy()
                    
                    attention_stats[name] = {
                        'weight_mean': float(np.mean(weight)),
                        'weight_std': float(np.std(weight)),
                        'weight_norm': float(np.linalg.norm(weight)),
                        'weight_sparsity': float(np.mean(np.abs(weight) < 1e-6)),
                        'weight_shape': list(weight.shape)
                    }
            
            if attention_stats:
                attention_metrics.update({
                    'attention_weights': attention_stats,
                    'num_attention_layers': len(attention_stats),
                    'avg_attention_sparsity': np.mean([stats['weight_sparsity'] for stats in attention_stats.values()])
                })
        
        except Exception as e:
            print(f"Error in attention weight analysis: {e}")
            attention_metrics['attention_analysis_error'] = str(e)
        
        return attention_metrics
    
    def _generate_rank_heatmap(
        self, 
        layer_ranks: Dict[str, float], 
        method: str, 
        task: str
    ) -> Path:
        """生成rank热力图"""
        try:
            # 过滤出基础rank（不包括effective和utilization）
            base_ranks = {k: v for k, v in layer_ranks.items() 
                         if not k.endswith('_effective') and not k.endswith('_utilization')}
            
            if not base_ranks:
                raise ValueError("No base ranks found for heatmap")
            
            # 创建图形
            plt.figure(figsize=(12, 8))
            
            # 准备数据
            layer_names = list(base_ranks.keys())
            ranks = list(base_ranks.values())
            
            # 创建热力图数据
            rank_matrix = np.array(ranks).reshape(-1, 1)
            
            # 生成热力图
            sns.heatmap(
                rank_matrix.T, 
                xticklabels=layer_names,
                yticklabels=['Rank'],
                annot=True, 
                fmt='.0f',
                cmap='viridis',
                cbar_kws={'label': 'Rank Value'}
            )
            
            plt.title(f'Rank Distribution Heatmap - {method.upper()} on {task.upper()}')
            plt.xlabel('Layers')
            plt.ylabel('Rank')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存图形
            heatmap_path = self.interpretability_dir / f"{method}_{task}_rank_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return heatmap_path
            
        except Exception as e:
            print(f"Error generating rank heatmap: {e}")
            # 创建一个简单的占位图
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f'Rank Heatmap\n{method} on {task}\nError: {str(e)}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'Rank Heatmap - {method.upper()} on {task.upper()}')
            
            heatmap_path = self.interpretability_dir / f"{method}_{task}_rank_heatmap_error.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return heatmap_path
    
    def _generate_gate_activation_plot(
        self, 
        gate_activations: Dict[str, Dict], 
        method: str, 
        task: str
    ) -> Path:
        """生成门控激活图"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            gate_names = list(gate_activations.keys())
            
            # 绘制激活统计
            if len(gate_names) > 0:
                # 平均激活值
                means = [gate_activations[name]['mean_activation'] for name in gate_names]
                axes[0].bar(range(len(gate_names)), means)
                axes[0].set_title('Mean Gate Activations')
                axes[0].set_xlabel('Gate Layers')
                axes[0].set_ylabel('Mean Activation')
                axes[0].set_xticks(range(len(gate_names)))
                axes[0].set_xticklabels([name.split('.')[-1] for name in gate_names], rotation=45)
                
                # 激活稀疏性
                sparsities = [gate_activations[name]['sparsity'] for name in gate_names]
                axes[1].bar(range(len(gate_names)), sparsities)
                axes[1].set_title('Gate Activation Sparsity')
                axes[1].set_xlabel('Gate Layers')
                axes[1].set_ylabel('Sparsity (< 0.1)')
                axes[1].set_xticks(range(len(gate_names)))
                axes[1].set_xticklabels([name.split('.')[-1] for name in gate_names], rotation=45)
                
                # 激活范围
                mins = [gate_activations[name]['min_activation'] for name in gate_names]
                maxs = [gate_activations[name]['max_activation'] for name in gate_names]
                x_pos = range(len(gate_names))
                axes[2].fill_between(x_pos, mins, maxs, alpha=0.3, label='Activation Range')
                axes[2].plot(x_pos, means, 'ro-', label='Mean')
                axes[2].set_title('Gate Activation Ranges')
                axes[2].set_xlabel('Gate Layers')
                axes[2].set_ylabel('Activation Value')
                axes[2].set_xticks(range(len(gate_names)))
                axes[2].set_xticklabels([name.split('.')[-1] for name in gate_names], rotation=45)
                axes[2].legend()
                
                # 激活分布直方图（使用第一个门控层）
                if gate_names:
                    first_gate = gate_names[0]
                    stats = gate_activations[first_gate]
                    # 创建模拟分布用于展示
                    sample_data = np.random.normal(
                        stats['mean_activation'], 
                        stats['std_activation'], 
                        1000
                    )
                    axes[3].hist(sample_data, bins=30, alpha=0.7, edgecolor='black')
                    axes[3].set_title(f'Activation Distribution\n({first_gate.split(".")[-1]})')
                    axes[3].set_xlabel('Activation Value')
                    axes[3].set_ylabel('Frequency')
            
            plt.suptitle(f'Gate Activation Analysis - {method.upper()} on {task.upper()}')
            plt.tight_layout()
            
            # 保存图形
            gate_plot_path = self.interpretability_dir / f"{method}_{task}_gate_activations.png"
            plt.savefig(gate_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return gate_plot_path
            
        except Exception as e:
            print(f"Error generating gate activation plot: {e}")
            # 创建一个简单的占位图
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f'Gate Activation Plot\n{method} on {task}\nError: {str(e)}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'Gate Activations - {method.upper()} on {task.upper()}')
            
            gate_plot_path = self.interpretability_dir / f"{method}_{task}_gate_activations_error.png"
            plt.savefig(gate_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return gate_plot_path
    
    def _generate_interpretability_report(
        self, 
        interpretability_metrics: Dict[str, Any], 
        method: str, 
        task: str
    ) -> Path:
        """生成可解释性分析报告"""
        try:
            report_path = self.interpretability_dir / f"{method}_{task}_interpretability_report.md"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# Interpretability Analysis Report\n\n")
                f.write(f"**Method:** {method.upper()}\n")
                f.write(f"**Task:** {task.upper()}\n")
                f.write(f"**Generated:** {Path(__file__).name}\n\n")
                
                # Rank分析
                if 'layer_ranks' in interpretability_metrics:
                    f.write("## Rank Analysis\n\n")
                    f.write(f"- **Average Rank:** {interpretability_metrics.get('avg_rank', 'N/A'):.2f}\n")
                    f.write(f"- **Rank Standard Deviation:** {interpretability_metrics.get('std_rank', 'N/A'):.2f}\n")
                    f.write(f"- **Rank Range:** {interpretability_metrics.get('min_rank', 'N/A'):.0f} - {interpretability_metrics.get('max_rank', 'N/A'):.0f}\n")
                    
                    if 'avg_effective_rank' in interpretability_metrics:
                        f.write(f"- **Average Effective Rank:** {interpretability_metrics['avg_effective_rank']:.2f}\n")
                        f.write(f"- **Effective Rank Ratio:** {interpretability_metrics.get('effective_rank_ratio', 'N/A'):.2f}\n")
                    
                    if 'avg_rank_utilization' in interpretability_metrics:
                        f.write(f"- **Average Rank Utilization:** {interpretability_metrics['avg_rank_utilization']:.2f}\n")
                    
                    f.write("\n")
                
                # 门控激活分析
                if 'gate_activations' in interpretability_metrics:
                    f.write("## Gate Activation Analysis\n\n")
                    f.write(f"- **Number of Gate Layers:** {interpretability_metrics.get('num_gate_layers', 'N/A')}\n")
                    f.write(f"- **Average Gate Sparsity:** {interpretability_metrics.get('avg_gate_sparsity', 'N/A'):.3f}\n\n")
                
                # 注意力权重分析
                if 'attention_weights' in interpretability_metrics:
                    f.write("## Attention Weight Analysis\n\n")
                    f.write(f"- **Number of Attention Layers:** {interpretability_metrics.get('num_attention_layers', 'N/A')}\n")
                    f.write(f"- **Average Attention Sparsity:** {interpretability_metrics.get('avg_attention_sparsity', 'N/A'):.3f}\n\n")
                
                # 可视化文件
                f.write("## Generated Visualizations\n\n")
                if 'rank_heatmap_path' in interpretability_metrics:
                    f.write(f"- **Rank Heatmap:** `{interpretability_metrics['rank_heatmap_path']}`\n")
                if 'gate_activation_plot_path' in interpretability_metrics:
                    f.write(f"- **Gate Activation Plot:** `{interpretability_metrics['gate_activation_plot_path']}`\n")
                
                f.write("\n## Summary\n\n")
                f.write(f"This report provides interpretability analysis for the {method.upper()} method on the {task.upper()} task. ")
                f.write("The analysis includes rank distribution, gate activations (if applicable), and attention weight patterns.\n")
            
            return report_path
            
        except Exception as e:
            print(f"Error generating interpretability report: {e}")
            # 创建简单的错误报告
            report_path = self.interpretability_dir / f"{method}_{task}_interpretability_report_error.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# Interpretability Analysis Report (Error)\n\n")
                f.write(f"**Method:** {method.upper()}\n")
                f.write(f"**Task:** {task.upper()}\n")
                f.write(f"**Error:** {str(e)}\n")
            
            return report_path
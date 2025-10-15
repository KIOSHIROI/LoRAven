# 详细评估指标收集器
# 功能：收集和计算各种评估指标，包括任务性能、参数效率、资源消耗、动态性、收敛性、可解释性等

import time
import torch
import numpy as np
import psutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
from .flops_calculator import FLOPsCalculator
from .dynamic_analyzer import DynamicAnalyzer
from .convergence_analyzer import ConvergenceAnalyzer
from .interpretability_analyzer import InterpretabilityAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class DetailedMetrics:
    """详细评估指标数据结构"""
    # 任务性能指标
    task_performance: Dict[str, float] = field(default_factory=dict)
    
    # 参数效率指标
    parameter_efficiency: Dict[str, Any] = field(default_factory=dict)
    
    # 资源消耗指标
    resource_consumption: Dict[str, float] = field(default_factory=dict)
    
    # 动态性指标 (LoRAven独有)
    dynamic_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 收敛与稳定性指标
    convergence_stability: Dict[str, float] = field(default_factory=dict)
    
    # 可解释性指标
    interpretability: Dict[str, Any] = field(default_factory=dict)
    
    # 泛化性指标
    generalization: Dict[str, float] = field(default_factory=dict)

class MetricsCollector:
    """详细指标收集器"""
    
    def __init__(self, output_dir: str = None):
        self.training_history = []
        self.memory_usage_history = []
        self.start_time = None
        self.end_time = None
        self.output_dir = output_dir or "."
        self.flops_calculator = FLOPsCalculator()
        self.dynamic_analyzer = DynamicAnalyzer()
        self.convergence_analyzer = ConvergenceAnalyzer()
        self.interpretability_analyzer = InterpretabilityAnalyzer(self.output_dir)
        
    def start_collection(self):
        """开始指标收集"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
    def end_collection(self):
        """结束指标收集"""
        self.end_time = time.time()
        
    def collect_parameter_efficiency(self, model, method: str) -> Dict[str, Any]:
        """收集参数效率指标"""
        metrics = {}
        
        # 计算总参数数量
        total_params = sum(p.numel() for p in model.parameters())
        
        # 计算可训练参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算参数效率比例
        efficiency_ratio = trainable_params / total_params if total_params > 0 else 0.0
        
        metrics.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'efficiency_ratio': efficiency_ratio,
            'efficiency_percentage': efficiency_ratio * 100,
            'parameter_reduction': (1 - efficiency_ratio) * 100,
            'method': method
        })
        
        # 如果是PEFT方法，计算适配器参数
        if hasattr(model, 'peft_config'):
            adapter_params = 0
            for name, param in model.named_parameters():
                if 'lora' in name.lower() or 'adapter' in name.lower():
                    adapter_params += param.numel()
            
            metrics.update({
                'adapter_parameters': adapter_params,
                'adapter_ratio': adapter_params / total_params if total_params > 0 else 0.0
            })
            
        return metrics
    
    def collect_resource_consumption(self, model, tokenizer, sample_input=None) -> Dict[str, float]:
        """收集资源消耗指标"""
        metrics = {}
        
        # GPU内存使用量
        if torch.cuda.is_available():
            # 当前内存使用
            current_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            metrics.update({
                'gpu_memory_current_gb': current_memory,
                'gpu_memory_peak_gb': peak_memory,
                'gpu_memory_total_gb': total_memory,
                'gpu_memory_utilization': peak_memory / total_memory * 100
            })
        
        # CPU内存使用量
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / (1024**3)  # GB
        system_memory = psutil.virtual_memory()
        
        metrics.update({
            'cpu_memory_gb': cpu_memory,
            'system_memory_total_gb': system_memory.total / (1024**3),
            'system_memory_available_gb': system_memory.available / (1024**3),
            'system_memory_percent': system_memory.percent
        })
        
        # FLOPs计算
        if sample_input is not None:
            try:
                # 获取输入形状
                if isinstance(sample_input, dict):
                    input_shape = {k: v.shape for k, v in sample_input.items()}
                else:
                    input_shape = sample_input.shape
                
                device = next(model.parameters()).device
                flops_results = self.flops_calculator.calculate_model_flops(
                    model, input_shape, str(device)
                )
                metrics.update({
                    'total_flops': flops_results['total_flops'],
                    'total_flops_m': flops_results['total_flops_m'],
                    'total_flops_g': flops_results['total_flops_g'],
                    'flops_per_param': flops_results['flops_per_param']
                })
            except Exception as e:
                logger.warning(f"Failed to calculate FLOPs: {e}")
        
        # 推理延迟测试
        if sample_input is not None:
            latency_metrics = self._measure_inference_latency(model, sample_input)
            metrics.update(latency_metrics)
            
        # 训练时间
        if self.start_time and self.end_time:
            training_time = self.end_time - self.start_time
            metrics['training_time_seconds'] = training_time
            metrics['training_time_minutes'] = training_time / 60
            
        return metrics
    
    def _measure_inference_latency(self, model, sample_input, num_runs: int = 100) -> Dict[str, float]:
        """测量推理延迟"""
        model.eval()
        latencies = []
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                if isinstance(sample_input, dict):
                    _ = model(**sample_input)
                else:
                    _ = model(sample_input)
        
        # 测量延迟
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                if isinstance(sample_input, dict):
                    _ = model(**sample_input)
                else:
                    _ = model(sample_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # ms
        
        return {
            'inference_latency_mean_ms': np.mean(latencies),
            'inference_latency_std_ms': np.std(latencies),
            'inference_latency_min_ms': np.min(latencies),
            'inference_latency_max_ms': np.max(latencies),
            'inference_latency_p95_ms': np.percentile(latencies, 95),
            'inference_latency_p99_ms': np.percentile(latencies, 99)
        }
    
    def collect_dynamic_metrics(self, model, method: str, dataloader=None) -> Dict[str, Any]:
        """收集动态性指标 (主要针对LoRAven等动态方法)"""
        metrics = {}
        
        # 使用动态分析器收集rank动态信息
        rank_dynamics = self.dynamic_analyzer.analyze_rank_dynamics(model, method)
        metrics.update(rank_dynamics)
        
        # 如果提供了数据加载器，分析样本级别的动态性
        if dataloader is not None and method.lower() in ['loraven']:
            sample_dynamics = self.dynamic_analyzer.analyze_sample_wise_dynamics(
                model, dataloader, method, num_samples=50
            )
            metrics['sample_dynamics'] = sample_dynamics
            
        return metrics
    
    def _collect_rank_variations(self, model) -> Dict[str, Any]:
        """收集rank变化信息"""
        rank_info = {
            'layer_wise_ranks': {},
            'rank_statistics': {}
        }
        
        # 收集各层的rank信息
        layer_ranks = []
        for name, module in model.named_modules():
            if hasattr(module, 'r') or hasattr(module, 'rank'):
                rank = getattr(module, 'r', getattr(module, 'rank', None))
                if rank is not None:
                    rank_info['layer_wise_ranks'][name] = rank
                    layer_ranks.append(rank)
        
        # 计算rank统计信息
        if layer_ranks:
            rank_info['rank_statistics'] = {
                'mean_rank': np.mean(layer_ranks),
                'std_rank': np.std(layer_ranks),
                'min_rank': np.min(layer_ranks),
                'max_rank': np.max(layer_ranks),
                'rank_variance': np.var(layer_ranks)
            }
            
        return rank_info
    
    def collect_convergence_stability(self, training_history: List[Dict], 
                                    target_metric: str = 'eval_accuracy') -> Dict[str, float]:
        """收集收敛与稳定性指标"""
        return self.convergence_analyzer.analyze_convergence_stability(
            training_history, target_metric
        )
    
    def collect_interpretability_metrics(
        self, 
        model: torch.nn.Module, 
        method: str,
        task: str,
        dataloader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, Any]:
        """
        收集可解释性指标
        
        Args:
            model: 要分析的模型
            method: PEFT方法名称
            task: 任务名称
            dataloader: 数据加载器（用于样本级分析）
            
        Returns:
            可解释性指标字典
        """
        return self.interpretability_analyzer.analyze_interpretability(
            model, method, task, dataloader
        )
    
    def _analyze_weight_distributions(self, model) -> Dict[str, Any]:
        """分析权重分布"""
        weight_stats = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad and 'lora' in name.lower():
                param_data = param.data.cpu().numpy()
                weight_stats[name] = {
                    'mean': float(np.mean(param_data)),
                    'std': float(np.std(param_data)),
                    'min': float(np.min(param_data)),
                    'max': float(np.max(param_data)),
                    'norm': float(np.linalg.norm(param_data))
                }
                
        return weight_stats
    
    def _collect_gate_activations(self, model) -> Dict[str, Any]:
        """收集门控激活信息"""
        gate_info = {}
        
        # 这里需要根据具体的LoRAven实现来收集门控信息
        # 暂时返回空字典，具体实现需要根据LoRAven的架构调整
        
        return gate_info
    
    def generate_comprehensive_report(self, 
                                   method: str, 
                                   task: str, 
                                   model,
                                   eval_results: Dict,
                                   training_history: List[Dict] = None,
                                   sample_input: Dict = None,
                                   dataloader = None) -> DetailedMetrics:
        """生成综合评估报告"""
        
        # 任务性能指标
        task_performance = {
            'accuracy': eval_results.get('eval_accuracy', 0.0),
            'f1_score': eval_results.get('eval_f1', 0.0),
            'matthews_correlation': eval_results.get('eval_matthews_correlation', 0.0),
            'runtime': eval_results.get('eval_runtime', 0.0),
            'samples_per_second': eval_results.get('eval_samples_per_second', 0.0)
        }
        
        # 参数效率指标
        parameter_efficiency = self.collect_parameter_efficiency(model, method)
        
        # 资源消耗指标
        resource_consumption = self.collect_resource_consumption(model, None, sample_input)
        
        # 动态性指标
        dynamic_metrics = self.collect_dynamic_metrics(model, method, dataloader)
        
        # 收敛与稳定性指标
        convergence_stability = {}
        if training_history:
            convergence_stability = self.collect_convergence_stability(
                training_history, 'eval_accuracy'
            )
        
        # 可解释性指标
        interpretability = self.collect_interpretability_metrics(
            model, method, task, dataloader
        )
        
        return DetailedMetrics(
            task_performance=task_performance,
            parameter_efficiency=parameter_efficiency,
            resource_consumption=resource_consumption,
            dynamic_metrics=dynamic_metrics,
            convergence_stability=convergence_stability,
            interpretability=interpretability
        )
    
    def save_metrics(self, metrics: DetailedMetrics, output_path: str):
        """保存指标到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化的格式
        metrics_dict = {
            'task_performance': metrics.task_performance,
            'parameter_efficiency': metrics.parameter_efficiency,
            'resource_consumption': metrics.resource_consumption,
            'dynamic_metrics': metrics.dynamic_metrics,
            'convergence_stability': metrics.convergence_stability,
            'interpretability': metrics.interpretability,
            'generalization': metrics.generalization
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Detailed metrics saved to {output_path}")
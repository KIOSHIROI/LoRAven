"""
Base benchmark interface for pluggable benchmark system
Integrates with PyTorch's built-in profiling capabilities
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import time
import gc
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable, Type
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np

# 尝试导入psutil，如果不可用则使用替代方案
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution"""
    warmup_iterations: int = 10
    measurement_iterations: int = 100
    enable_cuda_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = True
    profile_memory: bool = True
    record_shapes: bool = True
    with_stack: bool = False
    export_chrome_trace: bool = False
    export_stacks: bool = False


@dataclass
class BenchmarkResult:
    """Standardized benchmark result container"""
    benchmark_name: str
    metrics: Dict[str, Any]
    profiler_data: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    memory_usage: Dict[str, float] = None
    error: Optional[str] = None
    
    # Additional fields for compatibility
    model_name: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.memory_usage is None:
            self.memory_usage = {}
        # Map error to error_message for compatibility
        if self.error and not self.error_message:
            self.error_message = self.error
        if self.error_message and not self.error:
            self.error = self.error_message
    
    @property
    def throughput(self) -> float:
        """Get throughput from metrics"""
        if self.success and self.metrics:
            # Try to get from first batch size results
            for key, value in self.metrics.items():
                if isinstance(value, dict):
                    # Look for throughput data in nested structure
                    for sub_key, sub_value in value.items():
                        if 'throughput' in sub_key and isinstance(sub_value, dict):
                            if 'samples_per_second' in sub_value:
                                return sub_value['samples_per_second']
        return 0.0
    
    @property
    def avg_latency(self) -> float:
        """Get average latency from metrics"""
        if self.success and self.metrics:
            # Try to get from first batch size results
            for key, value in self.metrics.items():
                if isinstance(value, dict):
                    # Look for latency data in nested structure
                    for sub_key, sub_value in value.items():
                        if 'latency' in sub_key and isinstance(sub_value, dict):
                            if 'mean' in sub_value:
                                return float(sub_value['mean'])
        return 0.0
    
    @property
    def memory_peak_mb(self) -> float:
        """Get peak memory usage from memory_usage or metrics"""
        # First try memory_usage field
        if hasattr(self, 'memory_usage') and self.memory_usage:
            if 'peak_memory_mb' in self.memory_usage:
                return self.memory_usage['peak_memory_mb']
            if 'allocated_mb' in self.memory_usage:
                return self.memory_usage['allocated_mb']
        
        # Then try metrics field
        if self.success and self.metrics:
            for key, value in self.metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if 'memory' in sub_key and isinstance(sub_value, dict):
                            if 'peak_memory_mb' in sub_value:
                                return sub_value['peak_memory_mb']
                    if 'memory_peak_mb' in value:
                        return value['memory_peak_mb']
                    if 'peak_memory_mb' in value:
                        return value['peak_memory_mb']
        return 0.0
    
    @property
    def peak_memory_mb(self) -> float:
        """Get peak memory usage from memory_usage or metrics (alias for memory_peak_mb)"""
        return self.memory_peak_mb
    
    @property
    def memory_allocated_mb(self) -> float:
        """Get allocated memory in MB"""
        if self.success and self.metrics:
            for key, value in self.metrics.items():
                if isinstance(value, dict) and 'memory_allocated_mb' in value:
                    return value['memory_allocated_mb']
        return 0.0
    
    @property
    def avg_memory_mb(self) -> float:
        """Get average memory usage from metrics"""
        if self.success and self.metrics:
            for key, value in self.metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if 'memory' in sub_key and isinstance(sub_value, dict):
                            if 'memory_used_mb' in sub_value:
                                return sub_value['memory_used_mb']
        return 0.0
    
    @property
    def memory_efficiency(self) -> float:
        """Get memory efficiency from metrics"""
        if self.success and self.metrics:
            for key, value in self.metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if 'memory' in sub_key and isinstance(sub_value, dict):
                            if 'memory_efficiency' in sub_value:
                                return sub_value['memory_efficiency']
        return 0.0
    
    @property
    def energy_consumption_j(self) -> float:
        """Get energy consumption in Joules"""
        if self.success and self.metrics:
            for key, value in self.metrics.items():
                if isinstance(value, dict) and 'energy_consumption_j' in value:
                    return value['energy_consumption_j']
        return 0.0
    
    @property
    def total_energy_j(self) -> float:
        """Get total energy consumption from metrics"""
        if self.success and self.metrics:
            for key, value in self.metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if 'energy' in sub_key and isinstance(sub_value, dict):
                            if 'mean_joules' in sub_value:
                                return sub_value['mean_joules']
        return 0.0
    
    @property
    def avg_power_w(self) -> float:
        """Get average power consumption from metrics"""
        if self.success and self.metrics:
            for key, value in self.metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if 'power' in sub_key and isinstance(sub_value, dict):
                            if 'mean_watts' in sub_value:
                                return sub_value['mean_watts']
        return 0.0
    
    @property
    def energy_efficiency(self) -> float:
        """Get energy efficiency from metrics"""
        if self.success and self.metrics:
            for key, value in self.metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if 'efficiency' in sub_key and isinstance(sub_value, dict):
                            if 'energy_efficiency_samples_per_joule' in sub_value:
                                return sub_value['energy_efficiency_samples_per_joule']
        return 0.0
    
    @property
    def batch_size_scaling(self) -> dict:
        """Get batch size scaling results from metrics"""
        if self.success and self.metrics:
            # Look for batch_scaling data in metrics
            if 'batch_scaling' in self.metrics:
                return self.metrics['batch_scaling']
            # Look for any batch size related data
            for key, value in self.metrics.items():
                if 'batch' in key and isinstance(value, dict):
                    return value
            # Return the entire metrics if it contains scaling data
            if any('batch_size' in str(k) for k in self.metrics.keys()):
                return self.metrics
        return {}
    
    @property
    def scalability_score(self) -> float:
        """Get scalability score from metrics"""
        if self.success and self.metrics:
            # Look in batch_scaling results
            if 'batch_scaling' in self.metrics:
                batch_scaling = self.metrics['batch_scaling']
                if isinstance(batch_scaling, dict):
                    if 'scaling_exponent' in batch_scaling:
                        return batch_scaling['scaling_exponent']
                    if 'efficiency_scores' in batch_scaling and batch_scaling['efficiency_scores']:
                        return np.mean(batch_scaling['efficiency_scores'])
            
            # Look in other metrics
            for key, value in self.metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if 'scalability_score' in sub_key:
                            return sub_value
                        if 'scaling_efficiency' in sub_key:
                            return sub_value
        return 0.0
    
    @property
    def scaling_efficiency(self) -> float:
        """Get scaling efficiency"""
        if self.success and self.metrics:
            for key, value in self.metrics.items():
                if isinstance(value, dict) and 'scaling_efficiency' in value:
                    return value['scaling_efficiency']
        return 0.0


class BaseBenchmark(ABC):
    """
    Abstract base class for all benchmarks
    Provides common functionality and PyTorch profiler integration
    """
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.profiler_data = {}
        
    @abstractmethod
    def run_benchmark(self, 
                     model: nn.Module, 
                     data_loader: DataLoader,
                     **kwargs) -> BenchmarkResult:
        """
        Run the specific benchmark
        
        Args:
            model: PyTorch model to benchmark
            data_loader: Data loader for benchmark data
            **kwargs: Additional benchmark-specific parameters
            
        Returns:
            BenchmarkResult containing metrics and profiling data
        """
        pass
    
    @abstractmethod
    def get_benchmark_name(self) -> str:
        """Return the name of this benchmark"""
        pass
    
    def setup_profiler(self) -> torch.profiler.profile:
        """
        Setup PyTorch profiler with appropriate configuration
        
        Returns:
            Configured PyTorch profiler
        """
        activities = []
        if self.config.enable_cpu_profiling:
            activities.append(ProfilerActivity.CPU)
        if self.config.enable_cuda_profiling and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        
        return profile(
            activities=activities,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=True,
            with_modules=True
        )
    
    @contextmanager
    def memory_monitor(self):
        """Context manager for monitoring memory usage"""
        # Clear cache and collect garbage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Record initial memory state
        initial_memory = self._get_memory_stats()
        
        try:
            yield
        finally:
            # Record final memory state
            final_memory = self._get_memory_stats()
            
            # Calculate memory usage
            self.memory_usage = {
                'cpu_memory_delta': final_memory['cpu'] - initial_memory['cpu'],
                'gpu_memory_delta': final_memory['gpu'] - initial_memory['gpu'],
                'cpu_memory_peak': final_memory['cpu_peak'],
                'gpu_memory_peak': final_memory['gpu_peak']
            }
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """获取内存统计信息"""
        stats = {}
        
        if PSUTIL_AVAILABLE:
            stats.update({
                'cpu': psutil.Process().memory_info().rss / 1024**2,  # MB
                'cpu_peak': psutil.Process().memory_info().peak_wss / 1024**2 if hasattr(psutil.Process().memory_info(), 'peak_wss') else 0,
            })
        else:
            stats.update({
                'cpu': 100.0,  # 默认值
                'cpu_peak': 100.0,  # 默认值
            })
        
        if torch.cuda.is_available():
            stats['gpu'] = torch.cuda.memory_allocated() / 1024**2  # MB
            stats['gpu_peak'] = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            stats['gpu'] = 0
            stats['gpu_peak'] = 0
        
        return stats
    
    def warmup_model(self, model: nn.Module, data_loader: DataLoader):
        """
        Warmup model for stable benchmarking
        
        Args:
            model: Model to warmup
            data_loader: Data loader for warmup
        """
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= self.config.warmup_iterations:
                    break
                
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                    inputs = batch[0]
                else:
                    inputs = batch.to(self.device)
                
                # Forward pass
                _ = model(inputs)
                
                # Clear cache periodically
                if torch.cuda.is_available() and i % 5 == 0:
                    torch.cuda.empty_cache()
    
    def extract_profiler_metrics(self, prof: torch.profiler.profile) -> Dict[str, Any]:
        """
        Extract key metrics from PyTorch profiler
        
        Args:
            prof: PyTorch profiler instance
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}
        
        try:
            # Get key averages
            key_averages = prof.key_averages()
            
            # CPU time metrics
            cpu_time_total = sum([item.cpu_time_total for item in key_averages])
            cpu_time_avg = np.mean([item.cpu_time for item in key_averages if item.count > 0])
            
            metrics['cpu_time_total_us'] = cpu_time_total
            metrics['cpu_time_avg_us'] = cpu_time_avg
            
            # CUDA time metrics (if available)
            if torch.cuda.is_available():
                cuda_time_total = sum([item.cuda_time_total for item in key_averages])
                cuda_time_avg = np.mean([item.cuda_time for item in key_averages if item.count > 0])
                
                metrics['cuda_time_total_us'] = cuda_time_total
                metrics['cuda_time_avg_us'] = cuda_time_avg
            
            # Memory metrics
            if self.config.profile_memory:
                cpu_memory_usage = sum([item.cpu_memory_usage for item in key_averages if item.cpu_memory_usage > 0])
                cuda_memory_usage = sum([item.cuda_memory_usage for item in key_averages if item.cuda_memory_usage > 0])
                
                metrics['cpu_memory_usage_bytes'] = cpu_memory_usage
                metrics['cuda_memory_usage_bytes'] = cuda_memory_usage
            
            # FLOPs metrics
            flops_total = sum([item.flops for item in key_averages if item.flops > 0])
            metrics['flops_total'] = flops_total
            
            # Top operations by time
            top_ops = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:10]
            metrics['top_cpu_ops'] = [
                {
                    'name': op.key,
                    'cpu_time_total': op.cpu_time_total,
                    'count': op.count,
                    'cpu_time_avg': op.cpu_time
                }
                for op in top_ops
            ]
            
            if torch.cuda.is_available():
                top_cuda_ops = sorted(key_averages, key=lambda x: x.cuda_time_total, reverse=True)[:10]
                metrics['top_cuda_ops'] = [
                    {
                        'name': op.key,
                        'cuda_time_total': op.cuda_time_total,
                        'count': op.count,
                        'cuda_time_avg': op.cuda_time
                    }
                    for op in top_cuda_ops
                ]
        
        except Exception as e:
            metrics['profiler_error'] = str(e)
        
        return metrics
    
    def export_profiler_trace(self, prof: torch.profiler.profile, output_path: str):
        """
        Export profiler trace for visualization
        
        Args:
            prof: PyTorch profiler instance
            output_path: Path to save trace file
        """
        try:
            if self.config.export_chrome_trace:
                prof.export_chrome_trace(f"{output_path}_chrome_trace.json")
            
            if self.config.export_stacks:
                prof.export_stacks(f"{output_path}_stacks.txt", "self_cpu_time_total")
        
        except Exception as e:
            print(f"Warning: Failed to export profiler trace: {e}")
    
    def validate_inputs(self, model: nn.Module, data_loader: DataLoader) -> bool:
        """
        Validate benchmark inputs
        
        Args:
            model: Model to validate
            data_loader: Data loader to validate
            
        Returns:
            True if inputs are valid, False otherwise
        """
        if not isinstance(model, nn.Module):
            return False
        
        if not isinstance(data_loader, DataLoader):
            return False
        
        # Check if data loader has data
        try:
            next(iter(data_loader))
            return True
        except StopIteration:
            return False
        except Exception:
            return False


class BenchmarkRegistry:
    """Registry for benchmark implementations"""
    _benchmarks: Dict[str, Type[BaseBenchmark]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a benchmark"""
        def decorator(benchmark_class: Type[BaseBenchmark]):
            cls._benchmarks[name] = benchmark_class
            return benchmark_class
        return decorator
    
    @classmethod
    def get_benchmark(cls, name: str) -> Type[BaseBenchmark]:
        """Get a benchmark class by name"""
        if name not in cls._benchmarks:
            raise ValueError(f"Unknown benchmark: {name}")
        return cls._benchmarks[name]
    
    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all registered benchmark names"""
        return list(cls._benchmarks.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a benchmark is registered"""
        return name in cls._benchmarks
    
    @classmethod
    def create_benchmark(cls, name: str, config=None) -> BaseBenchmark:
        """Create a benchmark instance by name"""
        benchmark_class = cls.get_benchmark(name)
        return benchmark_class(config)
    
    def run_all_benchmarks(self, 
                          model: nn.Module, 
                          data_loader: DataLoader,
                          **kwargs) -> Dict[str, BenchmarkResult]:
        """
        Run all registered benchmarks
        
        Args:
            model: Model to benchmark
            data_loader: Data loader for benchmarking
            **kwargs: Additional parameters passed to benchmarks
            
        Returns:
            Dictionary mapping benchmark names to results
        """
        results = {}
        
        for name, benchmark in self._benchmarks.items():
            try:
                result = benchmark.run_benchmark(model, data_loader, **kwargs)
                results[name] = result
            except Exception as e:
                results[name] = BenchmarkResult(
                    benchmark_name=name,
                    metrics={},
                    error=str(e)
                )
        
        return results


# Global benchmark registry
benchmark_registry = BenchmarkRegistry()
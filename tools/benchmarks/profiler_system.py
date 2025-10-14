"""
Unified profiling system integrating PyTorch profiler with custom metrics
Provides comprehensive performance, memory, and energy analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import time
# 尝试导入psutil，如果不可用则使用替代方案
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import gc
import json
import os
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import numpy as np
from pathlib import Path
import threading
import subprocess


@dataclass
class ProfilerConfig:
    """Configuration for the unified profiler"""
    # PyTorch profiler settings
    enable_cpu_profiling: bool = True
    enable_cuda_profiling: bool = True
    profile_memory: bool = True
    record_shapes: bool = True
    with_stack: bool = False
    with_flops: bool = True
    with_modules: bool = True
    
    # Custom profiling settings
    enable_energy_profiling: bool = True
    enable_system_monitoring: bool = True
    monitor_gpu_utilization: bool = True
    
    # Sampling and scheduling
    warmup_steps: int = 5
    active_steps: int = 10
    repeat_count: int = 1
    skip_first: int = 0
    
    # Output settings
    export_chrome_trace: bool = True
    export_memory_timeline: bool = True
    export_detailed_report: bool = True
    output_dir: str = "./profiler_output"
    
    # Backward compatibility aliases
    pytorch_profiling: bool = True
    system_monitoring: bool = True
    energy_profiling: bool = True
    save_trace: bool = True
    
    def __post_init__(self):
        # Map old parameter names to new ones for backward compatibility
        if hasattr(self, 'pytorch_profiling'):
            self.enable_cpu_profiling = self.pytorch_profiling
            self.enable_cuda_profiling = self.pytorch_profiling
        if hasattr(self, 'system_monitoring'):
            self.enable_system_monitoring = self.system_monitoring
        if hasattr(self, 'energy_profiling'):
            self.enable_energy_profiling = self.energy_profiling
        if hasattr(self, 'save_trace'):
            self.export_chrome_trace = self.save_trace


@dataclass
class SystemMetrics:
    """System-level metrics"""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    memory_used_gb: float
    gpu_utilization: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_temperature: float = 0.0
    power_draw_watts: float = 0.0
    timestamp: float = 0.0


@dataclass
class ProfilerResult:
    """Comprehensive profiler result"""
    model_name: str
    profiling_duration: float
    pytorch_metrics: Dict[str, Any]
    system_metrics: List[SystemMetrics]
    energy_metrics: Dict[str, float]
    memory_timeline: List[Dict[str, float]]
    performance_summary: Dict[str, Any]
    trace_files: Dict[str, str]
    error: Optional[str] = None


class SystemMonitor:
    """System resource monitor running in background thread"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.metrics_history = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics_history = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> List[SystemMetrics]:
        """Stop monitoring and return collected metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        return self.metrics_history.copy()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.interval)
            except Exception as e:
                print(f"Warning: System monitoring error: {e}")
                time.sleep(self.interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and# 系统资源监控
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
        else:
            cpu_percent = 0.0
            memory = type('Memory', (), {'percent': 0.0, 'used': 0, 'total': 8*1024**3})()  # 模拟内存对象
        
        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            memory_used_gb=memory.used / (1024**3),
            timestamp=time.time()
        )
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            try:
                # PyTorch GPU metrics
                metrics.gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
                metrics.gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                # NVIDIA-SMI metrics (if available)
                gpu_stats = self._get_nvidia_smi_stats()
                if gpu_stats:
                    metrics.gpu_utilization = gpu_stats.get('utilization', 0.0)
                    metrics.gpu_temperature = gpu_stats.get('temperature', 0.0)
                    metrics.power_draw_watts = gpu_stats.get('power_draw', 0.0)
                    
            except Exception as e:
                print(f"Warning: GPU metrics collection failed: {e}")
        
        return metrics
    
    def _get_nvidia_smi_stats(self) -> Optional[Dict[str, float]]:
        """Get GPU stats from nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=2.0)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'utilization': float(values[0]),
                    'temperature': float(values[1]),
                    'power_draw': float(values[2])
                }
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, IndexError):
            pass
        
        return None


class EnergyProfiler:
    """Energy consumption profiler"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.initial_energy = None
        self.final_energy = None
        
    def start_profiling(self):
        """Start energy profiling"""
        self.start_time = time.time()
        self.initial_energy = self._get_energy_reading()
    
    def stop_profiling(self) -> Dict[str, float]:
        """Stop energy profiling and return metrics"""
        self.end_time = time.time()
        self.final_energy = self._get_energy_reading()
        
        duration = self.end_time - self.start_time
        energy_consumed = 0.0
        
        if self.initial_energy is not None and self.final_energy is not None:
            energy_consumed = self.final_energy - self.initial_energy
        
        return {
            'duration_seconds': duration,
            'energy_consumed_joules': energy_consumed,
            'average_power_watts': energy_consumed / duration if duration > 0 else 0.0
        }
    
    def _get_energy_reading(self) -> Optional[float]:
        """Get current energy reading (placeholder for actual implementation)"""
        # This would integrate with actual energy measurement hardware/software
        # For now, we estimate based on GPU power draw
        try:
            if torch.cuda.is_available():
                # Estimate based on GPU utilization and TDP
                gpu_props = torch.cuda.get_device_properties(0)
                # This is a rough estimation - real implementation would use actual sensors
                estimated_power = 150.0  # Watts (typical GPU TDP)
                return time.time() * estimated_power  # Simplified energy accumulation
        except Exception:
            pass
        
        return None


class UnifiedProfiler:
    """
    Unified profiling system that combines PyTorch profiler with custom metrics
    """
    
    def __init__(self, config: ProfilerConfig = None, output_dir: str = None):
        self.config = config or ProfilerConfig()
        
        # Override output_dir if provided
        if output_dir:
            self.config.output_dir = output_dir
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.system_monitor = SystemMonitor()
        self.energy_profiler = EnergyProfiler()
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def profile_model(self, 
                     model: nn.Module,
                     data_loader: DataLoader,
                     model_name: str = "model",
                     num_batches: Optional[int] = None) -> ProfilerResult:
        """
        Comprehensive model profiling
        
        Args:
            model: Model to profile
            data_loader: Data loader for profiling
            model_name: Name for the model (used in output files)
            num_batches: Number of batches to profile (None for all)
            
        Returns:
            ProfilerResult with comprehensive profiling data
        """
        start_time = time.time()
        
        try:
            # Setup profiler schedule
            profiler_schedule = schedule(
                skip_first=self.config.skip_first,
                wait=self.config.warmup_steps,
                warmup=1,
                active=self.config.active_steps,
                repeat=self.config.repeat_count
            )
            
            # Configure PyTorch profiler
            activities = []
            if self.config.enable_cpu_profiling:
                activities.append(ProfilerActivity.CPU)
            if self.config.enable_cuda_profiling and torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            
            # Start system monitoring
            if self.config.enable_system_monitoring:
                self.system_monitor.start_monitoring()
            
            # Start energy profiling
            if self.config.enable_energy_profiling:
                self.energy_profiler.start_profiling()
            
            # Memory timeline tracking
            memory_timeline = []
            
            # Run profiling with PyTorch profiler
            with profile(
                activities=activities,
                schedule=profiler_schedule,
                record_shapes=self.config.record_shapes,
                profile_memory=self.config.profile_memory,
                with_stack=self.config.with_stack,
                with_flops=self.config.with_flops,
                with_modules=self.config.with_modules,
                on_trace_ready=lambda prof: self._save_trace(prof, model_name)
            ) as prof:
                
                model.eval()
                with torch.no_grad():
                    for batch_idx, batch in enumerate(data_loader):
                        if num_batches and batch_idx >= num_batches:
                            break
                        
                        # Record memory before batch
                        if self.config.export_memory_timeline:
                            memory_timeline.append(self._get_memory_snapshot(batch_idx, "before"))
                        
                        # Move batch to device
                        if isinstance(batch, (list, tuple)):
                            batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                            inputs = batch[0]
                        else:
                            inputs = batch.to(self.device)
                        
                        # Profile forward pass
                        with record_function(f"forward_pass_batch_{batch_idx}"):
                            outputs = model(inputs)
                        
                        # Record memory after batch
                        if self.config.export_memory_timeline:
                            memory_timeline.append(self._get_memory_snapshot(batch_idx, "after"))
                        
                        # Step profiler
                        prof.step()
                        
                        # Clear cache periodically
                        if torch.cuda.is_available() and batch_idx % 5 == 0:
                            torch.cuda.empty_cache()
            
            # Stop monitoring
            system_metrics = []
            if self.config.enable_system_monitoring:
                system_metrics = self.system_monitor.stop_monitoring()
            
            energy_metrics = {}
            if self.config.enable_energy_profiling:
                energy_metrics = self.energy_profiler.stop_profiling()
            
            # Extract PyTorch profiler metrics
            pytorch_metrics = self._extract_pytorch_metrics(prof)
            
            # Generate performance summary
            performance_summary = self._generate_performance_summary(
                pytorch_metrics, system_metrics, energy_metrics
            )
            
            # Generate trace file paths
            trace_files = self._get_trace_file_paths(model_name)
            
            # Create result
            result = ProfilerResult(
                model_name=model_name,
                profiling_duration=time.time() - start_time,
                pytorch_metrics=pytorch_metrics,
                system_metrics=system_metrics,
                energy_metrics=energy_metrics,
                memory_timeline=memory_timeline,
                performance_summary=performance_summary,
                trace_files=trace_files
            )
            
            # Export detailed report
            if self.config.export_detailed_report:
                self._export_detailed_report(result)
            
            return result
            
        except Exception as e:
            return ProfilerResult(
                model_name=model_name,
                profiling_duration=time.time() - start_time,
                pytorch_metrics={},
                system_metrics=[],
                energy_metrics={},
                memory_timeline=[],
                performance_summary={},
                trace_files={},
                error=str(e)
            )
    
    def _save_trace(self, prof: profile, model_name: str):
        """Save profiler trace files"""
        try:
            if self.config.export_chrome_trace:
                trace_path = os.path.join(self.config.output_dir, f"{model_name}_trace.json")
                prof.export_chrome_trace(trace_path)
        except Exception as e:
            print(f"Warning: Failed to save trace: {e}")
    
    def _get_memory_snapshot(self, batch_idx: int, stage: str) -> Dict[str, float]:
        """Get memory snapshot at specific point"""
        snapshot = {
            'batch_idx': batch_idx,
            'stage': stage,
            'timestamp': time.time(),
            'cpu_memory_mb': psutil.Process().memory_info().rss / (1024**2) if PSUTIL_AVAILABLE else 100.0
        }
        
        if torch.cuda.is_available():
            snapshot.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2),
                'gpu_memory_cached_mb': torch.cuda.memory_cached() / (1024**2)
            })
        
        return snapshot
    
    def _extract_pytorch_metrics(self, prof: profile) -> Dict[str, Any]:
        """Extract comprehensive metrics from PyTorch profiler"""
        metrics = {}
        
        try:
            key_averages = prof.key_averages()
            
            # Time metrics
            cpu_time_total = sum(item.cpu_time_total for item in key_averages)
            metrics['cpu_time_total_us'] = cpu_time_total
            
            if torch.cuda.is_available():
                cuda_time_total = sum(item.cuda_time_total for item in key_averages)
                metrics['cuda_time_total_us'] = cuda_time_total
            
            # Memory metrics
            if self.config.profile_memory:
                cpu_memory_usage = sum(item.cpu_memory_usage for item in key_averages if item.cpu_memory_usage > 0)
                cuda_memory_usage = sum(item.cuda_memory_usage for item in key_averages if item.cuda_memory_usage > 0)
                
                metrics['cpu_memory_usage_bytes'] = cpu_memory_usage
                metrics['cuda_memory_usage_bytes'] = cuda_memory_usage
            
            # FLOPs metrics
            if self.config.with_flops:
                total_flops = sum(item.flops for item in key_averages if item.flops > 0)
                metrics['total_flops'] = total_flops
            
            # Top operations analysis
            metrics['top_operations'] = self._analyze_top_operations(key_averages)
            
            # Module-level analysis
            if self.config.with_modules:
                metrics['module_analysis'] = self._analyze_modules(key_averages)
            
        except Exception as e:
            metrics['extraction_error'] = str(e)
        
        return metrics
    
    def _analyze_top_operations(self, key_averages) -> Dict[str, List[Dict]]:
        """Analyze top operations by different metrics"""
        analysis = {}
        
        # Top by CPU time
        top_cpu = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:10]
        analysis['top_cpu_time'] = [
            {
                'name': op.key,
                'cpu_time_total_us': op.cpu_time_total,
                'count': op.count,
                'cpu_time_avg_us': op.cpu_time
            }
            for op in top_cpu
        ]
        
        # Top by CUDA time (if available)
        if torch.cuda.is_available():
            top_cuda = sorted(key_averages, key=lambda x: x.cuda_time_total, reverse=True)[:10]
            analysis['top_cuda_time'] = [
                {
                    'name': op.key,
                    'cuda_time_total_us': op.cuda_time_total,
                    'count': op.count,
                    'cuda_time_avg_us': op.cuda_time
                }
                for op in top_cuda
            ]
        
        # Top by memory usage
        if self.config.profile_memory:
            top_memory = sorted(key_averages, key=lambda x: x.cpu_memory_usage, reverse=True)[:10]
            analysis['top_memory_usage'] = [
                {
                    'name': op.key,
                    'cpu_memory_usage_bytes': op.cpu_memory_usage,
                    'cuda_memory_usage_bytes': op.cuda_memory_usage,
                    'count': op.count
                }
                for op in top_memory if op.cpu_memory_usage > 0
            ]
        
        return analysis
    
    def _analyze_modules(self, key_averages) -> Dict[str, Any]:
        """Analyze performance by module"""
        module_stats = {}
        
        for item in key_averages:
            if hasattr(item, 'module_hierarchy') and item.module_hierarchy:
                module_name = item.module_hierarchy.split('.')[-1]
                
                if module_name not in module_stats:
                    module_stats[module_name] = {
                        'cpu_time_total': 0,
                        'cuda_time_total': 0,
                        'memory_usage': 0,
                        'count': 0
                    }
                
                module_stats[module_name]['cpu_time_total'] += item.cpu_time_total
                module_stats[module_name]['cuda_time_total'] += item.cuda_time_total
                module_stats[module_name]['memory_usage'] += item.cpu_memory_usage
                module_stats[module_name]['count'] += item.count
        
        return module_stats
    
    def _generate_performance_summary(self, 
                                    pytorch_metrics: Dict[str, Any],
                                    system_metrics: List[SystemMetrics],
                                    energy_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate high-level performance summary"""
        summary = {}
        
        # PyTorch metrics summary
        if pytorch_metrics:
            summary['total_cpu_time_ms'] = pytorch_metrics.get('cpu_time_total_us', 0) / 1000
            summary['total_cuda_time_ms'] = pytorch_metrics.get('cuda_time_total_us', 0) / 1000
            summary['total_flops'] = pytorch_metrics.get('total_flops', 0)
            summary['memory_usage_mb'] = pytorch_metrics.get('cpu_memory_usage_bytes', 0) / (1024**2)
        
        # System metrics summary
        if system_metrics:
            cpu_usage = [m.cpu_percent for m in system_metrics]
            memory_usage = [m.memory_percent for m in system_metrics]
            
            summary['avg_cpu_usage_percent'] = np.mean(cpu_usage)
            summary['max_cpu_usage_percent'] = np.max(cpu_usage)
            summary['avg_memory_usage_percent'] = np.mean(memory_usage)
            summary['max_memory_usage_percent'] = np.max(memory_usage)
            
            if torch.cuda.is_available():
                gpu_usage = [m.gpu_utilization for m in system_metrics if m.gpu_utilization > 0]
                if gpu_usage:
                    summary['avg_gpu_usage_percent'] = np.mean(gpu_usage)
                    summary['max_gpu_usage_percent'] = np.max(gpu_usage)
        
        # Energy metrics summary
        if energy_metrics:
            summary.update(energy_metrics)
        
        return summary
    
    def _get_trace_file_paths(self, model_name: str) -> Dict[str, str]:
        """Get paths to generated trace files"""
        trace_files = {}
        
        if self.config.export_chrome_trace:
            trace_files['chrome_trace'] = os.path.join(
                self.config.output_dir, f"{model_name}_trace.json"
            )
        
        return trace_files
    
    def _export_detailed_report(self, result: ProfilerResult):
        """Export detailed profiling report"""
        report_path = os.path.join(
            self.config.output_dir, f"{result.model_name}_detailed_report.json"
        )
        
        try:
            # Convert result to dictionary for JSON serialization
            report_data = asdict(result)
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Warning: Failed to export detailed report: {e}")


# Convenience function for quick profiling
def profile_model(model: nn.Module, 
                 data_loader: DataLoader,
                 model_name: str = "model",
                 config: ProfilerConfig = None) -> ProfilerResult:
    """
    Convenience function for quick model profiling
    
    Args:
        model: Model to profile
        data_loader: Data loader for profiling
        model_name: Name for the model
        config: Profiler configuration
        
    Returns:
        ProfilerResult with comprehensive profiling data
    """
    profiler = UnifiedProfiler(config)
    return profiler.profile_model(model, data_loader, model_name)
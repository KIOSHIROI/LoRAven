"""
性能估算器：估算延时、能耗和内存使用
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod
import time


class PerfEstimator(ABC):
    """
    性能估算器基类
    """
    
    def __init__(self, hardware_profile: Dict[str, Any]):
        self.hardware_profile = hardware_profile
    
    @abstractmethod
    def estimate(self, layer_dims: Tuple[int, int], rank: int) -> float:
        """估算性能指标"""
        pass


class EnergyEstimator(PerfEstimator):
    """
    改进的能耗估算器：考虑硬件特性、缓存效应和并行度
    """
    
    def __init__(
        self, 
        hardware_profile: Dict[str, Any],
        flops_per_mac: float = 2.0,
        energy_per_flop: float = 1e-6,  # mJ per FLOP
        energy_per_memory_access: float = 1e-7  # mJ per memory access
    ):
        super().__init__(hardware_profile)
        self.flops_per_mac = flops_per_mac
        self.energy_per_flop = energy_per_flop
        self.energy_per_memory_access = energy_per_memory_access
        
        # 硬件特定参数（更详细的建模）
        self.dram_energy_per_byte = hardware_profile.get('dram_energy_per_byte', 1e-6)
        self.l2_cache_energy_per_byte = hardware_profile.get('l2_cache_energy_per_byte', 1e-7)
        self.l1_cache_energy_per_byte = hardware_profile.get('l1_cache_energy_per_byte', 1e-8)
        self.register_energy_per_byte = hardware_profile.get('register_energy_per_byte', 1e-9)
        self.compute_energy_per_flop = hardware_profile.get('compute_energy_per_flop', 1e-6)
        
        # 缓存参数
        self.l1_cache_size = hardware_profile.get('l1_cache_size', 64 * 1024)  # 64KB
        self.l2_cache_size = hardware_profile.get('l2_cache_size', 1024 * 1024)  # 1MB
        self.cache_line_size = hardware_profile.get('cache_line_size', 128)  # 128B
        
        # 并行度参数
        self.gpu_cores = hardware_profile.get('gpu_cores', 5120)
        self.memory_controllers = hardware_profile.get('memory_controllers', 8)
        self.warp_size = hardware_profile.get('warp_size', 32)
        
        # 动态电压频率调节（DVFS）
        self.base_frequency = hardware_profile.get('base_frequency', 1.5e9)  # 1.5GHz
        self.max_frequency = hardware_profile.get('max_frequency', 2.0e9)  # 2.0GHz
        self.voltage_scaling_factor = hardware_profile.get('voltage_scaling_factor', 2.0)
        
        # 温度和功耗管理
        self.thermal_design_power = hardware_profile.get('thermal_design_power', 250.0)  # 250W
        self.idle_power = hardware_profile.get('idle_power', 50.0)  # 50W
    
    def estimate(
        self, 
        layer_dims: Tuple[int, int], 
        rank: int,
        batch_size: int = 1,
        utilization: float = 0.8,
        temperature: float = 65.0
    ) -> float:
        """
        估算能耗
        
        数学公式:
        E_total = E_compute + E_memory + E_static
        其中:
        - E_compute = FLOPs × E_flop × η_parallel × f_DVFS
        - E_memory = Σ(Access_l × E_l × CacheMiss_l) for l ∈ {L1, L2, DRAM}
        - E_static = P_idle × t_exec
        
        Args:
            layer_dims: 层维度 (in_features, out_features)
            rank: 秩
            batch_size: 批次大小
            utilization: GPU利用率
            temperature: 温度 (°C)
            
        Returns:
            energy: 估算能耗 (mJ)
        """
        in_features, out_features = layer_dims
        
        # 计算 FLOPs 和并行效率
        flops = self._calculate_flops(in_features, out_features, rank, batch_size)
        parallel_efficiency = self._calculate_parallel_efficiency(batch_size, rank)
        
        # 计算分层内存访问能耗
        memory_energy = self._calculate_hierarchical_memory_energy(
            in_features, out_features, rank, batch_size
        )
        
        # 计算计算能耗（考虑DVFS和并行度）
        compute_energy = self._calculate_compute_energy(
            flops, parallel_efficiency, utilization
        )
        
        # 温度影响
        temperature_factor = self._calculate_temperature_factor(temperature)
        
        # 静态功耗（待机功耗）
        execution_time = self._estimate_execution_time(flops, parallel_efficiency)
        static_energy = self.idle_power * execution_time / 1000.0  # 转换为mJ
        
        # 总能耗
        total_energy = (
            compute_energy + memory_energy + static_energy
        ) * temperature_factor
        
        return total_energy
    
    def _calculate_parallel_efficiency(self, batch_size: int, rank: int) -> float:
        """
        计算并行效率
        """
        # 计算工作负载大小
        workload_size = batch_size * rank
        
        # 计算warp利用率
        warps_needed = np.ceil(workload_size / self.warp_size)
        warp_efficiency = min(1.0, workload_size / (warps_needed * self.warp_size))
        
        # 计算SM利用率
        sm_count = self.gpu_cores // 128  # 假设每个SM有128个核心
        sm_efficiency = min(1.0, warps_needed / sm_count)
        
        # 综合并行效率
        parallel_efficiency = warp_efficiency * sm_efficiency * 0.9  # 考虑调度开销
        
        return max(0.1, parallel_efficiency)  # 最小10%效率
    
    def _calculate_hierarchical_memory_energy(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int, 
        batch_size: int
    ) -> float:
        """
        计算分层内存系统的能耗
        """
        # 计算各种数据大小
        input_size = batch_size * in_features * 4
        weight_size = (in_features + out_features) * rank * 4 + rank * rank * 4
        output_size = batch_size * out_features * 4
        
        # 缓存命中率建模
        l1_hit_rate = self._calculate_cache_hit_rate(weight_size, self.l1_cache_size)
        l2_hit_rate = self._calculate_cache_hit_rate(weight_size, self.l2_cache_size)
        
        # 分层内存访问能耗
        l1_energy = input_size * l1_hit_rate * self.l1_cache_energy_per_byte
        l2_energy = input_size * (1 - l1_hit_rate) * l2_hit_rate * self.l2_cache_energy_per_byte
        dram_energy = input_size * (1 - l1_hit_rate) * (1 - l2_hit_rate) * self.dram_energy_per_byte
        
        # 权重访问（通常有更好的局部性）
        weight_l1_energy = weight_size * 0.8 * self.l1_cache_energy_per_byte
        weight_l2_energy = weight_size * 0.15 * self.l2_cache_energy_per_byte
        weight_dram_energy = weight_size * 0.05 * self.dram_energy_per_byte
        
        # 输出写入
        output_energy = output_size * self.l1_cache_energy_per_byte
        
        total_memory_energy = (
            l1_energy + l2_energy + dram_energy +
            weight_l1_energy + weight_l2_energy + weight_dram_energy +
            output_energy
        )
        
        return total_memory_energy
    
    def _calculate_cache_hit_rate(self, data_size: int, cache_size: int) -> float:
        """
        计算缓存命中率
        """
        if data_size <= cache_size:
            return 0.95  # 高命中率
        elif data_size <= cache_size * 2:
            return 0.7   # 中等命中率
        else:
            return 0.3   # 低命中率
    
    def _calculate_compute_energy(
        self, 
        flops: int, 
        parallel_efficiency: float, 
        utilization: float
    ) -> float:
        """
        计算计算能耗（考虑DVFS）
        """
        # 根据利用率调整频率
        frequency_ratio = min(1.0, utilization * 1.2)  # 允许超频
        actual_frequency = self.base_frequency + (self.max_frequency - self.base_frequency) * frequency_ratio
        
        # 电压随频率缩放
        voltage_ratio = (actual_frequency / self.base_frequency) ** (1.0 / self.voltage_scaling_factor)
        
        # 功耗随电压平方缩放
        power_scaling = voltage_ratio ** 2
        
        # 计算能耗
        base_compute_energy = flops * self.compute_energy_per_flop
        scaled_compute_energy = base_compute_energy * power_scaling / parallel_efficiency
        
        return scaled_compute_energy
    
    def _calculate_temperature_factor(self, temperature: float) -> float:
        """
        计算温度对能耗的影响
        """
        # 温度每升高10°C，漏电流大约增加2倍
        base_temp = 25.0  # 基准温度
        temp_factor = 1.0 + 0.02 * (temperature - base_temp)  # 简化模型
        return max(0.8, min(1.5, temp_factor))  # 限制在合理范围内
    
    def _estimate_execution_time(self, flops: int, parallel_efficiency: float) -> float:
        """
        估算执行时间（秒）
        """
        effective_flops_per_second = self.base_frequency * self.gpu_cores * parallel_efficiency
        execution_time = flops / effective_flops_per_second
        return execution_time
    
    def _calculate_flops(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int, 
        batch_size: int
    ) -> int:
        """计算 FLOPs"""
        # 低秩矩阵乘法: Y = (U @ S) @ (V^T @ X)^T
        # 步骤1: X @ V -> (batch_size, rank)
        flops_step1 = batch_size * in_features * rank * self.flops_per_mac
        
        # 步骤2: (X @ V) @ S.T -> (batch_size, rank)
        flops_step2 = batch_size * rank * rank * self.flops_per_mac
        
        # 步骤3: ((X @ V) @ S.T) @ U.T -> (batch_size, out_features)
        flops_step3 = batch_size * rank * out_features * self.flops_per_mac
        
        total_flops = flops_step1 + flops_step2 + flops_step3
        return total_flops
    
    def _calculate_memory_access(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int, 
        batch_size: int
    ) -> int:
        """计算内存访问量（字节）"""
        # 输入数据
        input_size = batch_size * in_features * 4  # 假设 float32
        
        # 权重矩阵访问
        U_size = out_features * rank * 4
        V_size = in_features * rank * 4
        S_size = rank * rank * 4
        
        # 中间结果
        intermediate_size = batch_size * rank * 4
        
        # 输出数据
        output_size = batch_size * out_features * 4
        
        total_memory = input_size + U_size + V_size + S_size + intermediate_size + output_size
        return total_memory


class LatencyEstimator(PerfEstimator):
    """
    延时估算器
    """
    
    def __init__(
        self, 
        hardware_profile: Dict[str, Any],
        flops_per_second: float = 1e12,  # 1 TFLOP/s
        memory_bandwidth: float = 1e12   # 1 TB/s
    ):
        super().__init__(hardware_profile)
        self.flops_per_second = flops_per_second
        self.memory_bandwidth = memory_bandwidth
        
        # 硬件特定参数
        self.compute_latency_per_flop = 1.0 / flops_per_second
        self.memory_latency_per_byte = 1.0 / memory_bandwidth
        
        # GPU 特定参数
        self.gpu_cores = hardware_profile.get('gpu_cores', 5120)
        self.memory_latency = hardware_profile.get('memory_latency', 1e-6)  # 1μs
        self.compute_latency = hardware_profile.get('compute_latency', 1e-9)  # 1ns
    
    def estimate(
        self, 
        layer_dims: Tuple[int, int], 
        rank: int,
        batch_size: int = 1
    ) -> float:
        """
        估算延时
        
        Args:
            layer_dims: 层维度 (in_features, out_features)
            rank: 秩
            batch_size: 批大小
            
        Returns:
            latency: 估算延时 (ms)
        """
        in_features, out_features = layer_dims
        
        # 计算计算延时
        compute_latency = self._calculate_compute_latency(in_features, out_features, rank, batch_size)
        
        # 计算内存延时
        memory_latency = self._calculate_memory_latency(in_features, out_features, rank, batch_size)
        
        # 总延时（取最大值，因为计算和内存访问可能并行）
        total_latency = max(compute_latency, memory_latency)
        
        return total_latency * 1000  # 转换为毫秒
    
    def _calculate_compute_latency(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int, 
        batch_size: int
    ) -> float:
        """计算计算延时"""
        flops = self._calculate_flops(in_features, out_features, rank, batch_size)
        compute_latency = flops * self.compute_latency_per_flop
        
        # 考虑并行化
        parallel_efficiency = min(1.0, batch_size / self.gpu_cores)
        compute_latency = compute_latency / parallel_efficiency
        
        return compute_latency
    
    def _calculate_memory_latency(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int, 
        batch_size: int
    ) -> float:
        """计算内存延时"""
        memory_access = self._calculate_memory_access(in_features, out_features, rank, batch_size)
        memory_latency = memory_access * self.memory_latency_per_byte
        
        return memory_latency
    
    def _calculate_flops(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int, 
        batch_size: int
    ) -> int:
        """计算 FLOPs"""
        flops_step1 = batch_size * in_features * rank * 2
        flops_step2 = batch_size * rank * rank * 2
        flops_step3 = batch_size * rank * out_features * 2
        return flops_step1 + flops_step2 + flops_step3
    
    def _calculate_memory_access(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int, 
        batch_size: int
    ) -> int:
        """计算内存访问量"""
        input_size = batch_size * in_features * 4
        U_size = out_features * rank * 4
        V_size = in_features * rank * 4
        S_size = rank * rank * 4
        intermediate_size = batch_size * rank * 4
        output_size = batch_size * out_features * 4
        
        return input_size + U_size + V_size + S_size + intermediate_size + output_size


class MemoryEstimator(PerfEstimator):
    """
    内存估算器
    """
    
    def __init__(self, hardware_profile: Dict[str, Any]):
        super().__init__(hardware_profile)
        self.dtype_size = hardware_profile.get('dtype_size', 4)  # float32 = 4 bytes
    
    def estimate(
        self, 
        layer_dims: Tuple[int, int], 
        rank: int,
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        估算内存使用
        
        Args:
            layer_dims: 层维度 (in_features, out_features)
            rank: 秩
            batch_size: 批大小
            
        Returns:
            memory_info: 内存信息字典
        """
        in_features, out_features = layer_dims
        
        # 权重内存
        U_memory = out_features * rank * self.dtype_size
        V_memory = in_features * rank * self.dtype_size
        S_memory = rank * rank * self.dtype_size
        weight_memory = U_memory + V_memory + S_memory
        
        # 激活内存
        input_memory = batch_size * in_features * self.dtype_size
        intermediate_memory = batch_size * rank * self.dtype_size
        output_memory = batch_size * out_features * self.dtype_size
        activation_memory = input_memory + intermediate_memory + output_memory
        
        # 总内存
        total_memory = weight_memory + activation_memory
        
        # 压缩比
        full_rank_memory = in_features * out_features * self.dtype_size
        compression_ratio = weight_memory / full_rank_memory
        
        return {
            'weight_memory_mb': weight_memory / (1024 * 1024),
            'activation_memory_mb': activation_memory / (1024 * 1024),
            'total_memory_mb': total_memory / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'memory_savings': 1.0 - compression_ratio
        }


class PerfEstimator:
    """
    综合性能估算器
    """
    
    def __init__(self, hardware_profile: Dict[str, Any]):
        self.hardware_profile = hardware_profile
        
        # 初始化子估算器
        self.energy_estimator = EnergyEstimator(hardware_profile)
        self.latency_estimator = LatencyEstimator(hardware_profile)
        self.memory_estimator = MemoryEstimator(hardware_profile)
    
    def estimate_all(
        self, 
        layer_dims: Tuple[int, int], 
        rank: int,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        估算所有性能指标
        
        Args:
            layer_dims: 层维度 (in_features, out_features)
            rank: 秩
            batch_size: 批大小
            
        Returns:
            performance_info: 性能信息字典
        """
        energy = self.energy_estimator.estimate(layer_dims, rank, batch_size)
        latency = self.latency_estimator.estimate(layer_dims, rank, batch_size)
        memory_info = self.memory_estimator.estimate(layer_dims, rank, batch_size)
        
        return {
            'energy_mj': energy,
            'latency_ms': latency,
            'memory_info': memory_info,
            'efficiency_score': self._calculate_efficiency_score(energy, latency, memory_info)
        }
    
    def _calculate_efficiency_score(
        self, 
        energy: float, 
        latency: float, 
        memory_info: Dict[str, Any]
    ) -> float:
        """计算效率分数"""
        # 归一化指标
        energy_score = 1.0 / (1.0 + energy)
        latency_score = 1.0 / (1.0 + latency)
        memory_score = memory_info['compression_ratio']
        
        # 加权平均
        efficiency_score = 0.4 * energy_score + 0.3 * latency_score + 0.3 * memory_score
        
        return efficiency_score
    
    def benchmark_layer(
        self, 
        layer, 
        input_tensor: torch.Tensor,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        基准测试层性能
        
        Args:
            layer: 层实例
            input_tensor: 输入张量
            num_runs: 运行次数
            
        Returns:
            benchmark_results: 基准测试结果
        """
        device = input_tensor.device
        
        # 预热
        for _ in range(3):
            _ = layer(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 计时
        start_time = time.time()
        for _ in range(num_runs):
            _ = layer(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # 计算平均延时
        avg_latency = (end_time - start_time) / num_runs * 1000  # 转换为毫秒
        
        # 估算能耗（简化）
        estimated_energy = self.energy_estimator.estimate(
            (layer.in_features, layer.out_features), 
            layer.r_curr,
            input_tensor.size(0)
        )
        
        return {
            'avg_latency_ms': avg_latency,
            'estimated_energy_mj': estimated_energy,
            'throughput_samples_per_sec': 1000.0 / avg_latency
        }

"""
ADLRNS: Adaptive Dynamic Low-Rank Neural Systems
一种面向类脑计算的运行时自适应低秩表示与能耗感知推理框架
"""

__version__ = "0.1.0"
__author__ = "ADLRNS Team"

# 核心组件
from .models.dynamic_lowrank_layer import DynamicLowRankLayer
from .models.gates import LightweightScorer, GateNetwork
from .schedulers.rank_scheduler import RankScheduler
from .utils.perf_estimator import PerfEstimator

# 简化接口
try:
    from .adlrns_simple import ADLRNS, create_adlrns_layer, replace_linear_with_adlrns
    _SIMPLE_AVAILABLE = True
except ImportError:
    _SIMPLE_AVAILABLE = False
    ADLRNS = None
    create_adlrns_layer = None
    replace_linear_with_adlrns = None

# 训练器和测试工具
try:
    from .trainers.train_adlrns import ADLRNSTrainer
    _TRAINER_AVAILABLE = True
except ImportError:
    _TRAINER_AVAILABLE = False
    ADLRNSTrainer = None

# 实验和基准测试
try:
    from .experiments.benchmark import run_benchmark
    from .experiments.visual_analysis import create_visualizations
    _EXPERIMENTS_AVAILABLE = True
except ImportError:
    _EXPERIMENTS_AVAILABLE = False
    run_benchmark = None
    create_visualizations = None

# 导出的公共接口
__all__ = [
    # 核心组件
    "DynamicLowRankLayer",
    "LightweightScorer", 
    "GateNetwork",
    "RankScheduler",
    "PerfEstimator",
]

# 添加简化接口（如果可用）
if _SIMPLE_AVAILABLE:
    __all__.extend([
        "ADLRNS",
        "create_adlrns_layer",
        "replace_linear_with_adlrns",
    ])

# 添加训练器（如果可用）
if _TRAINER_AVAILABLE:
    __all__.append("ADLRNSTrainer")

# 添加实验工具（如果可用）
if _EXPERIMENTS_AVAILABLE:
    __all__.extend([
        "run_benchmark",
        "create_visualizations",
    ])

# 版本和功能检查函数
def get_version():
    """获取ADLRNS版本信息"""
    return __version__

def check_features():
    """检查可用功能"""
    features = {
        "core": True,
        "simple_interface": _SIMPLE_AVAILABLE,
        "trainer": _TRAINER_AVAILABLE,
        "experiments": _EXPERIMENTS_AVAILABLE,
    }
    return features

def print_info():
    """打印ADLRNS信息"""
    print(f"ADLRNS v{__version__}")
    print("Adaptive Dynamic Low-Rank Neural Systems")
    print("自适应动态低秩神经网络系统")
    print("\n可用功能:")
    features = check_features()
    for feature, available in features.items():
        status = "✓" if available else "✗"
        print(f"  {status} {feature}")

# 添加到__all__
__all__.extend(["get_version", "check_features", "print_info"])

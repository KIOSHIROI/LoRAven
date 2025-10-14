# LoRAven API Reference

## Table of Contents

1. [Core Components](#1-core-components)
2. [Models](#2-models)
3. [Schedulers](#3-schedulers)
4. [Budget Management](#4-budget-management)
5. [Performance Estimation](#5-performance-estimation)
6. [Utilities](#6-utilities)
7. [Examples and Usage](#7-examples-and-usage)

## 1. Core Components

### 1.1 LoRAven (Simplified Interface)

```python
class LoRAven(nn.Module)
```

Simplified interface for LoRAven layers, providing PyTorch-like usage patterns.

#### Constructor

```python
def __init__(
    self,
    in_features: int,
    out_features: int,
    mode: str = 'balanced',
    max_rank: Optional[int] = None,
    min_rank: Optional[int] = None,
    energy_budget: Optional[float] = None,
    bias: bool = True,
    device: Optional[torch.device] = None
)
```

**Parameters:**
- `in_features` (int): Input feature dimensions
- `out_features` (int): Output feature dimensions  
- `mode` (str): Preset mode - 'high_performance', 'balanced', 'low_power', or 'custom'
- `max_rank` (int, optional): Maximum rank (required for 'custom' mode)
- `min_rank` (int, optional): Minimum rank (required for 'custom' mode)
- `energy_budget` (float, optional): Energy budget in mJ/sample
- `bias` (bool): Whether to use bias term
- `device` (torch.device, optional): Device for computation

**Preset Modes:**

| Mode | Rank Ratio | Min Rank Ratio | Energy Multiplier | Description |
|------|------------|----------------|-------------------|-------------|
| `high_performance` | 0.8 | 0.3 | 1.5 | Prioritizes accuracy over energy |
| `balanced` | 0.5 | 0.2 | 1.0 | Balanced accuracy-energy trade-off |
| `low_power` | 0.3 | 0.1 | 0.7 | Prioritizes energy savings |

#### Methods

##### forward

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through the LoRAven layer.

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape `(batch_size, in_features)`

**Returns:**
- `torch.Tensor`: Output tensor of shape `(batch_size, out_features)`

**Example:**
```python
layer = LoRAven(512, 256, mode='balanced')
x = torch.randn(32, 512)
output = layer(x)  # Shape: (32, 256)
```

##### get_current_rank

```python
def get_current_rank(self) -> int
```

Returns the current rank being used by the layer.

**Returns:**
- `int`: Current rank value

##### get_budget_usage

```python
def get_budget_usage(self) -> float
```

Returns the current budget utilization ratio.

**Returns:**
- `float`: Budget utilization ratio (0.0 to 1.0)

##### reset_budget

```python
def reset_budget(self) -> None
```

Resets the energy budget to initial state.

##### get_performance_stats

```python
def get_performance_stats(self) -> Dict[str, Any]
```

Returns comprehensive performance statistics.

**Returns:**
- `Dict[str, Any]`: Performance metrics including energy consumption, rank history, etc.

## 2. Models

### 2.1 DynamicLowRankLayer

```python
class DynamicLowRankLayer(nn.Module)
```

Core dynamic low-rank layer implementing runtime-adaptive matrix factorization.

#### Constructor

```python
def __init__(
    self,
    in_features: int,
    out_features: int,
    r_max: int,
    r_min: int = 4,
    init_rank: Optional[int] = None,
    bias: bool = True,
    scorer_hidden: int = 32,
    num_gate_blocks: int = 8,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
)
```

**Parameters:**
- `in_features` (int): Input feature dimensions
- `out_features` (int): Output feature dimensions
- `r_max` (int): Maximum rank
- `r_min` (int): Minimum rank (default: 4)
- `init_rank` (int, optional): Initial rank (defaults to r_min)
- `bias` (bool): Whether to include bias term
- `scorer_hidden` (int): Hidden dimension for complexity scorer
- `num_gate_blocks` (int): Number of gate network blocks
- `device` (torch.device, optional): Computation device
- `dtype` (torch.dtype, optional): Data type

#### Methods

##### forward

```python
def forward(
    self, 
    x: torch.Tensor, 
    budget: Optional[float] = None,
    mode: str = 'inference'
) -> Tuple[torch.Tensor, int]
```

Forward pass with dynamic rank adaptation.

**Parameters:**
- `x` (torch.Tensor): Input tensor `(batch_size, in_features)`
- `budget` (float, optional): Energy budget for this forward pass
- `mode` (str): Operation mode ('inference' or 'training')

**Returns:**
- `Tuple[torch.Tensor, int]`: Output tensor and current rank used

**Mathematical Formula:**
```
y = x @ V @ S^T @ U^T
where rank is dynamically determined by:
r(t) = RankScheduler(complexity_score(x), budget, history)
```

##### get_compression_ratio

```python
def get_compression_ratio(self) -> float
```

Returns the current compression ratio compared to full-rank matrix.

**Returns:**
- `float`: Compression ratio (0.0 to 1.0)

##### get_flops

```python
def get_flops(self, batch_size: int = 1) -> int
```

Calculates FLOPs for current configuration.

**Parameters:**
- `batch_size` (int): Batch size for FLOP calculation

**Returns:**
- `int`: Number of floating-point operations

##### visualize_rank_history

```python
def visualize_rank_history(self, save_path: Optional[str] = None) -> None
```

Visualizes the rank adaptation history.

**Parameters:**
- `save_path` (str, optional): Path to save the visualization

### 2.2 LightweightScorer

```python
class LightweightScorer(nn.Module)
```

Lightweight neural network for computing input complexity scores.

#### Constructor

```python
def __init__(
    self,
    in_features: int,
    hidden_dim: int = 32,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
)
```

**Parameters:**
- `in_features` (int): Input feature dimensions
- `hidden_dim` (int): Hidden layer dimension
- `device` (torch.device, optional): Computation device
- `dtype` (torch.dtype, optional): Data type

#### Methods

##### forward

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Computes complexity score for input.

**Parameters:**
- `x` (torch.Tensor): Input tensor `(batch_size, in_features)`

**Returns:**
- `torch.Tensor`: Complexity scores `(batch_size, 1)` in range [0, 1]

**Architecture:**
```
Input -> Linear(in_features, hidden_dim) -> ReLU -> Linear(hidden_dim, 1) -> Sigmoid -> Output
```

### 2.3 GateNetwork

```python
class GateNetwork(nn.Module)
```

Gate network for selective computation control.

#### Constructor

```python
def __init__(
    self,
    in_features: int,
    num_blocks: int = 8,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
)
```

**Parameters:**
- `in_features` (int): Input feature dimensions
- `num_blocks` (int): Number of gate blocks
- `device` (torch.device, optional): Computation device
- `dtype` (torch.dtype, optional): Data type

#### Methods

##### forward

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Computes gate activations.

**Parameters:**
- `x` (torch.Tensor): Input tensor `(batch_size, in_features)`

**Returns:**
- `torch.Tensor`: Gate activations `(batch_size, num_blocks)`

## 3. Schedulers

### 3.1 RankScheduler (Base Class)

```python
class RankScheduler(ABC)
```

Abstract base class for rank scheduling algorithms.

#### Methods

##### schedule_rank

```python
@abstractmethod
def schedule_rank(
    self, 
    complexity_scores: torch.Tensor, 
    budget: Optional[float] = None,
    **kwargs
) -> int
```

Abstract method for rank scheduling.

**Parameters:**
- `complexity_scores` (torch.Tensor): Complexity scores `(batch_size,)`
- `budget` (float, optional): Energy budget
- `**kwargs`: Additional parameters

**Returns:**
- `int`: Target rank

### 3.2 LinearRankScheduler

```python
class LinearRankScheduler(RankScheduler)
```

Linear mapping from complexity scores to rank values with diversity enhancement.

#### Constructor

```python
def __init__(
    self, 
    r_min: int = 4, 
    r_max: int = 128, 
    diversity_factor: float = 0.1
)
```

**Parameters:**
- `r_min` (int): Minimum rank
- `r_max` (int): Maximum rank
- `diversity_factor` (float): Diversity enhancement factor

#### Methods

##### schedule_rank

```python
def schedule_rank(
    self, 
    complexity_scores: torch.Tensor, 
    budget: Optional[float] = None,
    current_loss: Optional[float] = None,
    gradient_norm: Optional[float] = None,
    **kwargs
) -> int
```

Linear rank scheduling with enhancements.

**Algorithm:**
```python
s_avg = complexity_scores.mean()
s_std = complexity_scores.std()
r_base = r_min + s_avg * (r_max - r_min)
diversity_bonus = s_std * diversity_factor * (r_max - r_min)
gradient_bonus = min(gradient_norm / 10.0, 1.0) * 0.2 * (r_max - r_min)
r_final = enforce_diversity(r_base + diversity_bonus + gradient_bonus)
```

### 3.3 EnergyAwareRankScheduler

```python
class EnergyAwareRankScheduler(RankScheduler)
```

Energy-aware rank scheduler considering energy budgets and constraints.

#### Constructor

```python
def __init__(
    self, 
    r_min: int = 4, 
    r_max: int = 128,
    energy_model: Optional[Any] = None,
    alpha: float = 1.0,
    beta: float = 0.1
)
```

**Parameters:**
- `r_min` (int): Minimum rank
- `r_max` (int): Maximum rank
- `energy_model` (Any, optional): Energy estimation model
- `alpha` (float): Energy weight in optimization
- `beta` (float): Performance weight in optimization

#### Methods

##### schedule_rank

```python
def schedule_rank(
    self, 
    complexity_scores: torch.Tensor, 
    budget: Optional[float] = None,
    layer_dims: Optional[Tuple[int, int]] = None,
    **kwargs
) -> int
```

Energy-constrained rank scheduling.

**Optimization Problem:**
```
r* = argmax{r : E(r,x) ≤ budget} P(r,x)
where:
- E(r,x): Energy consumption function
- P(r,x): Performance function
- budget: Available energy budget
```

## 4. Budget Management

### 4.1 BudgetManager

```python
class BudgetManager
```

Manages energy budgets and resource allocation across layers.

#### Constructor

```python
def __init__(
    self,
    total_budget: float = 10.0,
    window_size: int = 100,
    safety_margin: float = 0.1,
    adaptation_rate: float = 0.01
)
```

**Parameters:**
- `total_budget` (float): Total energy budget in mJ/sample
- `window_size` (int): History window size for adaptation
- `safety_margin` (float): Safety margin for budget allocation
- `adaptation_rate` (float): Rate of budget adaptation

#### Methods

##### allocate_budget

```python
def allocate_budget(
    self, 
    layer_id: str, 
    complexity_score: float,
    layer_dims: Tuple[int, int],
    performance_priority: float = 1.0
) -> float
```

Allocates budget to a specific layer.

**Parameters:**
- `layer_id` (str): Unique layer identifier
- `complexity_score` (float): Layer complexity score
- `layer_dims` (Tuple[int, int]): Layer dimensions (in_features, out_features)
- `performance_priority` (float): Performance priority weight

**Returns:**
- `float`: Allocated budget in mJ/sample

**Algorithm:**
```python
base_budget = calculate_base_budget(complexity_score, layer_dims)
priority_factor = 0.5 + 0.5 * performance_priority
adjusted_budget = base_budget * priority_factor * budget_scaling_factor
final_budget = min(adjusted_budget, remaining_budget)
```

##### update_energy_consumption

```python
def update_energy_consumption(
    self, 
    layer_id: str, 
    actual_energy: float,
    performance_metric: Optional[float] = None
) -> None
```

Updates energy consumption records for adaptive budget management.

**Parameters:**
- `layer_id` (str): Layer identifier
- `actual_energy` (float): Actual energy consumed
- `performance_metric` (float, optional): Performance metric (e.g., accuracy)

##### get_budget_status

```python
def get_budget_status(self) -> Dict[str, float]
```

Returns current budget status and utilization.

**Returns:**
- `Dict[str, float]`: Budget status including utilization, remaining budget, etc.

##### reset

```python
def reset(self) -> None
```

Resets budget manager to initial state.

## 5. Performance Estimation

### 5.1 PerfEstimator (Base Class)

```python
class PerfEstimator(ABC)
```

Abstract base class for performance estimation.

#### Constructor

```python
def __init__(self, hardware_profile: Dict[str, Any])
```

**Parameters:**
- `hardware_profile` (Dict[str, Any]): Hardware configuration parameters

#### Methods

##### estimate

```python
@abstractmethod
def estimate(self, layer_dims: Tuple[int, int], rank: int) -> float
```

Abstract method for performance estimation.

### 5.2 EnergyEstimator

```python
class EnergyEstimator(PerfEstimator)
```

Estimates energy consumption considering hardware characteristics and memory hierarchy.

#### Constructor

```python
def __init__(
    self, 
    hardware_profile: Dict[str, Any],
    flops_per_mac: float = 2.0,
    energy_per_flop: float = 1e-6,
    energy_per_memory_access: float = 1e-7
)
```

**Parameters:**
- `hardware_profile` (Dict[str, Any]): Hardware configuration
- `flops_per_mac` (float): FLOPs per multiply-accumulate operation
- `energy_per_flop` (float): Energy per FLOP in mJ
- `energy_per_memory_access` (float): Energy per memory access in mJ

**Hardware Profile Keys:**
```python
{
    'dram_energy_per_byte': 1e-6,      # mJ per byte
    'l2_cache_energy_per_byte': 1e-7,  # mJ per byte
    'l1_cache_energy_per_byte': 1e-8,  # mJ per byte
    'compute_energy_per_flop': 1e-6,   # mJ per FLOP
    'l1_cache_size': 64 * 1024,        # bytes
    'l2_cache_size': 1024 * 1024,      # bytes
    'gpu_cores': 5120,                 # number of cores
    'base_frequency': 1.5e9,           # Hz
    'thermal_design_power': 250.0      # Watts
}
```

#### Methods

##### estimate

```python
def estimate(
    self, 
    layer_dims: Tuple[int, int], 
    rank: int,
    batch_size: int = 1,
    utilization: float = 0.8,
    temperature: float = 65.0
) -> float
```

Estimates energy consumption for given configuration.

**Parameters:**
- `layer_dims` (Tuple[int, int]): Layer dimensions
- `rank` (int): Matrix rank
- `batch_size` (int): Batch size
- `utilization` (float): GPU utilization ratio
- `temperature` (float): Operating temperature in Celsius

**Returns:**
- `float`: Estimated energy consumption in mJ

**Energy Model:**
```
E_total = E_compute + E_memory + E_static
where:
- E_compute = FLOPs × E_flop × η_parallel × f_DVFS
- E_memory = Σ(Access_l × E_l × CacheMiss_l)
- E_static = P_idle × t_exec
```

##### estimate_all

```python
def estimate_all(
    self, 
    layer_dims: Tuple[int, int], 
    rank: int, 
    batch_size: int = 1
) -> Dict[str, float]
```

Comprehensive performance estimation.

**Returns:**
- `Dict[str, float]`: Complete performance metrics
  - `'energy_mj'`: Energy in millijoules
  - `'latency_ms'`: Latency in milliseconds  
  - `'memory_mb'`: Memory usage in megabytes
  - `'flops'`: Number of FLOPs
  - `'params'`: Number of parameters

## 6. Utilities

### 6.1 Helper Functions

#### create_loraven_layer

```python
def create_loraven_layer(
    in_features: int,
    out_features: int,
    mode: str = 'balanced',
    **kwargs
) -> LoRAven
```

Convenience function for creating LoRAven layers.

**Parameters:**
- `in_features` (int): Input features
- `out_features` (int): Output features
- `mode` (str): Preset mode
- `**kwargs`: Additional parameters

**Returns:**
- `LoRAven`: Configured LoRAven layer

#### get_default_hardware_profile

```python
def get_default_hardware_profile() -> Dict[str, Any]
```

Returns default hardware profile for common GPU configurations.

**Returns:**
- `Dict[str, Any]`: Default hardware configuration

### 6.2 Visualization Utilities

#### plot_rank_evolution

```python
def plot_rank_evolution(
    rank_history: List[int], 
    complexity_history: List[float],
    save_path: Optional[str] = None
) -> None
```

Plots rank evolution over time.

#### plot_energy_efficiency

```python
def plot_energy_efficiency(
    energy_data: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None
```

Visualizes energy efficiency metrics.

## 7. Examples and Usage

### 7.1 Basic Usage

```python
import torch
from loraven import LoRAven

# Create a LoRAven layer
layer = LoRAven(
    in_features=512,
    out_features=256,
    mode='balanced'
)

# Forward pass
x = torch.randn(32, 512)
output = layer(x)

# Check current rank
current_rank = layer.get_current_rank()
print(f"Current rank: {current_rank}")
```

### 7.2 Custom Configuration

```python
from loraven import LoRAven

# Custom configuration
layer = LoRAven(
    in_features=1024,
    out_features=512,
    mode='custom',
    max_rank=128,
    min_rank=8,
    energy_budget=500.0  # mJ per sample
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        output = layer(batch)
        loss = criterion(output, targets)
        
        # Monitor performance
        stats = layer.get_performance_stats()
        print(f"Energy usage: {stats['energy_consumption']:.2f} mJ")
        print(f"Current rank: {stats['current_rank']}")
```

### 7.3 Advanced Usage with Budget Management

```python
from loraven import DynamicLowRankLayer, BudgetManager

# Create budget manager
budget_manager = BudgetManager(
    total_budget=1000.0,  # mJ per sample
    window_size=100
)

# Create layer with budget management
layer = DynamicLowRankLayer(
    in_features=2048,
    out_features=1024,
    r_max=256,
    r_min=16
)

# Training with budget awareness
for batch_idx, (data, target) in enumerate(dataloader):
    # Allocate budget for this batch
    budget = budget_manager.allocate_budget(
        layer_id=f"layer_{batch_idx}",
        complexity_score=0.7,  # Example complexity
        layer_dims=(2048, 1024),
        performance_priority=0.8
    )
    
    # Forward pass with budget constraint
    output, current_rank = layer(data, budget=budget, mode='training')
    
    # Update energy consumption
    actual_energy = estimate_actual_energy(output, current_rank)
    budget_manager.update_energy_consumption(
        layer_id=f"layer_{batch_idx}",
        actual_energy=actual_energy,
        performance_metric=calculate_accuracy(output, target)
    )
```

### 7.4 Integration with Existing Models

```python
import torch.nn as nn
from loraven import LoRAven

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loraven1 = LoRAven(784, 512, mode='high_performance')
        self.loraven2 = LoRAven(512, 256, mode='balanced')
        self.loraven3 = LoRAven(256, 128, mode='low_power')
        self.classifier = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.loraven1(x))
        x = torch.relu(self.loraven2(x))
        x = torch.relu(self.loraven3(x))
        return self.classifier(x)

# Usage
model = MyModel()
x = torch.randn(32, 1, 28, 28)  # MNIST-like input
output = model(x)
```

### 7.5 Performance Monitoring

```python
from loraven import LoRAven
import matplotlib.pyplot as plt

layer = LoRAven(512, 256, mode='balanced')

# Collect performance data
rank_history = []
energy_history = []

for i in range(100):
    x = torch.randn(32, 512)
    output = layer(x)
    
    stats = layer.get_performance_stats()
    rank_history.append(stats['current_rank'])
    energy_history.append(stats['energy_consumption'])

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(rank_history)
plt.title('Rank Evolution')
plt.xlabel('Iteration')
plt.ylabel('Rank')

plt.subplot(1, 2, 2)
plt.plot(energy_history)
plt.title('Energy Consumption')
plt.xlabel('Iteration')
plt.ylabel('Energy (mJ)')
plt.tight_layout()
plt.show()
```

## Error Handling and Best Practices

### Common Error Scenarios

1. **Invalid Rank Configuration**
   ```python
   # This will raise ValueError
   layer = LoRAven(512, 256, mode='custom', max_rank=600)  # max_rank > min(in_features, out_features)
   ```

2. **Budget Exhaustion**
   ```python
   # Handle budget exhaustion gracefully
   try:
       budget = budget_manager.allocate_budget(layer_id, complexity, dims)
       if budget <= 0:
           # Use minimum rank configuration
           output, rank = layer(x, budget=0, mode='inference')
   except BudgetExhaustedException:
       # Fallback to emergency mode
       pass
   ```

3. **Numerical Instability**
   ```python
   # LoRAven includes automatic NaN handling
   x = torch.tensor([[float('nan'), 1.0, 2.0]])
   output = layer(x)  # Automatically handles NaN values
   ```

### Performance Optimization Tips

1. **Batch Size Optimization**
   ```python
   # Larger batch sizes improve efficiency
   layer = LoRAven(512, 256, mode='balanced')
   
   # Efficient: batch_size = 32 or higher
   x = torch.randn(32, 512)
   output = layer(x)
   ```

2. **Memory Management**
   ```python
   # Use torch.no_grad() for inference
   with torch.no_grad():
       output = layer(x)
   ```

3. **Device Placement**
   ```python
   # Ensure consistent device placement
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   layer = LoRAven(512, 256, device=device)
   x = x.to(device)
   ```

This comprehensive API reference provides detailed documentation for all LoRAven components, enabling efficient integration and usage in various applications.
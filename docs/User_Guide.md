# LoRAven User Guide

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Configuration Guide](#4-configuration-guide)
5. [Usage Examples](#5-usage-examples)
6. [Performance Optimization](#6-performance-optimization)
7. [Troubleshooting](#7-troubleshooting)
8. [Best Practices](#8-best-practices)
9. [Advanced Features](#9-advanced-features)
10. [Integration Examples](#10-integration-examples)

## 1. Introduction

LoRAven (Low-Rank Adaptive Venture) is a PyTorch-based library that implements dynamic low-rank adaptation for neural networks. It provides energy-efficient neural network layers that automatically adjust their computational complexity based on input characteristics and available energy budgets.

### Key Features

- **Dynamic Rank Adaptation**: Automatically adjusts matrix rank based on input complexity
- **Energy-Aware Computing**: Considers energy budgets in computational decisions
- **Hardware-Aware Optimization**: Adapts to different hardware configurations
- **Easy Integration**: Drop-in replacement for standard PyTorch layers
- **Comprehensive Monitoring**: Built-in performance and energy tracking

### When to Use LoRAven

LoRAven is ideal for:
- Mobile and edge computing applications
- Energy-constrained environments
- Large-scale neural networks requiring efficiency
- Applications with varying computational demands
- Research in adaptive neural architectures

## 2. Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9.0 or higher
- NumPy 1.19.0 or higher
- Optional: CUDA for GPU acceleration

### Installation Methods

#### Method 1: From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/loraven.git
cd loraven

# Install in development mode
pip install -e .

# Or install dependencies manually
pip install torch numpy matplotlib seaborn
```

#### Method 2: Using pip (when available)

```bash
pip install loraven
```

### Verification

```python
import torch
from loraven import LoRAven

# Test installation
layer = LoRAven(10, 5, mode='balanced')
x = torch.randn(2, 10)
output = layer(x)
print(f"Installation successful! Output shape: {output.shape}")
```

## 3. Quick Start

### Basic Usage

```python
import torch
from loraven import LoRAven

# Create a LoRAven layer
layer = LoRAven(
    in_features=512,
    out_features=256,
    mode='balanced'  # Options: 'high_performance', 'balanced', 'low_power'
)

# Use like any PyTorch layer
x = torch.randn(32, 512)  # Batch of 32 samples
output = layer(x)         # Shape: (32, 256)

print(f"Current rank: {layer.get_current_rank()}")
print(f"Budget usage: {layer.get_budget_usage():.2%}")
```

### Integration with Existing Models

```python
import torch.nn as nn
from loraven import LoRAven

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Replace nn.Linear with LoRAven
        self.layer1 = LoRAven(784, 512, mode='high_performance')
        self.layer2 = LoRAven(512, 256, mode='balanced')
        self.layer3 = LoRAven(256, 10, mode='low_power')
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# Usage
model = MyModel()
x = torch.randn(32, 1, 28, 28)  # MNIST-like input
output = model(x)
```

## 4. Configuration Guide

### Preset Modes

LoRAven provides three preset modes for different use cases:

#### High Performance Mode
```python
layer = LoRAven(512, 256, mode='high_performance')
```
- **Rank Ratio**: 80% of maximum possible rank
- **Min Rank Ratio**: 30% of maximum possible rank
- **Energy Multiplier**: 1.5x (allows higher energy consumption)
- **Use Case**: Accuracy-critical applications

#### Balanced Mode (Default)
```python
layer = LoRAven(512, 256, mode='balanced')
```
- **Rank Ratio**: 50% of maximum possible rank
- **Min Rank Ratio**: 20% of maximum possible rank
- **Energy Multiplier**: 1.0x (standard energy budget)
- **Use Case**: General-purpose applications

#### Low Power Mode
```python
layer = LoRAven(512, 256, mode='low_power')
```
- **Rank Ratio**: 30% of maximum possible rank
- **Min Rank Ratio**: 10% of maximum possible rank
- **Energy Multiplier**: 0.7x (reduced energy consumption)
- **Use Case**: Battery-powered devices, edge computing

### Custom Configuration

```python
layer = LoRAven(
    in_features=1024,
    out_features=512,
    mode='custom',
    max_rank=128,           # Maximum rank
    min_rank=16,            # Minimum rank
    energy_budget=500.0,    # Energy budget in mJ/sample
    bias=True,              # Include bias term
    device=torch.device('cuda')  # Computation device
)
```

### Advanced Configuration

```python
from loraven import DynamicLowRankLayer, BudgetManager, EnergyAwareRankScheduler

# Create custom budget manager
budget_manager = BudgetManager(
    total_budget=1000.0,    # Total energy budget (mJ/sample)
    window_size=100,        # History window for adaptation
    safety_margin=0.1,      # Safety margin (10%)
    adaptation_rate=0.01    # Learning rate for budget adaptation
)

# Create custom rank scheduler
scheduler = EnergyAwareRankScheduler(
    r_min=8,
    r_max=256,
    alpha=1.0,              # Energy weight
    beta=0.1                # Performance weight
)

# Create layer with custom components
layer = DynamicLowRankLayer(
    in_features=2048,
    out_features=1024,
    r_max=256,
    r_min=8,
    scorer_hidden=64,       # Complexity scorer hidden dimension
    num_gate_blocks=16      # Number of gate network blocks
)
```

## 5. Usage Examples

### Example 1: Image Classification

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from loraven import LoRAven

class LoRAvenCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Use LoRAven for fully connected layers
        self.classifier = nn.Sequential(
            LoRAven(64 * 7 * 7, 512, mode='high_performance'),
            nn.ReLU(),
            nn.Dropout(0.5),
            LoRAven(512, 256, mode='balanced'),
            nn.ReLU(),
            nn.Dropout(0.5),
            LoRAven(256, num_classes, mode='low_power')
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Training setup
model = LoRAvenCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Monitor LoRAven layers
        if batch_idx % 100 == 0:
            for name, module in model.named_modules():
                if isinstance(module, LoRAven):
                    stats = module.get_performance_stats()
                    print(f"{name}: Rank={stats['current_rank']}, "
                          f"Energy={stats['energy_consumption']:.2f}mJ")
```

### Example 2: Natural Language Processing

```python
import torch
import torch.nn as nn
from loraven import LoRAven

class LoRAvenTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Custom transformer layer with LoRAven
        self.transformer_layers = nn.ModuleList([
            LoRAvenTransformerLayer(d_model, nhead)
            for _ in range(num_layers)
        ])
        
        self.output_projection = LoRAven(d_model, vocab_size, mode='balanced')
    
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        return self.output_projection(x)

class LoRAvenTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        
        # Replace standard linear layers with LoRAven
        self.feed_forward = nn.Sequential(
            LoRAven(d_model, 2048, mode='high_performance'),
            nn.ReLU(),
            nn.Dropout(0.1),
            LoRAven(2048, d_model, mode='balanced')
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### Example 3: Reinforcement Learning

```python
import torch
import torch.nn as nn
from loraven import LoRAven

class LoRAvenDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        # Value network with adaptive complexity
        self.value_net = nn.Sequential(
            LoRAven(state_dim, hidden_dim, mode='high_performance'),
            nn.ReLU(),
            LoRAven(hidden_dim, hidden_dim, mode='balanced'),
            nn.ReLU(),
            LoRAven(hidden_dim, action_dim, mode='low_power')
        )
        
        # Advantage network
        self.advantage_net = nn.Sequential(
            LoRAven(state_dim, hidden_dim, mode='balanced'),
            nn.ReLU(),
            LoRAven(hidden_dim, hidden_dim // 2, mode='balanced'),
            nn.ReLU(),
            LoRAven(hidden_dim // 2, action_dim, mode='low_power')
        )
    
    def forward(self, state):
        value = self.value_net(state)
        advantage = self.advantage_net(state)
        
        # Dueling DQN architecture
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# Training with energy monitoring
agent = LoRAvenDQN(state_dim=84*84*4, action_dim=4)
optimizer = torch.optim.Adam(agent.parameters(), lr=0.0001)

def train_step(state, action, reward, next_state, done):
    q_values = agent(state)
    next_q_values = agent(next_state)
    
    target = reward + 0.99 * next_q_values.max(1)[0] * (1 - done)
    loss = nn.MSELoss()(q_values.gather(1, action), target.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Monitor energy consumption
    total_energy = 0
    for module in agent.modules():
        if isinstance(module, LoRAven):
            stats = module.get_performance_stats()
            total_energy += stats['energy_consumption']
    
    return loss.item(), total_energy
```

## 6. Performance Optimization

### Batch Size Optimization

```python
# Larger batch sizes improve efficiency
layer = LoRAven(512, 256, mode='balanced')

# Inefficient: small batch size
x_small = torch.randn(1, 512)
output_small = layer(x_small)

# Efficient: larger batch size
x_large = torch.randn(64, 512)
output_large = layer(x_large)

# Measure performance difference
import time

# Small batch timing
start = time.time()
for _ in range(100):
    _ = layer(x_small)
small_batch_time = time.time() - start

# Large batch timing
start = time.time()
for _ in range(100):
    _ = layer(x_large)
large_batch_time = time.time() - start

print(f"Small batch: {small_batch_time:.3f}s")
print(f"Large batch: {large_batch_time:.3f}s")
print(f"Efficiency gain: {small_batch_time/large_batch_time:.2f}x")
```

### Memory Management

```python
# Use torch.no_grad() for inference
model = LoRAvenCNN()
model.eval()

with torch.no_grad():
    for data in test_loader:
        output = model(data)
        # Process output without gradient computation

# Clear cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Device Optimization

```python
# Automatic device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create layer on specific device
layer = LoRAven(512, 256, mode='balanced', device=device)

# Or move existing layer
layer = LoRAven(512, 256, mode='balanced')
layer = layer.to(device)

# Ensure input tensors are on same device
x = torch.randn(32, 512, device=device)
output = layer(x)
```

### Rank Configuration Optimization

```python
# Find optimal rank configuration
def find_optimal_rank(layer_dims, target_accuracy=0.95, max_energy=100.0):
    best_config = None
    best_score = 0
    
    for max_rank in [32, 64, 128, 256]:
        for min_rank in [4, 8, 16]:
            if min_rank >= max_rank:
                continue
                
            layer = LoRAven(
                layer_dims[0], layer_dims[1],
                mode='custom',
                max_rank=max_rank,
                min_rank=min_rank,
                energy_budget=max_energy
            )
            
            # Evaluate configuration
            accuracy = evaluate_accuracy(layer)
            energy = measure_energy_consumption(layer)
            
            # Score based on accuracy and energy efficiency
            score = accuracy * (max_energy / energy) if energy > 0 else 0
            
            if score > best_score and accuracy >= target_accuracy:
                best_score = score
                best_config = (max_rank, min_rank)
    
    return best_config

# Usage
optimal_ranks = find_optimal_rank((512, 256))
print(f"Optimal configuration: max_rank={optimal_ranks[0]}, min_rank={optimal_ranks[1]}")
```

## 7. Troubleshooting

### Common Issues and Solutions

#### Issue 1: NaN Values in Output

```python
# Problem: NaN values appearing in output
x = torch.tensor([[float('nan'), 1.0, 2.0]])
output = layer(x)  # May contain NaN

# Solution: LoRAven automatically handles NaN values
# Check if input contains NaN
if torch.isnan(x).any():
    print("Warning: Input contains NaN values")
    x = torch.nan_to_num(x, nan=0.0)  # Replace NaN with 0

# Alternative: Use built-in NaN handling
layer = LoRAven(512, 256, mode='balanced')
output = layer(x)  # Automatically preprocesses NaN values
```

#### Issue 2: Memory Issues with Large Models

```python
# Problem: Out of memory errors
try:
    large_layer = LoRAven(10000, 5000, mode='high_performance')
    x = torch.randn(1000, 10000)
    output = large_layer(x)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("Memory error detected")
        
        # Solution 1: Reduce batch size
        for i in range(0, x.size(0), 100):  # Process in chunks of 100
            batch = x[i:i+100]
            batch_output = large_layer(batch)
        
        # Solution 2: Use lower rank configuration
        efficient_layer = LoRAven(10000, 5000, mode='low_power')
        output = efficient_layer(x)
        
        # Solution 3: Enable gradient checkpointing
        torch.utils.checkpoint.checkpoint(large_layer, x)
```

#### Issue 3: Slow Performance

```python
# Problem: Slower than expected performance
import time

layer = LoRAven(512, 256, mode='balanced')
x = torch.randn(32, 512)

# Measure performance
start = time.time()
for _ in range(100):
    output = layer(x)
end = time.time()

if (end - start) > 1.0:  # If too slow
    print("Performance issue detected")
    
    # Solution 1: Check device placement
    if not x.is_cuda and torch.cuda.is_available():
        layer = layer.cuda()
        x = x.cuda()
    
    # Solution 2: Optimize rank configuration
    layer = LoRAven(512, 256, mode='low_power')  # Lower computational cost
    
    # Solution 3: Use mixed precision
    with torch.cuda.amp.autocast():
        output = layer(x)
```

#### Issue 4: Unstable Training

```python
# Problem: Training loss oscillates or diverges
class StableLoRAvenModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use gradient clipping and careful initialization
        self.layer1 = LoRAven(784, 512, mode='balanced')
        self.layer2 = LoRAven(512, 256, mode='balanced')
        self.layer3 = LoRAven(256, 10, mode='balanced')
        
        # Initialize with smaller learning rates for LoRAven layers
        for module in self.modules():
            if isinstance(module, LoRAven):
                # Custom initialization if needed
                pass
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# Training with stability measures
model = StableLoRAvenModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Monitor for instability
        if torch.isnan(loss) or loss.item() > 100:
            print(f"Unstable training detected at epoch {epoch}, batch {batch_idx}")
            # Reset or adjust learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
```

### Debugging Tools

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor layer statistics
def debug_layer_stats(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRAven):
            stats = module.get_performance_stats()
            print(f"Layer {name}:")
            print(f"  Current rank: {stats['current_rank']}")
            print(f"  Energy consumption: {stats['energy_consumption']:.2f} mJ")
            print(f"  Budget usage: {stats['budget_usage']:.2%}")
            print(f"  Compression ratio: {module.get_compression_ratio():.2%}")

# Usage during training
model = LoRAvenCNN()
debug_layer_stats(model)
```

## 8. Best Practices

### Model Design

1. **Layer Placement Strategy**
   ```python
   class OptimalLoRAvenModel(nn.Module):
       def __init__(self):
           super().__init__()
           # Use high-performance mode for critical layers
           self.feature_extractor = LoRAven(784, 512, mode='high_performance')
           
           # Use balanced mode for intermediate layers
           self.hidden_layers = nn.ModuleList([
               LoRAven(512, 512, mode='balanced') for _ in range(3)
           ])
           
           # Use low-power mode for final layers
           self.classifier = LoRAven(512, 10, mode='low_power')
   ```

2. **Progressive Complexity Reduction**
   ```python
   # Gradually reduce complexity through the network
   layers = [
       LoRAven(1024, 512, mode='high_performance'),  # 80% rank ratio
       LoRAven(512, 256, mode='balanced'),           # 50% rank ratio
       LoRAven(256, 128, mode='low_power'),          # 30% rank ratio
       LoRAven(128, 10, mode='low_power')            # 30% rank ratio
   ]
   ```

### Training Strategies

1. **Warm-up Training**
   ```python
   # Start with higher ranks, gradually reduce
   def warmup_schedule(epoch, total_epochs):
       if epoch < total_epochs * 0.1:  # First 10% of training
           return 'high_performance'
       elif epoch < total_epochs * 0.7:  # Next 60% of training
           return 'balanced'
       else:  # Final 30% of training
           return 'low_power'
   
   # Apply during training
   for epoch in range(num_epochs):
       mode = warmup_schedule(epoch, num_epochs)
       # Update layer modes if needed
   ```

2. **Adaptive Learning Rates**
   ```python
   # Different learning rates for different components
   loraven_params = []
   other_params = []
   
   for name, param in model.named_parameters():
       if 'loraven' in name.lower():
           loraven_params.append(param)
       else:
           other_params.append(param)
   
   optimizer = torch.optim.Adam([
       {'params': loraven_params, 'lr': 0.001},
       {'params': other_params, 'lr': 0.01}
   ])
   ```

### Energy Management

1. **Budget Allocation Strategy**
   ```python
   # Allocate more budget to critical layers
   budget_manager = BudgetManager(total_budget=1000.0)
   
   layer_priorities = {
       'feature_extractor': 1.5,  # High priority
       'hidden_layer_1': 1.0,     # Normal priority
       'hidden_layer_2': 1.0,     # Normal priority
       'classifier': 0.5          # Low priority
   }
   
   for layer_name, priority in layer_priorities.items():
       budget = budget_manager.allocate_budget(
           layer_id=layer_name,
           complexity_score=0.7,
           layer_dims=(512, 256),
           performance_priority=priority
       )
   ```

2. **Dynamic Budget Adjustment**
   ```python
   # Adjust budget based on performance
   def adjust_budget_based_on_accuracy(accuracy, base_budget):
       if accuracy < 0.8:  # Low accuracy, increase budget
           return base_budget * 1.2
       elif accuracy > 0.95:  # High accuracy, reduce budget
           return base_budget * 0.8
       else:
           return base_budget
   ```

### Monitoring and Evaluation

1. **Comprehensive Metrics Collection**
   ```python
   class MetricsCollector:
       def __init__(self):
           self.metrics = {
               'energy_consumption': [],
               'rank_history': [],
               'accuracy': [],
               'latency': [],
               'memory_usage': []
           }
       
       def collect(self, model, accuracy, latency):
           total_energy = 0
           avg_rank = 0
           layer_count = 0
           
           for module in model.modules():
               if isinstance(module, LoRAven):
                   stats = module.get_performance_stats()
                   total_energy += stats['energy_consumption']
                   avg_rank += stats['current_rank']
                   layer_count += 1
           
           self.metrics['energy_consumption'].append(total_energy)
           self.metrics['rank_history'].append(avg_rank / layer_count)
           self.metrics['accuracy'].append(accuracy)
           self.metrics['latency'].append(latency)
           
           if torch.cuda.is_available():
               memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB
               self.metrics['memory_usage'].append(memory_usage)
   
   # Usage
   collector = MetricsCollector()
   
   for epoch in range(num_epochs):
       # Training code...
       accuracy = evaluate_model(model, test_loader)
       latency = measure_inference_time(model, sample_input)
       collector.collect(model, accuracy, latency)
   ```

2. **Performance Visualization**
   ```python
   import matplotlib.pyplot as plt
   
   def plot_performance_metrics(collector):
       fig, axes = plt.subplots(2, 2, figsize=(12, 8))
       
       # Energy consumption over time
       axes[0, 0].plot(collector.metrics['energy_consumption'])
       axes[0, 0].set_title('Energy Consumption')
       axes[0, 0].set_ylabel('Energy (mJ)')
       
       # Rank evolution
       axes[0, 1].plot(collector.metrics['rank_history'])
       axes[0, 1].set_title('Average Rank Evolution')
       axes[0, 1].set_ylabel('Rank')
       
       # Accuracy vs Energy trade-off
       axes[1, 0].scatter(collector.metrics['energy_consumption'], 
                         collector.metrics['accuracy'])
       axes[1, 0].set_xlabel('Energy (mJ)')
       axes[1, 0].set_ylabel('Accuracy')
       axes[1, 0].set_title('Accuracy vs Energy Trade-off')
       
       # Memory usage
       if collector.metrics['memory_usage']:
           axes[1, 1].plot(collector.metrics['memory_usage'])
           axes[1, 1].set_title('Memory Usage')
           axes[1, 1].set_ylabel('Memory (MB)')
       
       plt.tight_layout()
       plt.show()
   ```

## 9. Advanced Features

### Custom Rank Schedulers

```python
from loraven.core.rank_scheduler import RankScheduler

class CustomRankScheduler(RankScheduler):
    def __init__(self, r_min=4, r_max=128, temperature=1.0):
        self.r_min = r_min
        self.r_max = r_max
        self.temperature = temperature
    
    def schedule_rank(self, complexity_scores, budget=None, **kwargs):
        # Custom scheduling logic
        avg_complexity = complexity_scores.mean().item()
        
        # Temperature-based scaling
        scaled_complexity = avg_complexity / self.temperature
        
        # Sigmoid-based rank mapping
        sigmoid_value = torch.sigmoid(torch.tensor(scaled_complexity))
        rank = int(self.r_min + sigmoid_value * (self.r_max - self.r_min))
        
        # Budget constraint
        if budget is not None and budget < 50.0:  # Low budget
            rank = min(rank, self.r_min + (self.r_max - self.r_min) // 4)
        
        return max(self.r_min, min(rank, self.r_max))

# Usage
custom_scheduler = CustomRankScheduler(r_min=8, r_max=256, temperature=2.0)
layer = DynamicLowRankLayer(
    in_features=512,
    out_features=256,
    r_max=256,
    r_min=8
)
# Apply custom scheduler to layer
```

### Custom Energy Estimators

```python
from loraven.utils.perf_estimator import PerfEstimator

class CustomEnergyEstimator(PerfEstimator):
    def __init__(self, hardware_profile):
        super().__init__(hardware_profile)
        self.base_energy_per_flop = 1e-6  # mJ per FLOP
        self.memory_energy_factor = 1e-7  # mJ per byte
    
    def estimate(self, layer_dims, rank, batch_size=1, **kwargs):
        in_features, out_features = layer_dims
        
        # Calculate FLOPs for low-rank factorization
        flops = batch_size * (in_features * rank + rank * out_features)
        
        # Calculate memory accesses
        memory_accesses = (in_features + out_features) * rank * 4  # 4 bytes per float
        
        # Custom energy model
        compute_energy = flops * self.base_energy_per_flop
        memory_energy = memory_accesses * self.memory_energy_factor
        
        # Add hardware-specific factors
        gpu_efficiency = self.hardware_profile.get('gpu_efficiency', 0.8)
        total_energy = (compute_energy + memory_energy) / gpu_efficiency
        
        return total_energy

# Usage
hardware_profile = {
    'gpu_efficiency': 0.85,
    'memory_bandwidth': 900e9,  # bytes/second
    'compute_capability': 7.5
}

custom_estimator = CustomEnergyEstimator(hardware_profile)
energy = custom_estimator.estimate((512, 256), rank=64, batch_size=32)
print(f"Estimated energy: {energy:.3f} mJ")
```

### Event-Driven Adaptation

```python
class EventDrivenLoRAven(LoRAven):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_threshold = 0.95
        self.energy_threshold = 100.0  # mJ
        self.adaptation_cooldown = 10
        self.last_adaptation = 0
    
    def forward(self, x):
        output = super().forward(x)
        
        # Check for adaptation triggers
        current_step = getattr(self, '_step_counter', 0)
        self._step_counter = current_step + 1
        
        if (current_step - self.last_adaptation) > self.adaptation_cooldown:
            stats = self.get_performance_stats()
            
            # Trigger adaptation based on events
            if stats['energy_consumption'] > self.energy_threshold:
                self._reduce_complexity()
                self.last_adaptation = current_step
            elif stats.get('accuracy', 1.0) < self.performance_threshold:
                self._increase_complexity()
                self.last_adaptation = current_step
        
        return output
    
    def _reduce_complexity(self):
        # Switch to lower power mode
        if hasattr(self, '_current_mode'):
            if self._current_mode == 'high_performance':
                self._current_mode = 'balanced'
            elif self._current_mode == 'balanced':
                self._current_mode = 'low_power'
    
    def _increase_complexity(self):
        # Switch to higher performance mode
        if hasattr(self, '_current_mode'):
            if self._current_mode == 'low_power':
                self._current_mode = 'balanced'
            elif self._current_mode == 'balanced':
                self._current_mode = 'high_performance'
```

## 10. Integration Examples

### Integration with Hugging Face Transformers

```python
from transformers import BertModel, BertConfig
from loraven import LoRAven
import torch.nn as nn

class LoRAvenBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        
        # Replace linear layers in BERT with LoRAven
        self._replace_linear_layers()
        
        self.classifier = LoRAven(
            config.hidden_size, 
            config.num_labels, 
            mode='balanced'
        )
    
    def _replace_linear_layers(self):
        for name, module in self.bert.named_modules():
            if isinstance(module, nn.Linear):
                # Replace with LoRAven layer
                loraven_layer = LoRAven(
                    module.in_features,
                    module.out_features,
                    mode='balanced',
                    bias=module.bias is not None
                )
                
                # Copy weights if needed
                with torch.no_grad():
                    # Initialize LoRAven with existing weights (approximation)
                    pass
                
                # Replace the module
                parent = self.bert
                for attr in name.split('.')[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, name.split('.')[-1], loraven_layer)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        return logits

# Usage
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2  # Binary classification
model = LoRAvenBERT(config)
```

### Integration with PyTorch Lightning

```python
import pytorch_lightning as pl
from loraven import LoRAven

class LoRAvenLightningModule(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.network = nn.Sequential(
            LoRAven(input_dim, hidden_dim, mode='high_performance'),
            nn.ReLU(),
            nn.Dropout(0.2),
            LoRAven(hidden_dim, hidden_dim, mode='balanced'),
            nn.ReLU(),
            nn.Dropout(0.2),
            LoRAven(hidden_dim, output_dim, mode='low_power')
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log LoRAven metrics
        self._log_loraven_metrics('train')
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Calculate accuracy
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        # Log LoRAven metrics
        self._log_loraven_metrics('val')
        
        return loss
    
    def _log_loraven_metrics(self, stage):
        total_energy = 0
        avg_rank = 0
        layer_count = 0
        
        for name, module in self.network.named_modules():
            if isinstance(module, LoRAven):
                stats = module.get_performance_stats()
                total_energy += stats['energy_consumption']
                avg_rank += stats['current_rank']
                layer_count += 1
        
        if layer_count > 0:
            self.log(f'{stage}_total_energy', total_energy)
            self.log(f'{stage}_avg_rank', avg_rank / layer_count)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

# Usage
model = LoRAvenLightningModule(784, 512, 10)
trainer = pl.Trainer(max_epochs=100, gpus=1)
trainer.fit(model, train_dataloader, val_dataloader)
```

### Integration with Ray Tune for Hyperparameter Optimization

```python
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from loraven import LoRAven

def train_loraven_model(config):
    # Create model with hyperparameters from config
    model = nn.Sequential(
        LoRAven(
            784, 
            config['hidden_dim'], 
            mode=config['mode'],
            max_rank=config['max_rank'],
            min_rank=config['min_rank']
        ),
        nn.ReLU(),
        LoRAven(config['hidden_dim'], 10, mode='balanced')
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(10):
        total_loss = 0
        total_energy = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.view(data.size(0), -1))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate energy consumption
            for module in model.modules():
                if isinstance(module, LoRAven):
                    stats = module.get_performance_stats()
                    total_energy += stats['energy_consumption']
        
        # Report metrics to Ray Tune
        avg_loss = total_loss / len(train_loader)
        avg_energy = total_energy / len(train_loader)
        
        # Calculate validation accuracy
        val_acc = evaluate_model(model, val_loader)
        
        # Multi-objective optimization: maximize accuracy, minimize energy
        score = val_acc - 0.1 * (avg_energy / 100.0)  # Balance accuracy and energy
        
        tune.report(
            loss=avg_loss,
            accuracy=val_acc,
            energy=avg_energy,
            score=score
        )

# Hyperparameter search space
search_space = {
    'hidden_dim': tune.choice([256, 512, 1024]),
    'mode': tune.choice(['balanced', 'high_performance', 'low_power']),
    'max_rank': tune.choice([64, 128, 256]),
    'min_rank': tune.choice([4, 8, 16]),
    'lr': tune.loguniform(1e-4, 1e-2)
}

# Run hyperparameter optimization
scheduler = ASHAScheduler(
    metric='score',
    mode='max',
    max_t=10,
    grace_period=1,
    reduction_factor=2
)

result = tune.run(
    train_loraven_model,
    config=search_space,
    num_samples=50,
    scheduler=scheduler,
    resources_per_trial={'cpu': 2, 'gpu': 0.5}
)

# Get best configuration
best_config = result.get_best_config(metric='score', mode='max')
print(f"Best configuration: {best_config}")
```

This comprehensive user guide provides detailed instructions for using LoRAven effectively across various scenarios and applications. The examples demonstrate both basic usage patterns and advanced integration techniques, helping users maximize the benefits of dynamic low-rank adaptation in their neural network applications.
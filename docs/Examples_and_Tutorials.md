# LoRAven Examples and Tutorials

## Table of Contents

1. [Getting Started Examples](#1-getting-started-examples)
2. [Computer Vision Applications](#2-computer-vision-applications)
3. [Natural Language Processing](#3-natural-language-processing)
4. [Reinforcement Learning](#4-reinforcement-learning)
5. [Time Series Analysis](#5-time-series-analysis)
6. [Energy-Aware Training](#6-energy-aware-training)
7. [Custom Components](#7-custom-components)
8. [Benchmarking and Evaluation](#8-benchmarking-and-evaluation)
9. [Production Deployment](#9-production-deployment)
10. [Advanced Optimization](#10-advanced-optimization)

## 1. Getting Started Examples

### 1.1 Hello World with LoRAven

```python
# Basic LoRAven layer usage
import torch
from loraven import LoRAven

# Create a simple LoRAven layer
layer = LoRAven(
    in_features=100,
    out_features=50,
    mode='balanced'
)

# Generate sample input
x = torch.randn(32, 100)  # Batch of 32 samples

# Forward pass
output = layer(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Current rank: {layer.get_current_rank()}")
print(f"Budget usage: {layer.get_budget_usage():.2%}")
```

### 1.2 Comparing Different Modes

```python
import torch
import matplotlib.pyplot as plt
from loraven import LoRAven

# Create layers with different modes
modes = ['high_performance', 'balanced', 'low_power']
layers = {mode: LoRAven(512, 256, mode=mode) for mode in modes}

# Test input
x = torch.randn(64, 512)

# Compare performance
results = {}
for mode, layer in layers.items():
    output = layer(x)
    stats = layer.get_performance_stats()
    
    results[mode] = {
        'rank': stats['current_rank'],
        'energy': stats['energy_consumption'],
        'compression': layer.get_compression_ratio()
    }

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['rank', 'energy', 'compression']
titles = ['Current Rank', 'Energy Consumption (mJ)', 'Compression Ratio']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    values = [results[mode][metric] for mode in modes]
    axes[i].bar(modes, values)
    axes[i].set_title(title)
    axes[i].set_ylabel(title.split('(')[0])

plt.tight_layout()
plt.show()

# Print detailed comparison
print("\nMode Comparison:")
print("-" * 50)
for mode in modes:
    print(f"{mode:>15}: Rank={results[mode]['rank']:>3}, "
          f"Energy={results[mode]['energy']:>6.2f}mJ, "
          f"Compression={results[mode]['compression']:>5.1%}")
```

### 1.3 Dynamic Rank Adaptation Visualization

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from loraven import LoRAven

# Create layer
layer = LoRAven(256, 128, mode='balanced')

# Simulate varying input complexity
num_steps = 100
rank_history = []
complexity_history = []

for step in range(num_steps):
    # Generate input with varying complexity
    if step < 30:
        # Simple patterns
        x = torch.randn(16, 256) * 0.5
    elif step < 70:
        # Complex patterns
        x = torch.randn(16, 256) * 2.0 + torch.sin(torch.arange(256).float()) * 0.5
    else:
        # Mixed complexity
        complexity_factor = np.sin(step * 0.1) + 1.0
        x = torch.randn(16, 256) * complexity_factor
    
    # Forward pass
    output = layer(x)
    
    # Record metrics
    current_rank = layer.get_current_rank()
    input_complexity = torch.std(x).item()
    
    rank_history.append(current_rank)
    complexity_history.append(input_complexity)

# Visualize adaptation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot rank evolution
ax1.plot(rank_history, 'b-', linewidth=2, label='Rank')
ax1.set_ylabel('Rank')
ax1.set_title('Dynamic Rank Adaptation')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot input complexity
ax2.plot(complexity_history, 'r-', linewidth=2, label='Input Complexity')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Complexity (Std Dev)')
ax2.set_title('Input Complexity Over Time')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

print(f"Rank adaptation range: {min(rank_history)} - {max(rank_history)}")
print(f"Average rank: {np.mean(rank_history):.1f}")
```

## 2. Computer Vision Applications

### 2.1 MNIST Classification with LoRAven

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from loraven import LoRAven

class LoRAvenMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Progressive complexity reduction
        self.network = nn.Sequential(
            LoRAven(784, 512, mode='high_performance'),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            LoRAven(512, 256, mode='balanced'),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            LoRAven(256, 128, mode='balanced'),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            LoRAven(128, 10, mode='low_power')
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LoRAvenMNIST().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training function
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_energy = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect energy metrics
        batch_energy = 0
        for module in model.modules():
            if isinstance(module, LoRAven):
                stats = module.get_performance_stats()
                batch_energy += stats['energy_consumption']
        total_energy += batch_energy
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx:>3}: Loss={loss.item():.4f}, '
                  f'Energy={batch_energy:.2f}mJ')
    
    return total_loss / len(train_loader), total_energy / len(train_loader)

# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total_energy = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Collect energy metrics
            for module in model.modules():
                if isinstance(module, LoRAven):
                    stats = module.get_performance_stats()
                    total_energy += stats['energy_consumption']
    
    accuracy = 100. * correct / len(test_loader.dataset)
    avg_energy = total_energy / len(test_loader)
    
    return accuracy, avg_energy

# Training loop
num_epochs = 10
train_losses = []
train_energies = []
test_accuracies = []
test_energies = []

print("Starting training...")
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
    # Train
    train_loss, train_energy = train_epoch(model, train_loader, optimizer, criterion, device)
    train_losses.append(train_loss)
    train_energies.append(train_energy)
    
    # Evaluate
    test_acc, test_energy = evaluate(model, test_loader, device)
    test_accuracies.append(test_acc)
    test_energies.append(test_energy)
    
    print(f"Train Loss: {train_loss:.4f}, Train Energy: {train_energy:.2f}mJ")
    print(f"Test Accuracy: {test_acc:.2f}%, Test Energy: {test_energy:.2f}mJ")

# Visualize training progress
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

ax1.plot(train_losses)
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.plot(test_accuracies)
ax2.set_title('Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')

ax3.plot(train_energies, label='Train')
ax3.plot(test_energies, label='Test')
ax3.set_title('Energy Consumption')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Energy (mJ)')
ax3.legend()

# Energy vs Accuracy trade-off
ax4.scatter(test_energies, test_accuracies)
ax4.set_xlabel('Energy (mJ)')
ax4.set_ylabel('Accuracy (%)')
ax4.set_title('Energy vs Accuracy Trade-off')

plt.tight_layout()
plt.show()

print(f"\nFinal Results:")
print(f"Best Test Accuracy: {max(test_accuracies):.2f}%")
print(f"Average Energy Consumption: {np.mean(test_energies):.2f}mJ")
```

### 2.2 CIFAR-10 with Convolutional LoRAven

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from loraven import LoRAven

class LoRAvenCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # LoRAven classifier with adaptive complexity
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            LoRAven(128 * 4 * 4, 512, mode='high_performance'),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            LoRAven(512, 256, mode='balanced'),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            LoRAven(256, num_classes, mode='low_power')
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Data preparation with augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10('data', train=False, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LoRAvenCNN().to(device)

# Use different learning rates for different components
conv_params = []
loraven_params = []

for name, param in model.named_parameters():
    if 'classifier' in name and any(isinstance(m, LoRAven) for m in model.classifier.modules()):
        loraven_params.append(param)
    else:
        conv_params.append(param)

optimizer = optim.Adam([
    {'params': conv_params, 'lr': 0.001},
    {'params': loraven_params, 'lr': 0.0005}  # Lower LR for LoRAven layers
])

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Training with detailed monitoring
def train_with_monitoring(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    layer_stats = {}
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Collect detailed layer statistics
        if batch_idx % 200 == 0:
            for name, module in model.named_modules():
                if isinstance(module, LoRAven):
                    stats = module.get_performance_stats()
                    if name not in layer_stats:
                        layer_stats[name] = []
                    layer_stats[name].append({
                        'rank': stats['current_rank'],
                        'energy': stats['energy_consumption'],
                        'compression': module.get_compression_ratio()
                    })
            
            print(f'Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}')
    
    return running_loss / len(train_loader), layer_stats

# Run training
num_epochs = 50
for epoch in range(num_epochs):
    train_loss, layer_stats = train_with_monitoring(
        model, train_loader, optimizer, criterion, device, epoch
    )
    
    # Evaluate
    test_acc, test_energy = evaluate(model, test_loader, device)
    
    # Adjust learning rate
    scheduler.step()
    
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
          f"Test Acc={test_acc:.2f}%, Test Energy={test_energy:.2f}mJ")
    
    # Print layer statistics every 10 epochs
    if epoch % 10 == 0 and layer_stats:
        print("\nLayer Statistics:")
        for layer_name, stats_list in layer_stats.items():
            if stats_list:
                latest_stats = stats_list[-1]
                print(f"  {layer_name}: Rank={latest_stats['rank']}, "
                      f"Energy={latest_stats['energy']:.2f}mJ, "
                      f"Compression={latest_stats['compression']:.1%}")
```

### 2.3 Transfer Learning with LoRAven

```python
import torch
import torch.nn as nn
from torchvision import models
from loraven import LoRAven

class LoRAvenResNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Add LoRAven-based classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            LoRAven(512, 256, mode='high_performance'),
            nn.ReLU(),
            nn.Dropout(0.3),
            LoRAven(256, 128, mode='balanced'),
            nn.ReLU(),
            LoRAven(128, num_classes, mode='low_power')
        )
        
        # Freeze backbone parameters for transfer learning
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def unfreeze_backbone(self, layers_to_unfreeze=2):
        """Unfreeze the last few layers of the backbone"""
        backbone_layers = list(self.backbone.children())
        for layer in backbone_layers[-layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

# Fine-tuning strategy
def fine_tune_with_loraven(model, train_loader, val_loader, device):
    # Phase 1: Train only LoRAven classifier
    print("Phase 1: Training LoRAven classifier only...")
    
    # Only optimize LoRAven parameters
    loraven_params = []
    for module in model.classifier.modules():
        if isinstance(module, LoRAven):
            loraven_params.extend(module.parameters())
    
    optimizer_phase1 = optim.Adam(loraven_params, lr=0.001)
    
    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer_phase1, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, Val Acc={val_acc:.2f}%")
    
    # Phase 2: Fine-tune with unfrozen backbone layers
    print("\nPhase 2: Fine-tuning with unfrozen backbone...")
    model.unfreeze_backbone(layers_to_unfreeze=2)
    
    # Lower learning rate for fine-tuning
    optimizer_phase2 = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 0.0001},
        {'params': model.classifier.parameters(), 'lr': 0.0005}
    ])
    
    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer_phase2, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, Val Acc={val_acc:.2f}%")

# Usage
model = LoRAvenResNet(num_classes=10)
fine_tune_with_loraven(model, train_loader, val_loader, device)
```

## 3. Natural Language Processing

### 3.1 Sentiment Analysis with LoRAven

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from loraven import LoRAven

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class LoRAvenSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(512, embed_dim))
        
        # LoRAven-based encoder layers
        self.encoder_layers = nn.ModuleList([
            LoRAvenEncoderLayer(embed_dim, hidden_dim) for _ in range(4)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            LoRAven(embed_dim, hidden_dim, mode='balanced'),
            nn.ReLU(),
            nn.Dropout(0.2),
            LoRAven(hidden_dim, num_classes, mode='low_power')
        )
    
    def forward(self, input_ids, attention_mask):
        # Embedding with positional encoding
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        
        # Global average pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        return self.classifier(pooled)

class LoRAvenEncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        
        # Self-attention with LoRAven
        self.self_attn_q = LoRAven(embed_dim, embed_dim, mode='high_performance')
        self.self_attn_k = LoRAven(embed_dim, embed_dim, mode='high_performance')
        self.self_attn_v = LoRAven(embed_dim, embed_dim, mode='high_performance')
        self.self_attn_out = LoRAven(embed_dim, embed_dim, mode='balanced')
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            LoRAven(embed_dim, hidden_dim, mode='balanced'),
            nn.ReLU(),
            nn.Dropout(0.1),
            LoRAven(hidden_dim, embed_dim, mode='balanced')
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, attention_mask):
        # Self-attention
        batch_size, seq_len, embed_dim = x.size()
        
        q = self.self_attn_q(x)
        k = self.self_attn_k(x)
        v = self.self_attn_v(x)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (embed_dim ** 0.5)
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = self.self_attn_out(attn_output)
        
        # Residual connection and normalization
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

# Training function for NLP
def train_sentiment_model():
    # Sample data (replace with real dataset)
    texts = ["This movie is great!", "I hate this film.", "Amazing performance!"]
    labels = [1, 0, 1]  # 1: positive, 0: negative
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Dataset and DataLoader
    dataset = SentimentDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Model
    model = LoRAvenSentimentClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_classes=2
    )
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(10):
        total_loss = 0
        total_energy = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Monitor energy consumption
            batch_energy = 0
            for module in model.modules():
                if isinstance(module, LoRAven):
                    stats = module.get_performance_stats()
                    batch_energy += stats['energy_consumption']
            total_energy += batch_energy
        
        print(f"Epoch {epoch}: Loss={total_loss/len(dataloader):.4f}, "
              f"Energy={total_energy/len(dataloader):.2f}mJ")

# Run training
train_sentiment_model()
```

### 3.2 Text Generation with LoRAven Transformer

```python
import torch
import torch.nn as nn
import math
from loraven import LoRAven

class LoRAvenTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, max_seq_len=1024):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # LoRAven-based transformer layers
        self.transformer_layers = nn.ModuleList([
            LoRAvenTransformerLayer(d_model, nhead) for _ in range(num_layers)
        ])
        
        self.output_projection = LoRAven(d_model, vocab_size, mode='balanced')
        
    def forward(self, x, mask=None):
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # Output projection
        return self.output_projection(x)
    
    def generate(self, start_tokens, max_length=100, temperature=1.0):
        """Generate text using the trained model"""
        self.eval()
        generated = start_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Create causal mask
                seq_len = generated.size(1)
                mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                
                # Forward pass
                logits = self.forward(generated, mask)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if end token is generated (assuming token 0 is end token)
                if next_token.item() == 0:
                    break
        
        return generated

class LoRAvenTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        
        # Multi-head attention with LoRAven
        self.self_attn = LoRAvenMultiHeadAttention(d_model, nhead)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            LoRAven(d_model, d_model * 4, mode='high_performance'),
            nn.GELU(),
            nn.Dropout(0.1),
            LoRAven(d_model * 4, d_model, mode='balanced')
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class LoRAvenMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Use LoRAven for attention projections
        self.w_q = LoRAven(d_model, d_model, mode='high_performance')
        self.w_k = LoRAven(d_model, d_model, mode='high_performance')
        self.w_v = LoRAven(d_model, d_model, mode='high_performance')
        self.w_o = LoRAven(d_model, d_model, mode='balanced')
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(attn_output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Training function for language model
def train_language_model():
    # Model setup
    vocab_size = 10000
    model = LoRAvenTransformerLM(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6
    )
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    # Sample training data (replace with real text data)
    batch_size = 32
    seq_len = 128
    
    for epoch in range(100):
        # Generate random training data (replace with real data)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}")
            
            # Monitor LoRAven layers
            total_energy = 0
            for module in model.modules():
                if isinstance(module, LoRAven):
                    stats = module.get_performance_stats()
                    total_energy += stats['energy_consumption']
            
            print(f"Total Energy: {total_energy:.2f}mJ")

# Run training
train_language_model()
```

## 4. Reinforcement Learning

### 4.1 Deep Q-Network with LoRAven

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from loraven import LoRAven

class LoRAvenDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        # Dueling DQN architecture with LoRAven
        self.feature_extractor = nn.Sequential(
            LoRAven(state_dim, hidden_dim, mode='high_performance'),
            nn.ReLU(),
            nn.Dropout(0.2),
            LoRAven(hidden_dim, hidden_dim, mode='balanced'),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            LoRAven(hidden_dim, hidden_dim // 2, mode='balanced'),
            nn.ReLU(),
            LoRAven(hidden_dim // 2, 1, mode='low_power')
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            LoRAven(hidden_dim, hidden_dim // 2, mode='balanced'),
            nn.ReLU(),
            LoRAven(hidden_dim // 2, action_dim, mode='low_power')
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class LoRAvenDQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Neural networks
        self.q_network = LoRAvenDQN(state_dim, action_dim)
        self.target_network = LoRAvenDQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(10000)
        
        # Update target network
        self.update_target_network()
        
        # Energy tracking
        self.energy_history = []
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def replay(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Track energy consumption
        total_energy = 0
        for module in self.q_network.modules():
            if isinstance(module, LoRAven):
                stats = module.get_performance_stats()
                total_energy += stats['energy_consumption']
        self.energy_history.append(total_energy)
        
        return loss.item()

# Training environment simulation
class SimpleEnvironment:
    def __init__(self, state_dim=10, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset()
    
    def reset(self):
        self.state = np.random.randn(self.state_dim)
        self.steps = 0
        return self.state
    
    def step(self, action):
        # Simple environment dynamics
        self.state += np.random.randn(self.state_dim) * 0.1
        reward = -np.sum(self.state ** 2) + action * 0.1  # Reward based on state and action
        self.steps += 1
        done = self.steps >= 100 or np.sum(self.state ** 2) > 10
        return self.state, reward, done, {}

# Training function
def train_dqn_agent():
    env = SimpleEnvironment(state_dim=20, action_dim=4)
    agent = LoRAvenDQNAgent(state_dim=20, action_dim=4)
    
    episodes = 1000
    scores = []
    energy_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_energy = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Train the agent
        if len(agent.replay_buffer) > 32:
            loss = agent.replay(batch_size=32)
        
        # Update target network periodically
        if episode % 100 == 0:
            agent.update_target_network()
        
        scores.append(total_reward)
        
        # Calculate episode energy consumption
        if agent.energy_history:
            episode_energy = sum(agent.energy_history[-env.steps:]) if len(agent.energy_history) >= env.steps else sum(agent.energy_history)
            energy_per_episode.append(episode_energy)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_energy = np.mean(energy_per_episode[-100:]) if energy_per_episode else 0
            print(f"Episode {episode}: Avg Score={avg_score:.2f}, "
                  f"Epsilon={agent.epsilon:.3f}, Avg Energy={avg_energy:.2f}mJ")
    
    return agent, scores, energy_per_episode

# Run training
agent, scores, energy_per_episode = train_dqn_agent()

# Visualize results
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Plot scores
ax1.plot(scores)
ax1.set_title('Training Scores')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Score')

# Plot energy consumption
if energy_per_episode:
    ax2.plot(energy_per_episode)
    ax2.set_title('Energy Consumption per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Energy (mJ)')

# Plot energy vs performance trade-off
if energy_per_episode and len(energy_per_episode) == len(scores):
    ax3.scatter(energy_per_episode, scores, alpha=0.6)
    ax3.set_xlabel('Energy (mJ)')
    ax3.set_ylabel('Score')
    ax3.set_title('Energy vs Performance Trade-off')

plt.tight_layout()
plt.show()
```

### 4.2 Policy Gradient with LoRAven

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from loraven import LoRAven

class LoRAvenPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Policy network with adaptive complexity
        self.policy_net = nn.Sequential(
            LoRAven(state_dim, hidden_dim, mode='high_performance'),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            LoRAven(hidden_dim, hidden_dim, mode='balanced'),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            LoRAven(hidden_dim, action_dim, mode='low_power')
        )
        
        # Value network for baseline
        self.value_net = nn.Sequential(
            LoRAven(state_dim, hidden_dim, mode='balanced'),
            nn.ReLU(),
            LoRAven(hidden_dim, hidden_dim // 2, mode='balanced'),
            nn.ReLU(),
            LoRAven(hidden_dim // 2, 1, mode='low_power')
        )
    
    def forward(self, state):
        policy_logits = self.policy_net(state)
        value = self.value_net(state)
        return policy_logits, value
    
    def get_action(self, state):
        policy_logits, value = self.forward(state)
        policy_dist = Categorical(logits=policy_logits)
        action = policy_dist.sample()
        return action.item(), policy_dist.log_prob(action), value

class LoRAvenREINFORCE:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.policy_net = LoRAvenPolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Storage for episode data
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.energy_consumption = []
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, value = self.policy_net.get_action(state_tensor)
        
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        # Track energy consumption
        episode_energy = 0
        for module in self.policy_net.modules():
            if isinstance(module, LoRAven):
                stats = module.get_performance_stats()
                episode_energy += stats['energy_consumption']
        self.energy_consumption.append(episode_energy)
        
        return action
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def update_policy(self):
        # Calculate discounted returns
        returns = []
        discounted_sum = 0
        for reward in reversed(self.rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate advantages
        values = torch.cat(self.values)
        advantages = returns - values.squeeze()
        
        # Policy loss (REINFORCE with baseline)
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * advantage.detach())
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Update parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Clear episode data
        self.log_probs = []
        self.values = []
        self.rewards = []
        
        return total_loss.item(), policy_loss.item(), value_loss.item()
    
    def get_energy_stats(self):
        if self.energy_consumption:
            return {
                'total_energy': sum(self.energy_consumption),
                'avg_energy': np.mean(self.energy_consumption),
                'max_energy': max(self.energy_consumption),
                'min_energy': min(self.energy_consumption)
            }
        return {}

# Training function for REINFORCE
def train_reinforce_agent():
    env = SimpleEnvironment(state_dim=15, action_dim=3)
    agent = LoRAvenREINFORCE(state_dim=15, action_dim=3)
    
    episodes = 2000
    scores = []
    energy_stats = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        # Run episode
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_reward(reward)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update policy
        total_loss, policy_loss, value_loss = agent.update_policy()
        scores.append(total_reward)
        
        # Collect energy statistics
        energy_stat = agent.get_energy_stats()
        energy_stats.append(energy_stat)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_energy = np.mean([stat.get('avg_energy', 0) for stat in energy_stats[-100:]])
            print(f"Episode {episode}: Avg Score={avg_score:.2f}, "
                  f"Loss={total_loss:.4f}, Avg Energy={avg_energy:.2f}mJ")
    
    return agent, scores, energy_stats

# Run training
agent, scores, energy_stats = train_reinforce_agent()

# Analyze results
print("\nTraining completed!")
print(f"Final average score: {np.mean(scores[-100:]):.2f}")
print(f"Final average energy: {np.mean([stat.get('avg_energy', 0) for stat in energy_stats[-100:]]):.2f}mJ")
```

## 5. Time Series Analysis

### 5.1 Time Series Forecasting with LoRAven LSTM

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from loraven import LoRAven

class LoRAvenLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, sequence_length):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # LoRAven-based output layers
        self.output_layers = nn.Sequential(
            LoRAven(hidden_dim, hidden_dim, mode='high_performance'),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            LoRAven(hidden_dim, hidden_dim // 2, mode='balanced'),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            LoRAven(hidden_dim // 2, output_dim, mode='low_power')
        )
        
        # Attention mechanism with LoRAven
        self.attention = LoRAvenAttention(hidden_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply attention
        attended_output = self.attention(lstm_out)
        
        # Final prediction
        output = self.output_layers(attended_output)
        
        return output

class LoRAvenAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.attention_weights = LoRAven(hidden_dim, 1, mode='balanced')
        
    def forward(self, lstm_output):
        # Calculate attention weights
        attention_scores = self.attention_weights(lstm_output)  # (batch, seq, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)  # (batch, hidden)
        
        return attended_output

class TimeSeriesDataset:
    def __init__(self, data, sequence_length, prediction_length=1):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
        # Normalize data
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        self.sequences, self.targets = self._create_sequences()
    
    def _create_sequences(self):
        sequences = []
        targets = []
        
        for i in range(len(self.normalized_data) - self.sequence_length - self.prediction_length + 1):
            seq = self.normalized_data[i:i + self.sequence_length]
            target = self.normalized_data[i + self.sequence_length:i + self.sequence_length + self.prediction_length]
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def get_data_loaders(self, train_ratio=0.8, batch_size=32):
        # Split data
        split_idx = int(len(self.sequences) * train_ratio)
        
        train_sequences = torch.FloatTensor(self.sequences[:split_idx]).unsqueeze(-1)
        train_targets = torch.FloatTensor(self.targets[:split_idx])
        
        test_sequences = torch.FloatTensor(self.sequences[split_idx:]).unsqueeze(-1)
        test_targets = torch.FloatTensor(self.targets[split_idx:])
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_sequences, train_targets)
        test_dataset = torch.utils.data.TensorDataset(test_sequences, test_targets)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()

# Generate synthetic time series data
def generate_synthetic_timeseries(length=1000):
    """Generate synthetic time series with trend, seasonality, and noise"""
    t = np.arange(length)
    
    # Trend component
    trend = 0.02 * t
    
    # Seasonal components
    seasonal1 = 10 * np.sin(2 * np.pi * t / 50)  # Period of 50
    seasonal2 = 5 * np.sin(2 * np.pi * t / 20)   # Period of 20
    
    # Noise
    noise = np.random.normal(0, 2, length)
    
    # Combine components
    timeseries = trend + seasonal1 + seasonal2 + noise
    
    return timeseries

# Training function
def train_timeseries_model():
    # Generate data
    data = generate_synthetic_timeseries(2000)
    dataset = TimeSeriesDataset(data, sequence_length=50, prediction_length=1)
    train_loader, test_loader = dataset.get_data_loaders(batch_size=64)
    
    # Model setup
    model = LoRAvenLSTM(
        input_dim=1,
        hidden_dim=128,
        num_layers=2,
        output_dim=1,
        sequence_length=50
    )
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    num_epochs = 100
    train_losses = []
    test_losses = []
    energy_consumption = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        epoch_energy = 0
        
        for batch_sequences, batch_targets in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_sequences)
            loss = criterion(outputs.squeeze(), batch_targets.squeeze())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Track energy consumption
            batch_energy = 0
            for module in model.modules():
                if isinstance(module, LoRAven):
                    stats = module.get_performance_stats()
                    batch_energy += stats['energy_consumption']
            epoch_energy += batch_energy
        
        train_loss /= len(train_loader)
        epoch_energy /= len(train_loader)
        
        # Validation
        model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for batch_sequences, batch_targets in test_loader:
                outputs = model(batch_sequences)
                loss = criterion(outputs.squeeze(), batch_targets.squeeze())
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        
        # Store metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        energy_consumption.append(epoch_energy)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, "
                  f"Test Loss={test_loss:.6f}, Energy={epoch_energy:.2f}mJ")
    
    return model, dataset, train_losses, test_losses, energy_consumption

# Forecasting function
def forecast_future(model, dataset, num_predictions=50):
    """Generate future predictions"""
    model.eval()
    
    # Use the last sequence from the dataset
    last_sequence = torch.FloatTensor(dataset.sequences[-1:]).unsqueeze(-1)
    predictions = []
    
    with torch.no_grad():
        current_sequence = last_sequence.clone()
        
        for _ in range(num_predictions):
            # Predict next value
            pred = model(current_sequence)
            predictions.append(pred.item())
            
            # Update sequence (sliding window)
            new_sequence = torch.cat([
                current_sequence[:, 1:, :],
                pred.unsqueeze(0).unsqueeze(-1)
            ], dim=1)
            current_sequence = new_sequence
    
    # Inverse transform predictions
    predictions = np.array(predictions)
    predictions = dataset.inverse_transform(predictions)
    
    return predictions

# Run training and forecasting
model, dataset, train_losses, test_losses, energy_consumption = train_timeseries_model()

# Generate forecasts
future_predictions = forecast_future(model, dataset, num_predictions=100)

# Visualize results
import matplotlib.pyplot as plt

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Training progress
ax1.plot(train_losses, label='Train Loss')
ax1.plot(test_losses, label='Test Loss')
ax1.set_title('Training Progress')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_yscale('log')

# Energy consumption
ax2.plot(energy_consumption)
ax2.set_title('Energy Consumption During Training')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Energy (mJ)')

# Original data and predictions
original_data = dataset.inverse_transform(dataset.normalized_data)
ax3.plot(original_data[-200:], label='Original Data', alpha=0.7)
ax3.plot(range(200, 200 + len(future_predictions)), future_predictions, 
         label='Predictions', color='red', linewidth=2)
ax3.set_title('Time Series Forecasting')
ax3.set_xlabel('Time')
ax3.set_ylabel('Value')
ax3.legend()

# Energy vs accuracy trade-off
final_test_loss = test_losses[-1]
final_energy = energy_consumption[-1]
ax4.scatter(energy_consumption, test_losses, alpha=0.6)
ax4.set_xlabel('Energy (mJ)')
ax4.set_ylabel('Test Loss')
ax4.set_title('Energy vs Accuracy Trade-off')
ax4.set_yscale('log')

plt.tight_layout()
plt.show()

print(f"Final test loss: {final_test_loss:.6f}")
print(f"Final energy consumption: {final_energy:.2f}mJ")
```

### 5.2 Anomaly Detection with LoRAven Autoencoder

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from loraven import LoRAven

class LoRAvenAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=64, hidden_dims=[128, 64]):
        super().__init__()
        
        # Encoder with progressive compression
        encoder_layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                LoRAven(current_dim, hidden_dim, mode='high_performance'),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        # Bottleneck layer
        encoder_layers.append(LoRAven(current_dim, encoding_dim, mode='balanced'))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder with progressive expansion
        decoder_layers = []
        current_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                LoRAven(current_dim, hidden_dim, mode='balanced'),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        # Output layer
        decoder_layers.append(LoRAven(current_dim, input_dim, mode='low_power'))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def get_reconstruction_error(self, x):
        """Calculate reconstruction error for anomaly detection"""
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
            return error

class AnomalyDetector:
    def __init__(self, input_dim, encoding_dim=32, threshold_percentile=95):
        self.model = LoRAvenAutoencoder(input_dim, encoding_dim)
        self.scaler = StandardScaler()
        self.threshold = None
        self.threshold_percentile = threshold_percentile
        
        # Energy tracking
        self.training_energy = []
        self.inference_energy = []
    
    def fit(self, X_train, epochs=100, batch_size=32, lr=0.001):
        """Train the autoencoder on normal data"""
        # Normalize data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Convert to tensor
        train_data = torch.FloatTensor(X_train_scaled)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_energy = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                reconstructed, encoded = self.model(batch)
                loss = criterion(reconstructed, batch)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Track energy consumption
                batch_energy = 0
                for module in self.model.modules():
                    if isinstance(module, LoRAven):
                        stats = module.get_performance_stats()
                        batch_energy += stats['energy_consumption']
                epoch_energy += batch_energy
            
            epoch_loss /= len(train_loader)
            epoch_energy /= len(train_loader)
            self.training_energy.append(epoch_energy)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss={epoch_loss:.6f}, Energy={epoch_energy:.2f}mJ")
        
        # Calculate threshold based on training data
        self.model.eval()
        train_errors = self.model.get_reconstruction_error(train_data)
        self.threshold = torch.percentile(train_errors, self.threshold_percentile).item()
        
        print(f"Anomaly threshold set to: {self.threshold:.6f}")
    
    def predict(self, X_test):
        """Predict anomalies in test data"""
        # Normalize test data
        X_test_scaled = self.scaler.transform(X_test)
        test_data = torch.FloatTensor(X_test_scaled)
        
        # Calculate reconstruction errors
        self.model.eval()
        errors = self.model.get_reconstruction_error(test_data)
        
        # Track inference energy
        inference_energy = 0
        for module in self.model.modules():
            if isinstance(module, LoRAven):
                stats = module.get_performance_stats()
                inference_energy += stats['energy_consumption']
        self.inference_energy.append(inference_energy)
        
        # Classify as anomaly if error > threshold
        anomalies = (errors > self.threshold).numpy()
        
        return anomalies, errors.numpy()
    
    def get_energy_stats(self):
        """Get energy consumption statistics"""
        return {
            'training_energy': {
                'total': sum(self.training_energy),
                'average': np.mean(self.training_energy),
                'max': max(self.training_energy) if self.training_energy else 0
            },
            'inference_energy': {
                'total': sum(self.inference_energy),
                'average': np.mean(self.inference_energy) if self.inference_energy else 0
            }
        }

# Generate synthetic data with anomalies
def generate_anomaly_data(n_normal=1000, n_anomalies=50, n_features=20):
    """Generate synthetic dataset with normal and anomalous samples"""
    np.random.seed(42)
    
    # Normal data (multivariate Gaussian)
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_normal
    )
    
    # Anomalous data (shifted mean and different covariance)
    anomaly_data = np.random.multivariate_normal(
        mean=np.ones(n_features) * 3,  # Shifted mean
        cov=np.eye(n_features) * 4,    # Different variance
        size=n_anomalies
    )
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])  # 0: normal, 1: anomaly
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    return X, y

# Training and evaluation
def evaluate_anomaly_detection():
    # Generate data
    X, y = generate_anomaly_data(n_normal=2000, n_anomalies=100, n_features=30)
    
    # Split data (use only normal data for training)
    normal_indices = np.where(y == 0)[0]
    anomaly_indices = np.where(y == 1)[0]
    
    # Training set (only normal data)
    train_size = int(0.7 * len(normal_indices))
    train_indices = normal_indices[:train_size]
    X_train = X[train_indices]
    
    # Test set (mix of normal and anomalous data)
    test_normal_indices = normal_indices[train_size:]
    test_indices = np.concatenate([test_normal_indices, anomaly_indices])
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    print(f"Training samples: {len(X_train)} (all normal)")
    print(f"Test samples: {len(X_test)} ({np.sum(y_test == 0)} normal, {np.sum(y_test == 1)} anomalous)")
    
    # Train anomaly detector
    detector = AnomalyDetector(input_dim=X.shape[1], encoding_dim=16)
    detector.fit(X_train, epochs=150, batch_size=64)
    
    # Predict anomalies
    predictions, errors = detector.predict(X_test)
    
    # Evaluate performance
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    # ROC AUC score
    auc_score = roc_auc_score(y_test, errors)
    print(f"\nROC AUC Score: {auc_score:.4f}")
    
    # Energy statistics
    energy_stats = detector.get_energy_stats()
    print(f"\nEnergy Statistics:")
    print(f"Training - Total: {energy_stats['training_energy']['total']:.2f}mJ, "
          f"Average: {energy_stats['training_energy']['average']:.2f}mJ")
    print(f"Inference - Total: {energy_stats['inference_energy']['total']:.2f}mJ")
    
    return detector, X_test, y_test, predictions, errors

# Run evaluation
detector, X_test, y_test, predictions, errors = evaluate_anomaly_detection()

# Visualize results
import matplotlib.pyplot as plt

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Training energy consumption
ax1.plot(detector.training_energy)
ax1.set_title('Training Energy Consumption')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Energy (mJ)')

# Reconstruction error distribution
normal_errors = errors[y_test == 0]
anomaly_errors = errors[y_test == 1]

ax2.hist(normal_errors, bins=30, alpha=0.7, label='Normal', density=True)
ax2.hist(anomaly_errors, bins=30, alpha=0.7, label='Anomaly', density=True)
ax2.axvline(detector.threshold, color='red', linestyle='--', label='Threshold')
ax2.set_title('Reconstruction Error Distribution')
ax2.set_xlabel('Reconstruction Error')
ax2.set_ylabel('Density')
ax2.legend()

# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, errors)
ax3.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, errors):.3f})')
ax3.plot([0, 1], [0, 1], 'k--', label='Random')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curve')
ax3.legend()

# Anomaly scores over time
ax4.scatter(range(len(errors)), errors, c=y_test, cmap='coolwarm', alpha=0.6)
ax4.axhline(detector.threshold, color='red', linestyle='--', label='Threshold')
ax4.set_title('Anomaly Scores')
ax4.set_xlabel('Sample Index')
ax4.set_ylabel('Reconstruction Error')
ax4.legend()

plt.tight_layout()
plt.show()
```

## 6. Energy-Aware Training

### 6.1 Dynamic Energy Budget Management

```python
import torch
import torch.nn as nn
from loraven import LoRAven, BudgetManager

class EnergyAwareTrainer:
    def __init__(self, model, initial_budget=1000.0, budget_decay=0.95):
        self.model = model
        self.budget_manager = BudgetManager(total_budget=initial_budget)
        self.budget_decay = budget_decay
        
        # Energy tracking
        self.energy_history = []
        self.performance_history = []
        self.budget_history = []
        
        # Adaptive parameters
        self.energy_weight = 0.1
        self.performance_weight = 0.9
    
    def train_epoch(self, train_loader, optimizer, criterion, device):
        """Train one epoch with energy awareness"""
        self.model.train()
        total_loss = 0
        total_energy = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Get current budget allocation
            current_budget = self.budget_manager.get_remaining_budget()
            
            # Adjust model complexity based on budget
            self._adjust_model_complexity(current_budget)
            
            # Forward pass
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            
            # Calculate energy consumption
            batch_energy = self._calculate_batch_energy()
            
            # Energy-aware loss
            energy_penalty = self.energy_weight * batch_energy / current_budget
            total_loss_with_penalty = loss + energy_penalty
            
            # Backward pass
            total_loss_with_penalty.backward()
            optimizer.step()
            
            # Update budget
            self.budget_manager.update_energy_consumption(batch_energy, loss.item())
            
            total_loss += loss.item()
            total_energy += batch_energy
            
            # Log progress
            if batch_idx % 100 == 0:
                remaining_budget = self.budget_manager.get_remaining_budget()
                print(f'Batch {batch_idx}: Loss={loss.item():.4f}, '
                      f'Energy={batch_energy:.2f}mJ, '
                      f'Budget Remaining={remaining_budget:.1f}mJ')
        
        # Update histories
        avg_loss = total_loss / len(train_loader)
        avg_energy = total_energy / len(train_loader)
        
        self.performance_history.append(avg_loss)
        self.energy_history.append(avg_energy)
        self.budget_history.append(self.budget_manager.get_remaining_budget())
        
        # Decay budget for next epoch
        self.budget_manager.total_budget *= self.budget_decay
        
        return avg_loss, avg_energy
    
    def _adjust_model_complexity(self, current_budget):
        """Adjust model complexity based on remaining budget"""
        budget_ratio = current_budget / self.budget_manager.total_budget
        
        for module in self.model.modules():
            if isinstance(module, LoRAven):
                if budget_ratio > 0.7:
                    # High budget: use high performance mode
                    module.set_mode('high_performance')
                elif budget_ratio > 0.3:
                    # Medium budget: use balanced mode
                    module.set_mode('balanced')
                else:
                    # Low budget: use low power mode
                    module.set_mode('low_power')
    
    def _calculate_batch_energy(self):
        """Calculate total energy consumption for current batch"""
        total_energy = 0
        for module in self.model.modules():
            if isinstance(module, LoRAven):
                stats = module.get_performance_stats()
                total_energy += stats['energy_consumption']
        return total_energy
    
    def get_energy_efficiency_metrics(self):
        """Calculate energy efficiency metrics"""
        if not self.energy_history or not self.performance_history:
            return {}
        
        # Energy efficiency = Performance improvement / Energy consumed
        initial_loss = self.performance_history[0]
        final_loss = self.performance_history[-1]
        performance_improvement = initial_loss - final_loss
        
        total_energy = sum(self.energy_history)
        
        efficiency = performance_improvement / total_energy if total_energy > 0 else 0
        
        return {
            'energy_efficiency': efficiency,
            'total_energy': total_energy,
            'performance_improvement': performance_improvement,
            'average_energy_per_epoch': np.mean(self.energy_history),
            'energy_variance': np.var(self.energy_history)
        }

# Example usage with MNIST
class EnergyAwareMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.network = nn.Sequential(
            LoRAven(784, 512, mode='balanced'),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            LoRAven(512, 256, mode='balanced'),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            LoRAven(256, 128, mode='balanced'),
            nn.ReLU(),
            
            LoRAven(128, 10, mode='balanced')
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

def train_energy_aware_model():
    # Data setup (using previous MNIST setup)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnergyAwareMNIST().to(device)
    
    # Energy-aware trainer
    trainer = EnergyAwareTrainer(model, initial_budget=5000.0, budget_decay=0.98)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 20
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train with energy awareness
        train_loss, train_energy = trainer.train_epoch(
            train_loader, optimizer, criterion, device
        )
        
        print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, "
              f"Energy={train_energy:.2f}mJ, "
              f"Budget Remaining={trainer.budget_manager.get_remaining_budget():.1f}mJ")
    
    # Get efficiency metrics
    efficiency_metrics = trainer.get_energy_efficiency_metrics()
    print(f"\nEnergy Efficiency Metrics:")
    for key, value in efficiency_metrics.items():
        print(f"{key}: {value:.6f}")
    
    return trainer, model

# Run energy-aware training
trainer, model = train_energy_aware_model()

# Visualize energy-aware training
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Training loss
ax1.plot(trainer.performance_history)
ax1.set_title('Training Loss Over Time')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

# Energy consumption
ax2.plot(trainer.energy_history)
ax2.set_title('Energy Consumption Per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Energy (mJ)')

# Budget remaining
ax3.plot(trainer.budget_history)
ax3.set_title('Remaining Energy Budget')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Budget (mJ)')

# Energy vs Performance trade-off
ax4.scatter(trainer.energy_history, trainer.performance_history)
ax4.set_xlabel('Energy (mJ)')
ax4.set_ylabel('Loss')
ax4.set_title('Energy vs Performance Trade-off')

plt.tight_layout()
plt.show()
```
"""
LoRAven 训练器实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import os
from tqdm import tqdm
import yaml

# 修复导入路径
try:
    from ..models.dynamic_lowrank_layer import DynamicLowRankLayer
    from ..schedulers.rank_scheduler import create_rank_scheduler
    from ..schedulers.budget_manager import BudgetManager
    from ..utils.perf_estimator import PerfEstimator
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.dynamic_lowrank_layer import DynamicLowRankLayer
    from schedulers.rank_scheduler import create_rank_scheduler
    from schedulers.budget_manager import BudgetManager
    from utils.perf_estimator import PerfEstimator


class LoRAvenTrainer:
    """
    LoRAven 训练器
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        save_dir: str = "./checkpoints"
    ):
        self.model = model
        self.config = config
        self.device = device
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化组件
        self.rank_scheduler = create_rank_scheduler(
            config.get('scheduler_type', 'energy_aware'),
            **config.get('scheduler_params', {})
        )
        
        self.budget_manager = BudgetManager(
            total_budget=config.get('energy_budget', 10.0),
            **config.get('budget_params', {})
        )
        
        self.perf_estimator = PerfEstimator(
            config.get('hardware_profile', {})
        )
        
        # 优化器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_lr_scheduler()
        
        # 损失函数
        self.criterion = self._create_criterion()
        
        # 训练状态
        self.current_epoch = 0
        self.best_performance = 0.0
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'energy': [],
            'latency': [],
            'rank_distribution': []
        }
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-4),
                weight_decay=optimizer_config.get('weight_decay', 1e-5),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-3),
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-5)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_lr_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_config = self.config.get('lr_scheduler', {})
        if not scheduler_config:
            return None
        
        scheduler_type = scheduler_config.get('type', 'step')
        
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 100)
            )
        else:
            return None
    
    def _create_criterion(self) -> nn.Module:
        """创建损失函数"""
        criterion_config = self.config.get('criterion', {})
        criterion_type = criterion_config.get('type', 'cross_entropy')
        
        if criterion_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif criterion_type == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown criterion type: {criterion_type}")
    
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前 epoch
            
        Returns:
            metrics: 训练指标
        """
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_energy = 0.0
        total_latency = 0.0
        rank_distribution = []
        
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                
                # 获取复杂度分数
                complexity_scores = self._get_complexity_scores(data)
                
                # 分配预算
                budget = self.budget_manager.allocate_budget(
                    f"layer_{batch_idx}", 
                    complexity_scores.mean().item(),
                    (data.size(-1), self.model.out_features)
                )
                
                # 模型前向传播
                output, current_rank = self.model(data, budget=budget, mode='training')
                
                # 计算损失
                loss = self.criterion(output, target)
                
                # 添加能耗正则化
                energy_penalty = self._calculate_energy_penalty(current_rank)
                total_loss_batch = loss + energy_penalty
                
                # 反向传播
                total_loss_batch.backward()
                self.optimizer.step()
                
                # 更新预算管理器
                actual_energy = self.perf_estimator.energy_estimator.estimate(
                    (data.size(-1), self.model.out_features), 
                    current_rank, 
                    data.size(0)
                )
                self.budget_manager.update_energy_consumption(
                    f"layer_{batch_idx}", 
                    actual_energy,
                    self._calculate_accuracy(output, target)
                )
                
                # 记录指标
                total_loss += loss.item()
                total_accuracy += self._calculate_accuracy(output, target)
                total_energy += actual_energy
                total_latency += self.perf_estimator.latency_estimator.estimate(
                    (data.size(-1), self.model.out_features), 
                    current_rank, 
                    data.size(0)
                )
                rank_distribution.append(current_rank)
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{self._calculate_accuracy(output, target):.4f}",
                    'Rank': current_rank,
                    'Energy': f"{actual_energy:.4f}"
                })
        
        # 计算平均指标
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_energy = total_energy / num_batches
        avg_latency = total_latency / num_batches
        
        # 更新学习率
        if self.scheduler:
            self.scheduler.step()
        
        # 记录历史
        self.training_history['loss'].append(avg_loss)
        self.training_history['accuracy'].append(avg_accuracy)
        self.training_history['energy'].append(avg_energy)
        self.training_history['latency'].append(avg_latency)
        self.training_history['rank_distribution'].append(rank_distribution)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'energy': avg_energy,
            'latency': avg_latency,
            'avg_rank': np.mean(rank_distribution),
            'rank_std': np.std(rank_distribution)
        }
    
    def validate(
        self, 
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            metrics: 验证指标
        """
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_energy = 0.0
        rank_distribution = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 获取复杂度分数
                complexity_scores = self._get_complexity_scores(data)
                
                # 分配预算
                budget = self.budget_manager.allocate_budget(
                    "validation", 
                    complexity_scores.mean().item(),
                    (data.size(-1), self.model.out_features)
                )
                
                # 模型前向传播
                output, current_rank = self.model(data, budget=budget, mode='inference')
                
                # 计算损失
                loss = self.criterion(output, target)
                
                # 记录指标
                total_loss += loss.item()
                total_accuracy += self._calculate_accuracy(output, target)
                
                # 估算能耗
                actual_energy = self.perf_estimator.energy_estimator.estimate(
                    (data.size(-1), self.model.out_features), 
                    current_rank, 
                    data.size(0)
                )
                total_energy += actual_energy
                rank_distribution.append(current_rank)
        
        num_batches = len(val_loader)
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches,
            'val_energy': total_energy / num_batches,
            'val_avg_rank': np.mean(rank_distribution),
            'val_rank_std': np.std(rank_distribution)
        }
    
    def _get_complexity_scores(self, data: torch.Tensor) -> torch.Tensor:
        """获取复杂度分数"""
        # 这里应该使用模型的复杂度评分器
        # 简化实现：使用输入的标准差作为复杂度指标
        complexity_scores = torch.std(data, dim=1)
        return torch.sigmoid(complexity_scores)
    
    def _calculate_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """计算准确率"""
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return correct / target.size(0)
    
    def _calculate_energy_penalty(self, rank: int) -> torch.Tensor:
        """计算能耗惩罚"""
        energy_penalty_weight = self.config.get('energy_penalty_weight', 0.01)
        
        # 简化的能耗惩罚：基于秩的平方
        penalty = energy_penalty_weight * (rank ** 2)
        return torch.tensor(penalty, device=self.device, requires_grad=True)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_performance': self.best_performance,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_performance = checkpoint['best_performance']
        self.training_history = checkpoint['training_history']
    
    def train(
        self, 
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100
    ) -> Dict[str, List[float]]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            
        Returns:
            training_history: 训练历史
        """
        print(f"开始训练 LoRAven 模型，共 {num_epochs} 个 epoch")
        
        for epoch in range(self.current_epoch, num_epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            if val_loader:
                val_metrics = self.validate(val_loader)
                
                # 检查是否是最佳性能
                current_performance = val_metrics['val_accuracy']
                is_best = current_performance > self.best_performance
                if is_best:
                    self.best_performance = current_performance
                
                # 保存检查点
                self.save_checkpoint(epoch, is_best)
                
                print(f"Epoch {epoch}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                      f"Avg Rank: {train_metrics['avg_rank']:.1f}, "
                      f"Energy: {train_metrics['energy']:.4f}")
            else:
                self.save_checkpoint(epoch)
                print(f"Epoch {epoch}: "
                      f"Loss: {train_metrics['loss']:.4f}, "
                      f"Acc: {train_metrics['accuracy']:.4f}, "
                      f"Avg Rank: {train_metrics['avg_rank']:.1f}, "
                      f"Energy: {train_metrics['energy']:.4f}")
        
        return self.training_history


def train_loraven(config_path: str, data_dir: str, save_dir: str) -> Dict[str, Any]:
    """
    训练 LoRAven 模型的便捷函数
    
    Args:
        config_path: 配置文件路径
        data_dir: 数据目录
        save_dir: 保存目录
        
    Returns:
        results: 训练结果
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型（这里需要根据具体任务实现）
    model = create_model_from_config(config['model'])
    model = model.to(device)
    
    # 创建训练器
    trainer = LoRAvenTrainer(model, config, device, save_dir)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(data_dir, config['data'])
    
    # 训练
    training_history = trainer.train(train_loader, val_loader, config['train']['epochs'])
    
    return {
        'training_history': training_history,
        'best_performance': trainer.best_performance,
        'config': config
    }


def create_model_from_config(model_config: Dict[str, Any]) -> nn.Module:
    """根据配置创建模型"""
    # 这里需要根据具体任务实现
    # 例如：创建 ResNet 或 Transformer 模型，并替换某些层为 DynamicLowRankLayer
    pass


def create_data_loaders(data_dir: str, data_config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """创建数据加载器"""
    # 这里需要根据具体任务实现
    # 例如：ImageNet、CIFAR-10 等数据集的加载
    pass

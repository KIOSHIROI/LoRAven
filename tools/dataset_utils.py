"""
数据集工具模块
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from typing import Tuple, Optional, Dict, Any


class DatasetLoader:
    """
    数据集加载器
    """
    
    def __init__(self, data_root: str, dataset_name: str = 'imagenet'):
        self.data_root = data_root
        self.dataset_name = dataset_name
    
    def load_dataset(self, split: str = 'train') -> Dataset:
        """
        加载数据集
        
        Args:
            split: 数据集分割 ('train', 'val', 'test')
            
        Returns:
            dataset: 数据集实例
        """
        if self.dataset_name == 'imagenet':
            return self._load_imagenet(split)
        elif self.dataset_name == 'cifar10':
            return self._load_cifar10(split)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_imagenet(self, split: str) -> Dataset:
        """加载 ImageNet 数据集"""
        from torchvision.datasets import ImageNet
        
        transform = self._get_transform(split)
        
        return ImageNet(
            root=self.data_root,
            split=split,
            transform=transform
        )
    
    def _load_cifar10(self, split: str) -> Dataset:
        """加载 CIFAR-10 数据集"""
        from torchvision.datasets import CIFAR10
        
        transform = self._get_transform(split)
        train = (split == 'train')
        
        return CIFAR10(
            root=self.data_root,
            train=train,
            transform=transform,
            download=True
        )
    
    def _get_transform(self, split: str) -> transforms.Compose:
        """获取数据变换"""
        if split == 'train':
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])


class DataPreprocessor:
    """
    数据预处理器
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """
        预处理数据
        
        Args:
            data: 输入数据
            
        Returns:
            processed_data: 预处理后的数据
        """
        # 标准化
        if 'normalize' in self.config:
            mean = self.config['normalize'].get('mean', [0.485, 0.456, 0.406])
            std = self.config['normalize'].get('std', [0.229, 0.224, 0.225])
            data = self._normalize(data, mean, std)
        
        return data
    
    def _normalize(self, data: torch.Tensor, mean: list, std: list) -> torch.Tensor:
        """标准化数据"""
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
        return (data - mean) / std


def create_data_loaders(
    data_root: str,
    dataset_name: str = 'imagenet',
    batch_size: int = 64,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        data_root: 数据根目录
        dataset_name: 数据集名称
        batch_size: 批大小
        num_workers: 工作进程数
        **kwargs: 其他参数
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    loader = DatasetLoader(data_root, dataset_name)
    
    # 创建数据集
    train_dataset = loader.load_dataset('train')
    val_dataset = loader.load_dataset('val')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )
    
    return train_loader, val_loader

"""
GLUE基准测试：LoRA、AdaLoRA、DoRA、LoRAven对比实验
支持SST-2、MNLI、QNLI、RTE、CoLA任务的参数高效微调
"""

import os
# 设置HF镜像站解决连接问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 设置多GPU并行训练，使用cuda:0, cuda:1, cuda:3
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 设置tokenizers并行化，避免fork警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置Hugging Face缓存目录到项目目录，避免在/home目录创建内容
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hf_cache_dir = os.path.join(project_root, ".hf_cache")
os.environ["HF_HOME"] = hf_cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_dir, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_cache_dir, "hub")

# 确保缓存目录存在
os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(os.path.join(hf_cache_dir, "transformers"), exist_ok=True)
os.makedirs(os.path.join(hf_cache_dir, "datasets"), exist_ok=True)
os.makedirs(os.path.join(hf_cache_dir, "hub"), exist_ok=True)

import json
import yaml
import math
import torch
import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, fields
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
from datasets import load_dataset, DatasetDict
from peft import (
    LoraConfig, AdaLoraConfig, get_peft_model, 
    TaskType, PeftModel, PeftConfig
)
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import logging

# 导入可视化工具
import sys
sys.path.append(os.path.join(project_root, 'tools'))
try:
    from benchmarks.visualization import BenchmarkVisualizer
    from benchmarks.base_benchmark import BaseBenchmark, BenchmarkConfig, BenchmarkResult
    from matplotlib_utils import setup_chinese_font
    from metrics_collector import MetricsCollector
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    # 在logger配置之前，使用print输出警告
    print(f"Warning: Failed to import visualization tools: {e}")
    VISUALIZATION_AVAILABLE = False
    BenchmarkVisualizer = None
    setup_chinese_font = None
    MetricsCollector = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """实验配置类"""
    model_name: str = 'bert-base-uncased'
    tasks: List[str] = None
    max_seq_length: int = 128  # 改名为max_seq_length以符合用户要求
    batch_size: int = 100
    learning_rate: float = 2e-4
    num_epochs: int = 5  # epochs
    warmup_epochs: float = 1  # warmup_epochs，默认为总epochs的10%
    optimizer: str = 'adamw_hf'  # optimizer类型
    weight_decay: float = 0.01  # weight_decay
    lr_scheduler: str = 'linear'  # lr_scheduler类型
    output_dir: str = './results'
    seed: int = 42
    use_mixed_precision: bool = True  # 启用混合精度训练
    dataloader_num_workers: int = 2  # 减少数据加载器工作进程以避免内存问题
    enable_visualization: bool = True  # 启用可视化功能
    gradient_accumulation_steps: int = 8  # 支持从YAML覆盖梯度累积步数
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = ["sst2", "mnli", "qnli", "rte", "cola"]

class GLUEDataProcessor:
    """GLUE数据处理器"""
    
    TASK_CONFIGS = {
        "sst2": {
            "dataset_name": "glue",
            "dataset_config": "sst2",
            "text_columns": ["sentence"],
            "label_column": "label",
            "num_labels": 2,
            "metric": "accuracy"
        },
        "mnli": {
            "dataset_name": "glue", 
            "dataset_config": "mnli",
            "text_columns": ["premise", "hypothesis"],
            "label_column": "label",
            "num_labels": 3,
            "metric": "accuracy"
        },
        "qnli": {
            "dataset_name": "glue",
            "dataset_config": "qnli", 
            "text_columns": ["question", "sentence"],
            "label_column": "label",
            "num_labels": 2,
            "metric": "accuracy"
        },
        "rte": {
            "dataset_name": "glue",
            "dataset_config": "rte",
            "text_columns": ["sentence1", "sentence2"], 
            "label_column": "label",
            "num_labels": 2,
            "metric": "accuracy"
        },
        "cola": {
            "dataset_name": "glue",
            "dataset_config": "cola",
            "text_columns": ["sentence"],
            "label_column": "label", 
            "num_labels": 2,
            "metric": "matthews_correlation"
        }
    }
    
    def __init__(self, tokenizer, max_seq_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_seq_length
    
    def load_dataset(self, task: str) -> DatasetDict:
        """加载指定任务的数据集"""
        if task not in self.TASK_CONFIGS:
            raise ValueError(f"Unsupported task: {task}")
        
        config = self.TASK_CONFIGS[task]
        dataset = load_dataset(config["dataset_name"], config["dataset_config"])
        
        # 对于MNLI，只使用matched验证集
        if task == "mnli":
            dataset["validation"] = dataset["validation_matched"]
            del dataset["validation_matched"]
            del dataset["validation_mismatched"]
        
        return dataset
    
    def preprocess_function(self, examples, task: str):
        """预处理函数"""
        config = self.TASK_CONFIGS[task]
        text_columns = config["text_columns"]
        
        if len(text_columns) == 1:
            # 单句任务
            texts = examples[text_columns[0]]
            inputs = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors=None
            )
        else:
            # 句子对任务
            texts1 = examples[text_columns[0]]
            texts2 = examples[text_columns[1]]
            inputs = self.tokenizer(
                texts1, texts2,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors=None
            )
        
        inputs["labels"] = examples[config["label_column"]]
        return inputs

class PEFTMethodFactory:
    """PEFT方法工厂"""
    
    # 定义不同方法的warmup策略
    WARMUP_STRATEGIES = {
        "lora": "lr_only",      # LoRA：只进行学习率warmup，算法机制不变
        "dora": "lr_only",      # DoRA：只进行学习率warmup，算法机制不变
        "adalora": "full",      # AdaLoRA：需要完整warmup，禁用rank变化
        "dylora": "full",       # DyLoRA：需要完整warmup，禁用动态机制
        "loraven": "full",      # LoRAven：需要完整warmup，禁用剪枝等动态机制
        "full_finetune": "lr_only"  # 全参数微调：只进行学习率warmup
    }
    
    @staticmethod
    def get_warmup_strategy(method: str) -> str:
        """获取指定方法的warmup策略"""
        return PEFTMethodFactory.WARMUP_STRATEGIES.get(method.lower(), "lr_only")
    
    @staticmethod
    def should_disable_dynamic_mechanisms(method: str, current_step: int, warmup_steps: int) -> bool:
        """判断是否应该禁用动态机制"""
        strategy = PEFTMethodFactory.get_warmup_strategy(method)
        if strategy == "full" and current_step < warmup_steps:
            return True
        return False
    
    @staticmethod
    def create_lora_config(num_labels: int, overrides: Optional[Dict[str, Any]] = None) -> LoraConfig:
        """创建LoRA配置，支持YAML覆盖"""
        cfg = dict(
            r=8,
            lora_alpha=32,
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        overrides = overrides or {}
        cfg.update(overrides)
        return LoraConfig(**cfg)
    
    @staticmethod
    def create_adalora_config(num_labels: int, overrides: Optional[Dict[str, Any]] = None) -> AdaLoraConfig:
        """创建AdaLoRA配置，支持YAML覆盖"""
        cfg = dict(
            r=8,
            lora_alpha=32,
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            init_r=12,
            target_r=4,
            beta1=0.85,
            beta2=0.85,
            tinit=100,
            tfinal=500,
            deltaT=10,
            total_step=1000,
        )
        overrides = overrides or {}
        cfg.update(overrides)
        return AdaLoraConfig(**cfg)
    
    @staticmethod
    def create_dora_config(num_labels: int, overrides: Optional[Dict[str, Any]] = None) -> LoraConfig:
        """创建DoRA配置，支持YAML覆盖"""
        cfg = dict(
            r=8,
            lora_alpha=32,
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            use_dora=True,
        )
        overrides = overrides or {}
        cfg.update(overrides)
        return LoraConfig(**cfg)

class GLUEEvaluator:
    """GLUE评估器"""
    
    @staticmethod
    def compute_metrics(eval_pred: EvalPrediction, task: str) -> Dict[str, float]:
        """计算评估指标并添加eval_前缀"""
        try:
            # 正确处理EvalPrediction对象
            if hasattr(eval_pred, 'predictions') and hasattr(eval_pred, 'label_ids'):
                predictions = eval_pred.predictions
                labels = eval_pred.label_ids
            else:
                predictions, labels = eval_pred
            
            # 确保是numpy数组
            if torch.is_tensor(predictions):
                predictions = predictions.cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
                
            # 获取预测类别
            if predictions.ndim > 1:
                predictions = np.argmax(predictions, axis=1)
            
            if task == "cola":
                # CoLA使用Matthews相关系数
                return {
                    "eval_matthews_correlation": matthews_corrcoef(labels, predictions),
                    "eval_accuracy": accuracy_score(labels, predictions)
                }
            else:
                # 其他任务使用准确率
                return {
                    "eval_accuracy": accuracy_score(labels, predictions),
                    "eval_f1": f1_score(labels, predictions, average="macro")
                }
        except Exception as e:
            logger.error(f"compute_metrics函数中发生错误: {e}")
            # 返回包含eval_accuracy的默认指标，避免训练中断
            return {"eval_accuracy": 0.0}

class GLUEBenchmark:
    """GLUE基准测试主类"""
    
    def __init__(self, config: ExperimentConfig, yaml_overrides: Optional[Dict[str, Any]] = None):
        # 支持从YAML合并覆盖的配置
        self.base_config = config
        self.yaml_overrides = yaml_overrides or {}
        # 根据模型/任务/方法动态生成运行时配置，默认先使用base_config
        self.config = config
        self.tokenizer = None
        self.data_processor = None
        self.evaluator = GLUEEvaluator()
        self.results = {}
        
        # 初始化详细指标收集器
        if VISUALIZATION_AVAILABLE and MetricsCollector:
            try:
                self.metrics_collector = MetricsCollector(config.output_dir)
                logger.info("Metrics collector initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize metrics collector: {e}")
                self.metrics_collector = None
        else:
            self.metrics_collector = None
        
        # 初始化可视化器
        if config.enable_visualization and VISUALIZATION_AVAILABLE:
            try:
                # 设置中文字体支持
                if setup_chinese_font:
                    setup_chinese_font()
                self.visualizer = BenchmarkVisualizer(Path(config.output_dir))
                logger.info("Visualization tools initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize visualization tools: {e}")
                self.visualizer = None
        else:
            if config.enable_visualization and not VISUALIZATION_AVAILABLE:
                logger.warning("Visualization requested but tools not available")
            self.visualizer = None
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # 确保输出目录存在
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _compose_runtime_config(self, method: str, task: str) -> ExperimentConfig:
        """根据YAML覆盖生成当前模型/任务/方法的局部配置"""
        cfg_dict = {
            'model_name': self.base_config.model_name,
            'tasks': [task],
            'max_seq_length': self.base_config.max_seq_length,
            'batch_size': self.base_config.batch_size,
            'learning_rate': self.base_config.learning_rate,
            'num_epochs': self.base_config.num_epochs,
            'warmup_epochs': self.base_config.warmup_epochs,
            'optimizer': self.base_config.optimizer,
            'weight_decay': self.base_config.weight_decay,
            'lr_scheduler': self.base_config.lr_scheduler,
            'output_dir': self.base_config.output_dir,
            'seed': self.base_config.seed,
            'use_mixed_precision': self.base_config.use_mixed_precision,
            'dataloader_num_workers': self.base_config.dataloader_num_workers,
            'enable_visualization': self.base_config.enable_visualization,
        }

        # YAML层级：global -> models[model_name] -> tasks[task] -> methods[method]
        def deep_update(dst: Dict[str, Any], src: Dict[str, Any]):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    deep_update(dst[k], v)
                else:
                    dst[k] = v

        # 逐层合并，仅合并training子字段，避免将peft等未知键注入ExperimentConfig
        global_training = self.yaml_overrides.get('global', {}).get('training', {})
        model_training = self.yaml_overrides.get('models', {}).get(self.base_config.model_name, {}).get('training', {})
        task_training = self.yaml_overrides.get('models', {}).get(self.base_config.model_name, {}).get('tasks', {}).get(task, {}).get('training', {})
        method_training = self.yaml_overrides.get('models', {}).get(self.base_config.model_name, {}).get('tasks', {}).get(task, {}).get('methods', {}).get(method, {}).get('training', {})

        deep_update(cfg_dict, global_training)
        deep_update(cfg_dict, model_training)
        deep_update(cfg_dict, task_training)
        deep_update(cfg_dict, method_training)

        # 过滤未知键，确保只传入ExperimentConfig定义的字段
        allowed_keys = {f.name for f in fields(ExperimentConfig)}
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in allowed_keys}

        # 生成新的ExperimentConfig
        runtime_config = ExperimentConfig(**filtered_cfg)
        return runtime_config

    def _get_peft_overrides(self, method: str, task: str) -> Dict[str, Any]:
        """组合不同层级的PEFT覆盖配置"""
        def deep_update(dst: Dict[str, Any], src: Dict[str, Any]):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    deep_update(dst[k], v)
                else:
                    dst[k] = v
        overrides: Dict[str, Any] = {}
        global_peft = self.yaml_overrides.get('global', {}).get('peft', {})
        model_peft = self.yaml_overrides.get('models', {}).get(self.base_config.model_name, {}).get('peft', {})
        task_peft = self.yaml_overrides.get('models', {}).get(self.base_config.model_name, {}).get('tasks', {}).get(task, {}).get('peft', {})
        method_peft = self.yaml_overrides.get('models', {}).get(self.base_config.model_name, {}).get('tasks', {}).get(task, {}).get('methods', {}).get(method, {}).get('peft', {})
        deep_update(overrides, global_peft)
        deep_update(overrides, model_peft)
        deep_update(overrides, task_peft)
        deep_update(overrides, method_peft)
        return overrides

    def run_experiment(self, method: str, task: str) -> Dict[str, float]:
        """运行单个实验"""
        logger.info(f"Running {method} on {task}")

        # 根据YAML覆盖组合运行时配置
        self.config = self._compose_runtime_config(method, task)

        # 确保tokenizer和数据处理器与当前配置一致
        if self.tokenizer is None or getattr(self.base_config, 'model_name', None) != self.config.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        # 重建数据处理器以应用当前max_seq_length
        self.data_processor = GLUEDataProcessor(self.tokenizer, self.config.max_seq_length)
        
        # 检查多GPU可用性
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for parallel training")
            device = torch.device("cuda")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # 加载数据集
        dataset = self.data_processor.load_dataset(task)
        task_config = self.data_processor.TASK_CONFIGS[task]
        
        # 预处理数据
        tokenized_dataset = dataset.map(
            lambda examples: self.data_processor.preprocess_function(examples, task),
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=4,  # 减少并行进程数以避免内存问题
            load_from_cache_file=True,  # 启用缓存以加速重复处理
            desc=f"Tokenizing {task} dataset"  # 添加进度描述
        )
        
        # 加载基础模型
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=task_config["num_labels"]
        )
        
        # 应用PEFT方法或全参数微调
        if method == "lora":
            peft_config = PEFTMethodFactory.create_lora_config(task_config["num_labels"], overrides=self._get_peft_overrides(method, task))
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        elif method == "adalora":
            peft_config = PEFTMethodFactory.create_adalora_config(task_config["num_labels"], overrides=self._get_peft_overrides(method, task))
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        elif method == "dora":
            peft_config = PEFTMethodFactory.create_dora_config(task_config["num_labels"], overrides=self._get_peft_overrides(method, task))
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        elif method == "loraven":
            # 使用LoRAven配置（需要从loraven.peft_adapter导入）
            try:
                from loraven.peft_adapter import LoRAvenConfig
                loraven_defaults = dict(
                    r_min=4,
                    r_max=16,
                    r=8,
                    lora_alpha=32,
                    target_modules=["query_proj", "key_proj", "value_proj", "dense"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.SEQ_CLS,
                    complexity_scorer_type="lightweight",
                    rank_scheduler_type="linear",
                    energy_budget=1000.0
                )
                overrides = self._get_peft_overrides(method, task)
                loraven_defaults.update(overrides)
                peft_config = LoRAvenConfig(**loraven_defaults)
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            except ImportError:
                logger.warning("LoRAven not available, using LoRA instead")
                peft_config = PEFTMethodFactory.create_lora_config(task_config["num_labels"])
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
        elif method == "full_finetune":
            # 全参数微调：不使用PEFT，直接训练整个模型
            logger.info("Using full fine-tuning (all parameters trainable)")
            # 打印可训练参数信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Trainable parameters percentage: {100 * trainable_params / total_params:.2f}%")
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # 创建PEFT模型（仅对PEFT方法）
        # 注意：full_finetune方法不需要get_peft_model包装
        
        # 多GPU并行训练配置
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for parallel training")
            # 使用DataParallel进行多GPU训练，避免标量张量gather警告
            model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            model = model.to(device)
        else:
            model = model.to(device)
        
        # 根据任务确定最佳模型指标
        if task == "cola":
            metric_for_best_model = "eval_matthews_correlation"  # 恢复eval_前缀
        else:
            metric_for_best_model = "eval_accuracy"  # 恢复eval_前缀
        
        # 计算warmup_steps（从warmup_epochs转换）
        # 估算每个epoch的步数：数据集大小 / (batch_size * GPU数量)
        train_dataset_size = len(tokenized_dataset["train"])
        effective_batch_size = self.config.batch_size * max(1, torch.cuda.device_count())
        steps_per_epoch = train_dataset_size // effective_batch_size
        
        # 计算优化器步数（考虑梯度累积）
        gradient_accumulation_steps = int(getattr(self.config, 'gradient_accumulation_steps', 8))
        optimizer_steps_per_epoch = math.ceil(steps_per_epoch / gradient_accumulation_steps)
        
        # 改进的warmup_epochs逻辑：
        # - 如果warmup_epochs >= 1，则视为具体的epoch数
        # - 如果warmup_epochs < 1，则视为总epochs的比例
        # - warmup_steps基于优化器步数计算，而非数据迭代步数
        if self.config.warmup_epochs >= 1:
            # 整数表示具体的warmup轮数
            warmup_steps = int(self.config.warmup_epochs * optimizer_steps_per_epoch)
            logger.info(f"Using absolute warmup epochs: {self.config.warmup_epochs}")
        else:
            # 小数表示总轮数的比例
            warmup_steps = int(self.config.warmup_epochs * self.config.num_epochs * optimizer_steps_per_epoch)
            logger.info(f"Using proportional warmup epochs: {self.config.warmup_epochs} of total {self.config.num_epochs} epochs")
        
        logger.info(f"Training steps calculation: train_size={train_dataset_size}, effective_batch_size={effective_batch_size}")
        logger.info(f"Steps per epoch: {steps_per_epoch} (data iterations), {optimizer_steps_per_epoch} (optimizer steps)")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"Calculated warmup_steps: {warmup_steps} (based on optimizer steps)")
        
        # 根据PEFT方法调整warmup策略
        warmup_strategy = PEFTMethodFactory.get_warmup_strategy(method)
        logger.info(f"Method {method} uses warmup strategy: {warmup_strategy}")
        
        # 对于只需要学习率warmup的方法（LoRA/DoRA），如果warmup_epochs=1，则有效
        if warmup_strategy == "lr_only" and self.config.warmup_epochs == 1:
            logger.info(f"Enabling learning rate warmup for {method} (warmup_epochs=1)")
        elif warmup_strategy == "lr_only" and self.config.warmup_epochs != 1:
            # 对于LoRA/DoRA，如果不是1，则禁用warmup
            warmup_steps = 0
            logger.info(f"Disabling warmup for {method} (warmup_epochs != 1)")
        
        logger.info(f"Training dataset size: {train_dataset_size}")
        logger.info(f"Effective batch size: {effective_batch_size}")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Final warmup steps: {warmup_steps}")
        
        # 训练参数配置
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/{method}_{task}",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=warmup_steps,  # 使用计算得到的warmup_steps
            weight_decay=self.config.weight_decay,  # 添加weight_decay参数
            optim=self.config.optimizer,  # 添加optimizer参数
            lr_scheduler_type=self.config.lr_scheduler,  # 添加lr_scheduler参数
            logging_steps=100,
            eval_strategy="epoch",  # 使用新的参数名
            save_strategy="no",  # 禁用checkpoint保存以避免modules_to_save错误
            save_steps=1000000,  # 设置很大的值确保不保存
            load_best_model_at_end=False,  # 禁用最佳模型加载以避免保存相关错误
            metric_for_best_model=None,  # 显式禁用最佳模型指标检查
            greater_is_better=True,
            seed=self.config.seed,
            dataloader_num_workers=self.config.dataloader_num_workers,
            # 混合精度训练配置
            fp16=self.config.use_mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,  # YAML可覆盖
            dataloader_pin_memory=True,    # 启用pin_memory以提高性能
            remove_unused_columns=False,  # 保留所有列，避免列名匹配问题
            # 多GPU优化
            ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
            report_to=[],  # 禁用报告以避免额外依赖
            run_name=f"{method}_{task}",  # 设置运行名称
        )
        
        # 创建训练器
        # 定义compute_metrics函数，确保正确计算评估指标
        def compute_metrics(eval_pred):
            """计算评估指标并添加eval_前缀"""
            try:
                from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
                import numpy as np
                
                logger.info(f"compute_metrics called for task: {task}")
                
                # 正确处理EvalPrediction对象
                if hasattr(eval_pred, 'predictions') and hasattr(eval_pred, 'label_ids'):
                    predictions = eval_pred.predictions
                    labels = eval_pred.label_ids
                else:
                    # 处理tuple格式的输入
                    if isinstance(eval_pred, tuple) and len(eval_pred) == 2:
                        predictions, labels = eval_pred
                    else:
                        predictions = eval_pred
                        labels = None
                
                logger.info(f"Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else type(predictions)}")
                logger.info(f"Labels shape: {labels.shape if hasattr(labels, 'shape') else type(labels)}")
                
                # 确保是numpy数组
                if torch.is_tensor(predictions):
                    predictions = predictions.cpu().numpy()
                if torch.is_tensor(labels):
                    labels = labels.cpu().numpy()
                
                # 处理predictions可能是tuple的情况
                if isinstance(predictions, tuple):
                    # 通常第一个元素是logits
                    predictions = predictions[0]
                    if torch.is_tensor(predictions):
                        predictions = predictions.cpu().numpy()
                
                # 确保predictions和labels都是numpy数组
                if torch.is_tensor(predictions):
                    predictions = predictions.cpu().numpy()
                if torch.is_tensor(labels):
                    labels = labels.cpu().numpy()
                    
                logger.info(f"After processing - Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else type(predictions)}")
                logger.info(f"After processing - Labels shape: {labels.shape if hasattr(labels, 'shape') else type(labels)}")
                    
                # 获取预测类别
                if hasattr(predictions, 'ndim') and predictions.ndim > 1:
                    logger.info(f"Applying argmax to predictions with shape: {predictions.shape}")
                    predictions = np.argmax(predictions, axis=1)
                    logger.info(f"After argmax - Predictions shape: {predictions.shape}")
                else:
                    logger.info(f"Predictions already 1D or no ndim attribute: {predictions.shape if hasattr(predictions, 'shape') else type(predictions)}")
                
                if task == "cola":
                    # CoLA使用Matthews相关系数
                    accuracy = accuracy_score(labels, predictions)
                    matthews_corr = matthews_corrcoef(labels, predictions)
                    result = {
                        "eval_accuracy": accuracy,
                        "eval_matthews_correlation": matthews_corr
                    }
                else:
                    # 其他任务使用准确率和F1
                    accuracy = accuracy_score(labels, predictions)
                    f1 = f1_score(labels, predictions, average="macro")
                    result = {
                        "eval_accuracy": accuracy,
                        "eval_f1": f1
                    }
                
                logger.info(f"Computed metrics for {task}: {result}")
                return result
                
            except Exception as e:
                logger.error(f"Error in compute_metrics: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # 返回包含eval_accuracy的默认指标，避免训练中断
                return {"eval_accuracy": 0.0}
        
        # 创建方法感知的Trainer类
        class MethodAwareTrainer(Trainer):
            def __init__(self, peft_method, warmup_steps, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.peft_method = peft_method
                self.warmup_steps = warmup_steps
                self.current_step = 0
                
            def training_step(self, model, inputs, num_items_in_batch=None):
                # 在训练步骤中检查是否需要禁用动态机制
                if PEFTMethodFactory.should_disable_dynamic_mechanisms(
                    self.peft_method, self.current_step, self.warmup_steps
                ):
                    # 对于需要完整warmup的方法，在warmup期间禁用动态机制
                    # 这里可以根据具体的PEFT方法实现相应的禁用逻辑
                    # 例如：暂时禁用AdaLoRA的rank变化、DyLoRA的动态机制等
                    logger.info(f"Step {self.current_step}: Disabling dynamic mechanisms for {self.peft_method} during warmup")
                
                result = super().training_step(model, inputs, num_items_in_batch)
                self.current_step += 1
                return result
        
        trainer = MethodAwareTrainer(
            peft_method=method,
            warmup_steps=warmup_steps,
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=compute_metrics,
        )
        
        # 训练
        trainer.train()
        
        # 评估
        try:
            eval_results = trainer.evaluate()
        except KeyError as e:
            # 捕获因metric_for_best_model导致的KeyError
            logger.error(f"trainer.evaluate() raised KeyError: {e}. Falling back to manual metrics computation.")
            eval_results = {}

        # 指标回退：若未返回核心评估指标，则使用predict手动计算并补充
        expected_key = "eval_matthews_correlation" if task == "cola" else "eval_accuracy"
        if expected_key not in eval_results:
            logger.warning(f"{expected_key} not found in eval_results; computing metrics via predict() fallback")
            try:
                # 确保验证集包含标签
                eval_dataset = tokenized_dataset["validation"]
                pred_output = trainer.predict(eval_dataset)
                
                # 检查是否有标签
                if pred_output.label_ids is not None:
                    # 使用与Trainer一致的compute_metrics逻辑
                    computed_metrics = compute_metrics((pred_output.predictions, pred_output.label_ids))
                    if isinstance(computed_metrics, dict):
                        eval_results.update(computed_metrics)
                        logger.info(f"Manually computed metrics added: {computed_metrics}")
                    else:
                        logger.warning("compute_metrics did not return a dict; skipping manual metric update")
                else:
                    # 如果predict没有返回标签，直接从数据集获取
                    logger.warning("pred_output.label_ids is None, extracting labels from dataset")
                    if "labels" in eval_dataset.column_names:
                        true_labels = np.array(eval_dataset["labels"])
                        logger.info(f"Extracted labels shape: {true_labels.shape}")
                        logger.info(f"Predictions type: {type(pred_output.predictions)}")
                        logger.info(f"Predictions shape: {pred_output.predictions.shape if hasattr(pred_output.predictions, 'shape') else 'No shape attribute'}")
                        
                        # 确保predictions是完整的预测结果
                        predictions = pred_output.predictions
                        
                        # 处理predictions可能是tuple的情况
                        if isinstance(predictions, tuple):
                            logger.info(f"Predictions is tuple with length: {len(predictions)}")
                            for i, elem in enumerate(predictions):
                                logger.info(f"Tuple element {i}: type={type(elem)}, shape={elem.shape if hasattr(elem, 'shape') else 'No shape'}")
                            
                            # 智能选择正确的logits元素
                            # 寻找形状为(样本数, 类别数)的元素
                            target_samples = true_labels.shape[0]
                            logits_elem = None
                            
                            for i, elem in enumerate(predictions):
                                if hasattr(elem, 'shape') and len(elem.shape) == 2:
                                    if elem.shape[0] == target_samples:
                                        logits_elem = elem
                                        logger.info(f"Found logits at tuple element {i} with shape {elem.shape}")
                                        break
                            
                            if logits_elem is not None:
                                predictions = logits_elem
                            else:
                                # 回退到第一个元素
                                predictions = predictions[0]
                                logger.warning("Could not find matching logits element, using first element")
                            
                            logger.info(f"Selected predictions shape: {predictions.shape if hasattr(predictions, 'shape') else type(predictions)}")
                        
                        if torch.is_tensor(predictions):
                            predictions = predictions.cpu().numpy()
                        
                        logger.info(f"After tensor conversion - Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else type(predictions)}")
                        
                        # 检查predictions的维度和内容
                        if hasattr(predictions, 'shape'):
                            logger.info(f"Predictions content (first 10): {predictions[:10] if len(predictions) > 10 else predictions}")
                            
                            # 如果predictions形状不匹配，可能需要重新获取完整的预测结果
                            if predictions.shape[0] != true_labels.shape[0]:
                                logger.warning(f"Predictions shape {predictions.shape} doesn't match labels shape {true_labels.shape}")
                                logger.info("Attempting to get full predictions by running model inference...")
                                
                                # 尝试直接使用模型进行推理
                                try:
                                    model = trainer.model
                                    model.eval()
                                    all_predictions = []
                                    
                                    # 使用DataLoader进行批量推理
                                    eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
                                    
                                    with torch.no_grad():
                                        for batch in eval_dataloader:
                                            # 将batch移到正确的设备
                                            batch = {k: v.to(trainer.args.device) for k, v in batch.items() if k != 'labels'}
                                            outputs = model(**batch)
                                            logits = outputs.logits
                                            predictions_batch = torch.argmax(logits, dim=-1)
                                            all_predictions.extend(predictions_batch.cpu().numpy())
                                    
                                    predictions = np.array(all_predictions)
                                    logger.info(f"Full inference predictions shape: {predictions.shape}")
                                    
                                except Exception as e:
                                    logger.error(f"Full inference failed: {e}")
                                    eval_results[expected_key] = 0.0
                        
                        # 如果predictions是2D的logits，需要应用argmax
                        if hasattr(predictions, 'ndim') and predictions.ndim > 1:
                            logger.info(f"Applying argmax to predictions with shape: {predictions.shape}")
                            predictions = np.argmax(predictions, axis=1)
                            logger.info(f"After argmax - Predictions shape: {predictions.shape}")
                        
                        # 直接计算准确率，不使用compute_metrics函数
                        if hasattr(predictions, 'shape') and predictions.shape[0] == true_labels.shape[0]:
                            try:
                                accuracy = accuracy_score(true_labels, predictions)
                                computed_metrics = {expected_key: accuracy}
                                eval_results.update(computed_metrics)
                                logger.info(f"Manually computed metrics added: {computed_metrics}")
                            except Exception as e:
                                logger.error(f"Manual metric computation failed: {e}")
                                eval_results[expected_key] = 0.0
                        else:
                            logger.error(f"Shape mismatch or invalid predictions: predictions {predictions.shape if hasattr(predictions, 'shape') else type(predictions)}, labels {true_labels.shape}")
                            eval_results[expected_key] = 0.0
                    else:
                        logger.error("No labels found in validation dataset")
                        eval_results[expected_key] = 0.0
                        
            except Exception as e:
                logger.error(f"Manual metric computation failed: {e}")
                # 兜底：至少补上核心指标键，避免后续报表或选择最优模型时报错
                eval_results[expected_key] = 0.0
        
        # 保存模型
        try:
            if method == "full_finetune":
                # 全参数微调：保存完整模型
                model_output_dir = f"{self.config.output_dir}/{method}_{task}_model"
                
                # 处理DataParallel包装的模型
                if hasattr(model, 'module'):
                    model_to_save = model.module
                else:
                    model_to_save = model
                
                # 创建输出目录
                import os
                os.makedirs(model_output_dir, exist_ok=True)
                
                # 保存完整模型
                model_to_save.save_pretrained(model_output_dir)
                logger.info(f"Full model saved to {model_output_dir}")
            else:
                # PEFT方法：只保存适配器
                adapter_output_dir = f"{self.config.output_dir}/{method}_{task}_adapter"
                
                # 获取要保存的模型
                model_to_save = model
                if hasattr(model, 'module'):
                    # 如果使用了DataParallel，需要先unwrap
                    model_to_save = model.module
                
                # 只保存PEFT适配器，避免完整模型保存时的键名冲突
                if hasattr(model_to_save, 'save_pretrained'):
                    # 创建输出目录
                    import os
                    os.makedirs(adapter_output_dir, exist_ok=True)
                    
                    # 保存适配器权重
                    model_to_save.save_pretrained(adapter_output_dir)
                    logger.info(f"PEFT adapter saved to {adapter_output_dir}")
                else:
                    logger.warning(f"Model does not have save_pretrained method")
                
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")
            # 不让保存失败影响训练结果
            pass
        
        # 保存结果
        result_key = f"{method}_{task}"
        self.results[result_key] = eval_results
        
        # 收集详细指标
        if self.metrics_collector:
            try:
                logger.info(f"Collecting detailed metrics for {method} on {task}")
                
                # 准备样本输入用于FLOPs计算
                sample_input = None
                if tokenized_dataset and "validation" in tokenized_dataset:
                    sample_batch = tokenized_dataset["validation"][:1]  # 取第一个样本
                    sample_input = {
                        'input_ids': torch.tensor(sample_batch['input_ids']).to(model.device),
                        'attention_mask': torch.tensor(sample_batch['attention_mask']).to(model.device)
                    }
                
                # 获取训练历史（如果可用）
                training_history = None
                if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
                    training_history = trainer.state.log_history
                
                # 获取验证数据加载器（用于动态性分析）
                eval_dataloader = None
                if tokenized_dataset and "validation" in tokenized_dataset:
                    eval_dataloader = trainer.get_eval_dataloader(tokenized_dataset["validation"])
                
                # 生成详细指标报告
                detailed_metrics = self.metrics_collector.generate_comprehensive_report(
                    model=model,
                    method=method,
                    task=task,
                    eval_results=eval_results,
                    sample_input=sample_input,
                    training_history=training_history,
                    dataloader=eval_dataloader
                )
                
                # 将详细指标添加到结果中
                self.results[result_key].update(detailed_metrics)
                
                logger.info(f"Detailed metrics collected successfully for {method} on {task}")
                
            except Exception as e:
                logger.warning(f"Failed to collect detailed metrics for {method} on {task}: {e}")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")
        
        # 移除完整模型保存以避免modules_to_save错误
        # model.save_pretrained(f"{self.config.output_dir}/{method}_{task}")
        
        # 输出详细指标
        self._print_experiment_metrics(method, task, eval_results, model, trainer)
        
        return eval_results
    
    def _print_experiment_metrics(self, method: str, task: str, eval_results: Dict[str, float], model, trainer):
        """打印实验的详细指标"""
        print("\n" + "="*80)
        print(f"实验完成: {method.upper()} on {task.upper()}")
        print("="*80)
        
        # 1. 基本性能指标
        print("\n📊 性能指标:")
        print("-" * 40)
        for key, value in eval_results.items():
            if key.startswith('eval_'):
                metric_name = key.replace('eval_', '').replace('_', ' ').title()
                if isinstance(value, float):
                    print(f"  {metric_name:<25}: {value:.4f}")
                else:
                    print(f"  {metric_name:<25}: {value}")
        
        # 2. 模型参数统计
        print("\n🔧 模型参数:")
        print("-" * 40)
        try:
            if hasattr(model, 'module'):
                # 处理DataParallel包装的模型
                model_to_analyze = model.module
            else:
                model_to_analyze = model
                
            total_params = sum(p.numel() for p in model_to_analyze.parameters())
            trainable_params = sum(p.numel() for p in model_to_analyze.parameters() if p.requires_grad)
            
            print(f"  {'总参数量':<25}: {total_params:,}")
            print(f"  {'可训练参数量':<25}: {trainable_params:,}")
            print(f"  {'可训练参数比例':<25}: {100 * trainable_params / total_params:.2f}%")
            
            # 对于PEFT方法，显示额外信息
            if method != "full_finetune" and hasattr(model_to_analyze, 'print_trainable_parameters'):
                print(f"  {'PEFT方法':<25}: {method}")
                
        except Exception as e:
            logger.warning(f"Failed to compute parameter statistics: {e}")
        
        # 3. 训练统计
        print("\n⏱️ 训练统计:")
        print("-" * 40)
        try:
            if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
                log_history = trainer.state.log_history
                if log_history:
                    # 获取训练时间
                    if 'train_runtime' in eval_results:
                        runtime = eval_results['train_runtime']
                        print(f"  {'训练时间':<25}: {runtime:.2f} 秒")
                    
                    # 获取训练步数
                    total_steps = trainer.state.global_step
                    print(f"  {'训练步数':<25}: {total_steps}")
                    
                    # 获取最终训练损失
                    train_losses = [log['train_loss'] for log in log_history if 'train_loss' in log]
                    if train_losses:
                        final_loss = train_losses[-1]
                        print(f"  {'最终训练损失':<25}: {final_loss:.4f}")
                    
                    # 获取学习率信息
                    learning_rates = [log['learning_rate'] for log in log_history if 'learning_rate' in log]
                    if learning_rates:
                        final_lr = learning_rates[-1]
                        print(f"  {'最终学习率':<25}: {final_lr:.2e}")
                        
        except Exception as e:
            logger.warning(f"Failed to extract training statistics: {e}")
        
        # 4. 评估统计
        print("\n📈 评估统计:")
        print("-" * 40)
        try:
            if 'eval_runtime' in eval_results:
                eval_runtime = eval_results['eval_runtime']
                print(f"  {'评估时间':<25}: {eval_runtime:.2f} 秒")
            
            if 'eval_samples_per_second' in eval_results:
                eval_speed = eval_results['eval_samples_per_second']
                print(f"  {'评估速度':<25}: {eval_speed:.2f} 样本/秒")
                
        except Exception as e:
            logger.warning(f"Failed to extract evaluation statistics: {e}")
        
        # 5. 硬件使用情况
        print("\n💻 硬件使用:")
        print("-" * 40)
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"  {'GPU数量':<25}: {gpu_count}")
                
                # 显示GPU内存使用情况
                for i in range(gpu_count):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                    print(f"  {'GPU ' + str(i) + ' 内存使用':<25}: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
            else:
                print(f"  {'设备':<25}: CPU")
                
        except Exception as e:
            logger.warning(f"Failed to get hardware information: {e}")
        
        # 6. 配置信息
        print("\n⚙️ 实验配置:")
        print("-" * 40)
        print(f"  {'模型':<25}: {self.config.model_name}")
        print(f"  {'任务':<25}: {task}")
        print(f"  {'方法':<25}: {method}")
        print(f"  {'批次大小':<25}: {self.config.batch_size}")
        print(f"  {'学习率':<25}: {self.config.learning_rate}")
        print(f"  {'训练轮数':<25}: {self.config.num_epochs}")
        print(f"  {'最大序列长度':<25}: {self.config.max_seq_length}")
        
        print("\n" + "="*80)
        print(f"实验 {method.upper()} on {task.upper()} 完成!")
        print("="*80 + "\n")
    
    def run_all_experiments(self, methods: List[str]) -> Dict[str, Dict[str, float]]:
        """运行所有实验"""
        all_results = {}
        visualization_data = []
        
        for method in methods:
            method_results = {}
            for task in self.config.tasks:
                try:
                    results = self.run_experiment(method, task)
                    method_results[task] = results
                    logger.info(f"{method} on {task}: {results}")
                    
                    # 收集可视化数据
                    if self.visualizer and 'eval_accuracy' in results:
                        viz_data = {
                            'method': method,
                            'task': task,
                            'benchmarks': {
                                'performance': {
                                    'accuracy': results['eval_accuracy'],
                                    'throughput': results.get('eval_samples_per_second', 0),
                                    'avg_latency': results.get('eval_runtime', 0) * 1000,  # 转换为毫秒
                                    'memory_peak_mb': results.get('train_memory_peak_mb', 0)
                                }
                            }
                        }
                        visualization_data.append(viz_data)
                        
                except Exception as e:
                    logger.error(f"Error running {method} on {task}: {e}")
                    method_results[task] = {"error": str(e)}
            
            all_results[method] = method_results
        
        # 保存所有结果
        with open(f"{self.config.output_dir}/all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # 生成可视化图表
        if self.visualizer and visualization_data:
            try:
                logger.info("Generating visualization charts...")
                
                # 性能对比图表
                perf_chart_path = self.visualizer.visualize_performance_comparison(
                    visualization_data, 
                    save_path=Path(self.config.output_dir) / "performance_comparison.png"
                )
                logger.info(f"Performance comparison chart saved to: {perf_chart_path}")
                
                # 创建交互式仪表板（如果支持plotly）
                try:
                    dashboard_path = self.visualizer.create_interactive_dashboard(
                        visualization_data,
                        save_path=Path(self.config.output_dir) / "interactive_dashboard.html"
                    )
                    logger.info(f"Interactive dashboard saved to: {dashboard_path}")
                except Exception as e:
                    logger.warning(f"Failed to create interactive dashboard: {e}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate visualization charts: {e}")
        
        return all_results
    
    def generate_comparison_report(self, results: Dict[str, Dict[str, float]]):
        """生成对比报告"""
        report = []
        report.append("# GLUE Benchmark Results\n")
        
        # 创建结果表格
        methods = list(results.keys())
        tasks = self.config.tasks
        
        # 表头
        header = "| Method | " + " | ".join(tasks) + " | Average |\n"
        separator = "|" + "|".join(["---"] * (len(tasks) + 2)) + "|\n"
        report.append(header)
        report.append(separator)
        
        # 每个方法的结果
        for method in methods:
            row = f"| {method} |"
            scores = []
            
            for task in tasks:
                if task in results[method] and "error" not in results[method][task]:
                    task_config = self.data_processor.TASK_CONFIGS[task]
                    metric_key = f"eval_{task_config['metric']}"
                    if metric_key in results[method][task]:
                        score = results[method][task][metric_key]
                        scores.append(score)
                        row += f" {score:.4f} |"
                    else:
                        row += " N/A |"
                        scores.append(0.0)
                else:
                    row += " Error |"
                    scores.append(0.0)
            
            # 计算平均分
            avg_score = np.mean(scores) if scores else 0.0
            row += f" {avg_score:.4f} |\n"
            report.append(row)
        
        # 保存报告
        with open(f"{self.config.output_dir}/comparison_report.md", "w") as f:
            f.writelines(report)
        
        logger.info("Comparison report saved to comparison_report.md")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GLUE基准测试：LoRA、AdaLoRA、DoRA、LoRAven对比实验')
    
    # 基础参数
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='预训练模型名称 (默认: bert-base-uncased)')
    parser.add_argument('--tasks', type=str, nargs='+', default=["sst2", "mnli", "qnli", "rte", "cola"],
                        help='要运行的GLUE任务列表 (默认: 所有任务)')
    parser.add_argument('--methods', type=str, nargs='+', default=["lora"],
                        help='要测试的方法列表 (可选: lora, adalora, dora, loraven, full_finetune) (默认: lora)')
    
    # 训练参数
    parser.add_argument('--num_epochs', '--epochs', type=int, default=5,
                        help='训练轮数 (默认: 5)')
    parser.add_argument('--warmup_epochs', type=float, default=1,
                        help='预热轮数：>=1表示具体轮数，<1表示总轮数的比例 (默认: 1)')
    parser.add_argument('--optimizer', type=str, default='adamw_hf',
                        choices=['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adafactor', 'sgd', 'adagrad', 'rmsprop'],
                        help='优化器类型 (默认: adamw_hf)')
    parser.add_argument('--lr', '--learning_rate', type=float, default=2e-4,
                        help='学习率 (默认: 2e-4)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小 (默认: 64)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减 (默认: 0.01)')
    parser.add_argument('--lr_scheduler', type=str, default='linear',
                        choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'],
                        help='学习率调度器类型 (默认: linear)')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='最大序列长度 (默认: 128)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./glue_results',
                        help='输出目录 (默认: ./glue_results)')
    
    # 其他参数
    parser.add_argument('--use_mixed_precision', action='store_true', default=True,
                        help='是否使用混合精度训练 (默认: True)')
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                        help='数据加载器工作进程数 (默认: 4)')
    
    # 可视化参数
    parser.add_argument('--enable_visualization', action='store_true', default=True,
                        help='是否启用可视化功能 (默认: True)')
    parser.add_argument('--disable_visualization', action='store_true', default=False,
                        help='禁用可视化功能')

    # YAML配置文件
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'glue_config.yaml'),
                        help='YAML配置文件路径 (默认: experiments/glue_config.yaml)')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 加载YAML配置
    yaml_overrides = {}
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                yaml_overrides = yaml.safe_load(f) or {}
            logger.info(f"加载YAML配置: {args.config}")
        except Exception as e:
            logger.warning(f"YAML配置加载失败，将使用CLI与默认参数: {e}")
    else:
        logger.info("未提供或找不到YAML配置文件，将使用CLI与默认参数")
    
    # 创建实验配置
    config = ExperimentConfig(
        model_name=args.model_name,
        tasks=args.tasks,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        output_dir=args.output_dir,
        seed=args.seed,
        use_mixed_precision=args.use_mixed_precision,
        dataloader_num_workers=args.dataloader_num_workers,
        enable_visualization=args.enable_visualization and not args.disable_visualization
    )
    
    # 打印配置信息
    logger.info("=== 实验配置 ===")
    logger.info(f"模型: {config.model_name}")
    logger.info(f"任务: {config.tasks}")
    logger.info(f"方法: {args.methods}")
    logger.info(f"训练轮数: {config.num_epochs}")
    logger.info(f"预热轮数: {config.warmup_epochs}")
    logger.info(f"优化器: {config.optimizer}")
    logger.info(f"学习率: {config.learning_rate}")
    logger.info(f"批次大小: {config.batch_size}")
    logger.info(f"权重衰减: {config.weight_decay}")
    logger.info(f"学习率调度器: {config.lr_scheduler}")
    logger.info(f"最大序列长度: {config.max_seq_length}")
    logger.info(f"随机种子: {config.seed}")
    logger.info(f"输出目录: {config.output_dir}")
    logger.info(f"可视化功能: {'启用' if config.enable_visualization else '禁用'}")
    logger.info("================")
    
    # 创建基准测试
    benchmark = GLUEBenchmark(config, yaml_overrides=yaml_overrides)
    
    # 运行所有实验
    results = benchmark.run_all_experiments(args.methods)
    
    # 生成对比报告
    benchmark.generate_comparison_report(results)
    
    logger.info("=== 实验完成 ===")
    if config.enable_visualization and benchmark.visualizer:
        logger.info("可视化图表已保存到输出目录中")
        logger.info(f"- 性能对比图表: {config.output_dir}/performance_comparison.png")
        logger.info(f"- 交互式仪表板: {config.output_dir}/interactive_dashboard.html")
    logger.info(f"详细结果请查看: {config.output_dir}/all_results.json")
    logger.info("================")
    
    print("GLUE benchmark completed!")
    print(f"Results saved to {config.output_dir}")

if __name__ == "__main__":
    main()
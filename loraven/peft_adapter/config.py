"""
LoRAven PEFT Configuration

Defines configuration class for LoRAven PEFT adapter integration.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict
from peft import PeftConfig, PeftType


@dataclass
class LoRAvenConfig(PeftConfig):
    """
    Configuration class for LoRAven PEFT adapter
    
    This class extends PeftConfig to provide LoRAven-specific parameters
    for dynamic rank adaptation and energy-aware optimization.
    """
    
    # Dynamic rank parameters
    r_min: int = field(default=4, metadata={"help": "Minimum rank for dynamic adaptation"})
    r_max: int = field(default=128, metadata={"help": "Maximum rank for dynamic adaptation"})
    
    # LoRA compatibility parameters
    r: int = field(default=64, metadata={"help": "Default rank (for PEFT compatibility)"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA scaling parameter"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout probability"})
    
    # PEFT compatibility patterns
    rank_pattern: Dict = field(default_factory=dict, metadata={"help": "Rank pattern for different modules"})
    alpha_pattern: Dict = field(default_factory=dict, metadata={"help": "Alpha pattern for different modules"})
    
    # Additional LoRA compatibility attributes
    fan_in_fan_out: bool = field(default=False, metadata={"help": "Set to True if the layer to replace stores weight like (fan_in, fan_out)"})
    bias: str = field(default="none", metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(default=None, metadata={"help": "List of modules apart from LoRA layers to be set as trainable"})
    init_lora_weights: bool = field(default=True, metadata={"help": "Whether to initialize the weights of the LoRA layers"})
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None, metadata={"help": "The layer indexes to transform"})
    layers_pattern: Optional[str] = field(default=None, metadata={"help": "The layer pattern name"})
    revision: Optional[str] = field(default=None, metadata={"help": "The specific model version to use"})
    megatron_config: Optional[Dict] = field(default=None, metadata={"help": "Megatron config for distributed training"})
    megatron_core: Optional[str] = field(default="megatron.core", metadata={"help": "Megatron core module"})
    loftq_config: Dict = field(default_factory=dict, metadata={"help": "LoftQ configuration"})
    
    # Target modules
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to replace with LoRAven"}
    )
    
    # LoRAven specific parameters
    complexity_scorer_type: str = field(
        default="lightweight",
        metadata={"help": "Type of complexity scorer ('lightweight', 'attention', 'gradient')"}
    )
    
    rank_scheduler_type: str = field(
        default="linear", 
        metadata={"help": "Type of rank scheduler ('linear', 'energy_aware', 'adaptive')"}
    )
    
    # Energy-aware parameters
    energy_budget: Optional[float] = field(
        default=None,
        metadata={"help": "Energy budget per sample (mJ/sample)"}
    )
    
    budget_allocation_strategy: str = field(
        default="uniform",
        metadata={"help": "Budget allocation strategy ('uniform', 'priority', 'adaptive')"}
    )
    
    # Adaptation parameters
    adaptation_threshold: float = field(
        default=0.1,
        metadata={"help": "Threshold for triggering rank adaptation"}
    )
    
    cooldown_steps: int = field(
        default=10,
        metadata={"help": "Cooldown steps between rank adaptations"}
    )
    
    # Performance optimization
    enable_caching: bool = field(
        default=True,
        metadata={"help": "Enable caching for performance optimization"}
    )
    
    cache_size: int = field(
        default=1000,
        metadata={"help": "Size of the complexity score cache"}
    )
    
    def __post_init__(self):
        # Set PEFT type - handle case where PEFT is not available
        try:
            if hasattr(PeftType, 'LORAVEN'):
                self.peft_type = PeftType.LORAVEN
            else:
                # Fallback when PEFT is not available or LORAVEN not registered
                self.peft_type = "LORAVEN"
        except (AttributeError, NameError):
            # Handle case where PeftType is not available
            self.peft_type = "LORAVEN"
        
        # Call parent's __post_init__ if it exists and is safe to call
        try:
            # Only call parent __post_init__ if PEFT is properly available
            if hasattr(PeftType, 'LORAVEN'):
                super().__post_init__()
        except Exception:
            # Skip parent __post_init__ if it fails
            pass
        
        # Validate parameters
        if self.r_min >= self.r_max:
            raise ValueError(f"r_min ({self.r_min}) must be less than r_max ({self.r_max})")
        
        if self.r < self.r_min or self.r > self.r_max:
            # Adjust default r to be within valid range
            self.r = min(max(self.r, self.r_min), self.r_max)
        
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError(f"lora_dropout must be between 0 and 1, got {self.lora_dropout}")
        
        if self.energy_budget is not None and self.energy_budget <= 0:
            raise ValueError(f"energy_budget must be positive, got {self.energy_budget}")
        
        # Set default target modules if not specified
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
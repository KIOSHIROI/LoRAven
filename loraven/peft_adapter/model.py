"""
LoRAven PEFT Model Implementation

Implements LoRAven as a PEFT adapter, providing dynamic rank adaptation
while maintaining compatibility with the PEFT ecosystem.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any, List, Union
import warnings

try:
    from peft.tuners.lora import LoraModel, LoraLayer
    from peft.utils import PeftType
except ImportError:
    warnings.warn("PEFT library not found. Please install with: pip install peft")
    # Fallback base classes
    class LoraModel:
        pass
    class LoraLayer:
        pass

from ..core.rank_scheduler import create_rank_scheduler
from ..core.budget_manager import BudgetManager
from .config import LoRAvenConfig


class LoRAvenLayer(LoraLayer):
    """
    LoRAven layer implementation with dynamic rank adaptation
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r_min: int = 4,
        r_max: int = 128,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        rank_scheduler_type: str = "linear",
        complexity_scorer_type: str = "lightweight",
        **kwargs
    ):
        super().__init__(base_layer, **kwargs)
        
        self.adapter_name = adapter_name
        self.r_min = r_min
        self.r_max = r_max
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Initialize LoRAven components
        self.rank_scheduler = create_rank_scheduler(
            scheduler_type=rank_scheduler_type,
            r_min=r_min,
            r_max=r_max
        )
        
        # Initialize complexity scorer
        self.complexity_scorer = self._create_complexity_scorer(
            complexity_scorer_type,
            base_layer
        )
        
        # Dynamic LoRA parameters - will be created on first forward pass
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.scaling = {}
        self.lora_dropout_layer = nn.ModuleDict()
        
        # Current rank tracking
        self.current_rank = r_min
        self.rank_history = []
        
        # Performance optimization
        self.enable_caching = kwargs.get('enable_caching', True)
        self.complexity_cache = {}
        
    def _create_complexity_scorer(self, scorer_type: str, base_layer: nn.Module):
        """Create complexity scorer based on type"""
        if scorer_type == "lightweight":
            return LightweightComplexityScorer(base_layer)
        elif scorer_type == "attention":
            return AttentionComplexityScorer(base_layer)
        elif scorer_type == "gradient":
            return GradientComplexityScorer(base_layer)
        else:
            raise ValueError(f"Unknown complexity scorer type: {scorer_type}")
    
    def _ensure_lora_params(self, rank: int):
        """Ensure LoRA parameters exist for the given rank"""
        adapter_name = self.adapter_name
        
        if adapter_name not in self.lora_A or self.lora_A[adapter_name].shape[0] != rank:
            # Get base layer dimensions
            if hasattr(self.base_layer, 'in_features') and hasattr(self.base_layer, 'out_features'):
                in_features = self.base_layer.in_features
                out_features = self.base_layer.out_features
            else:
                # For other layer types, try to infer dimensions
                weight_shape = self.base_layer.weight.shape
                out_features, in_features = weight_shape[0], weight_shape[1]
            
            # Create new LoRA parameters
            self.lora_A[adapter_name] = nn.Parameter(
                torch.randn(rank, in_features, device=self.base_layer.weight.device) * 0.01
            )
            self.lora_B[adapter_name] = nn.Parameter(
                torch.zeros(out_features, rank, device=self.base_layer.weight.device)
            )
            
            # Initialize scaling
            self.scaling[adapter_name] = self.lora_alpha / rank
            
            # Initialize dropout
            if adapter_name not in self.lora_dropout_layer:
                self.lora_dropout_layer[adapter_name] = nn.Dropout(p=self.lora_dropout)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass with dynamic rank adaptation"""
        # Get base layer output
        base_output = self.base_layer(x, *args, **kwargs)
        
        # Skip adaptation if not training
        if not self.training:
            if hasattr(self, '_cached_rank') and self._cached_rank is not None:
                rank = self._cached_rank
            else:
                rank = self.current_rank
        else:
            # Compute complexity scores
            complexity_scores = self.complexity_scorer(x)
            
            # Schedule rank based on complexity
            rank = self.rank_scheduler.schedule_rank(complexity_scores)
            
            # Update current rank
            self.current_rank = rank
            self.rank_history.append(rank)
            
            # Keep history bounded
            if len(self.rank_history) > 100:
                self.rank_history.pop(0)
        
        # Ensure LoRA parameters exist for current rank
        self._ensure_lora_params(rank)
        
        # Apply LoRA adaptation
        adapter_name = self.adapter_name
        if adapter_name in self.lora_A and adapter_name in self.lora_B:
            # Truncate parameters to current rank
            lora_A = self.lora_A[adapter_name][:rank, :]
            lora_B = self.lora_B[adapter_name][:, :rank]
            
            # Apply dropout
            dropout_layer = self.lora_dropout_layer.get(adapter_name, nn.Identity())
            
            # Compute LoRA output
            lora_output = dropout_layer(x) @ lora_A.T @ lora_B.T
            
            # Scale and add to base output
            scaling = self.scaling.get(adapter_name, 1.0)
            base_output = base_output + lora_output * scaling
        
        return base_output
    
    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """Merge LoRAven adapter weights into the base layer"""
        if hasattr(self, 'merged') and self.merged:
            warnings.warn("LoRAven adapter already merged")
            return
        
        try:
            # Use current rank for merging
            current_rank = getattr(self, 'current_rank', self.r_min)
            adapter_name = self.adapter_name
            
            if adapter_name in self.lora_A and adapter_name in self.lora_B:
                # Extract active LoRA weights
                lora_A = self.lora_A[adapter_name][:current_rank, :]
                lora_B = self.lora_B[adapter_name][:, :current_rank]
                
                # Compute delta weights
                scaling = self.scaling.get(adapter_name, 1.0)
                delta_weight = (lora_B @ lora_A) * scaling
                
                # Merge with base weights
                if hasattr(self.base_layer, 'weight'):
                    if safe_merge:
                        # Store original weights for potential rollback
                        self._original_weight = self.base_layer.weight.data.clone()
                    
                    self.base_layer.weight.data += delta_weight
                    self.merged = True
                else:
                    warnings.warn("No base layer weight found for merging")
            else:
                warnings.warn("No LoRA parameters found for merging")
                
        except Exception as e:
            warnings.warn(f"Failed to merge LoRAven adapter: {e}")
    
    def unmerge(self) -> None:
        """Unmerge LoRAven adapter weights from the base layer"""
        if not hasattr(self, 'merged') or not self.merged:
            warnings.warn("LoRAven adapter not merged")
            return
        
        try:
            # Use current rank for unmerging
            current_rank = getattr(self, 'current_rank', self.r_min)
            adapter_name = self.adapter_name
            
            if adapter_name in self.lora_A and adapter_name in self.lora_B:
                # Extract active LoRA weights
                lora_A = self.lora_A[adapter_name][:current_rank, :]
                lora_B = self.lora_B[adapter_name][:, :current_rank]
                
                # Compute delta weights
                scaling = self.scaling.get(adapter_name, 1.0)
                delta_weight = (lora_B @ lora_A) * scaling
                
                # Remove from base weights
                if hasattr(self.base_layer, 'weight'):
                    self.base_layer.weight.data -= delta_weight
                    self.merged = False
                    
                    # Clean up stored original weights
                    if hasattr(self, '_original_weight'):
                        delattr(self, '_original_weight')
                else:
                    warnings.warn("No base layer weight found for unmerging")
            else:
                warnings.warn("No LoRA parameters found for unmerging")
                
        except Exception as e:
            warnings.warn(f"Failed to unmerge LoRAven adapter: {e}")
    
    def get_delta_weight(self, adapter: str = None) -> torch.Tensor:
        """Get the delta weight matrix for the current adapter state"""
        current_rank = getattr(self, 'current_rank', self.r_min)
        adapter_name = adapter or self.adapter_name
        
        if current_rank == 0 or adapter_name not in self.lora_A or adapter_name not in self.lora_B:
            # Return zero tensor with appropriate shape
            if hasattr(self.base_layer, 'weight'):
                return torch.zeros_like(self.base_layer.weight)
            else:
                # Fallback dimensions
                return torch.zeros(self.r_max, self.r_max)
        
        # Extract active LoRA weights
        lora_A = self.lora_A[adapter_name][:current_rank, :]
        lora_B = self.lora_B[adapter_name][:, :current_rank]
        
        # Compute delta weights
        scaling = self.scaling.get(adapter_name, 1.0)
        delta_weight = (lora_B @ lora_A) * scaling
        
        return delta_weight
    
    def set_rank(self, rank: int):
        """Manually set the rank (for inference)"""
        rank = max(self.r_min, min(rank, self.r_max))
        self.current_rank = rank
        self._cached_rank = rank


class LoRAvenModel(LoraModel):
    """
    LoRAven PEFT Model with dynamic rank adaptation
    """
    
    def __init__(self, model: nn.Module, config: Union[LoRAvenConfig, Dict], adapter_name: str = "default"):
        # Handle case where config is passed as dict (PEFT compatibility)
        if isinstance(config, dict):
            # Extract the actual config from the dict
            if adapter_name in config:
                actual_config = config[adapter_name]
            else:
                # Fallback: use the first config in the dict
                actual_config = next(iter(config.values()))
        else:
            actual_config = config
        
        # Initialize parent with basic LoRA config for compatibility
        super().__init__(model, actual_config, adapter_name)
        
        self.config = actual_config
        self.adapter_name = adapter_name
        
        # Initialize budget manager if energy budget is specified
        if actual_config.energy_budget is not None:
            self.budget_manager = BudgetManager(
                total_budget=actual_config.energy_budget * 1000,  # Convert to total budget
                window_size=100,
                safety_margin=0.1,
                adaptation_rate=0.01
            )
        else:
            self.budget_manager = None
        
        # Replace LoRA layers with LoRAven layers
        self._replace_with_loraven_layers()
    
    def _replace_with_loraven_layers(self):
        """Replace standard LoRA layers with LoRAven layers"""
        # This method is called after PEFT has already created the LoRA layers
        # We need to enhance existing LoRA layers with LoRAven functionality
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # This is a LoRA layer, enhance it with LoRAven functionality
                if not hasattr(module, '_loraven_enhanced'):
                    # Add LoRAven-specific attributes
                    module.r_min = self.config.r_min
                    module.r_max = self.config.r_max
                    module.rank_scheduler_type = self.config.rank_scheduler_type
                    module.complexity_scorer_type = self.config.complexity_scorer_type
                    module.enable_caching = self.config.enable_caching
                    
                    # Create rank scheduler
                    module.rank_scheduler = create_rank_scheduler(
                        scheduler_type=self.config.rank_scheduler_type,
                        r_min=self.config.r_min,
                        r_max=self.config.r_max
                    )
                    
                    # Create complexity scorer
                    if self.config.complexity_scorer_type == "lightweight":
                        module.complexity_scorer = LightweightComplexityScorer(module)
                    elif self.config.complexity_scorer_type == "attention":
                        module.complexity_scorer = AttentionComplexityScorer(module)
                    elif self.config.complexity_scorer_type == "gradient":
                        module.complexity_scorer = GradientComplexityScorer(module)
                    else:
                        module.complexity_scorer = LightweightComplexityScorer(module)
                    
                    # Mark as enhanced
                    module._loraven_enhanced = True
    
    def set_global_rank(self, rank: int):
        """Set rank for all LoRAven layers"""
        for module in self.model.modules():
            if hasattr(module, '_loraven_enhanced') and hasattr(module, 'rank_scheduler'):
                # Use rank scheduler to set appropriate rank
                new_rank = module.rank_scheduler.get_rank(step=0)  # Use step 0 as default
                # Update LoRA parameters if needed
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    current_rank = module.lora_A[self.adapter_name].out_features
                    if new_rank != current_rank:
                        # Recreate LoRA parameters with new rank
                        self._update_lora_rank(module, new_rank)
    
    def _update_lora_rank(self, module, new_rank: int):
        """Update LoRA layer rank"""
        # This is a simplified implementation
        # In practice, you might want to preserve some weights or use more sophisticated resizing
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            device = module.lora_A[self.adapter_name].weight.device
            dtype = module.lora_A[self.adapter_name].weight.dtype
            
            # Get dimensions
            in_features = module.lora_A[self.adapter_name].in_features
            out_features = module.lora_B[self.adapter_name].out_features
            
            # Recreate LoRA layers with new rank
            module.lora_A[self.adapter_name] = nn.Linear(in_features, new_rank, bias=False, device=device, dtype=dtype)
            module.lora_B[self.adapter_name] = nn.Linear(new_rank, out_features, bias=False, device=device, dtype=dtype)
            
            # Initialize weights
            nn.init.kaiming_uniform_(module.lora_A[self.adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(module.lora_B[self.adapter_name].weight)
    
    def get_rank_statistics(self) -> Dict[str, Any]:
        """Get statistics about current ranks across all LoRAven layers"""
        ranks = []
        for module in self.model.modules():
            if hasattr(module, '_loraven_enhanced') and hasattr(module, 'lora_A'):
                if self.adapter_name in module.lora_A:
                    current_rank = module.lora_A[self.adapter_name].out_features
                    ranks.append(current_rank)
        
        if ranks:
            return {
                "mean_rank": sum(ranks) / len(ranks),
                "min_rank": min(ranks),
                "max_rank": max(ranks),
                "total_layers": len(ranks)
            }
        else:
            return {
                "mean_rank": 0,
                "min_rank": 0,
                "max_rank": 0,
                "total_layers": 0
            }


class LightweightComplexityScorer(nn.Module):
    """Lightweight complexity scorer for LoRAven"""
    
    def __init__(self, base_layer: nn.Module, hidden_dim: int = 32):
        super().__init__()
        
        # Get input dimension from base layer
        if hasattr(base_layer, 'in_features'):
            in_features = base_layer.in_features
        else:
            in_features = base_layer.weight.shape[1]
        
        self.scorer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute complexity scores for input batch"""
        # Compute mean across sequence dimension if present
        if x.dim() > 2:
            x_mean = x.mean(dim=1)  # (batch_size, features)
        else:
            x_mean = x
        
        # Get complexity scores
        scores = self.scorer(x_mean).squeeze(-1)  # (batch_size,)
        return scores


class AttentionComplexityScorer(nn.Module):
    """Attention-based complexity scorer"""
    
    def __init__(self, base_layer: nn.Module):
        super().__init__()
        
        if hasattr(base_layer, 'in_features'):
            in_features = base_layer.in_features
        else:
            in_features = base_layer.weight.shape[1]
        
        self.attention = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=8,
            batch_first=True
        )
        self.complexity_head = nn.Linear(in_features, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention-based complexity scores"""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Apply self-attention
        attn_output, attn_weights = self.attention(x, x, x)
        
        # Compute complexity from attention patterns
        complexity = self.complexity_head(attn_output.mean(dim=1))
        return torch.sigmoid(complexity).squeeze(-1)


class GradientComplexityScorer(nn.Module):
    """Gradient-based complexity scorer"""
    
    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.base_layer = base_layer
        self.gradient_buffer = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient-based complexity scores"""
        if not self.training:
            # Return cached scores during inference
            return torch.ones(x.shape[0], device=x.device) * 0.5
        
        # Compute gradients with respect to input
        x_grad = x.clone().detach().requires_grad_(True)
        output = self.base_layer(x_grad)
        
        # Compute gradient norm as complexity measure
        if output.requires_grad:
            grad_outputs = torch.ones_like(output)
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=x_grad,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
            
            # Compute complexity as gradient norm
            complexity = torch.norm(gradients, dim=-1)
            # Normalize to [0, 1]
            complexity = torch.sigmoid(complexity / complexity.max())
        else:
            complexity = torch.ones(x.shape[0], device=x.device) * 0.5
        
        return complexity
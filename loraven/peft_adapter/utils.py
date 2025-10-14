"""
Utility functions for LoRAven PEFT adapter

Provides merge, unload, and other utility functions for managing
LoRAven adapters in the PEFT ecosystem.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import warnings


def merge_loraven_weights(
    base_layer: nn.Module,
    loraven_layer: nn.Module,
    scaling: float = 1.0
) -> nn.Module:
    """
    Merge LoRAven adapter weights into the base layer
    
    Args:
        base_layer: Original layer (e.g., nn.Linear)
        loraven_layer: LoRAven adapter layer
        scaling: Scaling factor for merged weights
        
    Returns:
        Merged layer with adapter weights incorporated
    """
    if not hasattr(loraven_layer, 'lora_A') or not hasattr(loraven_layer, 'lora_B'):
        warnings.warn("LoRAven layer missing LoRA components, skipping merge")
        return base_layer
    
    try:
        # Get current rank
        current_rank = getattr(loraven_layer, 'current_rank', loraven_layer.r)
        
        # Extract active LoRA weights based on current rank
        lora_A = loraven_layer.lora_A[:current_rank, :]
        lora_B = loraven_layer.lora_B[:, :current_rank]
        
        # Compute LoRA delta weights
        lora_delta = (lora_B @ lora_A) * (loraven_layer.lora_alpha / current_rank) * scaling
        
        # Merge with base weights
        if hasattr(base_layer, 'weight'):
            base_layer.weight.data += lora_delta
        else:
            warnings.warn("Base layer has no weight parameter to merge with")
            
        return base_layer
        
    except Exception as e:
        warnings.warn(f"Failed to merge LoRAven weights: {e}")
        return base_layer


def unload_loraven_weights(
    merged_layer: nn.Module,
    loraven_layer: nn.Module,
    scaling: float = 1.0
) -> nn.Module:
    """
    Unload LoRAven adapter weights from a merged layer
    
    Args:
        merged_layer: Layer with merged adapter weights
        loraven_layer: Original LoRAven adapter layer
        scaling: Scaling factor used during merge
        
    Returns:
        Layer with adapter weights removed
    """
    if not hasattr(loraven_layer, 'lora_A') or not hasattr(loraven_layer, 'lora_B'):
        warnings.warn("LoRAven layer missing LoRA components, skipping unload")
        return merged_layer
    
    try:
        # Get current rank
        current_rank = getattr(loraven_layer, 'current_rank', loraven_layer.r)
        
        # Extract active LoRA weights
        lora_A = loraven_layer.lora_A[:current_rank, :]
        lora_B = loraven_layer.lora_B[:, :current_rank]
        
        # Compute LoRA delta weights
        lora_delta = (lora_B @ lora_A) * (loraven_layer.lora_alpha / current_rank) * scaling
        
        # Remove from merged weights
        if hasattr(merged_layer, 'weight'):
            merged_layer.weight.data -= lora_delta
        else:
            warnings.warn("Merged layer has no weight parameter to unload from")
            
        return merged_layer
        
    except Exception as e:
        warnings.warn(f"Failed to unload LoRAven weights: {e}")
        return merged_layer


def get_loraven_state_dict(model: nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
    """
    Extract LoRAven-specific state dict from a model
    
    Args:
        model: Model containing LoRAven layers
        prefix: Prefix for parameter names
        
    Returns:
        Dictionary containing LoRAven parameters
    """
    loraven_state = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            param_prefix = f"{prefix}{name}." if prefix else f"{name}."
            
            # Save LoRA weights
            loraven_state[f"{param_prefix}lora_A"] = module.lora_A.clone()
            loraven_state[f"{param_prefix}lora_B"] = module.lora_B.clone()
            
            # Save LoRAven-specific parameters
            if hasattr(module, 'current_rank'):
                loraven_state[f"{param_prefix}current_rank"] = torch.tensor(module.current_rank)
            
            if hasattr(module, 'complexity_scores'):
                loraven_state[f"{param_prefix}complexity_scores"] = module.complexity_scores.clone()
                
            if hasattr(module, 'energy_budget'):
                loraven_state[f"{param_prefix}energy_budget"] = torch.tensor(module.energy_budget)
    
    return loraven_state


def load_loraven_state_dict(
    model: nn.Module, 
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True
) -> None:
    """
    Load LoRAven-specific state dict into a model
    
    Args:
        model: Model to load state into
        state_dict: LoRAven state dictionary
        strict: Whether to strictly enforce parameter matching
    """
    missing_keys = []
    unexpected_keys = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Load LoRA weights
            lora_A_key = f"{name}.lora_A"
            lora_B_key = f"{name}.lora_B"
            
            if lora_A_key in state_dict:
                module.lora_A.data.copy_(state_dict[lora_A_key])
            elif strict:
                missing_keys.append(lora_A_key)
                
            if lora_B_key in state_dict:
                module.lora_B.data.copy_(state_dict[lora_B_key])
            elif strict:
                missing_keys.append(lora_B_key)
            
            # Load LoRAven-specific parameters
            rank_key = f"{name}.current_rank"
            if rank_key in state_dict:
                module.current_rank = state_dict[rank_key].item()
                
            complexity_key = f"{name}.complexity_scores"
            if complexity_key in state_dict and hasattr(module, 'complexity_scores'):
                module.complexity_scores.data.copy_(state_dict[complexity_key])
                
            budget_key = f"{name}.energy_budget"
            if budget_key in state_dict and hasattr(module, 'energy_budget'):
                module.energy_budget = state_dict[budget_key].item()
    
    # Check for unexpected keys
    model_keys = set()
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A'):
            model_keys.update([
                f"{name}.lora_A", f"{name}.lora_B", 
                f"{name}.current_rank", f"{name}.complexity_scores",
                f"{name}.energy_budget"
            ])
    
    unexpected_keys = [k for k in state_dict.keys() if k not in model_keys]
    
    if strict and (missing_keys or unexpected_keys):
        error_msg = []
        if missing_keys:
            error_msg.append(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            error_msg.append(f"Unexpected keys: {unexpected_keys}")
        raise RuntimeError(f"Error loading LoRAven state dict: {'; '.join(error_msg)}")
    
    if missing_keys:
        warnings.warn(f"Missing keys when loading LoRAven state: {missing_keys}")
    if unexpected_keys:
        warnings.warn(f"Unexpected keys when loading LoRAven state: {unexpected_keys}")


def validate_loraven_compatibility(model: nn.Module) -> Dict[str, Any]:
    """
    Validate LoRAven compatibility of a model
    
    Args:
        model: Model to validate
        
    Returns:
        Dictionary with compatibility information
    """
    info = {
        'is_compatible': True,
        'loraven_layers': 0,
        'total_layers': 0,
        'issues': []
    }
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            info['total_layers'] += 1
            
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                info['loraven_layers'] += 1
                
                # Check for required LoRAven components
                if not hasattr(module, 'current_rank'):
                    info['issues'].append(f"Layer {name} missing current_rank")
                    info['is_compatible'] = False
                    
                if not hasattr(module, 'rank_scheduler'):
                    info['issues'].append(f"Layer {name} missing rank_scheduler")
                    
                if not hasattr(module, 'complexity_scorer'):
                    info['issues'].append(f"Layer {name} missing complexity_scorer")
    
    info['coverage'] = info['loraven_layers'] / max(info['total_layers'], 1)
    
    return info


def get_memory_usage(model: nn.Module) -> Dict[str, float]:
    """
    Calculate memory usage of LoRAven components
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with memory usage information (in MB)
    """
    usage = {
        'base_model': 0.0,
        'lora_adapters': 0.0,
        'loraven_overhead': 0.0,
        'total': 0.0
    }
    
    for name, param in model.named_parameters():
        param_size = param.numel() * param.element_size() / (1024 * 1024)  # MB
        
        if 'lora_A' in name or 'lora_B' in name:
            usage['lora_adapters'] += param_size
        elif any(x in name for x in ['complexity_scores', 'energy_budget', 'rank_scheduler']):
            usage['loraven_overhead'] += param_size
        else:
            usage['base_model'] += param_size
    
    usage['total'] = sum(usage.values())
    
    return usage
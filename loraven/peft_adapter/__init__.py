"""
LoRAven PEFT Adapter

Provides PEFT (Parameter-Efficient Fine-Tuning) compatibility for LoRAven,
enabling seamless integration with the PEFT ecosystem.
"""

from .config import LoRAvenConfig
from .model import LoRAvenModel
from .utils import (
    merge_loraven_weights,
    unload_loraven_weights,
    get_loraven_state_dict,
    load_loraven_state_dict,
    validate_loraven_compatibility,
    get_memory_usage
)
from .registry import (
    register_loraven_peft,
    unregister_loraven_peft,
    is_loraven_registered,
    get_peft_compatibility_info,
    ensure_peft_compatibility,
    auto_register
)

# Auto-register LoRAven with PEFT when imported
auto_register()

__all__ = [
    'LoRAvenConfig',
    'LoRAvenModel', 
    'merge_loraven_weights',
    'unload_loraven_weights',
    'get_loraven_state_dict',
    'load_loraven_state_dict',
    'validate_loraven_compatibility',
    'get_memory_usage',
    'register_loraven_peft',
    'unregister_loraven_peft',
    'is_loraven_registered',
    'get_peft_compatibility_info',
    'ensure_peft_compatibility'
]
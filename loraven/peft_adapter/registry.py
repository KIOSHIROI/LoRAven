"""
PEFT Registry for LoRAven

Handles registration of LoRAven as a PEFT method, enabling seamless
integration with the PEFT ecosystem.
"""

import warnings
from typing import Dict, Any

try:
    from peft import PeftType
    from peft import PEFT_TYPE_TO_CONFIG_MAPPING
    from peft.mapping import PEFT_TYPE_TO_TUNER_MAPPING
    from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
    PEFT_AVAILABLE = True
except ImportError:
    warnings.warn("PEFT library not found. LoRAven PEFT integration will not be available.")
    PEFT_AVAILABLE = False
    
    # Create dummy classes for development
    class PeftType:
        pass
    
    PEFT_TYPE_TO_CONFIG_MAPPING = {}
    PEFT_TYPE_TO_TUNER_MAPPING = {}
    PEFT_TYPE_TO_MODEL_MAPPING = {}

from .config import LoRAvenConfig
from .model import LoRAvenModel


def register_loraven_with_peft():
    """
    Register LoRAven as a PEFT method
    
    This function adds LoRAven to the PEFT registry, allowing it to be used
    with standard PEFT APIs like get_peft_model().
    """
    return register_loraven_peft()


def register_loraven_peft():
    """
    Register LoRAven as a PEFT method
    
    This function adds LoRAven to the PEFT registry, allowing it to be used
    with standard PEFT APIs like get_peft_model().
    """
    if not PEFT_AVAILABLE:
        print("ℹ️  PEFT not available - registration skipped")
        return False
    
    try:
        # Create LORAVEN identifier as PeftType enum
        loraven_type = "LORAVEN"
        
        # Add LORAVEN to PeftType enum if not already present
        if not hasattr(PeftType, 'LORAVEN'):
            setattr(PeftType, 'LORAVEN', loraven_type)
            loraven_enum = getattr(PeftType, 'LORAVEN')
        else:
            loraven_enum = getattr(PeftType, 'LORAVEN')
        
        # Register config mapping using the enum value
        PEFT_TYPE_TO_CONFIG_MAPPING[loraven_enum] = LoRAvenConfig
        
        # Register tuner mapping using the enum value
        PEFT_TYPE_TO_TUNER_MAPPING[loraven_enum] = LoRAvenModel
        
        # Register model mapping using the enum value
        PEFT_TYPE_TO_MODEL_MAPPING[loraven_enum] = LoRAvenModel
        
        print("✅ LoRAven successfully registered with PEFT")
        return True
        
    except Exception as e:
        print(f"⚠️  Failed to register LoRAven with PEFT: {e}")
        return False


def unregister_loraven_peft():
    """
    Unregister LoRAven from PEFT (for cleanup/testing)
    """
    if not PEFT_AVAILABLE:
        return False
    
    try:
        # Remove from mappings
        if hasattr(PeftType, 'LORAVEN'):
            if PeftType.LORAVEN in PEFT_TYPE_TO_CONFIG_MAPPING:
                del PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.LORAVEN]
            
            if PeftType.LORAVEN in PEFT_TYPE_TO_MODEL_MAPPING:
                del PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORAVEN]
        
        print("✅ LoRAven unregistered from PEFT")
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to unregister LoRAven from PEFT: {e}")
        return False


def is_loraven_registered() -> bool:
    """
    Check if LoRAven is registered with PEFT
    
    Returns:
        bool: True if LoRAven is registered, False otherwise
    """
    if not PEFT_AVAILABLE:
        return False
    
    return (
        hasattr(PeftType, 'LORAVEN') and
        PeftType.LORAVEN in PEFT_TYPE_TO_CONFIG_MAPPING and
        PeftType.LORAVEN in PEFT_TYPE_TO_MODEL_MAPPING
    )


def get_peft_compatibility_info() -> Dict[str, Any]:
    """
    Get information about PEFT compatibility
    
    Returns:
        Dict containing compatibility information
    """
    info = {
        'peft_available': PEFT_AVAILABLE,
        'loraven_registered': is_loraven_registered(),
        'supported_features': []
    }
    
    if PEFT_AVAILABLE:
        info['supported_features'].extend([
            'get_peft_model',
            'PeftConfig',
            'save_pretrained',
            'load_adapter'
        ])
        
        if is_loraven_registered():
            info['supported_features'].extend([
                'dynamic_rank_adaptation',
                'energy_aware_optimization',
                'complexity_scoring'
            ])
    
    return info


# Compatibility layer for different PEFT versions
def ensure_peft_compatibility():
    """
    Ensure compatibility with different PEFT versions
    """
    if not PEFT_AVAILABLE:
        return False
    
    try:
        # Check PEFT version and adapt accordingly
        import peft
        peft_version = getattr(peft, '__version__', '0.0.0')
        
        # Handle version-specific compatibility issues
        major_version = int(peft_version.split('.')[0])
        
        if major_version >= 1:
            # Modern PEFT version - use standard registration
            return register_loraven_peft()
        else:
            # Legacy PEFT version - may need different approach
            warnings.warn(f"PEFT version {peft_version} may not be fully supported")
            return register_loraven_peft()
            
    except Exception as e:
        warnings.warn(f"PEFT compatibility check failed: {e}")
        return False


# Auto-registration hook
def auto_register():
    """
    Automatically register LoRAven when module is imported
    """
    if PEFT_AVAILABLE:
        success = ensure_peft_compatibility()
        if success:
            print("🚀 LoRAven PEFT adapter ready!")
        else:
            print("⚠️  LoRAven PEFT registration failed")
    else:
        print("ℹ️  PEFT not available - LoRAven will work in standalone mode")


# Register on import
if __name__ != "__main__":
    # Only auto-register when imported, not when run directly
    pass  # Will be called from __init__.py
# -*- coding: utf-8 -*-
"""
AEGIS-OS v5.0 Utilities Package
Mathematical, data processing, and visualization utilities
"""

__version__ = "5.0"
__author__ = "AEGIS-OS Development Team"
__description__ = "Utilities for orbital mechanics calculations, data processing, and visualization"

# Import all utility modules for convenient access
try:
    from . import math_utils
    from . import data_utils
    from . import visualization_utils
except ImportError as e:
    print(f"Warning: Could not import all utility modules: {e}")

__all__ = [
    'math_utils',
    'data_utils',
    'visualization_utils'
]


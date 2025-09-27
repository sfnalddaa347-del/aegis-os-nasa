# -*- coding: utf-8 -*-
"""
AEGIS-OS v5.0 Modules Package
Advanced Orbital Debris Intelligence & Sustainability Platform
"""

__version__ = "5.0"
__author__ = "AEGIS-OS Development Team"
__description__ = "Advanced orbital debris intelligence and sustainability platform with AI integration"

# Import all modules for convenient access
try:
    from . import constants
    from . import data_sources  
    from . import orbital_mechanics
    from . import atmospheric_models
    from . import ai_models
    from . import collision_detection
    from . import economic_analysis
    from . import compliance_monitor
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")

__all__ = [
    'constants',
    'data_sources',
    'orbital_mechanics', 
    'atmospheric_models',
    'ai_models',
    'collision_detection',
    'economic_analysis',
    'compliance_monitor'
]

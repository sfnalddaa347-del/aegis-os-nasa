# -*- coding: utf-8 -*-
"""
AEGIS-OS v5.0 Enhanced Constants
Advanced Orbital Debris Intelligence & Sustainability Platform
Scientific Constants and Parameters for High-Precision Orbital Mechanics
"""

import numpy as np

# ===========================
# FUNDAMENTAL PHYSICAL CONSTANTS (CODATA 2018)
# ===========================
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m³/kg/s²
SPEED_OF_LIGHT = 299792458.0  # m/s (exact)
STEFAN_BOLTZMANN_CONSTANT = 5.670374419e-8  # W/m²/K⁴
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s
ELEMENTARY_CHARGE = 1.602176634e-19  # C
AVOGADRO_CONSTANT = 6.02214076e23  # mol⁻¹

# ===========================
# EARTH SYSTEM PARAMETERS (WGS84 & EGM2008)
# ===========================
EARTH_GRAVITATIONAL_PARAMETER = 398600.4418  # km³/s² (EGM2008)
EARTH_RADIUS_EQUATORIAL = 6378.137  # km (WGS84)
EARTH_RADIUS_POLAR = 6356.752314245  # km (WGS84)
EARTH_RADIUS_MEAN = 6371.0  # km (volumetric mean)
EARTH_RADIUS = EARTH_RADIUS_MEAN  # Alias for backward compatibility
EARTH_MASS = 5.9722e24  # kg (IAU 2015)
EARTH_ANGULAR_VELOCITY = 7.2921159e-5  # rad/s (sidereal)
EARTH_FLATTENING = 1/298.257223563  # WGS84
EARTH_SURFACE_AREA = 5.10072e14  # m²
EARTH_VOLUME = 1.08321e21  # m³

# Earth Gravity Field Coefficients (EGM2008 complete to J8)
J2_EARTH = 1.08262668e-3    # Primary oblateness
J3_EARTH = -2.53265648e-6   # Pear-shaped component
J4_EARTH = -1.61996214e-6   # Higher-order oblateness
J5_EARTH = -2.27296082e-7   # Asymmetric component
J6_EARTH = 5.40681239e-7    # Additional higher-order term
J7_EARTH = -3.52640e-7      # J7 coefficient
J8_EARTH = 2.03046e-7       # J8 coefficient

# ===========================
# CELESTIAL BODY PARAMETERS (IAU 2015)
# ===========================
# Moon parameters
MOON_GRAVITATIONAL_PARAMETER = 4902.800  # km³/s²
MOON_MASS = 7.342e22  # kg
MOON_RADIUS = 1737.4  # km
MOON_MEAN_DISTANCE = 384400.0  # km
MOON_ORBITAL_PERIOD = 27.321661  # days (sidereal)
MOON_ECCENTRICITY = 0.0549  # orbital eccentricity

# Sun parameters
SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e11  # km³/s²
SUN_MASS = 1.98847e30  # kg
SUN_RADIUS = 695700.0  # km
AU = 149597870.7  # km (astronomical unit)
SOLAR_LUMINOSITY = 3.828e26  # W
SOLAR_TEMPERATURE = 5778  # K (effective temperature)
SOLAR_CONSTANT = 1361  # W/m² (solar irradiance at 1 AU)

# Planetary parameters for N-body calculations
PLANETARY_PARAMETERS = {
    'mercury': {'mass': 3.3011e23, 'distance_au': 0.387, 'mu': 22032},
    'venus': {'mass': 4.8675e24, 'distance_au': 0.723, 'mu': 324859},
    'mars': {'mass': 6.4171e23, 'distance_au': 1.524, 'mu': 42828},
    'jupiter': {'mass': 1.8982e27, 'distance_au': 5.204, 'mu': 126686534},
    'saturn': {'mass': 5.6834e26, 'distance_au': 9.573, 'mu': 37931187}
}

# ===========================
# ATMOSPHERIC MODEL PARAMETERS (NRLMSISE-00 Enhanced)
# ===========================
ATMOSPHERIC_SCALE_HEIGHT = 8500.0  # m (standard atmosphere)
SEA_LEVEL_DENSITY = 1.225  # kg/m³
SEA_LEVEL_PRESSURE = 101325.0  # Pa
STANDARD_TEMPERATURE = 288.15  # K (15°C)
MOLECULAR_WEIGHT_AIR = 28.9644  # g/mol
GAS_CONSTANT_AIR = 287.0  # J/(kg⋅K)

# NRLMSISE-00 Model Enhanced Parameters
NRLMSISE_MIN_ALTITUDE = 0.0  # km
NRLMSISE_MAX_ALTITUDE = 2500.0  # km (extended range)
NRLMSISE00_BASE_DENSITY = SEA_LEVEL_DENSITY  # kg/m³ baseline
THERMOSPHERE_BASE = 90.0  # km
EXOSPHERE_BASE = 600.0  # km
KARMAN_LINE = 100.0  # km (space boundary)

# Atmospheric composition by altitude
ATMOSPHERIC_COMPOSITION = {
    'sea_level': {'N2': 0.78084, 'O2': 0.20946, 'Ar': 0.00934, 'CO2': 0.000415},
    'thermosphere': {'N2': 0.20, 'O': 0.65, 'O2': 0.10, 'He': 0.05},
    'exosphere': {'H': 0.90, 'He': 0.10}
}

# Enhanced atmospheric layers with precise boundaries
ATMOSPHERIC_LAYERS = {
    'troposphere': (0, 12),      # km
    'stratosphere': (12, 50),    # km
    'mesosphere': (50, 90),      # km
    'thermosphere': (90, 600),   # km
    'exosphere': (600, 10000)    # km
}

# ===========================
# SPACE DEBRIS PARAMETERS (ORDEM 3.3 Enhanced)
# ===========================
MIN_TRACKABLE_DEBRIS_SIZE = 0.1  # cm (next-gen radar limit)
MIN_CATALOGED_DEBRIS_SIZE = 5.0  # cm (current SSN capability)
CRITICAL_DEBRIS_SIZE = 1.0  # cm (significant damage threshold)
LETHAL_DEBRIS_SIZE = 10.0  # cm (spacecraft mission kill)

# Enhanced debris size categories with NASA/ESA classifications
DEBRIS_SIZE_CATEGORIES = {
    'nano': (0.01, 0.1),      # cm - paint flecks, small particles
    'micro': (0.1, 1.0),      # cm - small fragments
    'small': (1.0, 10.0),     # cm - medium fragments
    'medium': (10.0, 100.0),  # cm - large fragments
    'large': (100.0, 1000.0), # cm - defunct satellites/upper stages
    'xlarge': (1000.0, 10000.0)  # cm - large spacecraft/stations
}

# Material properties database (enhanced)
MATERIAL_PROPERTIES = {
    'aluminum': {
        'density': 2700.0,  # kg/m³
        'melting_point': 933.47,  # K
        'thermal_conductivity': 237,  # W/(m⋅K)
        'specific_heat': 897,  # J/(kg⋅K)
        'yield_strength': 276e6,  # Pa
        'value_per_ton': 2000,  # USD
        'ballistic_coefficient': 0.15  # m²/kg
    },
    'titanium': {
        'density': 4500.0,
        'melting_point': 1941.0,
        'thermal_conductivity': 21.9,
        'specific_heat': 523,
        'yield_strength': 880e6,
        'value_per_ton': 8000,
        'ballistic_coefficient': 0.12
    },
    'steel': {
        'density': 7850.0,
        'melting_point': 1811.0,
        'thermal_conductivity': 50.2,
        'specific_heat': 490,
        'yield_strength': 250e6,
        'value_per_ton': 500,
        'ballistic_coefficient': 0.08
    },
    'carbon_fiber': {
        'density': 1600.0,
        'melting_point': 3800.0,
        'thermal_conductivity': 1000,
        'specific_heat': 710,
        'yield_strength': 3500e6,
        'value_per_ton': 15000,
        'ballistic_coefficient': 0.25
    },
    'copper': {
        'density': 8960.0,
        'melting_point': 1357.77,
        'thermal_conductivity': 401,
        'specific_heat': 385,
        'yield_strength': 70e6,
        'value_per_ton': 6000,
        'ballistic_coefficient': 0.07
    }
}

# Economic parameters (updated market values)
RECYCLING_EFFICIENCY = 0.942  # 94.2% advanced recycling
PROCESSING_COST_FACTOR = 0.30  # 30% of material value
TRANSPORTATION_COST_PER_KG = 5000  # USD to LEO
DEBRIS_REMOVAL_COST_PER_KG = 15000  # USD per kg removed

# ===========================
# ORBITAL ZONES (Enhanced Classification)
# ===========================
ORBITAL_ZONES = {
    'VEO': (100, 160),        # Very Low Earth Orbit
    'LEO': (160, 2000),       # Low Earth Orbit
    'MEO': (2000, 35786),     # Medium Earth Orbit
    'GEO': (35786, 35786),    # Geostationary Orbit
    'HEO': (35786, 100000),   # High Earth Orbit
    'LUNAR': (100000, 400000) # Cislunar space
}

# Critical orbital regions with enhanced coverage
CRITICAL_ORBITAL_REGIONS = {
    'ISS_REGION': (400, 420),        # International Space Station
    'STARLINK_SHELL_1': (540, 570),  # Starlink constellation shell 1
    'STARLINK_SHELL_2': (1110, 1130), # Starlink constellation shell 2
    'ONEWEB_SHELL': (1200, 1200),    # OneWeb constellation
    'GPS_SHELL': (20200, 20200),     # GPS constellation
    'GALILEO_SHELL': (23222, 23222), # Galileo constellation
    'GEO_BELT': (35786, 35786),      # Geostationary belt
    'GRAVEYARD_ORBIT': (36086, 36586) # GEO graveyard
}

# ===========================
# COLLISION AND RISK PARAMETERS (Enhanced)
# ===========================
CRITICAL_COLLISION_RISK = 0.001         # 0.1% probability threshold
KESSLER_THRESHOLD = 100000               # Critical debris count
COLLISION_ALERT_HOURS = 12               # Alert lead time
REENTRY_ALERT_HOURS = 24                 # Reentry warning time
MIN_CONJUNCTION_DISTANCE = 1.0           # km (close approach)
PROBABILITY_COLLISION_THRESHOLD = 1e-6   # Minimum reportable probability
HARD_BODY_RADIUS = 0.005                # km (5m combined object radius)

# Enhanced conjunction analysis parameters
CONJUNCTION_ANALYSIS_PARAMETERS = {
    'screening_volume': {
        'primary_threshold': 20.0,    # km (primary screening)
        'secondary_threshold': 5.0,   # km (secondary screening)
        'final_threshold': 1.0        # km (final screening)
    },
    'probability_calculation': {
        'monte_carlo_samples': 100000,  # MC iterations
        'sigma_multiplier': 3.0,        # Uncertainty bounds
        'covariance_scaling': 1.2       # Safety factor
    }
}

# ===========================
# SIMULATION PARAMETERS (High-Precision)
# ===========================
SIMULATION_TIME_STEP = 60                   # seconds (1 minute)
MAX_PROPAGATION_TIME = 86400 * 365 * 10     # 10 years in seconds
INTEGRATION_TOLERANCE = 1e-14                # Numerical integration tolerance
MAX_ITERATIONS_KEPLER = 200                  # Kepler equation solver iterations
CONVERGENCE_CRITERIA = 1e-12                 # General convergence tolerance

# Enhanced Monte Carlo parameters
MC_SIMULATION_PARAMETERS = {
    'sample_sizes': {
        'quick': 1000,
        'standard': 10000,
        'high_precision': 100000,
        'ultra_high': 1000000
    },
    'confidence_levels': [0.68, 0.90, 0.95, 0.99, 0.999],
    'parallel_processes': 8,
    'chunk_size': 1000
}

# Kalman filter parameters for enhanced tracking
KALMAN_FILTER_PARAMETERS = {
    'process_noise': 1e-8,          # Process noise covariance
    'measurement_noise': 1e-6,      # Measurement noise covariance
    'initial_uncertainty': 1e-4,    # Initial state uncertainty
    'innovation_gate': 9.21,        # Chi-square gate (99% confidence)
    'minimum_track_length': 3       # Minimum observations for track
}

# ===========================
# SOLAR ACTIVITY PARAMETERS (Enhanced)
# ===========================
SOLAR_FLUX_QUIET = 70                       # sfu (solar flux units)
SOLAR_FLUX_MODERATE = 100                   # sfu
SOLAR_FLUX_ACTIVE = 150                     # sfu
SOLAR_FLUX_STORM = 200                      # sfu
SOLAR_FLUX_EXTREME = 300                    # sfu

# Solar cycle parameters with historical data
SOLAR_CYCLE_PARAMETERS = {
    'average_period': 11.0,                 # years
    'minimum_period': 9.0,                  # years
    'maximum_period': 14.0,                 # years
    'amplitude_variation': 0.3,             # 30% variation
    'schwabe_cycle': 11.0,                  # Primary cycle
    'gleissberg_cycle': 87.0,               # Secondary cycle
    'suess_cycle': 210.0                    # Long-term cycle
}

# Geomagnetic indices with enhanced classification
GEOMAGNETIC_INDICES = {
    'quiet': {'kp_range': (0, 2), 'description': 'Quiet conditions'},
    'unsettled': {'kp_range': (2, 3), 'description': 'Unsettled conditions'},
    'active': {'kp_range': (3, 4), 'description': 'Active conditions'},
    'minor_storm': {'kp_range': (4, 5), 'description': 'Minor geomagnetic storm'},
    'moderate_storm': {'kp_range': (5, 6), 'description': 'Moderate storm'},
    'strong_storm': {'kp_range': (6, 7), 'description': 'Strong storm'},
    'severe_storm': {'kp_range': (7, 8), 'description': 'Severe storm'},
    'extreme_storm': {'kp_range': (8, 9), 'description': 'Extreme storm'}
}

# ===========================
# SENSOR AND OBSERVATION PARAMETERS
# ===========================
# Ground-based radar systems
RADAR_SYSTEMS = {
    'PFISR': {  # Poker Flat Incoherent Scatter Radar
        'frequency_mhz': 449,
        'power_kw': 2000,
        'min_detectable_size_cm': 1.0,
        'max_range_km': 2000
    },
    'ALTAIR': {  # ALTAIR radar at Kwajalein
        'frequency_mhz': 422,
        'power_mw': 6,
        'min_detectable_size_cm': 0.5,
        'max_range_km': 3000
    },
    'HAYSTACK': {  # Haystack radar
        'frequency_ghz': 10,
        'power_mw': 2,
        'min_detectable_size_cm': 0.3,
        'max_range_km': 5000
    }
}

# Space-based sensors
SPACE_SENSORS = {
    'optical_telescopes': {
        'limiting_magnitude': 18.0,
        'angular_resolution_arcsec': 0.1,
        'field_of_view_deg': 2.0,
        'temporal_resolution_s': 1.0
    },
    'lidar_systems': {
        'range_precision_m': 0.1,
        'velocity_precision_ms': 0.001,
        'max_range_km': 2000,
        'pulse_rate_hz': 20
    }
}

# ===========================
# COMPLIANCE AND REGULATORY PARAMETERS
# ===========================
# ISO 27852 Compliance Requirements (Updated 2023)
ISO_27852_REQUIREMENTS = {
    'LEO_disposal': {
        'max_lifetime_years': 25,           # Post-mission disposal time
        'disposal_altitude_km': 300,        # Minimum disposal altitude
        'collision_probability_max': 1e-4,  # Maximum collision risk
        'debris_release_max': 0,            # Zero intentional debris release
        'passivation_required': True        # Mandatory passivation
    },
    'MEO_disposal': {
        'max_lifetime_years': 100,          # Extended for MEO
        'graveyard_altitude_km': 300,       # Above operational region
        'collision_probability_max': 1e-4,
        'debris_release_max': 0,
        'passivation_required': True
    },
    'GEO_disposal': {
        'graveyard_altitude_km': 300,       # Above GEO (36,086 km)
        'station_keeping_margin_km': 75,    # Longitude slot margin
        'collision_probability_max': 1e-4,
        'debris_release_max': 0,
        'passivation_required': True
    }
}

# IADC Guidelines Parameters
IADC_GUIDELINES = {
    'design_requirements': {
        'explosion_prevention': True,
        'breakup_assessment': True,
        'collision_avoidance': True,
        'disposal_planning': True
    },
    'operational_requirements': {
        'tracking_accuracy_m': 100,
        'conjunction_screening': True,
        'maneuver_capability': True,
        'end_of_life_disposal': True
    }
}

# ===========================
# API AND DATA SOURCE PARAMETERS
# ===========================
API_ENDPOINTS = {
    'celestrak': {
        'base_url': 'https://celestrak.org/NORAD/elements/',
        'rate_limit_per_hour': 1000,
        'timeout_seconds': 30
    },
    'space_track': {
        'base_url': 'https://www.space-track.org/',
        'rate_limit_per_hour': 200,
        'timeout_seconds': 60
    },
    'noaa_space_weather': {
        'base_url': 'https://services.swpc.noaa.gov/',
        'rate_limit_per_hour': 3600,
        'timeout_seconds': 10
    },
    'esa_debris': {
        'base_url': 'https://sdup.esoc.esa.int/',
        'rate_limit_per_hour': 100,
        'timeout_seconds': 45
    }
}

# Data quality parameters
DATA_QUALITY_THRESHOLDS = {
    'tle_age_hours': 24,                # Maximum TLE age
    'position_accuracy_m': 1000,        # Required position accuracy
    'velocity_accuracy_ms': 10,         # Required velocity accuracy
    'covariance_determinant_max': 1e12,  # Maximum covariance determinant
    'innovation_threshold': 5.0         # Innovation threshold for outliers
}

# ===========================
# AI AND MACHINE LEARNING PARAMETERS
# ===========================
# Transformer model parameters
TRANSFORMER_PARAMETERS = {
    'sequence_length': 168,              # 7 days of hourly data
    'feature_dimensions': 12,            # Number of input features
    'hidden_dimensions': 256,            # Hidden layer size
    'num_attention_heads': 8,            # Multi-head attention
    'num_layers': 6,                     # Transformer layers
    'dropout_rate': 0.1,                 # Dropout for regularization
    'learning_rate': 1e-4,               # Adam optimizer learning rate
    'batch_size': 32                     # Training batch size
}

# Graph Neural Network parameters
GNN_PARAMETERS = {
    'node_features': 8,                  # Number of node features
    'edge_features': 4,                  # Number of edge features
    'hidden_dimensions': 128,            # Hidden layer dimensions
    'num_gnn_layers': 4,                 # GNN propagation layers
    'aggregation_method': 'mean',        # Node aggregation method
    'activation_function': 'relu',       # Activation function
    'regularization_weight': 1e-5        # L2 regularization
}

# ===========================
# PERFORMANCE AND OPTIMIZATION
# ===========================
# Computational limits for real-time processing
PERFORMANCE_LIMITS = {
    'max_objects_real_time': 10000,      # Max objects for real-time analysis
    'max_objects_batch': 100000,         # Max objects for batch processing
    'max_conjunction_pairs': 1000000,    # Max conjunction pairs
    'parallel_workers': 8,               # Number of parallel workers
    'memory_limit_gb': 16,               # Memory usage limit
    'computation_timeout_s': 300         # Maximum computation time
}

# Cache parameters
CACHE_PARAMETERS = {
    'tle_cache_hours': 6,                # TLE data cache duration
    'weather_cache_minutes': 30,         # Space weather cache duration
    'computation_cache_hours': 24,       # Computation results cache
    'visualization_cache_minutes': 15,   # Visualization cache
    'max_cache_size_mb': 1000           # Maximum cache size
}

# ===========================
# BACKUP CONSTANTS (Compatibility)
# ===========================
# Legacy constants for backward compatibility
ALUMINUM_VALUE_PER_TON = MATERIAL_PROPERTIES['aluminum']['value_per_ton']
TITANIUM_VALUE_PER_TON = MATERIAL_PROPERTIES['titanium']['value_per_ton']
CARBON_FIBER_VALUE_PER_TON = MATERIAL_PROPERTIES['carbon_fiber']['value_per_ton']

LEO_ALTITUDE_MIN = ORBITAL_ZONES['LEO'][0]
LEO_ALTITUDE_MAX = ORBITAL_ZONES['LEO'][1]

CELESTRAK_BASE_URL = API_ENDPOINTS['celestrak']['base_url']
NOAA_SPACE_WEATHER_URL = API_ENDPOINTS['noaa_space_weather']['base_url']
ESA_DEBRIS_URL = API_ENDPOINTS['esa_debris']['base_url']

# System version
AEGIS_VERSION = "5.0"
LAST_UPDATED = "2024-09-24"

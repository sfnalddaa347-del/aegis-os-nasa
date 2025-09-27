# -*- coding: utf-8 -*-
"""
Enhanced atmospheric models including NRLMSISE-00, drag calculations,
and solar radiation pressure with high-precision density calculations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import math
import logging
from .constants import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NRLMSISE00Model:
    """
    Enhanced NRLMSISE-00 atmospheric model for high-precision density calculations.
    Includes solar cycle effects, geomagnetic variations, and seasonal changes.
    """
    
    def __init__(self):
        self.model_version = "NRLMSISE-00 Enhanced"
        self.valid_altitude_range = (NRLMSISE_MIN_ALTITUDE, NRLMSISE_MAX_ALTITUDE)
        
        # Standard atmosphere parameters
        self.sea_level_density = SEA_LEVEL_DENSITY
        self.scale_height = ATMOSPHERIC_SCALE_HEIGHT
        self.standard_temp = STANDARD_TEMPERATURE
        
        # Atmospheric layer boundaries and properties
        self.atmospheric_layers = ATMOSPHERIC_LAYERS
        
        # Solar cycle model parameters
        self.solar_cycle_params = SOLAR_CYCLE_PARAMETERS
        
    def calculate_density(self, altitude_km: float, 
                         solar_flux_f107: float = 150,
                         solar_flux_f107_avg: float = 150,
                         geomagnetic_ap: float = 15,
                         local_solar_time: float = 12.0,
                         latitude_deg: float = 0.0,
                         longitude_deg: float = 0.0,
                         day_of_year: int = 180) -> float:
        """
        Calculate atmospheric density using enhanced NRLMSISE-00 model.
        
        Args:
            altitude_km: Altitude in kilometers
            solar_flux_f107: Current F10.7 solar flux
            solar_flux_f107_avg: 81-day average F10.7 solar flux
            geomagnetic_ap: Geomagnetic Ap index
            local_solar_time: Local solar time (hours)
            latitude_deg: Latitude in degrees
            longitude_deg: Longitude in degrees
            day_of_year: Day of year (1-365)
            
        Returns:
            Atmospheric density in kg/m³
        """
        try:
            # Validate inputs
            altitude_km = max(0, min(altitude_km, self.valid_altitude_range[1]))
            solar_flux_f107 = max(50, min(400, solar_flux_f107))
            
            if altitude_km > 2000:
                return 1e-15  # Near vacuum at very high altitudes
            
            # Base exponential atmosphere
            base_density = self._calculate_base_density(altitude_km)
            
            # Solar activity effects
            solar_factor = self._calculate_solar_effects(
                altitude_km, solar_flux_f107, solar_flux_f107_avg
            )
            
            # Geomagnetic effects
            geomagnetic_factor = self._calculate_geomagnetic_effects(
                altitude_km, geomagnetic_ap, latitude_deg
            )
            
            # Diurnal variations
            diurnal_factor = self._calculate_diurnal_variations(
                altitude_km, local_solar_time, latitude_deg, day_of_year
            )
            
            # Seasonal variations
            seasonal_factor = self._calculate_seasonal_variations(
                altitude_km, day_of_year, latitude_deg
            )
            
            # Combine all effects
            total_density = (base_density * solar_factor * geomagnetic_factor * 
                           diurnal_factor * seasonal_factor)
            
            # Apply minimum density floor
            return max(total_density, 1e-15)
            
        except Exception as e:
            logger.error(f"Error calculating atmospheric density: {e}")
            return self._fallback_density(altitude_km)
    
    def _calculate_base_density(self, altitude_km: float) -> float:
        """Calculate base atmospheric density using layer-specific models."""
        try:
            # Determine atmospheric layer
            layer = self._get_atmospheric_layer(altitude_km)
            
            if layer == 'troposphere':
                return self._troposphere_density(altitude_km)
            elif layer == 'stratosphere':
                return self._stratosphere_density(altitude_km)
            elif layer == 'mesosphere':
                return self._mesosphere_density(altitude_km)
            elif layer == 'thermosphere':
                return self._thermosphere_density(altitude_km)
            else:  # exosphere
                return self._exosphere_density(altitude_km)
                
        except Exception as e:
            logger.error(f"Error in base density calculation: {e}")
            return self._simple_exponential_density(altitude_km)
    
    def _get_atmospheric_layer(self, altitude_km: float) -> str:
        """Determine atmospheric layer for given altitude."""
        for layer, (min_alt, max_alt) in self.atmospheric_layers.items():
            if min_alt <= altitude_km < max_alt:
                return layer
        return 'exosphere'  # Default for very high altitudes
    
    def _troposphere_density(self, altitude_km: float) -> float:
        """Density calculation for troposphere (0-12 km)."""
        # Linear temperature profile
        temp_lapse_rate = -6.5e-3  # K/m
        temperature = self.standard_temp + temp_lapse_rate * altitude_km * 1000
        
        # Hydrostatic equilibrium
        exponent = -GRAVITATIONAL_CONSTANT * EARTH_MASS * 1000 * altitude_km / (GAS_CONSTANT_AIR * self.standard_temp)
        density = self.sea_level_density * np.exp(exponent)
        
        return density
    
    def _stratosphere_density(self, altitude_km: float) -> float:
        """Density calculation for stratosphere (12-50 km)."""
        # Temperature inversion in stratosphere
        base_altitude = 12.0  # km
        base_density = self._troposphere_density(base_altitude)
        
        # Simplified stratospheric profile
        scale_height_strat = 7000  # m
        altitude_diff = (altitude_km - base_altitude) * 1000  # Convert to meters
        
        density = base_density * np.exp(-altitude_diff / scale_height_strat)
        return density
    
    def _mesosphere_density(self, altitude_km: float) -> float:
        """Density calculation for mesosphere (50-90 km)."""
        base_altitude = 50.0  # km
        base_density = self._stratosphere_density(base_altitude)
        
        # Mesospheric scale height
        scale_height_meso = 6000  # m
        altitude_diff = (altitude_km - base_altitude) * 1000
        
        density = base_density * np.exp(-altitude_diff / scale_height_meso)
        return density
    
    def _thermosphere_density(self, altitude_km: float) -> float:
        """Enhanced density calculation for thermosphere (90-600 km)."""
        base_altitude = 90.0  # km
        base_density = 3.396e-6  # kg/m³ at 90 km
        
        # Temperature-dependent scale height
        exospheric_temp = 1000  # K (standard)
        temperature = self._calculate_thermosphere_temperature(altitude_km, exospheric_temp)
        
        # Variable scale height
        scale_height = GAS_CONSTANT_AIR * temperature / (GRAVITATIONAL_CONSTANT * EARTH_MASS / (EARTH_RADIUS * 1000 + altitude_km * 1000)**2)
        
        # Barometric formula with variable scale height
        altitude_diff = (altitude_km - base_altitude) * 1000
        density = base_density * np.exp(-altitude_diff / scale_height)
        
        return density
    
    def _exosphere_density(self, altitude_km: float) -> float:
        """Density calculation for exosphere (>600 km)."""
        base_altitude = 600.0  # km
        base_density = 1e-12  # kg/m³ at 600 km
        
        # Very large scale height in exosphere
        scale_height_exo = 50000  # m
        altitude_diff = (altitude_km - base_altitude) * 1000
        
        density = base_density * np.exp(-altitude_diff / scale_height_exo)
        return max(density, 1e-15)
    
    def _calculate_thermosphere_temperature(self, altitude_km: float, exospheric_temp: float) -> float:
        """Calculate temperature in thermosphere."""
        base_altitude = 120.0  # km
        base_temp = 355.0  # K at 120 km
        
        if altitude_km <= base_altitude:
            return base_temp
        
        # Asymptotic approach to exospheric temperature
        temp_factor = 1 - np.exp(-(altitude_km - base_altitude) / 50.0)
        temperature = base_temp + (exospheric_temp - base_temp) * temp_factor
        
        return temperature
    
    def _simple_exponential_density(self, altitude_km: float) -> float:
        """Fallback simple exponential atmosphere model."""
        return self.sea_level_density * np.exp(-altitude_km * 1000 / self.scale_height)
    
    def _calculate_solar_effects(self, altitude_km: float, f107: float, f107_avg: float) -> float:
        """Calculate solar activity effects on atmospheric density."""
        try:
            # Solar flux effect is strongest in thermosphere
            if altitude_km < 90:
                return 1.0  # Minimal solar effect below thermosphere
            
            # Reference solar flux
            f107_ref = 150.0
            
            # Solar effect coefficients (altitude-dependent)
            if altitude_km < 200:
                solar_coeff = 0.2
            elif altitude_km < 400:
                solar_coeff = 0.4
            elif altitude_km < 600:
                solar_coeff = 0.6
            else:
                solar_coeff = 0.8
            
            # Current and average solar flux effects
            current_effect = 1.0 + solar_coeff * (f107 - f107_ref) / f107_ref
            avg_effect = 1.0 + 0.5 * solar_coeff * (f107_avg - f107_ref) / f107_ref
            
            # Combine effects
            solar_factor = (current_effect + avg_effect) / 2
            
            return max(0.5, min(3.0, solar_factor))
            
        except Exception as e:
            logger.error(f"Error calculating solar effects: {e}")
            return 1.0
    
    def _calculate_geomagnetic_effects(self, altitude_km: float, ap_index: float, latitude_deg: float) -> float:
        """Calculate geomagnetic activity effects on density."""
        try:
            # Geomagnetic effects primarily in polar regions and high altitudes
            if altitude_km < 150:
                return 1.0
            
            # Latitude dependence (stronger at poles)
            lat_factor = 1.0 + 0.3 * (abs(latitude_deg) / 90.0)**2
            
            # Ap index effect
            ap_ref = 15.0  # Quiet conditions
            ap_effect = 1.0 + 0.1 * (ap_index - ap_ref) / ap_ref
            
            # Altitude dependence
            alt_factor = min(1.0, (altitude_km - 150) / 350)  # Effect increases with altitude
            
            geomag_factor = 1.0 + alt_factor * lat_factor * (ap_effect - 1.0)
            
            return max(0.8, min(2.0, geomag_factor))
            
        except Exception as e:
            logger.error(f"Error calculating geomagnetic effects: {e}")
            return 1.0
    
    def _calculate_diurnal_variations(self, altitude_km: float, local_solar_time: float, 
                                    latitude_deg: float, day_of_year: int) -> float:
        """Calculate day/night density variations."""
        try:
            if altitude_km < 120:
                return 1.0  # Minimal diurnal effect below 120 km
            
            # Diurnal amplitude (altitude and latitude dependent)
            base_amplitude = 0.3  # 30% variation
            alt_factor = min(1.0, (altitude_km - 120) / 280)  # Effect increases with altitude
            lat_factor = np.cos(np.radians(latitude_deg))  # Stronger at equator
            
            amplitude = base_amplitude * alt_factor * lat_factor
            
            # Solar zenith angle approximation
            hour_angle = 15.0 * (local_solar_time - 12.0)  # degrees
            
            # Simple cosine variation
            diurnal_factor = 1.0 + amplitude * np.cos(np.radians(hour_angle))
            
            return max(0.7, min(1.3, diurnal_factor))
            
        except Exception as e:
            logger.error(f"Error calculating diurnal variations: {e}")
            return 1.0
    
    def _calculate_seasonal_variations(self, altitude_km: float, day_of_year: int, latitude_deg: float) -> float:
        """Calculate seasonal density variations."""
        try:
            if altitude_km < 200:
                return 1.0  # Minimal seasonal effect below 200 km
            
            # Seasonal amplitude
            base_amplitude = 0.15  # 15% variation
            alt_factor = min(1.0, (altitude_km - 200) / 400)
            
            # Hemispheric effect
            if latitude_deg >= 0:  # Northern hemisphere
                phase_shift = 0
            else:  # Southern hemisphere
                phase_shift = np.pi
            
            # Annual variation
            annual_phase = 2 * np.pi * (day_of_year - 80) / 365.25 + phase_shift  # Peak around day 80 (March 21)
            seasonal_factor = 1.0 + base_amplitude * alt_factor * np.cos(annual_phase)
            
            return max(0.85, min(1.15, seasonal_factor))
            
        except Exception as e:
            logger.error(f"Error calculating seasonal variations: {e}")
            return 1.0
    
    def _fallback_density(self, altitude_km: float) -> float:
        """Fallback density calculation if main model fails."""
        return max(1e-15, self.sea_level_density * np.exp(-altitude_km * 1000 / self.scale_height))
    
    def get_density_profile(self, altitude_range: Tuple[float, float], 
                           num_points: int = 100, **kwargs) -> pd.DataFrame:
        """Generate atmospheric density profile over altitude range."""
        try:
            min_alt, max_alt = altitude_range
            altitudes = np.linspace(min_alt, max_alt, num_points)
            
            densities = []
            temperatures = []
            pressures = []
            
            for alt in altitudes:
                density = self.calculate_density(alt, **kwargs)
                densities.append(density)
                
                # Approximate temperature and pressure
                layer = self._get_atmospheric_layer(alt)
                if layer == 'thermosphere':
                    temp = self._calculate_thermosphere_temperature(alt, 1000)
                else:
                    temp = max(180, self.standard_temp - 6.5e-3 * alt * 1000)  # Simple lapse rate
                
                temperatures.append(temp)
                
                # Hydrostatic pressure
                if alt == 0:
                    pressure = SEA_LEVEL_PRESSURE
                else:
                    pressure = density * GAS_CONSTANT_AIR * temp
                pressures.append(pressure)
            
            return pd.DataFrame({
                'altitude_km': altitudes,
                'density_kg_m3': densities,
                'temperature_k': temperatures,
                'pressure_pa': pressures,
                'scale_height_km': [self.scale_height / 1000] * len(altitudes)
            })
            
        except Exception as e:
            logger.error(f"Error generating density profile: {e}")
            return pd.DataFrame()

class AtmosphericDragModel:
    """Enhanced atmospheric drag model with detailed coefficient calculations."""
    
    def __init__(self):
        self.atmospheric_model = NRLMSISE00Model()
        
        # Drag coefficient database for different object types
        self.drag_coefficients = {
            'sphere': 2.2,
            'cube': 2.1,
            'cylinder': 1.3,
            'flat_plate': 2.0,
            'complex_spacecraft': 2.2,
            'debris_fragment': 2.0
        }
    
    def calculate_drag_acceleration(self, position_eci: np.ndarray, 
                                  velocity_eci: np.ndarray,
                                  cross_sectional_area: float,
                                  mass: float,
                                  drag_coefficient: float = 2.2,
                                  space_weather: Optional[Dict] = None) -> np.ndarray:
        """
        Calculate atmospheric drag acceleration with high precision.
        
        Args:
            position_eci: Position vector in ECI frame [km]
            velocity_eci: Velocity vector in ECI frame [km/s]
            cross_sectional_area: Cross-sectional area [m²]
            mass: Object mass [kg]
            drag_coefficient: Drag coefficient
            space_weather: Current space weather conditions
            
        Returns:
            Drag acceleration vector [km/s²]
        """
        try:
            # Calculate altitude
            altitude_km = np.linalg.norm(position_eci) - EARTH_RADIUS
            
            if altitude_km > 1000 or altitude_km < 80:
                return np.zeros(3)  # No significant drag outside this range
            
            # Get atmospheric density
            if space_weather:
                f107 = space_weather.get('solar_flux_f107', 150)
                f107_avg = space_weather.get('solar_flux_f107_adj', f107)
                ap = space_weather.get('ap_index', 15)
            else:
                f107 = f107_avg = 150
                ap = 15
            
            # Calculate position-dependent parameters
            lat, lon, _ = self._eci_to_geodetic(position_eci)
            
            density = self.atmospheric_model.calculate_density(
                altitude_km, f107, f107_avg, ap,
                latitude_deg=np.degrees(lat),
                longitude_deg=np.degrees(lon)
            )
            
            # Atmospheric velocity (Earth rotation effect)
            atmospheric_velocity = self._calculate_atmospheric_velocity(position_eci)
            
            # Relative velocity
            relative_velocity = velocity_eci - atmospheric_velocity
            relative_speed = np.linalg.norm(relative_velocity)
            
            if relative_speed == 0:
                return np.zeros(3)
            
            # Drag force calculation
            # F_drag = -0.5 * ρ * C_d * A * V_rel * |V_rel|
            drag_force_magnitude = 0.5 * density * drag_coefficient * cross_sectional_area * relative_speed**2
            drag_force_direction = -relative_velocity / relative_speed
            
            # Convert to acceleration [km/s²]
            drag_acceleration = (drag_force_magnitude / mass) * drag_force_direction * 1e-3  # Convert m/s² to km/s²
            
            return drag_acceleration
            
        except Exception as e:
            logger.error(f"Error calculating drag acceleration: {e}")
            return np.zeros(3)
    
    def _eci_to_geodetic(self, position_eci: np.ndarray) -> Tuple[float, float, float]:
        """Convert ECI position to geodetic coordinates."""
        try:
            x, y, z = position_eci
            
            # Longitude
            longitude = np.arctan2(y, x)
            
            # Latitude (simplified spherical Earth)
            r_magnitude = np.linalg.norm(position_eci)
            latitude = np.arcsin(z / r_magnitude)
            
            # Altitude
            altitude = r_magnitude - EARTH_RADIUS
            
            return latitude, longitude, altitude
            
        except Exception as e:
            logger.error(f"Error in ECI to geodetic conversion: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_atmospheric_velocity(self, position_eci: np.ndarray) -> np.ndarray:
        """Calculate atmospheric velocity due to Earth rotation."""
        try:
            # Earth rotation vector (z-axis)
            omega_earth = np.array([0, 0, EARTH_ANGULAR_VELOCITY])  # rad/s
            
            # Atmospheric velocity = ω × r
            atmospheric_velocity = np.cross(omega_earth, position_eci)
            
            return atmospheric_velocity
            
        except Exception as e:
            logger.error(f"Error calculating atmospheric velocity: {e}")
            return np.zeros(3)
    
    def estimate_orbital_lifetime(self, initial_altitude: float,
                                 ballistic_coefficient: float,
                                 solar_activity_level: str = 'moderate') -> float:
        """
        Estimate orbital lifetime due to atmospheric drag.
        
        Args:
            initial_altitude: Initial altitude [km]
            ballistic_coefficient: B = m/(C_d * A) [kg/m²]
            solar_activity_level: Solar activity level
            
        Returns:
            Orbital lifetime in days
        """
        try:
            if initial_altitude > 800:
                return 365 * 100  # Very long lifetime above 800 km
            
            if initial_altitude < 200:
                return 1  # Very short lifetime below 200 km
            
            # Solar activity density enhancement
            density_multipliers = {
                'low': 0.7,
                'moderate': 1.0,
                'high': 1.5,
                'very_high': 2.5
            }
            
            density_factor = density_multipliers.get(solar_activity_level.lower(), 1.0)
            
            # Empirical lifetime model (King-Hele approximation)
            # Lifetime ∝ (ballistic_coefficient) / (density * altitude_factor)
            
            base_density = self.atmospheric_model.calculate_density(initial_altitude)
            effective_density = base_density * density_factor
            
            # Scale height and altitude factor
            scale_height_km = ATMOSPHERIC_SCALE_HEIGHT / 1000
            altitude_factor = np.exp(-initial_altitude / scale_height_km)
            
            # Lifetime calculation (empirical formula)
            if initial_altitude < 300:
                # Exponential decay regime
                lifetime_days = ballistic_coefficient * 100 / (effective_density * 1e6 * altitude_factor)
            else:
                # Power law regime
                lifetime_days = (ballistic_coefficient / 50) * (initial_altitude / 300)**3 / density_factor
            
            # Clamp to reasonable bounds
            return max(1, min(365 * 100, lifetime_days))
            
        except Exception as e:
            logger.error(f"Error estimating orbital lifetime: {e}")
            return 365  # Default 1 year

class SolarRadiationPressureModel:
    """Solar radiation pressure model for high-precision orbit determination."""
    
    def __init__(self):
        self.solar_constant = SOLAR_CONSTANT  # W/m² at 1 AU
        self.speed_of_light = SPEED_OF_LIGHT  # m/s
        self.au_km = AU  # km
    
    def calculate_srp_acceleration(self, position_eci: np.ndarray,
                                 cross_sectional_area: float,
                                 mass: float,
                                 reflectivity: float = 0.3,
                                 sun_position: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate solar radiation pressure acceleration.
        
        Args:
            position_eci: Satellite position in ECI frame [km]
            cross_sectional_area: Cross-sectional area [m²]
            mass: Satellite mass [kg]
            reflectivity: Surface reflectivity (0-1)
            sun_position: Sun position in ECI frame [km] (optional)
            
        Returns:
            SRP acceleration vector [km/s²]
        """
        try:
            # Default sun position (simplified - along x-axis at 1 AU)
            if sun_position is None:
                sun_position = np.array([self.au_km, 0, 0])
            
            # Vector from satellite to sun
            sun_satellite_vector = sun_position - position_eci
            sun_distance = np.linalg.norm(sun_satellite_vector)
            
            if sun_distance == 0:
                return np.zeros(3)
            
            # Unit vector towards sun
            sun_direction = sun_satellite_vector / sun_distance
            
            # Check if satellite is in Earth's shadow
            if self._is_in_shadow(position_eci, sun_position):
                return np.zeros(3)
            
            # Solar flux at satellite position
            solar_flux = self.solar_constant * (self.au_km / sun_distance)**2  # W/m²
            
            # Radiation pressure
            radiation_pressure = solar_flux / self.speed_of_light  # N/m²
            
            # SRP force
            # F = P * A * (1 + ρ) * cos(θ)
            # For simplicity, assume θ = 0 (normal incidence)
            srp_force = radiation_pressure * cross_sectional_area * (1 + reflectivity)
            
            # SRP acceleration towards sun
            srp_acceleration_ms2 = (srp_force / mass) * sun_direction  # m/s²
            
            # Convert to km/s²
            srp_acceleration = srp_acceleration_ms2 * 1e-3
            
            return srp_acceleration
            
        except Exception as e:
            logger.error(f"Error calculating SRP acceleration: {e}")
            return np.zeros(3)
    
    def _is_in_shadow(self, satellite_pos: np.ndarray, sun_pos: np.ndarray) -> bool:
        """Check if satellite is in Earth's shadow (simplified cylindrical model)."""
        try:
            # Vector from Earth to satellite
            earth_sat_vector = satellite_pos
            
            # Vector from Earth to Sun
            earth_sun_vector = sun_pos
            earth_sun_distance = np.linalg.norm(earth_sun_vector)
            
            if earth_sun_distance == 0:
                return False
            
            # Unit vector towards sun
            sun_direction = earth_sun_vector / earth_sun_distance
            
            # Project satellite position onto sun direction
            projection_length = np.dot(earth_sat_vector, sun_direction)
            
            # If projection is negative, satellite is on sunlit side
            if projection_length < 0:
                return False
            
            # Calculate perpendicular distance from satellite to Earth-Sun line
            projection_vector = projection_length * sun_direction
            perpendicular_vector = earth_sat_vector - projection_vector
            perpendicular_distance = np.linalg.norm(perpendicular_vector)
            
            # Check if within Earth's shadow cylinder
            earth_radius_km = EARTH_RADIUS
            return perpendicular_distance < earth_radius_km
            
        except Exception as e:
            logger.error(f"Error checking shadow condition: {e}")
            return False
    
    def calculate_srp_perturbations(self, orbital_elements: Dict,
                                   object_properties: Dict,
                                   simulation_days: int = 30) -> Dict:
        """Calculate long-term SRP orbital perturbations."""
        try:
            # Semi-major axis change due to SRP
            a = orbital_elements['semi_major_axis']
            e = orbital_elements['eccentricity']
            
            area_to_mass = object_properties['cross_sectional_area'] / object_properties['mass']
            reflectivity = object_properties.get('reflectivity', 0.3)
            
            # SRP parameter
            beta = (self.solar_constant / self.speed_of_light) * area_to_mass * (1 + reflectivity) * 1e-3  # km/s²
            
            # Mean motion
            n = np.sqrt(EARTH_GRAVITATIONAL_PARAMETER / a**3)  # rad/s
            
            # Secular changes per orbit (Vallado)
            # These are simplified expressions for circular orbits
            da_dt = 2 * beta * a**2 / EARTH_GRAVITATIONAL_PARAMETER  # Change in semi-major axis
            
            # Calculate changes over simulation period
            orbits_per_day = n * 86400 / (2 * np.pi)
            total_orbits = orbits_per_day * simulation_days
            
            delta_a = da_dt * total_orbits
            
            return {
                'delta_semi_major_axis': delta_a,
                'delta_altitude': delta_a,
                'srp_parameter_beta': beta,
                'orbital_period_change_s': (2 * np.pi / n - 2 * np.pi / np.sqrt(EARTH_GRAVITATIONAL_PARAMETER / (a + delta_a)**3)),
                'total_orbits': total_orbits
            }
            
        except Exception as e:
            logger.error(f"Error calculating SRP perturbations: {e}")
            return {}

# Global instances for easy access
nrlmsise_model = NRLMSISE00Model()
drag_model = AtmosphericDragModel()
srp_model = SolarRadiationPressureModel()

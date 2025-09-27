# -*- coding: utf-8 -*-
"""
Advanced atmospheric models for space debris analysis
NRLMSISE-00, drag calculations, and atmospheric density modeling
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import math
from modules.constants import *

class NRLMSISE00AtmosphericModel:
    """
    Enhanced NRLMSISE-00 atmospheric model implementation.
    Includes solar activity effects and geomagnetic influences.
    """
    
    def __init__(self):
        self.model_name = "NRLMSISE-00 Enhanced"
        self.valid_altitude_range = (0, 2000)  # km
        
        # Atmospheric composition parameters
        self.species_data = {
            'N2': {'molecular_mass': 28.014, 'fraction': 0.781},
            'O2': {'molecular_mass': 31.998, 'fraction': 0.209},
            'O': {'molecular_mass': 15.999, 'fraction': 0.0},
            'He': {'molecular_mass': 4.003, 'fraction': 0.0},
            'H': {'molecular_mass': 1.008, 'fraction': 0.0},
        }
        
        # Temperature profile coefficients
        self.temp_coefficients = {
            'troposphere': {'lapse_rate': -6.5e-3, 'h_base': 0, 'T_base': 288.15},
            'stratosphere': {'lapse_rate': 1.0e-3, 'h_base': 11, 'T_base': 216.65},
            'mesosphere': {'lapse_rate': -2.8e-3, 'h_base': 20, 'T_base': 216.65},
            'thermosphere': {'lapse_rate': 0, 'h_base': 47, 'T_base': 270.65}
        }
    
    def calculate_density(self, altitude_km: float, 
                         solar_flux_f107: float = 150,
                         solar_flux_f107_avg: float = 150,
                         geomagnetic_ap: float = 15,
                         local_solar_time: float = 12.0,
                         day_of_year: int = 80,
                         latitude_deg: float = 0.0,
                         longitude_deg: float = 0.0) -> Dict[str, float]:
        """
        Calculate atmospheric density using enhanced NRLMSISE-00 model.
        
        Args:
            altitude_km: Altitude in kilometers
            solar_flux_f107: Current 10.7 cm solar flux
            solar_flux_f107_avg: 81-day average solar flux
            geomagnetic_ap: Geomagnetic activity index
            local_solar_time: Local solar time (hours)
            day_of_year: Day of year (1-365)
            latitude_deg: Latitude in degrees
            longitude_deg: Longitude in degrees
            
        Returns:
            Dictionary with atmospheric parameters
        """
        if altitude_km < 0 or altitude_km > 2000:
            raise ValueError(f"Altitude {altitude_km} km outside valid range {self.valid_altitude_range}")
        
        # Base density calculation
        base_density = self._calculate_base_density(altitude_km)
        
        # Solar activity corrections
        solar_correction = self._calculate_solar_correction(
            altitude_km, solar_flux_f107, solar_flux_f107_avg
        )
        
        # Geomagnetic corrections
        geomagnetic_correction = self._calculate_geomagnetic_correction(
            altitude_km, geomagnetic_ap
        )
        
        # Diurnal variations
        diurnal_correction = self._calculate_diurnal_correction(
            altitude_km, local_solar_time, latitude_deg
        )
        
        # Seasonal variations
        seasonal_correction = self._calculate_seasonal_correction(
            altitude_km, day_of_year, latitude_deg
        )
        
        # Combined density
        total_density = (base_density * solar_correction * geomagnetic_correction * 
                        diurnal_correction * seasonal_correction)
        
        # Temperature calculation
        temperature = self._calculate_temperature(
            altitude_km, solar_flux_f107, geomagnetic_ap
        )
        
        # Pressure calculation
        pressure = self._calculate_pressure(altitude_km, total_density, temperature)
        
        # Scale height
        scale_height = self._calculate_scale_height(temperature, altitude_km)
        
        # Composition
        composition = self._calculate_composition(altitude_km, temperature)
        
        return {
            'total_density_kg_m3': total_density,
            'temperature_k': temperature,
            'pressure_pa': pressure,
            'scale_height_km': scale_height,
            'composition': composition,
            'solar_correction_factor': solar_correction,
            'geomagnetic_correction_factor': geomagnetic_correction,
            'diurnal_correction_factor': diurnal_correction,
            'seasonal_correction_factor': seasonal_correction
        }
    
    def _calculate_base_density(self, altitude_km: float) -> float:
        """Calculate base atmospheric density using exponential model."""
        if altitude_km <= 0:
            return NRLMSISE00_BASE_DENSITY
        
        # Multi-layer exponential atmosphere
        if altitude_km <= 100:
            # Lower atmosphere - exponential decay
            scale_height = 8.5  # km
            return NRLMSISE00_BASE_DENSITY * np.exp(-altitude_km / scale_height)
        
        elif altitude_km <= 500:
            # Thermosphere - slower decay with temperature dependence
            h0 = 100
            rho0 = NRLMSISE00_BASE_DENSITY * np.exp(-h0 / 8.5)
            
            # Temperature-dependent scale height
            temp = 1000 + 600 * (1 - np.exp(-(altitude_km - 100) / 50))
            scale_height = 16 + temp / 50  # km
            
            return rho0 * np.exp(-(altitude_km - h0) / scale_height)
        
        else:
            # High altitude - very low density
            h0 = 500
            rho0 = self._calculate_base_density(h0)
            scale_height = 60  # km
            
            return rho0 * np.exp(-(altitude_km - h0) / scale_height)
    
    def _calculate_solar_correction(self, altitude_km: float, 
                                  f107: float, f107_avg: float) -> float:
        """Calculate solar activity correction factor."""
        # Solar flux influence increases with altitude
        altitude_factor = min(1.0, altitude_km / 400)
        
        # Current vs average flux
        flux_ratio = f107 / 150  # Normalized to typical value
        avg_flux_ratio = f107_avg / 150
        
        # Combined solar influence (empirical model)
        solar_effect = 1.0 + altitude_factor * (
            0.3 * (flux_ratio - 1.0) + 0.1 * (avg_flux_ratio - 1.0)
        )
        
        return max(0.3, min(5.0, solar_effect))
    
    def _calculate_geomagnetic_correction(self, altitude_km: float, ap: float) -> float:
        """Calculate geomagnetic activity correction factor."""
        # Geomagnetic effects stronger at higher altitudes
        altitude_factor = min(1.0, max(0, (altitude_km - 200) / 300))
        
        # Ap index effect (logarithmic)
        ap_effect = 1.0 + altitude_factor * 0.1 * np.log10(max(1, ap / 15))
        
        return max(0.5, min(3.0, ap_effect))
    
    def _calculate_diurnal_correction(self, altitude_km: float, 
                                    local_solar_time: float, latitude_deg: float) -> float:
        """Calculate diurnal (daily) variation correction."""
        # Diurnal effects stronger in thermosphere
        if altitude_km < 100:
            return 1.0
        
        altitude_factor = min(1.0, (altitude_km - 100) / 400)
        
        # Solar heating cycle (peak at 14:00 local time)
        phase = 2 * np.pi * (local_solar_time - 14) / 24
        diurnal_amplitude = 0.3 * altitude_factor
        
        # Latitude dependence
        lat_factor = np.cos(np.radians(latitude_deg))
        
        diurnal_correction = 1.0 + diurnal_amplitude * lat_factor * np.cos(phase)
        
        return max(0.5, min(2.0, diurnal_correction))
    
    def _calculate_seasonal_correction(self, altitude_km: float, 
                                     day_of_year: int, latitude_deg: float) -> float:
        """Calculate seasonal variation correction."""
        if altitude_km < 100:
            return 1.0
        
        # Seasonal phase (maximum around day 100 in Northern Hemisphere)
        phase = 2 * np.pi * (day_of_year - 100) / 365
        
        # Hemisphere and latitude effects
        hemisphere_factor = np.sign(latitude_deg) if latitude_deg != 0 else 1
        lat_amplitude = 0.15 * abs(np.sin(np.radians(latitude_deg)))
        
        altitude_factor = min(1.0, (altitude_km - 100) / 400)
        
        seasonal_correction = 1.0 + altitude_factor * lat_amplitude * hemisphere_factor * np.sin(phase)
        
        return max(0.7, min(1.5, seasonal_correction))
    
    def _calculate_temperature(self, altitude_km: float, 
                             solar_flux: float, geomagnetic_ap: float) -> float:
        """Calculate atmospheric temperature."""
        if altitude_km <= 11:
            # Troposphere
            return 288.15 - 6.5 * altitude_km
        
        elif altitude_km <= 20:
            # Stratosphere (lower)
            return 216.65
        
        elif altitude_km <= 32:
            # Stratosphere (upper)
            return 216.65 + (altitude_km - 20) * 1.0
        
        elif altitude_km <= 47:
            # Stratosphere (top)
            return 228.65 + (altitude_km - 32) * 2.8
        
        elif altitude_km <= 100:
            # Mesosphere
            T_base = 270.65
            return T_base - 2.8 * (altitude_km - 47)
        
        else:
            # Thermosphere - temperature increases with solar activity
            T_base = 180 + 2 * altitude_km  # Base temperature profile
            
            # Solar heating
            solar_effect = (solar_flux - 150) * 2.0
            geomagnetic_effect = (geomagnetic_ap - 15) * 10.0
            
            temperature = T_base + solar_effect + geomagnetic_effect
            
            return max(180, min(2000, temperature))
    
    def _calculate_pressure(self, altitude_km: float, density: float, temperature: float) -> float:
        """Calculate atmospheric pressure using ideal gas law."""
        # Average molecular mass of air
        M_avg = 28.97e-3  # kg/mol
        R = 8.314  # J/(mol·K)
        
        # Pressure from ideal gas law
        pressure = density * R * temperature / M_avg
        
        return pressure
    
    def _calculate_scale_height(self, temperature: float, altitude_km: float) -> float:
        """Calculate atmospheric scale height."""
        # Constants
        R = 8.314  # J/(mol·K)
        M_avg = 28.97e-3  # kg/mol (average molecular mass)
        g = 9.81 * (6371 / (6371 + altitude_km))**2  # Gravity at altitude
        
        scale_height = R * temperature / (M_avg * g) / 1000  # Convert to km
        
        return scale_height
    
    def _calculate_composition(self, altitude_km: float, temperature: float) -> Dict[str, float]:
        """Calculate atmospheric composition by altitude."""
        composition = {}
        
        if altitude_km <= 100:
            # Well-mixed atmosphere
            composition = {
                'N2': 0.781,
                'O2': 0.209,
                'Ar': 0.0093,
                'CO2': 0.0004,
                'other': 0.0003
            }
        
        elif altitude_km <= 200:
            # Dissociation begins
            o2_fraction = 0.209 * np.exp(-(altitude_km - 100) / 20)
            o_fraction = 0.209 - o2_fraction
            
            composition = {
                'N2': 0.781,
                'O2': o2_fraction,
                'O': o_fraction,
                'Ar': 0.0093 * np.exp(-(altitude_km - 100) / 15),
                'other': 0.0007
            }
        
        else:
            # High altitude - atomic oxygen dominates
            total_o = 0.6
            n2_fraction = 0.781 * np.exp(-(altitude_km - 200) / 50)
            
            composition = {
                'O': total_o,
                'N2': n2_fraction,
                'He': min(0.2, 0.01 * np.exp((altitude_km - 200) / 100)),
                'H': min(0.1, 0.001 * np.exp((altitude_km - 500) / 200)),
                'other': max(0, 1 - total_o - n2_fraction)
            }
        
        return composition

class AtmosphericDragModel:
    """
    Advanced atmospheric drag modeling for orbital debris.
    """
    
    def __init__(self):
        self.atmosphere_model = NRLMSISE00AtmosphericModel()
    
    def calculate_drag_acceleration(self, position: np.ndarray, velocity: np.ndarray,
                                  cross_sectional_area: float, mass: float,
                                  drag_coefficient: float = 2.2,
                                  space_weather: Optional[Dict] = None) -> np.ndarray:
        """
        Calculate atmospheric drag acceleration.
        
        Args:
            position: Position vector in ECI frame [km]
            velocity: Velocity vector in ECI frame [km/s]
            cross_sectional_area: Cross-sectional area [m²]
            mass: Object mass [kg]
            drag_coefficient: Drag coefficient (typical: 2.2)
            space_weather: Current space weather conditions
            
        Returns:
            Drag acceleration vector [km/s²]
        """
        # Calculate altitude
        altitude_km = np.linalg.norm(position) - EARTH_RADIUS
        
        if altitude_km > 1000:  # Negligible drag above 1000 km
            return np.zeros(3)
        
        # Get atmospheric density
        if space_weather:
            atm_data = self.atmosphere_model.calculate_density(
                altitude_km,
                solar_flux_f107=space_weather.get('solar_flux_f107', 150),
                geomagnetic_ap=space_weather.get('kp_index', 3) * 3  # Convert Kp to Ap
            )
        else:
            atm_data = self.atmosphere_model.calculate_density(altitude_km)
        
        density = atm_data['total_density_kg_m3']
        
        # Relative velocity (subtract Earth's rotation)
        earth_rotation_vector = np.array([0, 0, 7.2921159e-5])  # rad/s
        position_unit = position / np.linalg.norm(position)
        rotation_velocity = np.cross(earth_rotation_vector, position)
        
        relative_velocity = velocity - rotation_velocity  # km/s
        relative_speed = np.linalg.norm(relative_velocity)
        
        if relative_speed == 0:
            return np.zeros(3)
        
        # Drag force calculation
        drag_force_magnitude = 0.5 * density * (relative_speed * 1000)**2 * cross_sectional_area * drag_coefficient
        
        # Drag acceleration (opposite to velocity direction)
        velocity_unit = relative_velocity / relative_speed
        drag_acceleration = -drag_force_magnitude * velocity_unit / mass / 1000  # Convert to km/s²
        
        return drag_acceleration
    
    def estimate_orbital_lifetime(self, initial_altitude_km: float,
                                 cross_sectional_area: float, mass: float,
                                 solar_activity_level: str = "moderate") -> float:
        """
        Estimate orbital lifetime due to atmospheric drag.
        
        Args:
            initial_altitude_km: Initial orbital altitude
            cross_sectional_area: Cross-sectional area [m²]
            mass: Object mass [kg]
            solar_activity_level: Solar activity level
            
        Returns:
            Estimated lifetime in years
        """
        if initial_altitude_km > 1000:
            return 100  # Very long lifetime above 1000 km
        
        # Ballistic coefficient
        ballistic_coefficient = mass / (2.2 * cross_sectional_area)  # kg/m²
        
        # Solar activity factors
        activity_factors = {
            "low": 0.7,
            "moderate": 1.0,
            "high": 1.8,
            "very_high": 3.0
        }
        
        solar_factor = activity_factors.get(solar_activity_level.lower(), 1.0)
        
        # Empirical lifetime model (based on NASA studies)
        if initial_altitude_km <= 300:
            base_lifetime_days = 10 * np.exp(initial_altitude_km / 40)
        elif initial_altitude_km <= 600:
            base_lifetime_days = 365 * np.exp((initial_altitude_km - 300) / 80)
        else:
            base_lifetime_days = 365 * 25 * np.exp((initial_altitude_km - 600) / 150)
        
        # Adjust for ballistic coefficient
        bc_factor = ballistic_coefficient / 100  # Normalized factor
        
        # Adjust for solar activity
        lifetime_days = base_lifetime_days * bc_factor / solar_factor
        
        return max(0.001, lifetime_days / 365.25)  # Convert to years
    
    def calculate_reentry_heating(self, velocity_km_s: float, altitude_km: float,
                                cross_sectional_area: float, mass: float) -> Dict[str, float]:
        """
        Calculate heating during atmospheric reentry.
        
        Args:
            velocity_km_s: Reentry velocity [km/s]
            altitude_km: Current altitude [km]
            cross_sectional_area: Cross-sectional area [m²]
            mass: Object mass [kg]
            
        Returns:
            Dictionary with heating parameters
        """
        if altitude_km > 120:
            return {'heat_flux': 0, 'temperature': 300, 'survival_probability': 1.0}
        
        # Get atmospheric density
        atm_data = self.atmosphere_model.calculate_density(altitude_km)
        density = atm_data['total_density_kg_m3']
        
        # Stagnation point heat flux (Detra-Kemp-Riddell correlation)
        velocity_m_s = velocity_km_s * 1000
        heat_flux = 1.7415e-4 * np.sqrt(density) * (velocity_m_s)**3.15  # W/m²
        
        # Surface temperature (simplified radiative equilibrium)
        stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        emissivity = 0.8  # Typical for metals
        
        # Radiative cooling
        if heat_flux > 0:
            surface_temperature = (heat_flux / (emissivity * stefan_boltzmann))**(1/4)
        else:
            surface_temperature = 300  # Ambient temperature
        
        # Material survival assessment
        melting_points = {
            'aluminum': 933,  # K
            'steel': 1811,
            'titanium': 1941,
            'carbon_fiber': 3800
        }
        
        # Assume aluminum for survival calculation
        melting_point = melting_points['aluminum']
        survival_probability = max(0, 1 - (surface_temperature - melting_point) / melting_point)
        
        return {
            'heat_flux_w_m2': heat_flux,
            'surface_temperature_k': surface_temperature,
            'survival_probability': max(0, min(1, survival_probability)),
            'altitude_km': altitude_km,
            'density_kg_m3': density
        }

class SolarRadiationPressureModel:
    """
    Solar radiation pressure modeling for orbital debris.
    """
    
    def __init__(self):
        self.solar_constant = 1361  # W/m² at 1 AU
        self.speed_of_light = 299792458  # m/s
        self.au_km = 149597870.7  # km
    
    def calculate_srp_acceleration(self, position: np.ndarray, 
                                 cross_sectional_area: float, mass: float,
                                 reflectivity: float = 0.3,
                                 sun_position: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate solar radiation pressure acceleration.
        
        Args:
            position: Satellite position in ECI frame [km]
            cross_sectional_area: Cross-sectional area [m²]
            mass: Object mass [kg]
            reflectivity: Surface reflectivity (0-1)
            sun_position: Sun position vector [km] (if None, uses simplified model)
            
        Returns:
            SRP acceleration vector [km/s²]
        """
        # Simplified sun position (assumes circular orbit)
        if sun_position is None:
            # Use current date to estimate sun direction
            day_of_year = datetime.now().timetuple().tm_yday
            sun_angle = 2 * np.pi * day_of_year / 365.25
            sun_position = self.au_km * np.array([np.cos(sun_angle), np.sin(sun_angle), 0])
        
        # Vector from satellite to sun
        sun_vector = sun_position - position
        sun_distance = np.linalg.norm(sun_vector)
        sun_unit = sun_vector / sun_distance
        
        # Check if satellite is in Earth's shadow
        earth_satellite_vector = -position
        shadow_angle = np.arccos(np.dot(sun_unit, earth_satellite_vector / np.linalg.norm(earth_satellite_vector)))
        
        earth_angular_radius = np.arcsin(EARTH_RADIUS / np.linalg.norm(position))
        
        if shadow_angle < earth_angular_radius:
            return np.zeros(3)  # In shadow, no SRP
        
        # Solar flux at satellite distance
        solar_flux = self.solar_constant * (self.au_km / sun_distance)**2  # W/m²
        
        # Radiation pressure
        radiation_pressure = solar_flux / self.speed_of_light  # N/m²
        
        # SRP force (accounting for reflection and absorption)
        absorption_factor = 1 - reflectivity
        reflection_factor = 2 * reflectivity
        
        total_factor = absorption_factor + reflection_factor
        srp_force = radiation_pressure * cross_sectional_area * total_factor  # N
        
        # SRP acceleration
        srp_acceleration = srp_force * sun_unit / mass / 1000  # km/s²
        
        return srp_acceleration

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
        self.valid_altitude_range = (0, 2500)  # km (extended range)
        
        # Atmospheric composition parameters
        self.species_data = {
            'N2': {'molecular_mass': 28.014, 'fraction': 0.781},
            'O2': {'molecular_mass': 31.998, 'fraction': 0.209},
            'O': {'molecular_mass': 15.999, 'fraction': 0.0},
            'He': {'molecular_mass': 4.003, 'fraction': 0.0},
            'H': {'molecular_mass': 1.008, 'fraction': 0.0},
        }
        
        # Enhanced temperature profile coefficients
        self.temp_coefficients = {
            'troposphere': {'lapse_rate': -6.5e-3, 'h_base': 0, 'T_base': 288.15},
            'stratosphere_lower': {'lapse_rate': 0.0, 'h_base': 11, 'T_base': 216.65},
            'stratosphere_upper': {'lapse_rate': 1.0e-3, 'h_base': 20, 'T_base': 216.65},
            'stratopause': {'lapse_rate': 2.8e-3, 'h_base': 32, 'T_base': 228.65},
            'mesosphere': {'lapse_rate': -2.8e-3, 'h_base': 47, 'T_base': 270.65},
            'mesopause': {'lapse_rate': 0, 'h_base': 86, 'T_base': 186.87},
            'thermosphere': {'lapse_rate': 0, 'h_base': 91, 'T_base': 186.87}
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
        if altitude_km < 0 or altitude_km > self.valid_altitude_range[1]:
            raise ValueError(f"Altitude {altitude_km} km outside valid range {self.valid_altitude_range}")
        
        # Base density calculation with multi-layer approach
        base_density = self._calculate_base_density(altitude_km)
        
        # Solar activity corrections
        solar_correction = self._calculate_solar_correction(
            altitude_km, solar_flux_f107, solar_flux_f107_avg
        )
        
        # Geomagnetic corrections
        geomagnetic_correction = self._calculate_geomagnetic_correction(
            altitude_km, geomagnetic_ap
        )
        
        # Diurnal variations (day/night effects)
        diurnal_correction = self._calculate_diurnal_correction(
            altitude_km, local_solar_time, latitude_deg
        )
        
        # Seasonal variations
        seasonal_correction = self._calculate_seasonal_correction(
            altitude_km, day_of_year, latitude_deg
        )
        
        # Semi-annual variations
        semiannual_correction = self._calculate_semiannual_correction(
            altitude_km, day_of_year
        )
        
        # Combined density
        total_density = (base_density * solar_correction * geomagnetic_correction * 
                        diurnal_correction * seasonal_correction * semiannual_correction)
        
        # Temperature calculation with enhanced modeling
        temperature = self._calculate_temperature(
            altitude_km, solar_flux_f107, geomagnetic_ap, local_solar_time, day_of_year
        )
        
        # Pressure calculation using hydrostatic equilibrium
        pressure = self._calculate_pressure(altitude_km, total_density, temperature)
        
        # Scale height calculation
        scale_height = self._calculate_scale_height(temperature, altitude_km)
        
        # Detailed atmospheric composition by altitude
        composition = self._calculate_composition(altitude_km, temperature, solar_flux_f107)
        
        # Mean molecular weight
        mean_molecular_weight = self._calculate_mean_molecular_weight(composition)
        
        return {
            'total_density_kg_m3': total_density,
            'temperature_k': temperature,
            'pressure_pa': pressure,
            'scale_height_km': scale_height,
            'composition': composition,
            'mean_molecular_weight': mean_molecular_weight,
            'solar_correction_factor': solar_correction,
            'geomagnetic_correction_factor': geomagnetic_correction,
            'diurnal_correction_factor': diurnal_correction,
            'seasonal_correction_factor': seasonal_correction,
            'semiannual_correction_factor': semiannual_correction,
            'model_version': self.model_name
        }
    
    def _calculate_base_density(self, altitude_km: float) -> float:
        """Calculate base atmospheric density using enhanced exponential model."""
        if altitude_km <= 0:
            return SEA_LEVEL_DENSITY
        
        # Multi-layer exponential atmosphere with realistic transitions
        if altitude_km <= 90:
            # Lower atmosphere - use standard atmosphere model
            if altitude_km <= 11:  # Troposphere
                T = 288.15 - 6.5 * altitude_km
                P = 101325 * (T / 288.15) ** 5.256
            elif altitude_km <= 20:  # Lower stratosphere
                T = 216.65
                P = 22632 * np.exp(-0.1577 * (altitude_km - 11))
            elif altitude_km <= 32:  # Upper stratosphere
                T = 216.65 + (altitude_km - 20)
                P = 5474.9 * (T / 216.65) ** (-34.163)
            else:  # Stratopause to mesosphere
                scale_height = 8.5 * (1 + altitude_km / 100)  # Variable scale height
                P = SEA_LEVEL_PRESSURE * np.exp(-altitude_km * 1000 / scale_height)
            
            # Convert pressure to density using ideal gas law
            T = max(180, 288.15 - 6.5 * min(altitude_km, 11))
            density = P * MOLECULAR_WEIGHT_AIR / (GAS_CONSTANT_AIR * 1000 * T)
            return density
            
        elif altitude_km <= 500:
            # Thermosphere - exponential decay with temperature effects
            h0 = 90
            rho0 = self._calculate_base_density(h0)
            
            # Temperature-dependent scale height in thermosphere
            temp_exo = self._calculate_exospheric_temperature(altitude_km)
            scale_height = (GAS_CONSTANT_AIR * temp_exo) / (9.81 * MOLECULAR_WEIGHT_AIR / 1000)
            scale_height_km = scale_height / 1000
            
            return rho0 * np.exp(-(altitude_km - h0) / scale_height_km)
        
        else:
            # Exosphere - very low density with H and He dominance
            h0 = 500
            rho0 = self._calculate_base_density(h0)
            scale_height = 50 + altitude_km / 20  # Increasing scale height
            
            return rho0 * np.exp(-(altitude_km - h0) / scale_height)
    
    def _calculate_exospheric_temperature(self, altitude_km: float) -> float:
        """Calculate exospheric temperature based on solar activity."""
        if altitude_km <= 120:
            return 180 + (altitude_km - 90) * 10  # Linear increase
        else:
            # Asymptotic approach to exospheric temperature
            T_inf = 1000  # Exospheric temperature (can vary with solar activity)
            return T_inf * (1 - np.exp(-(altitude_km - 120) / 70))
    
    def _calculate_solar_correction(self, altitude_km: float, 
                                  f107: float, f107_avg: float) -> float:
        """Enhanced solar activity correction factor."""
        # Solar flux influence increases exponentially with altitude above 200 km
        if altitude_km < 200:
            altitude_factor = 0.1 * (altitude_km / 200) ** 2
        else:
            altitude_factor = min(1.0, 0.1 + 0.9 * (altitude_km - 200) / 300)
        
        # Immediate and averaged flux effects
        f107_ref = 150  # Reference solar flux
        immediate_effect = (f107 / f107_ref - 1) * 0.4
        averaged_effect = (f107_avg / f107_ref - 1) * 0.2
        
        # Combined solar influence with altitude weighting
        solar_effect = 1.0 + altitude_factor * (immediate_effect + averaged_effect)
        
        return max(0.3, min(5.0, solar_effect))
    
    def _calculate_geomagnetic_correction(self, altitude_km: float, ap: float) -> float:
        """Enhanced geomagnetic activity correction factor."""
        # Geomagnetic effects stronger at higher altitudes and during storms
        if altitude_km < 200:
            altitude_factor = 0
        else:
            altitude_factor = min(1.0, (altitude_km - 200) / 300)
        
        # Ap index effect (enhanced during storms)
        ap_ref = 15
        if ap > 30:  # Storm conditions
            ap_effect = 1.0 + altitude_factor * 0.3 * np.log10(ap / ap_ref)
        else:
            ap_effect = 1.0 + altitude_factor * 0.1 * np.log10(max(1, ap / ap_ref))
        
        return max(0.5, min(4.0, ap_effect))
    
    def _calculate_diurnal_correction(self, altitude_km: float, 
                                    local_solar_time: float, latitude_deg: float) -> float:
        """Enhanced diurnal variation correction."""
        if altitude_km < 120:
            return 1.0
        
        altitude_factor = min(1.0, (altitude_km - 120) / 300)
        
        # Enhanced solar heating cycle with realistic phase
        phase = 2 * np.pi * (local_solar_time - 14) / 24  # Peak at 14:00 LT
        
        # Latitude-dependent amplitude
        lat_factor = np.cos(np.radians(latitude_deg)) ** 0.5
        
        # Altitude-dependent amplitude
        diurnal_amplitude = 0.15 + 0.25 * altitude_factor  # Up to 40% variation
        
        diurnal_correction = 1.0 + diurnal_amplitude * lat_factor * np.cos(phase)
        
        return max(0.4, min(2.5, diurnal_correction))
    
    def _calculate_seasonal_correction(self, altitude_km: float, 
                                     day_of_year: int, latitude_deg: float) -> float:
        """Enhanced seasonal variation correction."""
        if altitude_km < 100:
            return 1.0
        
        # Seasonal phase with proper timing for each hemisphere
        if latitude_deg >= 0:  # Northern hemisphere
            phase = 2 * np.pi * (day_of_year - 100) / 365  # Maximum around April
        else:  # Southern hemisphere
            phase = 2 * np.pi * (day_of_year - 280) / 365  # Maximum around October
        
        # Latitude-dependent amplitude
        lat_amplitude = 0.1 + 0.1 * abs(np.sin(np.radians(latitude_deg)))
        
        # Altitude-dependent enhancement
        altitude_factor = min(1.0, (altitude_km - 100) / 400)
        
        seasonal_correction = 1.0 + altitude_factor * lat_amplitude * np.sin(phase)
        
        return max(0.7, min(1.4, seasonal_correction))
    
    def _calculate_semiannual_correction(self, altitude_km: float, day_of_year: int) -> float:
        """Semi-annual density variation (important for thermosphere)."""
        if altitude_km < 200:
            return 1.0
        
        # Semi-annual oscillation with maxima around equinoxes
        phase = 2 * np.pi * (day_of_year - 80) / 182.5  # Period = 6 months
        
        altitude_factor = min(1.0, (altitude_km - 200) / 300)
        amplitude = 0.08 * altitude_factor  # Up to 8% variation
        
        semiannual_correction = 1.0 + amplitude * np.cos(phase)
        
        return max(0.9, min(1.1, semiannual_correction))
    
    def _calculate_temperature(self, altitude_km: float, solar_flux: float, 
                             geomagnetic_ap: float, local_solar_time: float, 
                             day_of_year: int) -> float:
        """Enhanced temperature calculation with comprehensive effects."""
        if altitude_km <= 11:
            # Troposphere
            return 288.15 - 6.5 * altitude_km
        elif altitude_km <= 20:
            # Lower stratosphere (isothermal)
            return 216.65
        elif altitude_km <= 32:
            # Upper stratosphere
            return 216.65 + (altitude_km - 20) * 1.0
        elif altitude_km <= 47:
            # Stratopause
            return 228.65 + (altitude_km - 32) * 2.8
        elif altitude_km <= 86:
            # Mesosphere
            return 270.65 - 2.8 * (altitude_km - 47)
        else:
            # Thermosphere with comprehensive modeling
            T_base = 186.87  # Temperature at mesopause
            
            # Exospheric temperature (varies with solar activity)
            T_inf = 500 + (solar_flux - 150) * 5 + (geomagnetic_ap - 15) * 20
            T_inf = max(500, min(2000, T_inf))
            
            # Exponential approach to exospheric temperature
            temp = T_inf - (T_inf - T_base) * np.exp(-(altitude_km - 86) / 60)
            
            # Add diurnal variation in thermosphere
            if altitude_km > 200:
                diurnal_var = 50 * np.cos(2 * np.pi * (local_solar_time - 14) / 24)
                temp += diurnal_var
            
            return max(T_base, min(T_inf, temp))
    
    def _calculate_pressure(self, altitude_km: float, density: float, temperature: float) -> float:
        """Calculate atmospheric pressure using ideal gas law and hydrostatic equilibrium."""
        # Average molecular mass varies with altitude
        if altitude_km < 100:
            M_avg = MOLECULAR_WEIGHT_AIR / 1000  # kg/mol (well-mixed atmosphere)
        else:
            # Composition changes with altitude
            M_avg = self._calculate_average_molecular_mass(altitude_km) / 1000
        
        # Ideal gas law: P = ρRT/M
        R = 8.314  # J/(mol·K)
        pressure = density * R * temperature / M_avg
        
        return pressure
    
    def _calculate_scale_height(self, temperature: float, altitude_km: float) -> float:
        """Calculate atmospheric scale height with variable molecular mass."""
        R = 8.314  # J/(mol·K)
        M_avg = self._calculate_average_molecular_mass(altitude_km) / 1000  # kg/mol
        
        # Gravity varies with altitude
        g = 9.81 * (EARTH_RADIUS / (EARTH_RADIUS + altitude_km)) ** 2
        
        scale_height = R * temperature / (M_avg * g) / 1000  # Convert to km
        
        return scale_height
    
    def _calculate_average_molecular_mass(self, altitude_km: float) -> float:
        """Calculate altitude-dependent average molecular mass."""
        if altitude_km < 100:
            return MOLECULAR_WEIGHT_AIR  # Well-mixed atmosphere
        elif altitude_km < 500:
            # Transition region with O2 dissociation
            base_mass = MOLECULAR_WEIGHT_AIR
            atomic_o_mass = 16.0
            # Linear interpolation between mixed air and atomic oxygen
            factor = min(1.0, (altitude_km - 100) / 200)
            return base_mass * (1 - factor) + atomic_o_mass * factor
        else:
            # High altitude - dominated by light species (He, H)
            return 8.0  # Approximate for He-H mixture
    
    def _calculate_composition(self, altitude_km: float, temperature: float,
                             solar_flux: float) -> Dict[str, float]:
        """Calculate detailed atmospheric composition by altitude."""
        composition = {}
        
        if altitude_km <= 100:
            # Well-mixed atmosphere (homosphere)
            composition = {
                'N2': 0.781,
                'O2': 0.209,
                'Ar': 0.0093,
                'CO2': 0.0004,
                'other_trace': 0.0003
            }
        
        elif altitude_km <= 200:
            # Heterosphere - molecular oxygen dissociation begins
            dissociation_factor = min(1.0, (altitude_km - 100) / 100)
            
            # O2 dissociates to atomic O
            o2_remaining = 0.209 * np.exp(-dissociation_factor * 2)
            atomic_o = 0.209 - o2_remaining
            
            # N2 remains mostly intact
            n2_fraction = 0.781 * np.exp(-dissociation_factor * 0.5)
            
            composition = {
                'N2': n2_fraction,
                'O2': o2_remaining,
                'O': atomic_o,
                'Ar': 0.0093 * np.exp(-dissociation_factor),
                'other': max(0, 1 - n2_fraction - o2_remaining - atomic_o - 0.0093 * np.exp(-dissociation_factor))
            }
        
        elif altitude_km <= 600:
            # Thermosphere - atomic oxygen dominates
            o_fraction = 0.5 + 0.3 * np.exp(-(altitude_km - 200) / 100)
            n2_fraction = 0.3 * np.exp(-(altitude_km - 200) / 150)
            he_fraction = min(0.15, 0.01 * np.exp((altitude_km - 400) / 100))
            
            composition = {
                'O': o_fraction,
                'N2': n2_fraction,
                'He': he_fraction,
                'other': max(0, 1 - o_fraction - n2_fraction - he_fraction)
            }
        
        else:
            # Exosphere - helium and hydrogen dominance
            he_fraction = 0.4 * np.exp(-(altitude_km - 600) / 200)
            h_fraction = min(0.6, 0.1 * np.exp((altitude_km - 800) / 300))
            
            composition = {
                'He': he_fraction,
                'H': h_fraction,
                'O': max(0, 0.8 - he_fraction - h_fraction),
                'other': max(0, 1 - he_fraction - h_fraction - max(0, 0.8 - he_fraction - h_fraction))
            }
        
        return composition
    
    def _calculate_mean_molecular_weight(self, composition: Dict[str, float]) -> float:
        """Calculate mean molecular weight from composition."""
        molecular_weights = {
            'N2': 28.014, 'O2': 31.998, 'O': 15.999, 'Ar': 39.948,
            'He': 4.003, 'H': 1.008, 'CO2': 44.01, 'other': 28.0, 'other_trace': 28.0
        }
        
        mean_weight = 0
        for species, fraction in composition.items():
            if species in molecular_weights:
                mean_weight += fraction * molecular_weights[species]
        
        return mean_weight


class AtmosphericDragModel:
    """
    Advanced atmospheric drag modeling for orbital debris.
    Includes variable drag coefficients and solar activity effects.
    """
    
    def __init__(self):
        self.atmosphere_model = NRLMSISE00AtmosphericModel()
    
    def calculate_drag_acceleration(self, position: np.ndarray, velocity: np.ndarray,
                                  cross_sectional_area: float, mass: float,
                                  drag_coefficient: float = 2.2,
                                  space_weather: Optional[Dict] = None) -> np.ndarray:
        """
        Calculate atmospheric drag acceleration with enhanced modeling.
        
        Args:
            position: Position vector in ECI frame [km]
            velocity: Velocity vector in ECI frame [km/s]
            cross_sectional_area: Cross-sectional area [m²]
            mass: Object mass [kg]
            drag_coefficient: Drag coefficient (default 2.2 for debris)
            space_weather: Current space weather conditions
            
        Returns:
            Drag acceleration vector [km/s²]
        """
        # Calculate altitude
        altitude_km = np.linalg.norm(position) - EARTH_RADIUS
        
        if altitude_km < 80:  # Below significant atmosphere
            return np.zeros(3)
        
        # Get atmospheric conditions
        if space_weather:
            atm_data = self.atmosphere_model.calculate_density(
                altitude_km=altitude_km,
                solar_flux_f107=space_weather.get('solar_flux_f107', 150),
                geomagnetic_ap=space_weather.get('kp_index', 3) * 3.33  # Convert Kp to Ap approximately
            )
        else:
            atm_data = self.atmosphere_model.calculate_density(altitude_km)
        
        density = atm_data['total_density_kg_m3']
        
        if density <= 1e-15:  # Negligible atmosphere
            return np.zeros(3)
        
        # Calculate relative velocity (atmospheric co-rotation)
        earth_rotation = np.array([0, 0, EARTH_ANGULAR_VELOCITY])
        rotation_velocity = np.cross(earth_rotation, position)  # km/s
        relative_velocity = velocity - rotation_velocity
        
        # Drag force calculation
        v_rel_mag = np.linalg.norm(relative_velocity)
        
        if v_rel_mag == 0:
            return np.zeros(3)
        
        # Enhanced drag coefficient based on flow regime
        enhanced_cd = self._calculate_enhanced_drag_coefficient(
            drag_coefficient, altitude_km, v_rel_mag, atm_data['temperature_k']
        )
        
        # Drag acceleration (convert units appropriately)
        drag_acceleration = -(0.5 * density * enhanced_cd * cross_sectional_area / mass * 
                            v_rel_mag * relative_velocity * 1e-6)  # Convert to km/s²
        
        return drag_acceleration
    
    def _calculate_enhanced_drag_coefficient(self, base_cd: float, altitude_km: float,
                                           velocity_magnitude: float, temperature_k: float) -> float:
        """
        Calculate enhanced drag coefficient based on flow regime and atmospheric conditions.
        """
        # Molecular flow regime at very high altitudes
        if altitude_km > 600:
            # Molecular flow - different physics
            return base_cd * 1.5  # Increased drag in molecular flow
        
        elif altitude_km > 300:
            # Transition regime
            transition_factor = 1 + 0.5 * (altitude_km - 300) / 300
            return base_cd * transition_factor
        
        else:
            # Continuum flow regime with Reynolds number effects
            # This is a simplified model - real implementation would be more complex
            return base_cd


class SolarRadiationPressureModel:
    """
    Solar radiation pressure modeling for space debris.
    """
    
    def __init__(self):
        self.solar_constant = 1361  # W/m² at 1 AU
        self.speed_of_light = 299792458  # m/s
    
    def calculate_srp_acceleration(self, position: np.ndarray, 
                                 cross_sectional_area: float, mass: float,
                                 reflectivity: float = 0.3) -> np.ndarray:
        """
        Calculate solar radiation pressure acceleration.
        
        Args:
            position: Position vector in ECI frame [km]
            cross_sectional_area: Cross-sectional area facing sun [m²]
            mass: Object mass [kg]
            reflectivity: Surface reflectivity (0-1)
            
        Returns:
            SRP acceleration vector [km/s²]
        """
        # Simplified sun position (should use ephemerides in production)
        sun_position = np.array([AU, 0, 0])  # km, simplified
        
        # Vector from object to sun
        to_sun = sun_position - position
        distance_to_sun = np.linalg.norm(to_sun)
        
        if distance_to_sun == 0:
            return np.zeros(3)
        
        sun_direction = to_sun / distance_to_sun
        
        # Solar flux at object's distance
        solar_flux = self.solar_constant * (AU / distance_to_sun) ** 2  # W/m²
        
        # Radiation pressure
        pressure = solar_flux / self.speed_of_light  # N/m²
        
        # Force accounting for reflection and absorption
        # Perfect absorber: F = PA, Perfect reflector: F = 2PA
        force_magnitude = pressure * cross_sectional_area * (1 + reflectivity)
        
        # Acceleration (convert to km/s²)
        srp_acceleration = force_magnitude * sun_direction / mass * 1e-3
        
        return srp_acceleration

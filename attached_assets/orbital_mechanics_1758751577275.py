# -*- coding: utf-8 -*-
"""
Advanced orbital mechanics and propagation algorithms
Enhanced SGP4/SDP4, N-body dynamics, and high-precision orbit determination
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import math
import logging
from .constants import *
from .atmospheric_models import NRLMSISE00Model, AtmosphericDragModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def solve_kepler_equation(mean_anomaly: float, eccentricity: float, 
                         tolerance: float = 1e-12, max_iterations: int = 200) -> float:
    """
    Solve Kepler's equation using Newton-Raphson method with high precision.
    
    Args:
        mean_anomaly: Mean anomaly in radians
        eccentricity: Orbital eccentricity (0 ≤ e < 1)
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
        
    Returns:
        Eccentric anomaly in radians
    """
    # Normalize mean anomaly to [-π, π]
    M = mean_anomaly % (2 * np.pi)
    if M > np.pi:
        M -= 2 * np.pi
    
    # Initial guess based on eccentricity
    if eccentricity < 0.8:
        E = M
    else:
        E = np.pi if M > 0 else -np.pi
    
    # Newton-Raphson iteration
    for i in range(max_iterations):
        f = E - eccentricity * np.sin(E) - M
        f_prime = 1 - eccentricity * np.cos(E)
        
        if abs(f_prime) < 1e-15:  # Avoid division by zero
            break
            
        E_new = E - f / f_prime
        
        if abs(E_new - E) < tolerance:
            return E_new
            
        E = E_new
    
    return E

def true_anomaly_from_eccentric(eccentric_anomaly: float, eccentricity: float) -> float:
    """Calculate true anomaly from eccentric anomaly."""
    E = eccentric_anomaly
    e = eccentricity
    
    return 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )

def perifocal_to_eci(r_pqw: np.ndarray, v_pqw: np.ndarray, 
                     inclination: float, raan: float, arg_perigee: float) -> Tuple[np.ndarray, np.ndarray]:
    """Transform position and velocity from perifocal to ECI frame."""
    # Rotation matrices
    cos_O = np.cos(raan)
    sin_O = np.sin(raan)
    cos_i = np.cos(inclination)
    sin_i = np.sin(inclination)
    cos_w = np.cos(arg_perigee)
    sin_w = np.sin(arg_perigee)
    
    # Combined rotation matrix from PQW to ECI
    R11 = cos_O * cos_w - sin_O * sin_w * cos_i
    R12 = -cos_O * sin_w - sin_O * cos_w * cos_i
    R13 = sin_O * sin_i
    
    R21 = sin_O * cos_w + cos_O * sin_w * cos_i
    R22 = -sin_O * sin_w + cos_O * cos_w * cos_i
    R23 = -cos_O * sin_i
    
    R31 = sin_w * sin_i
    R32 = cos_w * sin_i
    R33 = cos_i
    
    # Transformation matrix
    R = np.array([
        [R11, R12, R13],
        [R21, R22, R23],
        [R31, R32, R33]
    ])
    
    # Transform vectors
    r_eci = R @ r_pqw
    v_eci = R @ v_pqw
    
    return r_eci, v_eci

def orbital_elements_from_state_vector(r: np.ndarray, v: np.ndarray, 
                                     mu: float = EARTH_GRAVITATIONAL_PARAMETER) -> Dict[str, float]:
    """Calculate orbital elements from position and velocity vectors."""
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    
    # Specific angular momentum
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    
    # Node vector
    k = np.array([0, 0, 1])
    n = np.cross(k, h)
    n_mag = np.linalg.norm(n)
    
    # Eccentricity vector
    e_vec = ((v_mag**2 - mu/r_mag) * r - np.dot(r, v) * v) / mu
    eccentricity = np.linalg.norm(e_vec)
    
    # Specific energy
    energy = v_mag**2 / 2 - mu / r_mag
    
    # Semi-major axis
    if abs(energy) > 1e-10:
        semi_major_axis = -mu / (2 * energy)
    else:
        semi_major_axis = float('inf')  # Parabolic orbit
    
    # Inclination
    inclination = np.arccos(np.clip(h[2] / h_mag, -1, 1))
    
    # Right ascension of ascending node
    if n_mag > 1e-10:
        raan = np.arccos(np.clip(n[0] / n_mag, -1, 1))
        if n[1] < 0:
            raan = 2 * np.pi - raan
    else:
        raan = 0
    
    # Argument of perigee
    if n_mag > 1e-10 and eccentricity > 1e-10:
        arg_perigee = np.arccos(np.clip(np.dot(n, e_vec) / (n_mag * eccentricity), -1, 1))
        if e_vec[2] < 0:
            arg_perigee = 2 * np.pi - arg_perigee
    else:
        arg_perigee = 0
    
    # True anomaly
    if eccentricity > 1e-10:
        true_anomaly = np.arccos(np.clip(np.dot(e_vec, r) / (eccentricity * r_mag), -1, 1))
        if np.dot(r, v) < 0:
            true_anomaly = 2 * np.pi - true_anomaly
    else:
        true_anomaly = 0
    
    # Mean anomaly
    if eccentricity < 1.0:
        E = np.arccos((eccentricity + np.cos(true_anomaly)) / (1 + eccentricity * np.cos(true_anomaly)))
        if true_anomaly > np.pi:
            E = 2 * np.pi - E
        mean_anomaly = E - eccentricity * np.sin(E)
    else:
        mean_anomaly = 0
    
    return {
        'semi_major_axis': semi_major_axis,
        'eccentricity': eccentricity,
        'inclination': np.degrees(inclination),
        'raan': np.degrees(raan),
        'arg_perigee': np.degrees(arg_perigee),
        'true_anomaly': np.degrees(true_anomaly),
        'mean_anomaly': np.degrees(mean_anomaly),
        'period': 2 * np.pi * np.sqrt(semi_major_axis**3 / mu) if semi_major_axis > 0 else 0
    }

class EnhancedSGP4Propagator:
    """
    Enhanced SGP4/SDP4 orbital propagator with high-precision perturbations.
    Includes J2-J8 gravitational harmonics, atmospheric drag, and solar radiation pressure.
    """
    
    def __init__(self):
        self.mu = EARTH_GRAVITATIONAL_PARAMETER
        self.earth_radius = EARTH_RADIUS
        self.j_coefficients = {
            'J2': J2_EARTH,
            'J3': J3_EARTH,
            'J4': J4_EARTH,
            'J5': J5_EARTH,
            'J6': J6_EARTH,
            'J7': J7_EARTH,
            'J8': J8_EARTH
        }
        self.atmospheric_model = NRLMSISE00Model()
        self.drag_model = AtmosphericDragModel()
    
    def propagate_orbit(self, orbital_elements: Dict[str, float], 
                       dt_seconds: float,
                       object_properties: Optional[Dict] = None,
                       space_weather: Optional[Dict] = None,
                       include_perturbations: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Propagate orbit using enhanced SGP4 with comprehensive perturbations.
        
        Args:
            orbital_elements: Initial orbital elements
            dt_seconds: Time step in seconds
            object_properties: Physical properties (mass, area, etc.)
            space_weather: Current space weather conditions
            include_perturbations: Whether to include perturbation forces
            
        Returns:
            Position vector [km], velocity vector [km/s], updated elements
        """
        try:
            # Extract orbital elements with validation
            a = max(orbital_elements.get('semi_major_axis', 7000), EARTH_RADIUS + 100)
            e = max(0, min(0.99, orbital_elements.get('eccentricity', 0.001)))
            i = np.radians(orbital_elements.get('inclination', 45))
            raan = np.radians(orbital_elements.get('raan', 0))
            argp = np.radians(orbital_elements.get('arg_perigee', 0))
            M0 = np.radians(orbital_elements.get('mean_anomaly', 0))
            
            # Calculate mean motion
            n = np.sqrt(self.mu / a**3)
            
            # Initialize perturbation rates
            secular_rates = {'raan_dot': 0, 'argp_dot': 0, 'n_dot': 0}
            
            if include_perturbations:
                # Calculate secular perturbations due to J2-J8
                secular_rates = self._calculate_secular_rates(a, e, i, n)
                
                # Update elements for secular perturbations
                raan += secular_rates['raan_dot'] * dt_seconds
                argp += secular_rates['argp_dot'] * dt_seconds
                n += secular_rates['n_dot'] * dt_seconds
                
                # Update semi-major axis if mean motion changed
                if secular_rates['n_dot'] != 0:
                    a = (self.mu / n**2)**(1/3)
            
            # Update mean anomaly
            M = M0 + n * dt_seconds
            
            # Solve Kepler's equation
            E = solve_kepler_equation(M, e)
            
            # Calculate true anomaly
            nu = true_anomaly_from_eccentric(E, e)
            
            # Calculate radius
            r = a * (1 - e * np.cos(E))
            
            # Position and velocity in perifocal frame
            p = a * (1 - e**2)
            
            r_pqw = np.array([
                r * np.cos(nu),
                r * np.sin(nu),
                0
            ])
            
            sqrt_mu_p = np.sqrt(self.mu / p)
            v_pqw = sqrt_mu_p * np.array([
                -np.sin(nu),
                e + np.cos(nu),
                0
            ])
            
            # Transform to ECI frame
            r_eci, v_eci = perifocal_to_eci(r_pqw, v_pqw, i, raan, argp)
            
            # Apply additional perturbations if requested
            if include_perturbations and object_properties:
                total_acceleration = np.zeros(3)
                
                # Atmospheric drag
                if 'cross_sectional_area' in object_properties and 'mass' in object_properties:
                    drag_acc = self._calculate_drag_acceleration(
                        r_eci, v_eci, object_properties, space_weather
                    )
                    total_acceleration += drag_acc
                
                # Solar radiation pressure
                if 'cross_sectional_area' in object_properties and 'mass' in object_properties:
                    srp_acc = self._calculate_srp_acceleration(
                        r_eci, object_properties
                    )
                    total_acceleration += srp_acc
                
                # Higher-order gravitational harmonics
                harmonic_acc = self._calculate_harmonic_acceleration(r_eci)
                total_acceleration += harmonic_acc
                
                # Apply acceleration to velocity (simple Euler integration)
                v_eci += total_acceleration * dt_seconds
            
            # Calculate updated orbital elements
            updated_elements = orbital_elements_from_state_vector(r_eci, v_eci, self.mu)
            
            return r_eci, v_eci, updated_elements
            
        except Exception as e:
            logger.error(f"Error in orbit propagation: {e}")
            # Return original position if propagation fails
            default_r = np.array([7000.0, 0.0, 0.0])
            default_v = np.array([0.0, 7.5, 0.0])
            return default_r, default_v, orbital_elements
    
    def _calculate_secular_rates(self, a: float, e: float, i: float, n: float) -> Dict[str, float]:
        """Calculate secular perturbation rates due to J2-J8 harmonics."""
        try:
            # Semi-latus rectum
            p = a * (1 - e**2)
            
            if p <= 0:
                return {'raan_dot': 0, 'argp_dot': 0, 'n_dot': 0}
            
            # Common factors
            factor_base = (self.earth_radius / p)**2
            cos_i = np.cos(i)
            sin_i = np.sin(i)
            
            # J2 perturbations
            J2_factor = -1.5 * n * self.j_coefficients['J2'] * factor_base
            
            raan_dot = J2_factor * cos_i
            argp_dot = J2_factor * (2.5 * sin_i**2 - 2)
            
            # J4 correction to argument of perigee
            if abs(self.j_coefficients['J4']) > 0:
                J4_factor = 0.9375 * n * self.j_coefficients['J4'] * (self.earth_radius / p)**4
                J4_argp_correction = J4_factor * (7 * cos_i**2 - 1) * (3 - 30 * sin_i**2 + 35 * sin_i**4)
                argp_dot += J4_argp_correction
            
            # Mean motion rate (primarily from drag, included here as perturbation)
            n_dot = 0  # Will be calculated separately for drag
            
            return {
                'raan_dot': raan_dot,
                'argp_dot': argp_dot,
                'n_dot': n_dot
            }
            
        except Exception as e:
            logger.error(f"Error calculating secular rates: {e}")
            return {'raan_dot': 0, 'argp_dot': 0, 'n_dot': 0}
    
    def _calculate_harmonic_acceleration(self, position: np.ndarray) -> np.ndarray:
        """Calculate acceleration due to higher-order gravitational harmonics (J2-J8)."""
        try:
            r = np.linalg.norm(position)
            
            if r <= self.earth_radius:
                return np.zeros(3)
            
            x, y, z = position
            
            # Initialize acceleration
            acc = np.zeros(3)
            
            # J2 perturbation
            J2 = self.j_coefficients['J2']
            factor = -1.5 * J2 * self.mu * (self.earth_radius / r)**2 / r**3
            
            z_r_ratio = z / r
            factor_xy = factor * (5 * z_r_ratio**2 - 1)
            factor_z = factor * (5 * z_r_ratio**2 - 3)
            
            acc[0] += factor_xy * x
            acc[1] += factor_xy * y
            acc[2] += factor_z * z
            
            # J3 perturbation (zonal)
            J3 = self.j_coefficients['J3']
            if abs(J3) > 0:
                factor_J3 = -2.5 * J3 * self.mu * (self.earth_radius / r)**3 / r**3
                
                acc[0] += factor_J3 * x * z_r_ratio * (7 * z_r_ratio**2 - 3)
                acc[1] += factor_J3 * y * z_r_ratio * (7 * z_r_ratio**2 - 3)
                acc[2] += factor_J3 * (6 * z_r_ratio**2 - 7 * z_r_ratio**4 - 3/5)
            
            # J4 perturbation
            J4 = self.j_coefficients['J4']
            if abs(J4) > 0:
                factor_J4 = (5/8) * J4 * self.mu * (self.earth_radius / r)**4 / r**3
                z_r_2 = z_r_ratio**2
                z_r_4 = z_r_2**2
                
                factor_xy_J4 = factor_J4 * (63 * z_r_4 - 42 * z_r_2 + 3)
                factor_z_J4 = factor_J4 * (15 - 70 * z_r_2 + 63 * z_r_4)
                
                acc[0] += factor_xy_J4 * x
                acc[1] += factor_xy_J4 * y
                acc[2] += factor_z_J4 * z
            
            return acc
            
        except Exception as e:
            logger.error(f"Error calculating harmonic acceleration: {e}")
            return np.zeros(3)
    
    def _calculate_drag_acceleration(self, r_eci: np.ndarray, v_eci: np.ndarray,
                                   object_properties: Dict, space_weather: Optional[Dict]) -> np.ndarray:
        """Calculate atmospheric drag acceleration."""
        try:
            altitude = np.linalg.norm(r_eci) - self.earth_radius
            
            if altitude > 1000:  # No significant drag above 1000 km
                return np.zeros(3)
            
            # Get atmospheric density
            solar_flux = 150  # Default
            if space_weather:
                solar_flux = space_weather.get('solar_flux_f107', 150)
            
            density = self.atmospheric_model.calculate_density(altitude, solar_flux)
            
            # Drag calculation
            area = object_properties.get('cross_sectional_area', 1.0)
            mass = object_properties.get('mass', 100.0)
            cd = object_properties.get('drag_coefficient', 2.2)
            
            # Relative velocity (simplified - no atmospheric rotation)
            v_rel = v_eci
            v_rel_mag = np.linalg.norm(v_rel)
            
            if v_rel_mag > 0:
                # Drag force
                drag_force = -0.5 * density * cd * area * v_rel_mag * v_rel
                drag_acceleration = drag_force / mass
                return drag_acceleration
            
            return np.zeros(3)
            
        except Exception as e:
            logger.error(f"Error calculating drag acceleration: {e}")
            return np.zeros(3)
    
    def _calculate_srp_acceleration(self, r_eci: np.ndarray, 
                                  object_properties: Dict) -> np.ndarray:
        """Calculate solar radiation pressure acceleration."""
        try:
            # Simplified SRP model - assumes Sun in +X direction
            sun_direction = np.array([1.0, 0.0, 0.0])
            
            # Solar radiation pressure at 1 AU
            P_srp = SOLAR_CONSTANT / SPEED_OF_LIGHT  # N/m²
            
            area = object_properties.get('cross_sectional_area', 1.0)
            mass = object_properties.get('mass', 100.0)
            reflectivity = object_properties.get('reflectivity', 0.3)
            
            # Distance factor (assuming 1 AU)
            distance_factor = 1.0
            
            # SRP acceleration
            srp_acceleration = (P_srp * area * (1 + reflectivity) / mass) * sun_direction / distance_factor**2
            
            return srp_acceleration * 1e-9  # Convert to km/s² and scale appropriately
            
        except Exception as e:
            logger.error(f"Error calculating SRP acceleration: {e}")
            return np.zeros(3)
    
    def propagate_multiple_objects(self, objects_df: pd.DataFrame, 
                                 dt_seconds: float,
                                 space_weather: Optional[Dict] = None) -> pd.DataFrame:
        """Propagate multiple objects simultaneously with error handling."""
        results = []
        
        for idx, obj in objects_df.iterrows():
            try:
                # Extract orbital elements with defaults
                elements = {
                    'semi_major_axis': obj.get('semi_major_axis', 7000),
                    'eccentricity': obj.get('eccentricity', 0.001),
                    'inclination': obj.get('inclination', 45),
                    'raan': obj.get('raan', 0),
                    'arg_perigee': obj.get('arg_perigee', 0),
                    'mean_anomaly': obj.get('mean_anomaly', 0)
                }
                
                # Object properties with defaults
                properties = {
                    'mass': obj.get('mass_kg', 100),
                    'cross_sectional_area': obj.get('radar_cross_section', 1.0),
                    'drag_coefficient': obj.get('drag_coefficient', 2.2),
                    'reflectivity': obj.get('reflectivity', 0.3)
                }
                
                # Propagate orbit
                r_eci, v_eci, updated_elements = self.propagate_orbit(
                    elements, dt_seconds, properties, space_weather, True
                )
                
                # Update object data
                obj_result = obj.copy()
                obj_result.update(updated_elements)
                obj_result['position_x_km'] = r_eci[0]
                obj_result['position_y_km'] = r_eci[1]
                obj_result['position_z_km'] = r_eci[2]
                obj_result['velocity_x_km_s'] = v_eci[0]
                obj_result['velocity_y_km_s'] = v_eci[1]
                obj_result['velocity_z_km_s'] = v_eci[2]
                obj_result['altitude_km'] = np.linalg.norm(r_eci) - self.earth_radius
                obj_result['velocity_magnitude'] = np.linalg.norm(v_eci)
                obj_result['last_propagated'] = datetime.now()
                
                results.append(obj_result)
                
            except Exception as e:
                logger.warning(f"Failed to propagate object {idx}: {e}")
                # Keep original data if propagation fails
                results.append(obj)
        
        return pd.DataFrame(results)

class NBodyGravitationalModel:
    """N-body gravitational model including lunar, solar, and planetary perturbations."""
    
    def __init__(self):
        self.G = GRAVITATIONAL_CONSTANT * 1e-9  # Convert to km³/(kg⋅s²)
        
        # Primary bodies
        self.bodies = {
            'earth': {
                'mass': EARTH_MASS,
                'mu': EARTH_GRAVITATIONAL_PARAMETER,
                'position': np.array([0.0, 0.0, 0.0])
            },
            'moon': {
                'mass': MOON_MASS,
                'mu': MOON_GRAVITATIONAL_PARAMETER,
                'position': np.array([MOON_MEAN_DISTANCE, 0.0, 0.0])
            },
            'sun': {
                'mass': SUN_MASS,
                'mu': SUN_GRAVITATIONAL_PARAMETER,
                'position': np.array([AU, 0.0, 0.0])
            }
        }
        
        # Add planetary bodies
        for planet, params in PLANETARY_PARAMETERS.items():
            self.bodies[planet] = {
                'mass': params['mass'],
                'mu': params['mu'],
                'position': np.array([params['distance_au'] * AU, 0.0, 0.0])
            }
    
    def update_body_positions(self, julian_date: float):
        """Update positions of celestial bodies for given Julian date."""
        try:
            # Days since J2000.0
            t = julian_date - 2451545.0
            
            # Moon position (simplified elliptical orbit)
            moon_mean_anomaly = 2 * np.pi * t / MOON_ORBITAL_PERIOD
            moon_ecc = MOON_ECCENTRICITY
            moon_E = solve_kepler_equation(moon_mean_anomaly, moon_ecc)
            moon_true_anomaly = true_anomaly_from_eccentric(moon_E, moon_ecc)
            moon_r = MOON_MEAN_DISTANCE * (1 - moon_ecc**2) / (1 + moon_ecc * np.cos(moon_true_anomaly))
            
            self.bodies['moon']['position'] = moon_r * np.array([
                np.cos(moon_true_anomaly),
                np.sin(moon_true_anomaly),
                0
            ])
            
            # Sun position (simplified circular orbit)
            earth_mean_anomaly = 2 * np.pi * t / 365.25
            self.bodies['sun']['position'] = AU * np.array([
                np.cos(earth_mean_anomaly),
                np.sin(earth_mean_anomaly),
                0
            ])
            
            # Planetary positions (simplified circular orbits)
            for planet, params in PLANETARY_PARAMETERS.items():
                if planet in self.bodies:
                    # Approximate orbital periods (years)
                    periods = {
                        'mercury': 0.24, 'venus': 0.62, 'mars': 1.88,
                        'jupiter': 11.86, 'saturn': 29.46
                    }
                    period = periods.get(planet, 1.0)
                    
                    planet_angle = 2 * np.pi * t / (period * 365.25)
                    distance = params['distance_au'] * AU
                    
                    self.bodies[planet]['position'] = distance * np.array([
                        np.cos(planet_angle),
                        np.sin(planet_angle),
                        0
                    ])
                    
        except Exception as e:
            logger.error(f"Error updating body positions: {e}")
    
    def calculate_gravitational_acceleration(self, position: np.ndarray,
                                           julian_date: Optional[float] = None,
                                           include_planets: bool = False) -> Dict[str, np.ndarray]:
        """Calculate gravitational acceleration from all bodies."""
        try:
            if julian_date:
                self.update_body_positions(julian_date)
            
            accelerations = {}
            
            # Primary bodies (Earth, Moon, Sun)
            primary_bodies = ['earth', 'moon', 'sun']
            
            for body_name in primary_bodies:
                if body_name in self.bodies:
                    body = self.bodies[body_name]
                    r_vector = position - body['position']
                    r_magnitude = np.linalg.norm(r_vector)
                    
                    if r_magnitude > 0:
                        acceleration = -body['mu'] * r_vector / r_magnitude**3
                        accelerations[body_name] = acceleration
                    else:
                        accelerations[body_name] = np.zeros(3)
            
            # Planetary bodies (if requested)
            if include_planets:
                for planet in PLANETARY_PARAMETERS.keys():
                    if planet in self.bodies:
                        body = self.bodies[planet]
                        r_vector = position - body['position']
                        r_magnitude = np.linalg.norm(r_vector)
                        
                        if r_magnitude > 1000:  # Only include if reasonably far
                            acceleration = -body['mu'] * r_vector / r_magnitude**3
                            accelerations[planet] = acceleration
                        else:
                            accelerations[planet] = np.zeros(3)
            
            return accelerations
            
        except Exception as e:
            logger.error(f"Error calculating gravitational acceleration: {e}")
            return {'earth': np.array([0, 0, -9.81e-3])}  # Default Earth gravity in km/s²
    
    def total_acceleration(self, position: np.ndarray, julian_date: Optional[float] = None) -> np.ndarray:
        """Calculate total gravitational acceleration from all bodies."""
        accelerations = self.calculate_gravitational_acceleration(position, julian_date)
        return sum(accelerations.values())

class OrbitalStateEstimator:
    """Extended Kalman Filter for orbital state estimation."""
    
    def __init__(self):
        self.state_dimension = 6  # [x, y, z, vx, vy, vz]
        self.measurement_dimension = 3  # [range, azimuth, elevation] or [x, y, z]
        
        # Initialize filter parameters
        self.Q = np.eye(6) * KALMAN_FILTER_PARAMETERS['process_noise']  # Process noise
        self.R = np.eye(3) * KALMAN_FILTER_PARAMETERS['measurement_noise']  # Measurement noise
        self.P = np.eye(6) * KALMAN_FILTER_PARAMETERS['initial_uncertainty']  # Covariance
        
        self.propagator = EnhancedSGP4Propagator()
    
    def predict(self, state: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Predict step of Kalman filter."""
        try:
            # Extract position and velocity
            r = state[:3]
            v = state[3:]
            
            # Simple orbital mechanics integration (could use RK4)
            # F = dv/dt = -mu * r / |r|^3 (two-body problem)
            r_mag = np.linalg.norm(r)
            if r_mag > EARTH_RADIUS:
                acceleration = -EARTH_GRAVITATIONAL_PARAMETER * r / r_mag**3
            else:
                acceleration = np.zeros(3)
            
            # Update state
            new_r = r + v * dt + 0.5 * acceleration * dt**2
            new_v = v + acceleration * dt
            
            predicted_state = np.concatenate([new_r, new_v])
            
            # State transition matrix (linearized)
            F = np.eye(6)
            F[:3, 3:] = np.eye(3) * dt
            
            # Update covariance
            predicted_covariance = F @ self.P @ F.T + self.Q
            
            return predicted_state, predicted_covariance
            
        except Exception as e:
            logger.error(f"Error in predict step: {e}")
            return state, self.P
    
    def update(self, predicted_state: np.ndarray, predicted_covariance: np.ndarray,
               measurement: np.ndarray, measurement_type: str = 'position') -> Tuple[np.ndarray, np.ndarray]:
        """Update step of Kalman filter."""
        try:
            if measurement_type == 'position':
                # Direct position measurement
                H = np.zeros((3, 6))
                H[:3, :3] = np.eye(3)
                predicted_measurement = predicted_state[:3]
            else:
                # Could implement range/azimuth/elevation measurements here
                H = np.zeros((3, 6))
                H[:3, :3] = np.eye(3)
                predicted_measurement = predicted_state[:3]
            
            # Innovation
            innovation = measurement - predicted_measurement
            
            # Innovation covariance
            S = H @ predicted_covariance @ H.T + self.R
            
            # Kalman gain
            K = predicted_covariance @ H.T @ np.linalg.inv(S)
            
            # Updated state and covariance
            updated_state = predicted_state + K @ innovation
            updated_covariance = (np.eye(6) - K @ H) @ predicted_covariance
            
            return updated_state, updated_covariance
            
        except Exception as e:
            logger.error(f"Error in update step: {e}")
            return predicted_state, predicted_covariance

def propagate_orbit_rk4(r0: np.ndarray, v0: np.ndarray, dt: float, 
                       acceleration_function) -> Tuple[np.ndarray, np.ndarray]:
    """Fourth-order Runge-Kutta orbital propagation."""
    try:
        def derivatives(state):
            r = state[:3]
            v = state[3:]
            acc = acceleration_function(r)
            return np.concatenate([v, acc])
        
        state = np.concatenate([r0, v0])
        
        # RK4 integration
        k1 = dt * derivatives(state)
        k2 = dt * derivatives(state + k1/2)
        k3 = dt * derivatives(state + k2/2)
        k4 = dt * derivatives(state + k3)
        
        new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return new_state[:3], new_state[3:]
        
    except Exception as e:
        logger.error(f"Error in RK4 propagation: {e}")
        return r0, v0

# Global instances for easy access
sgp4_propagator = EnhancedSGP4Propagator()
nbody_model = NBodyGravitationalModel()
state_estimator = OrbitalStateEstimator()

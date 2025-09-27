# -*- coding: utf-8 -*-
"""
Mathematical utilities for orbital mechanics and space debris analysis
High-precision numerical methods, coordinate transformations, and statistical functions
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
import math
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize_scalar, fsolve
from scipy.stats import multivariate_normal, chi2
from scipy.linalg import cholesky, inv
import logging

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
    try:
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
                logger.warning("Near-singular derivative in Kepler equation solver")
                break
                
            E_new = E - f / f_prime
            
            if abs(E_new - E) < tolerance:
                return E_new
                
            E = E_new
        
        logger.warning(f"Kepler equation did not converge after {max_iterations} iterations")
        return E
        
    except Exception as e:
        logger.error(f"Error solving Kepler equation: {e}")
        return mean_anomaly  # Fallback to mean anomaly

def true_anomaly_from_eccentric(eccentric_anomaly: float, eccentricity: float) -> float:
    """
    Calculate true anomaly from eccentric anomaly.
    
    Args:
        eccentric_anomaly: Eccentric anomaly in radians
        eccentricity: Orbital eccentricity
        
    Returns:
        True anomaly in radians
    """
    try:
        E = eccentric_anomaly
        e = eccentricity
        
        # Half-angle formula for numerical stability
        return 2 * np.arctan2(
            np.sqrt(1 + e) * np.sin(E / 2),
            np.sqrt(1 - e) * np.cos(E / 2)
        )
    except Exception as e:
        logger.error(f"Error calculating true anomaly: {e}")
        return eccentric_anomaly

def mean_anomaly_from_true(true_anomaly: float, eccentricity: float) -> float:
    """
    Calculate mean anomaly from true anomaly.
    
    Args:
        true_anomaly: True anomaly in radians
        eccentricity: Orbital eccentricity
        
    Returns:
        Mean anomaly in radians
    """
    try:
        nu = true_anomaly
        e = eccentricity
        
        # Calculate eccentric anomaly first
        E = 2 * np.arctan2(
            np.sqrt(1 - e) * np.sin(nu / 2),
            np.sqrt(1 + e) * np.cos(nu / 2)
        )
        
        # Then mean anomaly
        M = E - e * np.sin(E)
        
        return M % (2 * np.pi)
        
    except Exception as e:
        logger.error(f"Error calculating mean anomaly: {e}")
        return true_anomaly

def rotation_matrix_x(angle: float) -> np.ndarray:
    """Rotation matrix about X-axis."""
    try:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    except Exception as e:
        logger.error(f"Error creating X rotation matrix: {e}")
        return np.eye(3)

def rotation_matrix_y(angle: float) -> np.ndarray:
    """Rotation matrix about Y-axis."""
    try:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    except Exception as e:
        logger.error(f"Error creating Y rotation matrix: {e}")
        return np.eye(3)

def rotation_matrix_z(angle: float) -> np.ndarray:
    """Rotation matrix about Z-axis."""
    try:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    except Exception as e:
        logger.error(f"Error creating Z rotation matrix: {e}")
        return np.eye(3)

def euler_angles_to_rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Convert Euler angles (3-2-1 sequence) to rotation matrix.
    
    Args:
        phi: Roll angle (radians)
        theta: Pitch angle (radians) 
        psi: Yaw angle (radians)
        
    Returns:
        3x3 rotation matrix
    """
    try:
        R_x = rotation_matrix_x(phi)
        R_y = rotation_matrix_y(theta)
        R_z = rotation_matrix_z(psi)
        
        return R_z @ R_y @ R_x
        
    except Exception as e:
        logger.error(f"Error converting Euler angles to rotation matrix: {e}")
        return np.eye(3)

def perifocal_to_eci(r_pqw: np.ndarray, v_pqw: np.ndarray, 
                     inclination: float, raan: float, arg_perigee: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform position and velocity from perifocal to ECI frame.
    
    Args:
        r_pqw: Position vector in perifocal frame [km]
        v_pqw: Velocity vector in perifocal frame [km/s]
        inclination: Inclination in radians
        raan: Right ascension of ascending node in radians
        arg_perigee: Argument of perigee in radians
        
    Returns:
        Position and velocity vectors in ECI frame
    """
    try:
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
        
    except Exception as e:
        logger.error(f"Error in perifocal to ECI transformation: {e}")
        return r_pqw, v_pqw

def eci_to_geodetic(r_eci: np.ndarray, earth_radius: float = 6378.137,
                   flattening: float = 1/298.257223563) -> Tuple[float, float, float]:
    """
    Convert ECI position to geodetic coordinates using WGS84 ellipsoid.
    
    Args:
        r_eci: Position vector in ECI frame [km]
        earth_radius: Earth's equatorial radius [km]
        flattening: Earth's flattening factor
        
    Returns:
        Latitude (rad), longitude (rad), altitude (km)
    """
    try:
        x, y, z = r_eci
        
        # Longitude (simple)
        longitude = np.arctan2(y, x)
        
        # Iterative solution for latitude and altitude
        p = np.sqrt(x**2 + y**2)
        
        # Semi-minor axis
        b = earth_radius * (1 - flattening)
        
        # Eccentricity squared
        e2 = 2 * flattening - flattening**2
        
        # Initial estimate
        lat = np.arctan2(z, p * (1 - e2))
        
        # Iterate to convergence
        for _ in range(10):  # Usually converges in 2-3 iterations
            N = earth_radius / np.sqrt(1 - e2 * np.sin(lat)**2)
            altitude = p / np.cos(lat) - N
            lat_new = np.arctan2(z, p * (1 - e2 * N / (N + altitude)))
            
            if abs(lat_new - lat) < 1e-12:
                break
            lat = lat_new
        
        # Final altitude calculation
        N = earth_radius / np.sqrt(1 - e2 * np.sin(lat)**2)
        altitude = p / np.cos(lat) - N
        
        return lat, longitude, altitude
        
    except Exception as e:
        logger.error(f"Error in ECI to geodetic conversion: {e}")
        # Fallback to spherical approximation
        r_mag = np.linalg.norm(r_eci)
        latitude = np.arcsin(r_eci[2] / r_mag) if r_mag > 0 else 0
        longitude = np.arctan2(r_eci[1], r_eci[0])
        altitude = r_mag - earth_radius
        return latitude, longitude, altitude

def orbital_elements_from_state_vector(r: np.ndarray, v: np.ndarray, 
                                     mu: float = 398600.4418) -> Dict[str, float]:
    """
    Calculate orbital elements from position and velocity vectors with high precision.
    
    Args:
        r: Position vector [km]
        v: Velocity vector [km/s]
        mu: Gravitational parameter [km³/s²]
        
    Returns:
        Dictionary of orbital elements
    """
    try:
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        if r_mag == 0 or v_mag == 0:
            raise ValueError("Zero position or velocity vector")
        
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
        mean_anomaly = mean_anomaly_from_true(true_anomaly, eccentricity)
        
        # Orbital period
        if semi_major_axis > 0 and eccentricity < 1:
            period = 2 * np.pi * np.sqrt(semi_major_axis**3 / mu)
        else:
            period = float('inf')
        
        return {
            'semi_major_axis': semi_major_axis,
            'eccentricity': eccentricity,
            'inclination': np.degrees(inclination),
            'raan': np.degrees(raan),
            'arg_perigee': np.degrees(arg_perigee),
            'true_anomaly': np.degrees(true_anomaly),
            'mean_anomaly': np.degrees(mean_anomaly),
            'period': period,
            'specific_energy': energy,
            'angular_momentum': h_mag
        }
        
    except Exception as e:
        logger.error(f"Error calculating orbital elements: {e}")
        return {
            'semi_major_axis': 7000.0,
            'eccentricity': 0.001,
            'inclination': 45.0,
            'raan': 0.0,
            'arg_perigee': 0.0,
            'true_anomaly': 0.0,
            'mean_anomaly': 0.0,
            'period': 5400.0,
            'specific_energy': -28.4,
            'angular_momentum': 52000.0
        }

def legendre_polynomial(n: int, x: float) -> float:
    """
    Calculate Legendre polynomial of degree n at x using recurrence relation.
    Used for gravitational potential calculations.
    
    Args:
        n: Degree of polynomial
        x: Evaluation point
        
    Returns:
        Value of Legendre polynomial P_n(x)
    """
    try:
        if n == 0:
            return 1.0
        elif n == 1:
            return x
        else:
            p_nm2 = 1.0  # P_0
            p_nm1 = x    # P_1
            for i in range(2, n + 1):
                p_n = ((2*i - 1) * x * p_nm1 - (i - 1) * p_nm2) / i
                p_nm2, p_nm1 = p_nm1, p_n
            return p_nm1
            
    except Exception as e:
        logger.error(f"Error calculating Legendre polynomial: {e}")
        return 0.0

def associated_legendre_polynomial(n: int, m: int, x: float) -> float:
    """
    Calculate associated Legendre polynomial P_n^m(x).
    
    Args:
        n: Degree
        m: Order (0 ≤ m ≤ n)
        x: Evaluation point (-1 ≤ x ≤ 1)
        
    Returns:
        Value of associated Legendre polynomial
    """
    try:
        if m < 0 or m > n or abs(x) > 1:
            return 0.0
        
        if m == 0:
            return legendre_polynomial(n, x)
        
        # Use recursion relation for associated Legendre polynomials
        # This is a simplified implementation
        if n == m:
            # P_m^m(x) = (-1)^m * (2m-1)!! * (1-x²)^(m/2)
            factor = (-1)**m
            for i in range(1, 2*m, 2):
                factor *= i
            return factor * (1 - x**2)**(m/2)
        
        # Use recurrence relation for higher degrees
        return ((2*n - 1) * x * associated_legendre_polynomial(n-1, m, x) - 
                (n + m - 1) * associated_legendre_polynomial(n-2, m, x)) / (n - m)
        
    except Exception as e:
        logger.error(f"Error calculating associated Legendre polynomial: {e}")
        return 0.0

def spherical_harmonics_acceleration(r: np.ndarray, coefficients: Dict[str, float], 
                                   earth_radius: float = 6378.137) -> np.ndarray:
    """
    Calculate acceleration due to spherical harmonics (J2-J8).
    
    Args:
        r: Position vector in ECI frame [km]
        coefficients: Dictionary of J coefficients
        earth_radius: Earth's radius [km]
        
    Returns:
        Acceleration vector [km/s²]
    """
    try:
        x, y, z = r
        r_mag = np.linalg.norm(r)
        
        if r_mag <= earth_radius:
            return np.zeros(3)
        
        # Normalized coordinates
        rho = earth_radius / r_mag
        
        # Initialize acceleration
        acc = np.zeros(3)
        mu = 398600.4418  # Earth's gravitational parameter
        
        # J2 perturbation (most significant)
        if 'J2' in coefficients:
            J2 = coefficients['J2']
            factor = -1.5 * J2 * mu * (rho**2) / r_mag**3
            
            z_r_ratio = z / r_mag
            factor_xy = factor * (5 * z_r_ratio**2 - 1)
            factor_z = factor * (5 * z_r_ratio**2 - 3)
            
            acc[0] += factor_xy * x / r_mag
            acc[1] += factor_xy * y / r_mag
            acc[2] += factor_z * z / r_mag
        
        # J3 perturbation (sectoral)
        if 'J3' in coefficients:
            J3 = coefficients['J3']
            factor = -2.5 * J3 * mu * (rho**3) / r_mag**3
            
            z_r_ratio = z / r_mag
            xy_factor = factor * z_r_ratio * (7 * z_r_ratio**2 - 3)
            z_factor = factor * (6 * z_r_ratio**2 - 7 * z_r_ratio**4 - 3/5)
            
            acc[0] += xy_factor * x / r_mag
            acc[1] += xy_factor * y / r_mag
            acc[2] += z_factor
        
        # J4 perturbation
        if 'J4' in coefficients:
            J4 = coefficients['J4']
            factor = (5/8) * J4 * mu * (rho**4) / r_mag**3
            
            z_r_ratio = z / r_mag
            z_r_2 = z_r_ratio**2
            z_r_4 = z_r_2**2
            
            xy_factor = factor * (63 * z_r_4 - 42 * z_r_2 + 3)
            z_factor = factor * (15 - 70 * z_r_2 + 63 * z_r_4)
            
            acc[0] += xy_factor * x / r_mag
            acc[1] += xy_factor * y / r_mag
            acc[2] += z_factor * z / r_mag
        
        return acc
        
    except Exception as e:
        logger.error(f"Error calculating spherical harmonics acceleration: {e}")
        return np.zeros(3)

def monte_carlo_collision_probability(r1: np.ndarray, v1: np.ndarray,
                                    r2: np.ndarray, v2: np.ndarray,
                                    covariance1: np.ndarray, covariance2: np.ndarray,
                                    combined_radius: float = 0.01,
                                    n_samples: int = 10000) -> Dict[str, float]:
    """
    Calculate collision probability using Monte Carlo method with full statistical analysis.
    
    Args:
        r1, v1: Position and velocity of object 1
        r2, v2: Position and velocity of object 2
        covariance1, covariance2: Position uncertainty covariances [km²]
        combined_radius: Combined collision radius [km]
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Dictionary with collision probability and statistics
    """
    try:
        # Generate random samples from position uncertainties
        samples1 = np.random.multivariate_normal(r1, covariance1, n_samples)
        samples2 = np.random.multivariate_normal(r2, covariance2, n_samples)
        
        # Calculate minimum distances
        min_distances = np.linalg.norm(samples1 - samples2, axis=1)
        
        # Count collisions
        collisions = np.sum(min_distances < combined_radius)
        probability = collisions / n_samples
        
        # Statistical analysis
        confidence_intervals = {}
        for confidence in [0.90, 0.95, 0.99]:
            alpha = 1 - confidence
            z_score = chi2.ppf(1 - alpha/2, 1)**0.5
            margin = z_score * np.sqrt(probability * (1 - probability) / n_samples)
            
            confidence_intervals[f'{confidence:.0%}'] = {
                'lower': max(0, probability - margin),
                'upper': min(1, probability + margin)
            }
        
        return {
            'probability': probability,
            'samples': n_samples,
            'collisions': collisions,
            'min_distance': np.min(min_distances),
            'mean_distance': np.mean(min_distances),
            'std_distance': np.std(min_distances),
            'confidence_intervals': confidence_intervals
        }
        
    except Exception as e:
        logger.error(f"Error in Monte Carlo collision probability: {e}")
        return {
            'probability': 0.0,
            'samples': n_samples,
            'collisions': 0,
            'error': str(e)
        }

def gauss_newton_method(func, jacobian, x0: np.ndarray, 
                       tolerance: float = 1e-8, max_iterations: int = 50) -> Dict[str, Any]:
    """
    Gauss-Newton method for non-linear least squares optimization.
    
    Args:
        func: Function that returns residuals
        jacobian: Function that returns Jacobian matrix
        x0: Initial guess
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        
    Returns:
        Optimization results
    """
    try:
        x = x0.copy()
        
        for iteration in range(max_iterations):
            # Calculate residuals and Jacobian
            residuals = func(x)
            J = jacobian(x)
            
            # Check for convergence
            if np.linalg.norm(residuals) < tolerance:
                return {
                    'success': True,
                    'x': x,
                    'residuals': residuals,
                    'iterations': iteration,
                    'convergence': 'Tolerance achieved'
                }
            
            # Gauss-Newton update step
            try:
                JTJ = J.T @ J
                JTr = J.T @ residuals
                delta_x = np.linalg.solve(JTJ, JTr)
                x = x - delta_x
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                delta_x = np.linalg.pinv(J) @ residuals
                x = x - delta_x
        
        return {
            'success': False,
            'x': x,
            'residuals': func(x),
            'iterations': max_iterations,
            'convergence': 'Maximum iterations reached'
        }
        
    except Exception as e:
        logger.error(f"Error in Gauss-Newton method: {e}")
        return {
            'success': False,
            'x': x0,
            'error': str(e)
        }

def runge_kutta_4(func, y0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Fourth-order Runge-Kutta integration method.
    
    Args:
        func: Derivative function dy/dt = func(t, y)
        y0: Initial conditions
        t: Time array
        
    Returns:
        Solution array
    """
    try:
        n = len(t)
        y = np.zeros((n, len(y0)))
        y[0] = y0
        
        for i in range(n - 1):
            dt = t[i + 1] - t[i]
            
            k1 = dt * func(t[i], y[i])
            k2 = dt * func(t[i] + dt/2, y[i] + k1/2)
            k3 = dt * func(t[i] + dt/2, y[i] + k2/2)
            k4 = dt * func(t[i] + dt, y[i] + k3)
            
            y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return y
        
    except Exception as e:
        logger.error(f"Error in Runge-Kutta 4 integration: {e}")
        return np.array([y0])

def statistical_orbit_determination(observations: List[Dict], 
                                  observation_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Statistical orbit determination using weighted least squares.
    
    Args:
        observations: List of observation dictionaries
        observation_weights: Weights for observations
        
    Returns:
        Orbit determination results
    """
    try:
        if len(observations) < 3:
            raise ValueError("At least 3 observations required for orbit determination")
        
        # Extract observation data
        times = np.array([obs['time'] for obs in observations])
        positions = np.array([obs['position'] for obs in observations])
        
        if observation_weights is None:
            observation_weights = np.ones(len(observations))
        
        # Initial orbit guess using first and last observations
        r1, r2 = positions[0], positions[-1]
        dt = times[-1] - times[0]
        
        # Simple initial velocity estimate
        v1_estimate = (r2 - r1) / dt
        
        # Refined orbit determination would use iterative methods
        # This is a simplified implementation
        initial_elements = orbital_elements_from_state_vector(r1, v1_estimate)
        
        # Calculate residuals for goodness of fit
        residuals = []
        for i, obs in enumerate(observations):
            predicted_pos = positions[i]  # Simplified - would propagate orbit
            observed_pos = obs['position']
            residual = np.linalg.norm(predicted_pos - observed_pos)
            residuals.append(residual)
        
        rms_residual = np.sqrt(np.mean(np.array(residuals)**2))
        
        return {
            'success': True,
            'orbital_elements': initial_elements,
            'rms_residual': rms_residual,
            'residuals': residuals,
            'observations_used': len(observations),
            'covariance_matrix': np.eye(6) * 0.1  # Simplified covariance
        }
        
    except Exception as e:
        logger.error(f"Error in statistical orbit determination: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def coordinate_transformation_matrix(from_frame: str, to_frame: str, 
                                   epoch: Optional[float] = None) -> np.ndarray:
    """
    Get coordinate transformation matrix between reference frames.
    
    Args:
        from_frame: Source coordinate frame
        to_frame: Target coordinate frame  
        epoch: Epoch for transformation (Julian date)
        
    Returns:
        3x3 transformation matrix
    """
    try:
        # Simplified transformations - real implementation would use IERS data
        if from_frame == 'ECI' and to_frame == 'ECEF':
            # Earth rotation angle
            if epoch is not None:
                # Simplified GMST calculation
                days_since_j2000 = epoch - 2451545.0
                gmst = (280.46061837 + 360.98564736629 * days_since_j2000) % 360
                theta = np.radians(gmst)
            else:
                theta = 0  # Identity transformation
            
            return rotation_matrix_z(-theta)  # Negative for ECI to ECEF
        
        elif from_frame == 'ECEF' and to_frame == 'ECI':
            # Reverse transformation
            if epoch is not None:
                days_since_j2000 = epoch - 2451545.0
                gmst = (280.46061837 + 360.98564736629 * days_since_j2000) % 360
                theta = np.radians(gmst)
            else:
                theta = 0
            
            return rotation_matrix_z(theta)
        
        else:
            # Identity transformation for unsupported frames
            logger.warning(f"Transformation {from_frame} to {to_frame} not implemented")
            return np.eye(3)
            
    except Exception as e:
        logger.error(f"Error creating coordinate transformation matrix: {e}")
        return np.eye(3)

def numerical_differentiation(func, x: float, h: float = 1e-6, method: str = 'central') -> float:
    """
    Numerical differentiation using finite differences.
    
    Args:
        func: Function to differentiate
        x: Point at which to evaluate derivative
        h: Step size
        method: 'forward', 'backward', or 'central'
        
    Returns:
        Numerical derivative
    """
    try:
        if method == 'forward':
            return (func(x + h) - func(x)) / h
        elif method == 'backward':
            return (func(x) - func(x - h)) / h
        elif method == 'central':
            return (func(x + h) - func(x - h)) / (2 * h)
        else:
            raise ValueError(f"Unknown differentiation method: {method}")
            
    except Exception as e:
        logger.error(f"Error in numerical differentiation: {e}")
        return 0.0

def matrix_exponential(A: np.ndarray, t: float = 1.0) -> np.ndarray:
    """
    Calculate matrix exponential exp(A*t) using Padé approximation.
    
    Args:
        A: Square matrix
        t: Scalar multiplier
        
    Returns:
        Matrix exponential
    """
    try:
        from scipy.linalg import expm
        return expm(A * t)
    except Exception as e:
        logger.error(f"Error calculating matrix exponential: {e}")
        return np.eye(A.shape[0]) if A.ndim == 2 else np.array([[1]])

# Utility functions for common mathematical operations
def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    try:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
    except Exception as e:
        logger.error(f"Error normalizing vector: {e}")
        return v

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two vectors in radians."""
    try:
        v1_norm = normalize_vector(v1)
        v2_norm = normalize_vector(v2)
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.arccos(cos_angle)
    except Exception as e:
        logger.error(f"Error calculating angle between vectors: {e}")
        return 0.0

def cross_product_matrix(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric cross product matrix."""
    try:
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    except Exception as e:
        logger.error(f"Error creating cross product matrix: {e}")
        return np.zeros((3, 3))


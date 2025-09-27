# -*- coding: utf-8 -*-
"""
Advanced Mathematical Calculations and Utility Functions
Orbital mechanics, statistical analysis, and scientific computations
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize, integrate
from scipy.spatial import distance
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Callable
import math
from modules.constants import *

class OrbitalCalculations:
    """
    Advanced orbital mechanics calculations and analysis.
    """
    
    @staticmethod
    def calculate_orbital_period(semi_major_axis: float, 
                               gravitational_parameter: float = EARTH_GRAVITATIONAL_PARAMETER) -> float:
        """
        Calculate orbital period using Kepler's third law.
        
        Args:
            semi_major_axis: Semi-major axis in km
            gravitational_parameter: Gravitational parameter in km³/s²
            
        Returns:
            Orbital period in seconds
        """
        return 2 * np.pi * np.sqrt(semi_major_axis**3 / gravitational_parameter)
    
    @staticmethod
    def calculate_escape_velocity(altitude: float, 
                                mass: float = EARTH_MASS,
                                radius: float = EARTH_RADIUS_EQUATORIAL) -> float:
        """
        Calculate escape velocity at given altitude.
        
        Args:
            altitude: Altitude in km
            mass: Central body mass in kg
            radius: Central body radius in km
            
        Returns:
            Escape velocity in km/s
        """
        r = radius + altitude
        return np.sqrt(2 * GRAVITATIONAL_CONSTANT * mass / (r * 1000)) / 1000
    
    @staticmethod
    def calculate_orbital_velocity(semi_major_axis: float, 
                                 current_radius: float,
                                 gravitational_parameter: float = EARTH_GRAVITATIONAL_PARAMETER) -> float:
        """
        Calculate orbital velocity at current position using vis-viva equation.
        
        Args:
            semi_major_axis: Semi-major axis in km
            current_radius: Current radius from center in km
            gravitational_parameter: Gravitational parameter in km³/s²
            
        Returns:
            Orbital velocity in km/s
        """
        return np.sqrt(gravitational_parameter * (2/current_radius - 1/semi_major_axis))
    
    @staticmethod
    def calculate_sphere_of_influence(primary_mass: float, 
                                    secondary_mass: float, 
                                    separation: float) -> float:
        """
        Calculate sphere of influence radius.
        
        Args:
            primary_mass: Primary body mass in kg
            secondary_mass: Secondary body mass in kg
            separation: Distance between bodies in km
            
        Returns:
            Sphere of influence radius in km
        """
        return separation * (secondary_mass / primary_mass)**(2/5)
    
    @staticmethod
    def calculate_hill_sphere(primary_mass: float, 
                            secondary_mass: float, 
                            separation: float, 
                            eccentricity: float = 0) -> float:
        """
        Calculate Hill sphere radius.
        
        Args:
            primary_mass: Primary body mass in kg
            secondary_mass: Secondary body mass in kg
            separation: Semi-major axis of secondary's orbit in km
            eccentricity: Orbital eccentricity
            
        Returns:
            Hill sphere radius in km
        """
        return separation * (1 - eccentricity) * (secondary_mass / (3 * primary_mass))**(1/3)
    
    @staticmethod
    def calculate_roche_limit(primary_mass: float, 
                            secondary_mass: float, 
                            primary_radius: float,
                            secondary_density: float = 2000) -> float:
        """
        Calculate Roche limit for tidal disruption.
        
        Args:
            primary_mass: Primary body mass in kg
            secondary_mass: Secondary body mass in kg
            primary_radius: Primary body radius in km
            secondary_density: Secondary body density in kg/m³
            
        Returns:
            Roche limit distance in km
        """
        primary_density = primary_mass / ((4/3) * np.pi * (primary_radius * 1000)**3)
        return 2.44 * primary_radius * (primary_density / secondary_density)**(1/3)
    
    @staticmethod
    def calculate_delta_v_hohmann(r1: float, r2: float,
                                gravitational_parameter: float = EARTH_GRAVITATIONAL_PARAMETER) -> Tuple[float, float]:
        """
        Calculate delta-v for Hohmann transfer.
        
        Args:
            r1: Initial orbital radius in km
            r2: Final orbital radius in km
            gravitational_parameter: Gravitational parameter in km³/s²
            
        Returns:
            Tuple of (delta_v1, delta_v2) in km/s
        """
        # Transfer orbit semi-major axis
        a_transfer = (r1 + r2) / 2
        
        # Initial and final circular velocities
        v1_circular = np.sqrt(gravitational_parameter / r1)
        v2_circular = np.sqrt(gravitational_parameter / r2)
        
        # Transfer orbit velocities
        v1_transfer = np.sqrt(gravitational_parameter * (2/r1 - 1/a_transfer))
        v2_transfer = np.sqrt(gravitational_parameter * (2/r2 - 1/a_transfer))
        
        # Delta-v requirements
        delta_v1 = abs(v1_transfer - v1_circular)
        delta_v2 = abs(v2_circular - v2_transfer)
        
        return delta_v1, delta_v2
    
    @staticmethod
    def calculate_synodic_period(period1: float, period2: float) -> float:
        """
        Calculate synodic period between two orbits.
        
        Args:
            period1: First orbit period in seconds
            period2: Second orbit period in seconds
            
        Returns:
            Synodic period in seconds
        """
        return abs(1 / (1/period1 - 1/period2))
    
    @staticmethod
    def calculate_ground_track(orbital_elements: Dict, 
                             time_span: float = 86400,
                             time_step: float = 60) -> Dict:
        """
        Calculate ground track of satellite.
        
        Args:
            orbital_elements: Dictionary with orbital elements
            time_span: Time span in seconds
            time_step: Time step in seconds
            
        Returns:
            Dictionary with latitude/longitude arrays
        """
        times = np.arange(0, time_span, time_step)
        latitudes = []
        longitudes = []
        
        for t in times:
            # This is a simplified calculation
            # Real implementation would use full orbital propagation
            
            # Mean motion
            n = np.sqrt(EARTH_GRAVITATIONAL_PARAMETER / orbital_elements['semi_major_axis']**3)
            
            # Mean anomaly at time t
            M = orbital_elements['mean_anomaly'] + n * t
            
            # Simplified position calculation
            lat = orbital_elements['inclination'] * np.sin(M) * RAD_TO_DEG
            lon = (M * RAD_TO_DEG + EARTH_ANGULAR_VELOCITY * t * RAD_TO_DEG) % 360
            
            latitudes.append(lat)
            longitudes.append(lon)
        
        return {
            'times': times,
            'latitudes': np.array(latitudes),
            'longitudes': np.array(longitudes)
        }

class StatisticalAnalysis:
    """
    Advanced statistical analysis tools for space debris data.
    """
    
    @staticmethod
    def calculate_distribution_statistics(data: Union[np.ndarray, pd.Series]) -> Dict:
        """
        Calculate comprehensive distribution statistics.
        
        Args:
            data: Input data array or series
            
        Returns:
            Dictionary with statistical measures
        """
        if len(data) == 0:
            return {'error': 'Empty dataset'}
        
        data_clean = pd.Series(data).dropna()
        
        if len(data_clean) == 0:
            return {'error': 'No valid data points'}
        
        return {
            'count': len(data_clean),
            'mean': float(np.mean(data_clean)),
            'median': float(np.median(data_clean)),
            'mode': float(stats.mode(data_clean, keepdims=True)[0][0]) if len(data_clean) > 1 else float(data_clean.iloc[0]),
            'std': float(np.std(data_clean)),
            'variance': float(np.var(data_clean)),
            'skewness': float(stats.skew(data_clean)),
            'kurtosis': float(stats.kurtosis(data_clean)),
            'min': float(np.min(data_clean)),
            'max': float(np.max(data_clean)),
            'range': float(np.max(data_clean) - np.min(data_clean)),
            'q25': float(np.percentile(data_clean, 25)),
            'q75': float(np.percentile(data_clean, 75)),
            'iqr': float(np.percentile(data_clean, 75) - np.percentile(data_clean, 25)),
            'coefficient_of_variation': float(np.std(data_clean) / np.mean(data_clean)) if np.mean(data_clean) != 0 else 0
        }
    
    @staticmethod
    def perform_normality_test(data: Union[np.ndarray, pd.Series]) -> Dict:
        """
        Perform normality tests on data.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with test results
        """
        data_clean = pd.Series(data).dropna()
        
        if len(data_clean) < 3:
            return {'error': 'Insufficient data for normality testing'}
        
        results = {}
        
        # Shapiro-Wilk test (best for small samples)
        if len(data_clean) <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data_clean)
                results['shapiro_wilk'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
            except:
                results['shapiro_wilk'] = {'error': 'Test failed'}
        
        # D'Agostino's normality test
        try:
            dagostino_stat, dagostino_p = stats.normaltest(data_clean)
            results['dagostino'] = {
                'statistic': float(dagostino_stat),
                'p_value': float(dagostino_p),
                'is_normal': dagostino_p > 0.05
            }
        except:
            results['dagostino'] = {'error': 'Test failed'}
        
        # Anderson-Darling test
        try:
            anderson_result = stats.anderson(data_clean)
            results['anderson_darling'] = {
                'statistic': float(anderson_result.statistic),
                'critical_values': anderson_result.critical_values.tolist(),
                'significance_levels': anderson_result.significance_level.tolist()
            }
        except:
            results['anderson_darling'] = {'error': 'Test failed'}
        
        return results
    
    @staticmethod
    def calculate_correlation_matrix(data: pd.DataFrame, 
                                   method: str = 'pearson') -> Dict:
        """
        Calculate correlation matrix with significance tests.
        
        Args:
            data: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dictionary with correlation matrix and p-values
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'error': 'No numeric columns found'}
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr(method=method)
        
        # Calculate p-values
        n = len(numeric_data)
        p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
        
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                if i != j:
                    if method == 'pearson':
                        _, p_val = stats.pearsonr(numeric_data.iloc[:, i], numeric_data.iloc[:, j])
                    elif method == 'spearman':
                        _, p_val = stats.spearmanr(numeric_data.iloc[:, i], numeric_data.iloc[:, j])
                    elif method == 'kendall':
                        _, p_val = stats.kendalltau(numeric_data.iloc[:, i], numeric_data.iloc[:, j])
                    else:
                        p_val = np.nan
                    
                    p_values.iloc[i, j] = p_val
                else:
                    p_values.iloc[i, j] = 0.0
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'p_values': p_values.to_dict(),
            'method': method,
            'sample_size': n
        }
    
    @staticmethod
    def detect_outliers(data: Union[np.ndarray, pd.Series], 
                       method: str = 'iqr') -> Dict:
        """
        Detect outliers using various methods.
        
        Args:
            data: Input data
            method: Detection method ('iqr', 'zscore', 'modified_zscore', 'isolation_forest')
            
        Returns:
            Dictionary with outlier information
        """
        data_clean = pd.Series(data).dropna()
        
        if len(data_clean) == 0:
            return {'error': 'No valid data points'}
        
        outlier_mask = np.zeros(len(data_clean), dtype=bool)
        
        if method == 'iqr':
            Q1 = np.percentile(data_clean, 25)
            Q3 = np.percentile(data_clean, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (data_clean < lower_bound) | (data_clean > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data_clean))
            outlier_mask = z_scores > 3
            
        elif method == 'modified_zscore':
            median = np.median(data_clean)
            mad = np.median(np.abs(data_clean - median))
            modified_z_scores = 0.6745 * (data_clean - median) / mad
            outlier_mask = np.abs(modified_z_scores) > 3.5
        
        outlier_indices = np.where(outlier_mask)[0]
        outlier_values = data_clean.iloc[outlier_indices]
        
        return {
            'method': method,
            'outlier_count': int(np.sum(outlier_mask)),
            'outlier_percentage': float(np.sum(outlier_mask) / len(data_clean) * 100),
            'outlier_indices': outlier_indices.tolist(),
            'outlier_values': outlier_values.tolist(),
            'outlier_mask': outlier_mask.tolist()
        }
    
    @staticmethod
    def fit_distribution(data: Union[np.ndarray, pd.Series], 
                        distributions: List[str] = None) -> Dict:
        """
        Fit various probability distributions to data.
        
        Args:
            data: Input data
            distributions: List of distribution names to test
            
        Returns:
            Dictionary with best-fit distribution information
        """
        if distributions is None:
            distributions = ['norm', 'lognorm', 'exponential', 'gamma', 'weibull_min']
        
        data_clean = pd.Series(data).dropna()
        
        if len(data_clean) < 10:
            return {'error': 'Insufficient data for distribution fitting'}
        
        results = {}
        best_distribution = None
        best_aic = np.inf
        
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                
                # Fit distribution
                params = dist.fit(data_clean)
                
                # Calculate AIC
                log_likelihood = np.sum(dist.logpdf(data_clean, *params))
                aic = 2 * len(params) - 2 * log_likelihood
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(data_clean, lambda x: dist.cdf(x, *params))
                
                results[dist_name] = {
                    'parameters': params,
                    'aic': float(aic),
                    'log_likelihood': float(log_likelihood),
                    'ks_statistic': float(ks_stat),
                    'ks_p_value': float(ks_p),
                    'fits_well': ks_p > 0.05
                }
                
                if aic < best_aic:
                    best_aic = aic
                    best_distribution = dist_name
                    
            except Exception as e:
                results[dist_name] = {'error': str(e)}
        
        return {
            'distribution_results': results,
            'best_distribution': best_distribution,
            'best_aic': float(best_aic) if best_aic != np.inf else None
        }

class DebrisAnalysis:
    """
    Specialized analysis functions for space debris data.
    """
    
    @staticmethod
    def calculate_debris_density_leo(debris_df: pd.DataFrame) -> float:
        """
        Calculate debris density in LEO region.
        
        Args:
            debris_df: DataFrame with debris data
            
        Returns:
            Debris density in objects per km³
        """
        if debris_df.empty or 'altitude_km' not in debris_df.columns:
            return 0.0
        
        # Filter LEO objects (160-2000 km)
        leo_objects = debris_df[
            (debris_df['altitude_km'] >= LEO_ALTITUDE_MIN) & 
            (debris_df['altitude_km'] <= LEO_ALTITUDE_MAX)
        ]
        
        if len(leo_objects) == 0:
            return 0.0
        
        # Calculate LEO volume (spherical shell)
        r_inner = EARTH_RADIUS_EQUATORIAL + LEO_ALTITUDE_MIN
        r_outer = EARTH_RADIUS_EQUATORIAL + LEO_ALTITUDE_MAX
        leo_volume = (4/3) * np.pi * (r_outer**3 - r_inner**3)
        
        return len(leo_objects) / leo_volume
    
    @staticmethod
    def calculate_spatial_distribution(debris_df: pd.DataFrame) -> Dict:
        """
        Calculate spatial distribution of debris.
        
        Args:
            debris_df: DataFrame with debris data
            
        Returns:
            Dictionary with spatial distribution analysis
        """
        if debris_df.empty:
            return {'error': 'No debris data available'}
        
        results = {}
        
        # Altitude distribution
        if 'altitude_km' in debris_df.columns:
            alt_bins = np.linspace(100, 50000, 50)
            alt_hist, alt_edges = np.histogram(debris_df['altitude_km'], bins=alt_bins)
            
            results['altitude_distribution'] = {
                'bin_edges': alt_edges.tolist(),
                'counts': alt_hist.tolist(),
                'peak_altitude': float(alt_edges[np.argmax(alt_hist)]),
                'mean_altitude': float(np.mean(debris_df['altitude_km']))
            }
        
        # Inclination distribution
        if 'inclination' in debris_df.columns:
            inc_bins = np.linspace(0, 180, 36)
            inc_hist, inc_edges = np.histogram(debris_df['inclination'], bins=inc_bins)
            
            results['inclination_distribution'] = {
                'bin_edges': inc_edges.tolist(),
                'counts': inc_hist.tolist(),
                'peak_inclination': float(inc_edges[np.argmax(inc_hist)]),
                'mean_inclination': float(np.mean(debris_df['inclination']))
            }
        
        # Orbital zone counts
        if 'orbital_zone' in debris_df.columns:
            zone_counts = debris_df['orbital_zone'].value_counts()
            results['orbital_zones'] = zone_counts.to_dict()
        
        return results
    
    @staticmethod
    def calculate_risk_distribution(collision_risks: Union[np.ndarray, pd.Series]) -> List[int]:
        """
        Calculate risk level distribution.
        
        Args:
            collision_risks: Array of collision risk values
            
        Returns:
            List with counts for [low, medium, high, critical] risk levels
        """
        risks = pd.Series(collision_risks).dropna()
        
        if len(risks) == 0:
            return [0, 0, 0, 0]
        
        low_count = len(risks[risks <= 0.3])
        medium_count = len(risks[(risks > 0.3) & (risks <= 0.5)])
        high_count = len(risks[(risks > 0.5) & (risks <= 0.7)])
        critical_count = len(risks[risks > 0.7])
        
        return [low_count, medium_count, high_count, critical_count]
    
    @staticmethod
    def calculate_collision_probability_matrix(debris_df: pd.DataFrame, 
                                            sample_size: int = 1000) -> Dict:
        """
        Calculate collision probability matrix for debris population.
        
        Args:
            debris_df: DataFrame with debris data
            sample_size: Number of objects to sample for analysis
            
        Returns:
            Dictionary with collision probability analysis
        """
        if debris_df.empty or len(debris_df) < 2:
            return {'error': 'Insufficient data for collision analysis'}
        
        # Sample data for computational efficiency
        if len(debris_df) > sample_size:
            sample_df = debris_df.sample(n=sample_size)
        else:
            sample_df = debris_df
        
        # Ensure we have position data
        position_cols = ['position_x_km', 'position_y_km', 'position_z_km']
        if not all(col in sample_df.columns for col in position_cols):
            # Generate approximate positions if not available
            sample_df = DebrisAnalysis._generate_approximate_positions(sample_df)
        
        # Calculate pairwise distances
        positions = sample_df[position_cols].values
        distances = distance.pdist(positions)
        distance_matrix = distance.squareform(distances)
        
        # Calculate collision probabilities (simplified model)
        collision_probs = 1.0 / (1.0 + distance_matrix**2)  # Inverse square model
        np.fill_diagonal(collision_probs, 0)  # No self-collision
        
        # Statistical analysis
        upper_triangle_mask = np.triu(np.ones_like(collision_probs, dtype=bool), k=1)
        upper_triangle_probs = collision_probs[upper_triangle_mask]
        
        return {
            'total_pairs': len(upper_triangle_probs),
            'mean_probability': float(np.mean(upper_triangle_probs)),
            'max_probability': float(np.max(upper_triangle_probs)),
            'high_risk_pairs': int(np.sum(upper_triangle_probs > 1e-4)),
            'probability_distribution': StatisticalAnalysis.calculate_distribution_statistics(upper_triangle_probs),
            'sample_size': len(sample_df)
        }
    
    @staticmethod
    def _generate_approximate_positions(debris_df: pd.DataFrame) -> pd.DataFrame:
        """Generate approximate positions from orbital elements."""
        enhanced_df = debris_df.copy()
        
        # Use altitude and random angles for approximate positions
        altitudes = enhanced_df.get('altitude_km', pd.Series([400] * len(enhanced_df)))
        radii = altitudes + EARTH_RADIUS_EQUATORIAL
        
        # Random distribution on sphere
        n_objects = len(enhanced_df)
        theta = np.random.uniform(0, 2*np.pi, n_objects)
        phi = np.random.uniform(0, np.pi, n_objects)
        
        enhanced_df['position_x_km'] = radii * np.sin(phi) * np.cos(theta)
        enhanced_df['position_y_km'] = radii * np.sin(phi) * np.sin(theta)
        enhanced_df['position_z_km'] = radii * np.cos(phi)
        
        return enhanced_df

class OptimizationUtilities:
    """
    Optimization and numerical methods for space debris analysis.
    """
    
    @staticmethod
    def optimize_debris_removal_sequence(debris_targets: pd.DataFrame,
                                       mission_constraints: Dict) -> Dict:
        """
        Optimize debris removal mission sequence.
        
        Args:
            debris_targets: DataFrame with target debris information
            mission_constraints: Dictionary with mission constraints
            
        Returns:
            Optimized removal sequence
        """
        if debris_targets.empty:
            return {'error': 'No targets provided'}
        
        # Extract target properties
        n_targets = len(debris_targets)
        
        # Value and cost arrays
        values = debris_targets.get('economic_value_usd', pd.Series([1000] * n_targets)).values
        removal_costs = debris_targets.get('removal_cost_usd', pd.Series([1e6] * n_targets)).values
        risk_levels = debris_targets.get('collision_risk', pd.Series([0.1] * n_targets)).values
        
        # Multi-objective optimization score
        # Maximize: value/cost ratio + risk reduction
        scores = (values / removal_costs) + (risk_levels * 1000)  # Weight risk highly
        
        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Apply mission constraints
        budget_limit = mission_constraints.get('budget_usd', 1e9)
        max_targets = mission_constraints.get('max_targets', 10)
        
        selected_indices = []
        total_cost = 0
        
        for idx in sorted_indices:
            if len(selected_indices) >= max_targets:
                break
            
            target_cost = removal_costs[idx]
            if total_cost + target_cost <= budget_limit:
                selected_indices.append(idx)
                total_cost += target_cost
        
        # Generate optimized sequence
        selected_targets = debris_targets.iloc[selected_indices]
        
        return {
            'selected_targets': selected_targets.to_dict('records'),
            'total_targets': len(selected_indices),
            'total_cost': total_cost,
            'total_value': float(np.sum(values[selected_indices])),
            'roi': float((np.sum(values[selected_indices]) - total_cost) / total_cost) if total_cost > 0 else 0,
            'budget_utilization': total_cost / budget_limit if budget_limit > 0 else 0,
            'optimization_score': float(np.sum(scores[selected_indices]))
        }
    
    @staticmethod
    def monte_carlo_orbital_uncertainty(orbital_elements: Dict,
                                      uncertainties: Dict,
                                      n_samples: int = 10000,
                                      propagation_time: float = 86400) -> Dict:
        """
        Monte Carlo analysis of orbital uncertainty propagation.
        
        Args:
            orbital_elements: Nominal orbital elements
            uncertainties: Uncertainty values for each element
            n_samples: Number of Monte Carlo samples
            propagation_time: Propagation time in seconds
            
        Returns:
            Uncertainty analysis results
        """
        # Generate sample distributions
        samples = {}
        for element, nominal_value in orbital_elements.items():
            uncertainty = uncertainties.get(element, nominal_value * 0.01)  # 1% default
            samples[element] = np.random.normal(nominal_value, uncertainty, n_samples)
        
        # Propagate each sample (simplified)
        final_positions = []
        final_altitudes = []
        
        for i in range(n_samples):
            # Simple propagation using mean motion
            a = samples['semi_major_axis'][i]
            e = samples['eccentricity'][i]
            
            # Calculate final altitude (very simplified)
            mean_motion = np.sqrt(EARTH_GRAVITATIONAL_PARAMETER / a**3)
            altitude_change = -0.1 * propagation_time / 86400  # Simplified decay
            final_altitude = a - EARTH_RADIUS_EQUATORIAL + altitude_change
            
            final_altitudes.append(final_altitude)
            
            # Approximate final position
            angle = mean_motion * propagation_time
            x = a * np.cos(angle)
            y = a * np.sin(angle)
            z = 0  # Simplified
            
            final_positions.append([x, y, z])
        
        final_positions = np.array(final_positions)
        final_altitudes = np.array(final_altitudes)
        
        # Calculate uncertainty statistics
        position_uncertainty = np.std(final_positions, axis=0)
        altitude_uncertainty = np.std(final_altitudes)
        
        return {
            'propagation_time_hours': propagation_time / 3600,
            'n_samples': n_samples,
            'position_uncertainty_km': {
                'x': float(position_uncertainty[0]),
                'y': float(position_uncertainty[1]),
                'z': float(position_uncertainty[2]),
                'total': float(np.linalg.norm(position_uncertainty))
            },
            'altitude_uncertainty_km': float(altitude_uncertainty),
            'final_altitude_statistics': StatisticalAnalysis.calculate_distribution_statistics(final_altitudes),
            'uncertainty_growth_factor': float(altitude_uncertainty / uncertainties.get('semi_major_axis', 1))
        }
    
    @staticmethod
    def numerical_integration_orbit(initial_state: np.ndarray,
                                  acceleration_function: Callable,
                                  time_span: Tuple[float, float],
                                  rtol: float = 1e-8) -> Dict:
        """
        High-precision numerical integration of orbital motion.
        
        Args:
            initial_state: Initial state vector [x, y, z, vx, vy, vz]
            acceleration_function: Function that returns acceleration
            time_span: Integration time span (start, end) in seconds
            rtol: Relative tolerance for integration
            
        Returns:
            Integration results
        """
        def derivatives(t, state):
            position = state[:3]
            velocity = state[3:]
            acceleration = acceleration_function(t, position, velocity)
            return np.concatenate([velocity, acceleration])
        
        # Solve ODE
        try:
            solution = integrate.solve_ivp(
                derivatives,
                time_span,
                initial_state,
                method='Radau',  # Implicit method for better stability
                rtol=rtol,
                atol=1e-12,
                dense_output=True
            )
            
            if solution.success:
                return {
                    'success': True,
                    'final_state': solution.y[:, -1].tolist(),
                    'integration_steps': len(solution.t),
                    'final_time': float(solution.t[-1]),
                    'max_error_estimate': float(np.max(np.abs(solution.y - solution.sol(solution.t))))
                }
            else:
                return {
                    'success': False,
                    'error': solution.message
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class MissionPlanningUtilities:
    """
    Utilities for space mission planning and analysis.
    """
    
    @staticmethod
    def calculate_launch_window(target_orbit: Dict,
                               launch_site_lat: float,
                               launch_constraints: Dict) -> List[Dict]:
        """
        Calculate optimal launch windows for debris removal missions.
        
        Args:
            target_orbit: Target orbital parameters
            launch_site_lat: Launch site latitude in degrees
            launch_constraints: Launch constraints dictionary
            
        Returns:
            List of launch windows
        """
        target_inclination = target_orbit.get('inclination', 45)
        
        # Check if launch site can reach target inclination
        min_inclination = abs(launch_site_lat)
        
        if target_inclination < min_inclination:
            return [{'error': f'Cannot reach inclination {target_inclination}° from latitude {launch_site_lat}°'}]
        
        # Calculate launch windows (simplified)
        windows = []
        
        # Daily launch opportunities
        for day in range(30):  # Next 30 days
            launch_date = datetime.now() + timedelta(days=day)
            
            # Morning and evening windows
            for hour in [6, 18]:  # 6 AM and 6 PM local time
                window = {
                    'date': launch_date.date().isoformat(),
                    'time': f"{hour:02d}:00",
                    'inclination_achievable': target_inclination,
                    'delta_v_penalty': abs(target_inclination - launch_site_lat) * 10,  # m/s per degree
                    'weather_probability': 0.8,  # 80% good weather assumption
                    'suitability_score': 100 - abs(target_inclination - launch_site_lat) * 2
                }
                windows.append(window)
        
        # Sort by suitability score
        windows.sort(key=lambda x: x['suitability_score'], reverse=True)
        
        return windows[:10]  # Return top 10 windows
    
    @staticmethod
    def analyze_mission_risk(mission_parameters: Dict) -> Dict:
        """
        Comprehensive mission risk analysis.
        
        Args:
            mission_parameters: Mission configuration parameters
            
        Returns:
            Risk analysis results
        """
        risks = {
            'technical': 0.1,    # 10% base technical risk
            'launch': 0.05,      # 5% launch risk
            'operational': 0.15, # 15% operational risk
            'weather': 0.1,      # 10% weather risk
            'regulatory': 0.05   # 5% regulatory risk
        }
        
        # Adjust risks based on mission parameters
        mission_duration = mission_parameters.get('duration_years', 2)
        complexity_factor = mission_parameters.get('complexity_factor', 1.0)
        
        # Duration effects
        risks['technical'] *= (1 + mission_duration * 0.1)
        risks['operational'] *= (1 + mission_duration * 0.05)
        
        # Complexity effects
        for risk_type in risks:
            risks[risk_type] *= complexity_factor
        
        # Calculate overall risk
        overall_risk = 1 - np.prod([1 - risk for risk in risks.values()])
        
        # Risk mitigation suggestions
        mitigation_strategies = []
        
        if risks['technical'] > 0.2:
            mitigation_strategies.append("Implement redundant systems")
        
        if risks['operational'] > 0.3:
            mitigation_strategies.append("Enhance crew training and procedures")
        
        if overall_risk > 0.4:
            mitigation_strategies.append("Consider mission redesign to reduce complexity")
        
        return {
            'individual_risks': risks,
            'overall_risk': float(overall_risk),
            'risk_level': 'High' if overall_risk > 0.4 else 'Medium' if overall_risk > 0.2 else 'Low',
            'mitigation_strategies': mitigation_strategies,
            'success_probability': float(1 - overall_risk)
        }

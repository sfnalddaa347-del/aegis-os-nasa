# -*- coding: utf-8 -*-
"""
Advanced collision detection and conjunction analysis
High-precision algorithms for space debris collision prediction
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

from .constants import *
from .orbital_mechanics import EnhancedSGP4Propagator, solve_kepler_equation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedCollisionDetector:
    """
    Advanced collision detection system with high-precision conjunction analysis.
    Implements multiple screening methods and Monte Carlo collision probability calculations.
    """
    
    def __init__(self):
        self.detector_version = "AEGIS-CollisionDetector-v5.0"
        self.propagator = EnhancedSGP4Propagator()
        
        # Screening parameters from constants
        self.screening_params = CONJUNCTION_ANALYSIS_PARAMETERS
        self.monte_carlo_params = MC_SIMULATION_PARAMETERS
        
        # Physical parameters
        self.hard_body_radius = HARD_BODY_RADIUS  # km
        self.min_reportable_probability = PROBABILITY_COLLISION_THRESHOLD
        
        # Screening thresholds
        self.primary_threshold = self.screening_params['screening_volume']['primary_threshold']
        self.secondary_threshold = self.screening_params['screening_volume']['secondary_threshold']
        self.final_threshold = self.screening_params['screening_volume']['final_threshold']
    
    def analyze_collision_risks(self, debris_data: pd.DataFrame,
                               time_horizon_hours: int = 72,
                               risk_threshold: float = 1e-6,
                               use_monte_carlo: bool = True) -> Dict[str, Any]:
        """
        Comprehensive collision risk analysis for debris population.
        
        Args:
            debris_data: DataFrame with orbital elements and object properties
            time_horizon_hours: Analysis time horizon in hours
            risk_threshold: Minimum collision probability to report
            use_monte_carlo: Whether to use Monte Carlo probability calculations
            
        Returns:
            Dictionary with collision analysis results
        """
        try:
            logger.info(f"Starting collision analysis for {len(debris_data)} objects over {time_horizon_hours} hours")
            
            if debris_data.empty:
                return self._get_default_analysis_result()
            
            # Step 1: Primary screening - identify potential conjunctions
            potential_conjunctions = self._primary_screening(debris_data, time_horizon_hours)
            logger.info(f"Primary screening identified {len(potential_conjunctions)} potential conjunctions")
            
            # Step 2: Secondary screening - refined orbital propagation
            refined_conjunctions = self._secondary_screening(potential_conjunctions, debris_data)
            logger.info(f"Secondary screening refined to {len(refined_conjunctions)} conjunctions")
            
            # Step 3: Final screening - detailed probability calculations
            final_conjunctions = self._final_screening(refined_conjunctions, debris_data, use_monte_carlo)
            logger.info(f"Final screening identified {len(final_conjunctions)} reportable conjunctions")
            
            # Step 4: Risk categorization and analysis
            risk_analysis = self._analyze_conjunction_risks(final_conjunctions, risk_threshold)
            
            # Step 5: Generate comprehensive results
            results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'time_horizon_hours': time_horizon_hours,
                'objects_analyzed': len(debris_data),
                'screening_results': {
                    'primary_conjunctions': len(potential_conjunctions),
                    'secondary_conjunctions': len(refined_conjunctions), 
                    'final_conjunctions': len(final_conjunctions)
                },
                'all_conjunctions': final_conjunctions,
                'high_risk_conjunctions': [c for c in final_conjunctions if c.get('probability', 0) > risk_threshold],
                'risk_analysis': risk_analysis,
                'screening_efficiency': self._calculate_screening_efficiency(potential_conjunctions, final_conjunctions),
                'computational_performance': self._get_performance_metrics()
            }
            
            logger.info(f"Collision analysis complete. Found {len(results['high_risk_conjunctions'])} high-risk conjunctions")
            return results
            
        except Exception as e:
            logger.error(f"Error in collision risk analysis: {e}")
            return self._get_default_analysis_result()
    
    def _primary_screening(self, debris_data: pd.DataFrame, time_horizon_hours: int) -> List[Dict]:
        """Primary screening using simple distance calculations."""
        try:
            conjunctions = []
            
            # Limit analysis size for performance
            max_objects = min(len(debris_data), PERFORMANCE_LIMITS['max_objects_real_time'])
            sample_data = debris_data.head(max_objects).copy()
            
            # Convert to numpy arrays for vectorized operations
            altitudes = sample_data['altitude_km'].values
            inclinations = sample_data['inclination'].values
            
            # Pairwise distance matrix (simplified using altitude and inclination)
            for i in range(len(sample_data)):
                for j in range(i + 1, len(sample_data)):
                    # Quick orbital distance approximation
                    alt_diff = abs(altitudes[i] - altitudes[j])
                    inc_diff = abs(inclinations[i] - inclinations[j])
                    
                    # Rough distance estimate in km
                    orbital_separation = np.sqrt(alt_diff**2 + (inc_diff * np.pi/180 * (altitudes[i] + altitudes[j])/2)**2)
                    
                    # Check if within primary screening threshold
                    if orbital_separation < self.primary_threshold:
                        # Estimate time to closest approach (simplified)
                        period_i = sample_data.iloc[i].get('period_minutes', 90)
                        period_j = sample_data.iloc[j].get('period_minutes', 90)
                        
                        # Simple relative motion estimate
                        rel_motion_rate = abs(1/period_i - 1/period_j)  # relative orbital rate
                        if rel_motion_rate > 0:
                            time_to_ca = min(time_horizon_hours, 24)  # Approximate
                        else:
                            time_to_ca = time_horizon_hours / 2
                        
                        if time_to_ca <= time_horizon_hours:
                            conjunctions.append({
                                'object1_idx': i,
                                'object2_idx': j,
                                'object1_id': sample_data.iloc[i].get('name', f'Object_{i}'),
                                'object2_id': sample_data.iloc[j].get('name', f'Object_{j}'),
                                'estimated_separation_km': orbital_separation,
                                'estimated_time_to_ca_hours': time_to_ca,
                                'screening_level': 'primary'
                            })
            
            return conjunctions
            
        except Exception as e:
            logger.error(f"Error in primary screening: {e}")
            return []
    
    def _secondary_screening(self, conjunctions: List[Dict], debris_data: pd.DataFrame) -> List[Dict]:
        """Secondary screening with improved orbital propagation."""
        try:
            refined_conjunctions = []
            
            for conjunction in conjunctions:
                try:
                    i = conjunction['object1_idx']
                    j = conjunction['object2_idx']
                    
                    obj1 = debris_data.iloc[i]
                    obj2 = debris_data.iloc[j]
                    
                    # Get orbital elements for both objects
                    elements1 = self._extract_orbital_elements(obj1)
                    elements2 = self._extract_orbital_elements(obj2)
                    
                    # Propagate orbits to find closest approach
                    closest_approach = self._find_closest_approach(elements1, elements2, 
                                                                 conjunction['estimated_time_to_ca_hours'])
                    
                    # Check if passes secondary screening threshold
                    if closest_approach['min_distance'] < self.secondary_threshold:
                        refined_conjunction = conjunction.copy()
                        refined_conjunction.update({
                            'min_distance_km': closest_approach['min_distance'],
                            'time_to_ca_hours': closest_approach['time_to_ca'],
                            'relative_velocity_km_s': closest_approach['relative_velocity'],
                            'ca_position1': closest_approach['position1'],
                            'ca_position2': closest_approach['position2'],
                            'screening_level': 'secondary'
                        })
                        refined_conjunctions.append(refined_conjunction)
                
                except Exception as e:
                    logger.warning(f"Error processing conjunction {conjunction.get('object1_id', 'unknown')}: {e}")
                    continue
            
            return refined_conjunctions
            
        except Exception as e:
            logger.error(f"Error in secondary screening: {e}")
            return conjunctions  # Return original if secondary screening fails
    
    def _final_screening(self, conjunctions: List[Dict], debris_data: pd.DataFrame, 
                        use_monte_carlo: bool = True) -> List[Dict]:
        """Final screening with detailed probability calculations."""
        try:
            final_conjunctions = []
            
            for conjunction in conjunctions:
                try:
                    # Check if passes final screening threshold
                    if conjunction['min_distance_km'] > self.final_threshold:
                        continue
                    
                    # Calculate collision probability
                    if use_monte_carlo:
                        probability = self._calculate_collision_probability_mc(conjunction, debris_data)
                    else:
                        probability = self._calculate_collision_probability_analytic(conjunction, debris_data)
                    
                    # Only include if above minimum reportable threshold
                    if probability >= self.min_reportable_probability:
                        final_conjunction = conjunction.copy()
                        final_conjunction.update({
                            'collision_probability': probability,
                            'calculation_method': 'monte_carlo' if use_monte_carlo else 'analytic',
                            'screening_level': 'final',
                            'risk_category': self._categorize_collision_risk(probability),
                            'confidence_level': 0.95 if use_monte_carlo else 0.80
                        })
                        final_conjunctions.append(final_conjunction)
                
                except Exception as e:
                    logger.warning(f"Error in final screening for conjunction: {e}")
                    continue
            
            return final_conjunctions
            
        except Exception as e:
            logger.error(f"Error in final screening: {e}")
            return []
    
    def _extract_orbital_elements(self, obj_data: pd.Series) -> Dict[str, float]:
        """Extract orbital elements from object data with defaults."""
        try:
            return {
                'semi_major_axis': obj_data.get('semi_major_axis', 
                                               obj_data.get('altitude_km', 500) + EARTH_RADIUS),
                'eccentricity': obj_data.get('eccentricity', 0.001),
                'inclination': obj_data.get('inclination', 45.0),
                'raan': obj_data.get('raan', 0.0),
                'arg_perigee': obj_data.get('arg_perigee', 0.0),
                'mean_anomaly': obj_data.get('mean_anomaly', 0.0)
            }
        except Exception as e:
            logger.error(f"Error extracting orbital elements: {e}")
            return {
                'semi_major_axis': 7000.0,
                'eccentricity': 0.001,
                'inclination': 45.0,
                'raan': 0.0,
                'arg_perigee': 0.0,
                'mean_anomaly': 0.0
            }
    
    def _find_closest_approach(self, elements1: Dict, elements2: Dict, 
                              max_time_hours: float) -> Dict[str, float]:
        """Find the time and distance of closest approach between two objects."""
        try:
            min_distance = float('inf')
            best_time = 0
            best_pos1 = np.zeros(3)
            best_pos2 = np.zeros(3)
            best_rel_vel = 0
            
            # Sample time points
            time_samples = np.linspace(0, max_time_hours * 3600, 100)  # Convert hours to seconds
            
            for dt in time_samples:
                # Propagate both objects
                try:
                    pos1, vel1, _ = self.propagator.propagate_orbit(elements1, dt)
                    pos2, vel2, _ = self.propagator.propagate_orbit(elements2, dt)
                    
                    # Calculate separation
                    separation = np.linalg.norm(pos1 - pos2)
                    
                    if separation < min_distance:
                        min_distance = separation
                        best_time = dt / 3600  # Convert back to hours
                        best_pos1 = pos1.copy()
                        best_pos2 = pos2.copy()
                        best_rel_vel = np.linalg.norm(vel1 - vel2)
                
                except Exception as e:
                    logger.warning(f"Error propagating orbit at dt={dt}: {e}")
                    continue
            
            return {
                'min_distance': min_distance,
                'time_to_ca': best_time,
                'position1': best_pos1,
                'position2': best_pos2,
                'relative_velocity': best_rel_vel
            }
            
        except Exception as e:
            logger.error(f"Error finding closest approach: {e}")
            return {
                'min_distance': 1000.0,  # Large distance if calculation fails
                'time_to_ca': max_time_hours / 2,
                'position1': np.zeros(3),
                'position2': np.zeros(3),
                'relative_velocity': 10.0
            }
    
    def _calculate_collision_probability_mc(self, conjunction: Dict, debris_data: pd.DataFrame) -> float:
        """Calculate collision probability using Monte Carlo simulation."""
        try:
            # Get object indices
            i = conjunction['object1_idx']
            j = conjunction['object2_idx']
            
            # Object properties
            obj1 = debris_data.iloc[i]
            obj2 = debris_data.iloc[j]
            
            # Estimate object radii (simplified)
            radius1 = self._estimate_object_radius(obj1)
            radius2 = self._estimate_object_radius(obj2)
            combined_radius = radius1 + radius2  # km
            
            # Position uncertainty (simplified)
            pos_uncertainty1 = np.eye(3) * (0.1)**2  # 100m uncertainty covariance
            pos_uncertainty2 = np.eye(3) * (0.1)**2
            
            # Monte Carlo simulation
            n_samples = self.monte_carlo_params['sample_sizes']['standard']
            collision_count = 0
            
            # Get closest approach positions
            ca_pos1 = np.array(conjunction.get('ca_position1', [0, 0, 0]))
            ca_pos2 = np.array(conjunction.get('ca_position2', [0, 0, 0]))
            
            for _ in range(n_samples):
                # Sample positions from uncertainty distributions
                sampled_pos1 = multivariate_normal.rvs(ca_pos1, pos_uncertainty1)
                sampled_pos2 = multivariate_normal.rvs(ca_pos2, pos_uncertainty2)
                
                # Calculate separation
                separation = np.linalg.norm(sampled_pos1 - sampled_pos2)
                
                # Check for collision
                if separation <= combined_radius:
                    collision_count += 1
            
            probability = collision_count / n_samples
            return max(probability, 1e-10)  # Avoid zero probability
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo probability calculation: {e}")
            return 1e-8  # Default low probability
    
    def _calculate_collision_probability_analytic(self, conjunction: Dict, debris_data: pd.DataFrame) -> float:
        """Calculate collision probability using analytic methods."""
        try:
            # Simplified analytic probability calculation
            min_distance = conjunction.get('min_distance_km', 1.0)
            rel_velocity = conjunction.get('relative_velocity_km_s', 10.0)
            
            # Get object radii
            i = conjunction['object1_idx']
            j = conjunction['object2_idx']
            
            obj1 = debris_data.iloc[i]
            obj2 = debris_data.iloc[j]
            
            radius1 = self._estimate_object_radius(obj1)
            radius2 = self._estimate_object_radius(obj2)
            combined_radius = radius1 + radius2
            
            # Position uncertainty (km)
            sigma_pos = 0.1  # 100m position uncertainty
            
            # Analytic collision probability (simplified Gaussian model)
            if min_distance > combined_radius + 3 * sigma_pos:
                # Well outside collision envelope
                probability = np.exp(-0.5 * ((min_distance - combined_radius) / sigma_pos)**2)
            else:
                # Within potential collision envelope
                probability = 1.0 - (min_distance / (combined_radius + 3 * sigma_pos))
                probability = max(0, probability)
            
            # Scale by relative velocity effect (higher velocity = lower probability)
            velocity_factor = 10.0 / max(rel_velocity, 1.0)
            probability *= velocity_factor
            
            return min(max(probability, 1e-10), 1e-3)  # Reasonable bounds
            
        except Exception as e:
            logger.error(f"Error in analytic probability calculation: {e}")
            return 1e-8
    
    def _estimate_object_radius(self, obj_data: pd.Series) -> float:
        """Estimate effective collision radius from object properties."""
        try:
            # Try to get size information
            if 'size_cm' in obj_data:
                return obj_data['size_cm'] / 200  # Convert cm to km
            elif 'radar_cross_section' in obj_data:
                # Estimate radius from RCS (assuming spherical)
                rcs = obj_data['radar_cross_section']  # m²
                radius_m = np.sqrt(rcs / np.pi)
                return radius_m / 1000  # Convert m to km
            elif 'mass_kg' in obj_data:
                # Estimate from mass (assuming aluminum sphere)
                mass = obj_data['mass_kg']
                density = 2700  # kg/m³ (aluminum)
                volume = mass / density  # m³
                radius_m = (3 * volume / (4 * np.pi))**(1/3)
                return radius_m / 1000  # Convert m to km
            else:
                # Default collision radius
                return self.hard_body_radius / 2  # km
                
        except Exception as e:
            logger.warning(f"Error estimating object radius: {e}")
            return self.hard_body_radius / 2
    
    def _categorize_collision_risk(self, probability: float) -> str:
        """Categorize collision risk level."""
        if probability >= 1e-3:
            return "Extreme"
        elif probability >= 1e-4:
            return "High" 
        elif probability >= 1e-5:
            return "Medium"
        elif probability >= 1e-6:
            return "Low"
        else:
            return "Minimal"
    
    def _analyze_conjunction_risks(self, conjunctions: List[Dict], risk_threshold: float) -> Dict[str, Any]:
        """Analyze overall conjunction risk statistics."""
        try:
            if not conjunctions:
                return {'total_conjunctions': 0, 'risk_summary': 'No conjunctions detected'}
            
            probabilities = [c.get('collision_probability', 0) for c in conjunctions]
            
            risk_categories = {
                'extreme': len([p for p in probabilities if p >= 1e-3]),
                'high': len([p for p in probabilities if 1e-4 <= p < 1e-3]),
                'medium': len([p for p in probabilities if 1e-5 <= p < 1e-4]),
                'low': len([p for p in probabilities if 1e-6 <= p < 1e-5]),
                'minimal': len([p for p in probabilities if p < 1e-6])
            }
            
            return {
                'total_conjunctions': len(conjunctions),
                'above_threshold': len([p for p in probabilities if p > risk_threshold]),
                'risk_categories': risk_categories,
                'statistics': {
                    'max_probability': max(probabilities) if probabilities else 0,
                    'mean_probability': np.mean(probabilities) if probabilities else 0,
                    'median_probability': np.median(probabilities) if probabilities else 0,
                    'std_probability': np.std(probabilities) if probabilities else 0
                },
                'risk_summary': self._generate_risk_summary(risk_categories),
                'urgent_actions_required': risk_categories['extreme'] > 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conjunction risks: {e}")
            return {'total_conjunctions': 0, 'error': str(e)}
    
    def _generate_risk_summary(self, risk_categories: Dict[str, int]) -> str:
        """Generate human-readable risk summary."""
        try:
            if risk_categories['extreme'] > 0:
                return f"CRITICAL: {risk_categories['extreme']} extreme risk conjunctions require immediate action"
            elif risk_categories['high'] > 0:
                return f"HIGH RISK: {risk_categories['high']} high-risk conjunctions detected"
            elif risk_categories['medium'] > 0:
                return f"MODERATE: {risk_categories['medium']} medium-risk conjunctions require monitoring"
            elif risk_categories['low'] > 0:
                return f"LOW RISK: {risk_categories['low']} low-risk conjunctions identified"
            else:
                return "MINIMAL RISK: No significant collision risks detected"
                
        except Exception as e:
            logger.error(f"Error generating risk summary: {e}")
            return "Risk assessment unavailable"
    
    def _calculate_screening_efficiency(self, primary_conjunctions: List[Dict], 
                                       final_conjunctions: List[Dict]) -> Dict[str, float]:
        """Calculate screening algorithm efficiency metrics."""
        try:
            primary_count = len(primary_conjunctions)
            final_count = len(final_conjunctions)
            
            if primary_count == 0:
                return {'efficiency': 0, 'reduction_ratio': 0}
            
            reduction_ratio = (primary_count - final_count) / primary_count
            efficiency = final_count / primary_count if primary_count > 0 else 0
            
            return {
                'screening_efficiency': efficiency,
                'reduction_ratio': reduction_ratio,
                'primary_count': primary_count,
                'final_count': final_count,
                'false_positive_reduction': reduction_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating screening efficiency: {e}")
            return {'efficiency': 0, 'error': str(e)}
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get computational performance metrics."""
        try:
            return {
                'screening_thresholds': {
                    'primary_km': self.primary_threshold,
                    'secondary_km': self.secondary_threshold,
                    'final_km': self.final_threshold
                },
                'monte_carlo_samples': self.monte_carlo_params['sample_sizes']['standard'],
                'min_reportable_probability': self.min_reportable_probability,
                'hard_body_radius_km': self.hard_body_radius,
                'algorithm_version': self.detector_version
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def _get_default_analysis_result(self) -> Dict[str, Any]:
        """Return default analysis result when detection fails."""
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'time_horizon_hours': 72,
            'objects_analyzed': 0,
            'screening_results': {
                'primary_conjunctions': 0,
                'secondary_conjunctions': 0,
                'final_conjunctions': 0
            },
            'all_conjunctions': [],
            'high_risk_conjunctions': [],
            'risk_analysis': {
                'total_conjunctions': 0,
                'risk_summary': 'No analysis performed - insufficient data',
                'urgent_actions_required': False
            },
            'screening_efficiency': {'efficiency': 0},
            'computational_performance': self._get_performance_metrics()
        }

class ConjunctionDataManager:
    """Manager for conjunction data persistence and historical analysis."""
    
    def __init__(self):
        self.data_cache = {}
        self.historical_conjunctions = []
    
    def store_conjunction_result(self, result: Dict[str, Any]):
        """Store conjunction analysis result for historical tracking."""
        try:
            timestamp = result.get('analysis_timestamp', datetime.now().isoformat())
            self.historical_conjunctions.append({
                'timestamp': timestamp,
                'total_conjunctions': result.get('screening_results', {}).get('final_conjunctions', 0),
                'high_risk_count': len(result.get('high_risk_conjunctions', [])),
                'max_probability': max([c.get('collision_probability', 0) 
                                      for c in result.get('all_conjunctions', [])], default=0)
            })
            
            # Keep only recent history (last 30 days)
            cutoff_time = datetime.now() - timedelta(days=30)
            self.historical_conjunctions = [
                h for h in self.historical_conjunctions 
                if datetime.fromisoformat(h['timestamp']) > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error storing conjunction result: {e}")
    
    def get_historical_trends(self) -> Dict[str, Any]:
        """Get historical conjunction trends."""
        try:
            if not self.historical_conjunctions:
                return {'trend': 'No historical data available'}
            
            df = pd.DataFrame(self.historical_conjunctions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate trends
            recent_count = df['total_conjunctions'].tail(7).mean()
            older_count = df['total_conjunctions'].head(7).mean() if len(df) > 7 else recent_count
            
            trend_direction = "increasing" if recent_count > older_count else "decreasing"
            
            return {
                'trend': trend_direction,
                'recent_average': recent_count,
                'historical_average': df['total_conjunctions'].mean(),
                'max_observed': df['total_conjunctions'].max(),
                'days_of_data': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error calculating historical trends: {e}")
            return {'trend': 'Analysis error', 'error': str(e)}

# Global instances
collision_detector = AdvancedCollisionDetector()
conjunction_manager = ConjunctionDataManager()


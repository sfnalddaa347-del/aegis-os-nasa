# -*- coding: utf-8 -*-
"""
AEGIS-OS v5.0 Kessler Syndrome Simulation Engine
Advanced Monte Carlo simulation of space debris cascade scenarios
Implements NASA Standard Break-up Model and evolutionary debris modeling
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp
from dataclasses import dataclass
import json
from modules.constants import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BreakupEvent:
    """Data class for collision/breakup events."""
    timestamp: datetime
    primary_object: str
    secondary_object: str
    collision_velocity: float  # km/s
    fragment_count: int
    total_mass: float  # kg
    altitude_km: float
    event_type: str  # 'collision', 'explosion', 'fragmentation'
    debris_generated: List[Dict[str, Any]]

@dataclass
class SimulationState:
    """Current state of the simulation."""
    time: datetime
    object_count: int
    collision_count: int
    fragmentation_count: int
    total_mass_kg: float
    active_objects: pd.DataFrame
    kessler_index: float
    critical_density_reached: bool

class KesslerSimulator:
    """Advanced Kessler Syndrome simulation with Monte Carlo methods."""
    
    def __init__(self):
        """Initialize Kessler simulation engine."""
        self.simulation_parameters = {
            'time_step_hours': 24,  # Daily time steps
            'max_simulation_years': 100,
            'monte_carlo_runs': 1000,
            'collision_probability_threshold': 1e-6,
            'fragmentation_model': 'NASA_SBM',  # NASA Standard Break-up Model
            'atmospheric_model': 'NRLMSISE00',
            'solar_activity_model': 'ENABLED'
        }
        
        # Collision cross-sections by object type
        self.collision_cross_sections = {
            'small_debris': 1e-8,    # km²
            'medium_debris': 1e-6,   # km²
            'large_debris': 1e-4,    # km²
            'satellite': 1e-5,       # km²
            'rocket_body': 1e-4      # km²
        }
        
        # NASA Standard Break-up Model parameters
        self.sbm_parameters = {
            'lc_coefficient': 0.1,        # Characteristic length coefficient
            'area_mass_ratio_mean': 0.02,  # m²/kg
            'area_mass_ratio_std': 0.01,   # Standard deviation
            'velocity_dispersion': 0.1,    # km/s
            'fragment_size_distribution': {
                'alpha': -2.5,  # Power law exponent for fragments
                'min_size_cm': 0.1,
                'max_size_cm': 100.0
            }
        }
        
        # Simulation state
        self.current_state = None
        self.simulation_history = []
        self.breakup_events = []
        
        logger.info("Kessler simulation engine initialized")
    
    def initialize_population(self, debris_catalog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Initialize space debris population for simulation."""
        try:
            if debris_catalog is not None and not debris_catalog.empty:
                # Use provided catalog
                population = debris_catalog.copy()
                logger.info(f"Using provided catalog with {len(population)} objects")
            else:
                # Generate synthetic population based on ORDEM/MASTER statistics
                population = self._generate_synthetic_population()
                logger.info(f"Generated synthetic population with {len(population)} objects")
            
            # Ensure required columns exist
            required_columns = [
                'object_id', 'altitude_km', 'mass_kg', 'size_category',
                'object_type', 'cross_section_m2', 'area_mass_ratio',
                'inclination', 'eccentricity', 'creation_date'
            ]
            
            for col in required_columns:
                if col not in population.columns:
                    population[col] = self._generate_missing_column(col, len(population))
            
            # Add simulation-specific attributes
            population['active'] = True
            population['collision_probability'] = 0.0
            population['time_to_decay'] = self._calculate_orbital_decay_times(population)
            population['last_updated'] = datetime.now()
            
            return population
            
        except Exception as e:
            logger.error(f"Error initializing population: {e}")
            return pd.DataFrame()
    
    def _generate_synthetic_population(self) -> pd.DataFrame:
        """Generate synthetic debris population based on statistical models."""
        objects = []
        object_id = 1
        
        # Define altitude shells with object densities (objects per km altitude)
        altitude_shells = [
            (200, 400, 50),    # VEO - Very Low Earth Orbit
            (400, 800, 200),   # LEO lower
            (800, 1200, 150),  # LEO upper
            (1200, 1600, 80),  # LEO/MEO transition
            (1600, 2000, 50),  # MEO lower
            (2000, 35786, 20), # MEO to GEO
            (35786, 35786, 100) # GEO belt
        ]
        
        for min_alt, max_alt, density in altitude_shells:
            num_objects = int(density * (max_alt - min_alt) / 100)  # Scale by altitude range
            
            for _ in range(num_objects):
                altitude = np.random.uniform(min_alt, max_alt)
                
                # Determine object type based on altitude and random distribution
                if altitude < 600:
                    object_types = ['small_debris', 'medium_debris', 'large_debris', 'satellite']
                    probabilities = [0.6, 0.25, 0.1, 0.05]
                elif altitude < 2000:
                    object_types = ['small_debris', 'medium_debris', 'satellite', 'rocket_body']
                    probabilities = [0.4, 0.3, 0.2, 0.1]
                else:
                    object_types = ['satellite', 'rocket_body', 'large_debris']
                    probabilities = [0.6, 0.3, 0.1]
                
                object_type = np.random.choice(object_types, p=probabilities)
                
                # Generate object properties based on type
                if object_type == 'small_debris':
                    mass = np.random.lognormal(mean=np.log(1), sigma=0.5)  # ~1 kg
                    size_category = 'small'
                elif object_type == 'medium_debris':
                    mass = np.random.lognormal(mean=np.log(50), sigma=0.8)  # ~50 kg
                    size_category = 'medium'
                elif object_type == 'large_debris':
                    mass = np.random.lognormal(mean=np.log(500), sigma=1.0)  # ~500 kg
                    size_category = 'large'
                elif object_type == 'satellite':
                    mass = np.random.lognormal(mean=np.log(1000), sigma=1.2)  # ~1000 kg
                    size_category = 'large'
                else:  # rocket_body
                    mass = np.random.lognormal(mean=np.log(3000), sigma=0.7)  # ~3000 kg
                    size_category = 'xlarge'
                
                # Calculate cross-section and area-to-mass ratio
                cross_section = self.collision_cross_sections.get(object_type, 1e-6) * np.random.uniform(0.5, 2.0)
                area_mass_ratio = cross_section / mass if mass > 0 else 0.01
                
                objects.append({
                    'object_id': f"SIM_{object_id:06d}",
                    'altitude_km': round(altitude, 2),
                    'mass_kg': round(mass, 2),
                    'size_category': size_category,
                    'object_type': object_type,
                    'cross_section_m2': cross_section,
                    'area_mass_ratio': area_mass_ratio,
                    'inclination': np.random.uniform(0, 180),
                    'eccentricity': np.random.exponential(0.05),  # Most orbits are nearly circular
                    'creation_date': datetime.now() - timedelta(days=np.random.randint(0, 365*20))
                })
                
                object_id += 1
        
        return pd.DataFrame(objects)
    
    def _generate_missing_column(self, column_name: str, length: int) -> List[Any]:
        """Generate missing column data based on column name."""
        if column_name == 'object_id':
            return [f"OBJ_{i:06d}" for i in range(length)]
        elif column_name == 'altitude_km':
            return np.random.uniform(200, 2000, length)
        elif column_name == 'mass_kg':
            return np.random.lognormal(mean=np.log(100), sigma=1.0, size=length)
        elif column_name == 'size_category':
            return np.random.choice(['small', 'medium', 'large'], size=length, p=[0.6, 0.3, 0.1])
        elif column_name == 'object_type':
            return np.random.choice(['debris', 'satellite', 'rocket_body'], size=length, p=[0.7, 0.2, 0.1])
        elif column_name == 'cross_section_m2':
            return np.random.uniform(1e-8, 1e-4, length)
        elif column_name == 'area_mass_ratio':
            return np.random.uniform(0.001, 0.1, length)
        elif column_name == 'inclination':
            return np.random.uniform(0, 180, length)
        elif column_name == 'eccentricity':
            return np.random.exponential(0.05, length)
        elif column_name == 'creation_date':
            return [datetime.now() - timedelta(days=np.random.randint(0, 365*10)) for _ in range(length)]
        else:
            return [None] * length
    
    def run_kessler_simulation(self, 
                              initial_population: pd.DataFrame,
                              simulation_years: int = 50,
                              monte_carlo_runs: int = 100) -> Dict[str, Any]:
        """
        Run comprehensive Kessler syndrome simulation.
        
        Args:
            initial_population: Initial debris/satellite population
            simulation_years: Years to simulate
            monte_carlo_runs: Number of Monte Carlo iterations
            
        Returns:
            Dictionary with simulation results and statistics
        """
        logger.info(f"Starting Kessler simulation: {simulation_years} years, {monte_carlo_runs} MC runs")
        
        try:
            # Initialize results storage
            mc_results = []
            
            # Run Monte Carlo simulations
            for run in range(monte_carlo_runs):
                if run % 10 == 0:
                    logger.info(f"Monte Carlo run {run}/{monte_carlo_runs}")
                
                # Run single simulation
                run_result = self._run_single_simulation(
                    initial_population.copy(), 
                    simulation_years
                )
                mc_results.append(run_result)
            
            # Analyze results
            simulation_results = self._analyze_monte_carlo_results(mc_results)
            
            # Calculate Kessler syndrome probability
            kessler_probability = self._calculate_kessler_probability(mc_results)
            
            # Generate summary report
            summary = {
                'simulation_parameters': {
                    'initial_objects': len(initial_population),
                    'simulation_years': simulation_years,
                    'monte_carlo_runs': monte_carlo_runs,
                    'time_step_hours': self.simulation_parameters['time_step_hours']
                },
                'kessler_syndrome_analysis': {
                    'probability_of_cascade': kessler_probability,
                    'critical_density_threshold': KESSLER_THRESHOLD,
                    'time_to_cascade_years': simulation_results.get('mean_time_to_cascade'),
                    'cascade_severity': simulation_results.get('cascade_severity_classification')
                },
                'population_evolution': simulation_results.get('population_statistics'),
                'collision_statistics': simulation_results.get('collision_statistics'),
                'economic_impact': self._calculate_economic_impact(simulation_results),
                'mitigation_recommendations': self._generate_mitigation_recommendations(simulation_results),
                'confidence_intervals': simulation_results.get('confidence_intervals'),
                'simulation_timestamp': datetime.now().isoformat()
            }
            
            logger.info("Kessler simulation completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error in Kessler simulation: {e}")
            return {'error': str(e), 'simulation_status': 'failed'}
    
    def _run_single_simulation(self, population: pd.DataFrame, years: int) -> Dict[str, Any]:
        """Run a single Monte Carlo simulation iteration."""
        try:
            # Initialize simulation state
            current_time = datetime.now()
            time_step = timedelta(hours=self.simulation_parameters['time_step_hours'])
            end_time = current_time + timedelta(days=years * 365)
            
            # Simulation history
            history = []
            collision_events = []
            breakup_events = []
            
            step = 0
            while current_time < end_time and len(population[population['active']]) > 0:
                active_pop = population[population['active']].copy()
                
                if len(active_pop) == 0:
                    break
                
                # Update orbital decay
                population = self._update_orbital_decay(population, time_step)
                
                # Calculate collision probabilities
                collision_matrix = self._calculate_collision_matrix(active_pop)
                
                # Process collisions
                new_collisions = self._process_collisions(population, collision_matrix)
                collision_events.extend(new_collisions)
                
                # Generate fragments from collisions
                for collision in new_collisions:
                    fragments = self._generate_collision_fragments(collision)
                    # Add fragments to population
                    if fragments:
                        fragment_df = pd.DataFrame(fragments)
                        population = pd.concat([population, fragment_df], ignore_index=True)
                
                # Record state
                active_objects = len(population[population['active']])
                total_mass = population[population['active']]['mass_kg'].sum()
                
                kessler_index = self._calculate_kessler_index(active_pop)
                
                history.append({
                    'time': current_time,
                    'active_objects': active_objects,
                    'total_mass_kg': total_mass,
                    'collisions_this_step': len(new_collisions),
                    'kessler_index': kessler_index,
                    'step': step
                })
                
                # Check for cascade conditions
                if kessler_index > 1.0 and not any(h.get('cascade_detected') for h in history):
                    history[-1]['cascade_detected'] = True
                    history[-1]['cascade_start_time'] = current_time
                
                current_time += time_step
                step += 1
                
                # Limit simulation length for performance
                if step > 1000:  # ~3 years at daily steps
                    break
            
            return {
                'history': history,
                'collision_events': collision_events,
                'breakup_events': breakup_events,
                'final_population': len(population[population['active']]),
                'cascade_occurred': any(h.get('cascade_detected') for h in history),
                'simulation_steps': step
            }
            
        except Exception as e:
            logger.error(f"Error in single simulation: {e}")
            return {'error': str(e), 'history': []}
    
    def _calculate_collision_matrix(self, population: pd.DataFrame) -> np.ndarray:
        """Calculate collision probability matrix between objects."""
        n = len(population)
        collision_matrix = np.zeros((n, n))
        
        if n < 2:
            return collision_matrix
        
        try:
            # Vectorized collision probability calculation
            altitudes = population['altitude_km'].values
            cross_sections = population['cross_section_m2'].values
            
            # Simple model: collision probability based on altitude proximity and cross-section
            for i in range(n):
                for j in range(i+1, n):
                    # Altitude difference factor
                    alt_diff = abs(altitudes[i] - altitudes[j])
                    alt_factor = np.exp(-alt_diff / 100)  # Exponential decay with altitude difference
                    
                    # Cross-section factor
                    combined_cross_section = cross_sections[i] + cross_sections[j]
                    
                    # Base collision probability (simplified)
                    base_prob = 1e-12 * combined_cross_section * alt_factor
                    
                    collision_matrix[i, j] = base_prob
                    collision_matrix[j, i] = base_prob
            
            return collision_matrix
            
        except Exception as e:
            logger.error(f"Error calculating collision matrix: {e}")
            return collision_matrix
    
    def _process_collisions(self, population: pd.DataFrame, collision_matrix: np.ndarray) -> List[BreakupEvent]:
        """Process potential collisions based on probability matrix."""
        collisions = []
        
        try:
            active_indices = population[population['active']].index.tolist()
            threshold = self.simulation_parameters['collision_probability_threshold']
            
            for i, idx_i in enumerate(active_indices):
                for j, idx_j in enumerate(active_indices[i+1:], i+1):
                    prob = collision_matrix[i, j] if i < len(collision_matrix) and j < len(collision_matrix[0]) else 0
                    
                    # Monte Carlo collision decision
                    if prob > threshold and np.random.random() < prob * 1e6:  # Scale probability
                        # Create collision event
                        obj1 = population.loc[idx_i]
                        obj2 = population.loc[idx_j]
                        
                        collision = BreakupEvent(
                            timestamp=datetime.now(),
                            primary_object=obj1['object_id'],
                            secondary_object=obj2['object_id'],
                            collision_velocity=np.random.uniform(5, 15),  # km/s
                            fragment_count=0,  # Will be calculated
                            total_mass=obj1['mass_kg'] + obj2['mass_kg'],
                            altitude_km=(obj1['altitude_km'] + obj2['altitude_km']) / 2,
                            event_type='collision',
                            debris_generated=[]
                        )
                        
                        collisions.append(collision)
                        
                        # Mark objects as inactive (destroyed in collision)
                        population.loc[idx_i, 'active'] = False
                        population.loc[idx_j, 'active'] = False
            
            return collisions
            
        except Exception as e:
            logger.error(f"Error processing collisions: {e}")
            return []
    
    def _generate_collision_fragments(self, collision: BreakupEvent) -> List[Dict[str, Any]]:
        """Generate debris fragments from collision using NASA SBM."""
        fragments = []
        
        try:
            # NASA Standard Break-up Model implementation
            total_mass = collision.total_mass
            collision_velocity = collision.collision_velocity
            
            # Calculate characteristic length
            lc = self.sbm_parameters['lc_coefficient'] * (total_mass ** (1/3))
            
            # Fragment size distribution (power law)
            alpha = self.sbm_parameters['fragment_size_distribution']['alpha']
            min_size = self.sbm_parameters['fragment_size_distribution']['min_size_cm']
            max_size = min(lc * 100, self.sbm_parameters['fragment_size_distribution']['max_size_cm'])
            
            # Number of fragments (empirical relation)
            num_fragments = min(1000, int(0.1 * (collision_velocity ** 1.5) * (total_mass ** 0.75)))
            
            collision.fragment_count = num_fragments
            
            for i in range(num_fragments):
                # Fragment size (power law distribution)
                size_cm = min_size * ((max_size/min_size) ** np.random.random()) ** (1/(alpha+1))
                
                # Fragment mass (assuming spherical with density)
                fragment_density = 2700  # kg/m³ (aluminum approximation)
                fragment_volume = (4/3) * np.pi * ((size_cm/200) ** 3)  # m³
                fragment_mass = fragment_density * fragment_volume
                
                # Fragment velocity (relative to collision)
                velocity_dispersion = self.sbm_parameters['velocity_dispersion']
                fragment_velocity_km_s = np.random.normal(0, velocity_dispersion)
                
                # Fragment orbital parameters (perturbed from collision location)
                altitude_perturbation = np.random.normal(0, 50)  # km
                fragment_altitude = max(100, collision.altitude_km + altitude_perturbation)
                
                fragment = {
                    'object_id': f"FRAG_{collision.primary_object}_{collision.secondary_object}_{i:04d}",
                    'altitude_km': round(fragment_altitude, 2),
                    'mass_kg': round(fragment_mass, 6),
                    'size_category': self._classify_fragment_size(size_cm),
                    'object_type': 'collision_debris',
                    'cross_section_m2': np.pi * ((size_cm/200) ** 2),  # m²
                    'area_mass_ratio': (np.pi * ((size_cm/200) ** 2)) / fragment_mass if fragment_mass > 0 else 0.01,
                    'inclination': np.random.uniform(0, 180),
                    'eccentricity': min(0.99, np.random.exponential(0.1)),
                    'creation_date': collision.timestamp,
                    'parent_collision': f"{collision.primary_object}_{collision.secondary_object}",
                    'fragment_velocity_km_s': fragment_velocity_km_s,
                    'active': True,
                    'collision_probability': 0.0,
                    'time_to_decay': self._estimate_fragment_decay_time(fragment_altitude, fragment_mass),
                    'last_updated': collision.timestamp
                }
                
                fragments.append(fragment)
            
            collision.debris_generated = fragments
            return fragments
            
        except Exception as e:
            logger.error(f"Error generating collision fragments: {e}")
            return []
    
    def _classify_fragment_size(self, size_cm: float) -> str:
        """Classify fragment size category."""
        if size_cm < 1:
            return 'nano'
        elif size_cm < 10:
            return 'small'
        elif size_cm < 100:
            return 'medium'
        else:
            return 'large'
    
    def _estimate_fragment_decay_time(self, altitude_km: float, mass_kg: float) -> Optional[float]:
        """Estimate orbital decay time for fragment in years."""
        if altitude_km > 800:
            return None  # Long-lived orbit
        
        # Simplified decay model
        base_time_years = (altitude_km - 200) / 100  # Rough approximation
        mass_factor = (mass_kg ** 0.2)  # Larger objects decay slower
        
        decay_time = max(0.1, base_time_years * mass_factor)
        return decay_time
    
    def _update_orbital_decay(self, population: pd.DataFrame, time_step: timedelta) -> pd.DataFrame:
        """Update population based on orbital decay."""
        try:
            step_years = time_step.total_seconds() / (365.25 * 24 * 3600)
            
            # Update time to decay for active objects
            active_mask = population['active'] == True
            decay_mask = population['time_to_decay'].notna()
            
            # Decrease time to decay
            update_mask = active_mask & decay_mask
            population.loc[update_mask, 'time_to_decay'] -= step_years
            
            # Mark objects as inactive if they've decayed
            decayed_mask = (population['time_to_decay'] <= 0) & active_mask
            population.loc[decayed_mask, 'active'] = False
            
            return population
            
        except Exception as e:
            logger.error(f"Error updating orbital decay: {e}")
            return population
    
    def _calculate_orbital_decay_times(self, population: pd.DataFrame) -> List[Optional[float]]:
        """Calculate orbital decay times for population."""
        decay_times = []
        
        for _, obj in population.iterrows():
            altitude = obj.get('altitude_km', 500)
            mass = obj.get('mass_kg', 100)
            area_mass_ratio = obj.get('area_mass_ratio', 0.01)
            
            if altitude > 800:
                decay_times.append(None)  # Stable orbit
            else:
                # Simplified decay model based on atmospheric drag
                base_time = (altitude - 200) / 50  # Years
                drag_factor = area_mass_ratio * 100  # Higher A/M ratio = faster decay
                
                decay_time = max(0.5, base_time / (1 + drag_factor))
                decay_times.append(decay_time)
        
        return decay_times
    
    def _calculate_kessler_index(self, population: pd.DataFrame) -> float:
        """Calculate Kessler syndrome index (simplified metric)."""
        try:
            if len(population) == 0:
                return 0.0
            
            # Factors contributing to Kessler syndrome risk
            # 1. Object density in critical regions
            critical_objects = population[
                (population['altitude_km'] >= 800) & 
                (population['altitude_km'] <= 1200)
            ]
            
            if len(critical_objects) == 0:
                return 0.0
            
            # 2. Mass density
            total_mass = critical_objects['mass_kg'].sum()
            altitude_range = 400  # km
            
            # 3. Collision cross-section density
            total_cross_section = critical_objects['cross_section_m2'].sum()
            
            # 4. Collision rate (simplified)
            collision_rate_factor = len(critical_objects) ** 1.5 / 1000
            
            # Normalized Kessler index (>1.0 indicates cascade conditions)
            kessler_index = (
                (len(critical_objects) / KESSLER_THRESHOLD) * 
                (total_mass / 1e6) * 
                (total_cross_section / 0.01) * 
                collision_rate_factor
            )
            
            return min(10.0, kessler_index)  # Cap at 10.0
            
        except Exception as e:
            logger.error(f"Error calculating Kessler index: {e}")
            return 0.0
    
    def _analyze_monte_carlo_results(self, mc_results: List[Dict]) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results."""
        try:
            if not mc_results:
                return {}
            
            # Extract key metrics from all runs
            final_populations = [r.get('final_population', 0) for r in mc_results if 'final_population' in r]
            cascade_occurred = [r.get('cascade_occurred', False) for r in mc_results if 'cascade_occurred' in r]
            collision_counts = []
            
            for result in mc_results:
                if 'collision_events' in result:
                    collision_counts.append(len(result['collision_events']))
                else:
                    collision_counts.append(0)
            
            # Calculate statistics
            analysis = {
                'population_statistics': {
                    'mean_final_population': np.mean(final_populations) if final_populations else 0,
                    'std_final_population': np.std(final_populations) if final_populations else 0,
                    'min_final_population': np.min(final_populations) if final_populations else 0,
                    'max_final_population': np.max(final_populations) if final_populations else 0,
                    'median_final_population': np.median(final_populations) if final_populations else 0
                },
                'collision_statistics': {
                    'mean_collisions': np.mean(collision_counts) if collision_counts else 0,
                    'std_collisions': np.std(collision_counts) if collision_counts else 0,
                    'total_collisions_simulated': np.sum(collision_counts) if collision_counts else 0
                },
                'cascade_statistics': {
                    'cascade_probability': np.mean(cascade_occurred) if cascade_occurred else 0,
                    'cascade_count': np.sum(cascade_occurred) if cascade_occurred else 0,
                    'total_runs': len(mc_results)
                },
                'confidence_intervals': self._calculate_confidence_intervals(final_populations)
            }
            
            # Classify cascade severity
            cascade_prob = analysis['cascade_statistics']['cascade_probability']
            if cascade_prob > 0.7:
                severity = 'Critical'
            elif cascade_prob > 0.3:
                severity = 'High'
            elif cascade_prob > 0.1:
                severity = 'Moderate'
            else:
                severity = 'Low'
            
            analysis['cascade_severity_classification'] = severity
            
            # Calculate mean time to cascade
            cascade_times = []
            for result in mc_results:
                if result.get('cascade_occurred') and 'history' in result:
                    for step in result['history']:
                        if step.get('cascade_detected'):
                            cascade_times.append(step['step'] * self.simulation_parameters['time_step_hours'] / (24 * 365))
                            break
            
            if cascade_times:
                analysis['mean_time_to_cascade'] = np.mean(cascade_times)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Monte Carlo results: {e}")
            return {}
    
    def _calculate_confidence_intervals(self, data: List[float]) -> Dict[str, float]:
        """Calculate confidence intervals for data."""
        if not data:
            return {}
        
        try:
            data = np.array(data)
            mean = np.mean(data)
            
            # Calculate percentiles for confidence intervals
            ci_90 = np.percentile(data, [5, 95])
            ci_95 = np.percentile(data, [2.5, 97.5])
            ci_99 = np.percentile(data, [0.5, 99.5])
            
            return {
                'mean': mean,
                'ci_90_lower': ci_90[0],
                'ci_90_upper': ci_90[1],
                'ci_95_lower': ci_95[0],
                'ci_95_upper': ci_95[1],
                'ci_99_lower': ci_99[0],
                'ci_99_upper': ci_99[1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            return {}
    
    def _calculate_kessler_probability(self, mc_results: List[Dict]) -> float:
        """Calculate probability of Kessler syndrome occurrence."""
        try:
            if not mc_results:
                return 0.0
            
            cascade_count = sum(1 for result in mc_results if result.get('cascade_occurred', False))
            total_runs = len(mc_results)
            
            return cascade_count / total_runs if total_runs > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Kessler probability: {e}")
            return 0.0
    
    def _calculate_economic_impact(self, simulation_results: Dict) -> Dict[str, float]:
        """Calculate economic impact of debris proliferation."""
        try:
            # Get collision statistics
            mean_collisions = simulation_results.get('collision_statistics', {}).get('mean_collisions', 0)
            cascade_probability = simulation_results.get('cascade_statistics', {}).get('cascade_probability', 0)
            
            # Economic impact factors
            collision_cost = 500e6  # $500M per collision (satellite loss + mission impact)
            cascade_cost = 50e9     # $50B for full cascade scenario
            insurance_factor = 1.5   # Insurance and indirect costs
            
            # Calculate expected economic impact
            collision_impact = mean_collisions * collision_cost * insurance_factor
            cascade_impact = cascade_probability * cascade_cost
            
            total_impact = collision_impact + cascade_impact
            
            return {
                'collision_economic_impact_usd': collision_impact,
                'cascade_economic_impact_usd': cascade_impact,
                'total_expected_impact_usd': total_impact,
                'impact_per_year_usd': total_impact / 50,  # Assuming 50-year simulation
                'insurance_premium_factor': insurance_factor
            }
            
        except Exception as e:
            logger.error(f"Error calculating economic impact: {e}")
            return {}
    
    def _generate_mitigation_recommendations(self, simulation_results: Dict) -> List[Dict[str, str]]:
        """Generate mitigation recommendations based on simulation results."""
        recommendations = []
        
        try:
            cascade_prob = simulation_results.get('cascade_statistics', {}).get('cascade_probability', 0)
            mean_collisions = simulation_results.get('collision_statistics', {}).get('mean_collisions', 0)
            
            # High cascade probability recommendations
            if cascade_prob > 0.5:
                recommendations.append({
                    'priority': 'Critical',
                    'category': 'Active Debris Removal',
                    'recommendation': 'Immediate implementation of active debris removal missions targeting high-risk objects in critical orbital regions (800-1200 km altitude).',
                    'estimated_cost_usd': '10B',
                    'time_frame': '5 years'
                })
                
                recommendations.append({
                    'priority': 'Critical',
                    'category': 'Launch Restrictions',
                    'recommendation': 'Temporary moratorium on new launches to critical orbital regions until debris density is reduced.',
                    'estimated_cost_usd': '5B',
                    'time_frame': '2 years'
                })
            
            # Medium collision rate recommendations
            if mean_collisions > 10:
                recommendations.append({
                    'priority': 'High',
                    'category': 'Collision Avoidance',
                    'recommendation': 'Enhanced conjunction assessment and mandatory collision avoidance maneuvers for all operational satellites.',
                    'estimated_cost_usd': '1B',
                    'time_frame': '1 year'
                })
            
            # General recommendations
            recommendations.extend([
                {
                    'priority': 'Medium',
                    'category': 'Design Standards',
                    'recommendation': 'Mandatory 25-year post-mission disposal rule for all new launches with enhanced compliance monitoring.',
                    'estimated_cost_usd': '500M',
                    'time_frame': 'Immediate'
                },
                {
                    'priority': 'Medium',
                    'category': 'Tracking Enhancement',
                    'recommendation': 'Deployment of next-generation space surveillance network with sub-centimeter tracking capability.',
                    'estimated_cost_usd': '2B',
                    'time_frame': '3 years'
                },
                {
                    'priority': 'Low',
                    'category': 'International Cooperation',
                    'recommendation': 'Establishment of global space traffic management authority with binding debris mitigation standards.',
                    'estimated_cost_usd': '100M',
                    'time_frame': '5 years'
                }
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating mitigation recommendations: {e}")
            return []
# -*- coding: utf-8 -*-
"""
Real-time data integration from space agencies and tracking networks
CelesTrak, Space-Track.org, NOAA, ESA data sources with enhanced error handling
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import os
from typing import Dict, List, Optional, Tuple, Any
import warnings
import logging
from .constants import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CelesTrakDataSource:
    """Enhanced interface to CelesTrak satellite tracking data with error handling."""
    
    def __init__(self):
        self.base_url = API_ENDPOINTS['celestrak']['base_url']
        self.timeout = API_ENDPOINTS['celestrak']['timeout_seconds']
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': f'AEGIS-OS/{AEGIS_VERSION} Space Debris Tracking System'
        })
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Rate limiting
    
    def _rate_limit(self):
        """Implement rate limiting to respect server resources."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _parse_tle_data(self, text_data: str) -> List[Dict[str, Any]]:
        """Parse TLE data from text format with enhanced error handling."""
        lines = text_data.strip().split('\n')
        objects_data = []
        
        for i in range(0, len(lines) - 2, 3):
            try:
                if i + 2 < len(lines):
                    name = lines[i].strip()
                    tle_line1 = lines[i + 1].strip()
                    tle_line2 = lines[i + 2].strip()
                    
                    # Validate TLE format
                    if not (tle_line1.startswith('1') and tle_line2.startswith('2')):
                        continue
                    
                    # Parse orbital elements from TLE
                    elements = self._parse_tle_elements(tle_line1, tle_line2)
                    if elements:
                        elements['name'] = name
                        elements['tle_line1'] = tle_line1
                        elements['tle_line2'] = tle_line2
                        elements['source'] = 'CelesTrak'
                        elements['last_updated'] = datetime.now()
                        objects_data.append(elements)
                        
            except Exception as e:
                logger.warning(f"Error parsing TLE for object {i//3 + 1}: {e}")
                continue
        
        return objects_data
    
    def _parse_tle_elements(self, line1: str, line2: str) -> Optional[Dict[str, float]]:
        """Extract orbital elements from TLE lines with validation."""
        try:
            # Parse from Line 1
            epoch_year = int(line1[18:20])
            epoch_day = float(line1[20:32])
            ndot_div2 = float(line1[33:43])
            nddot_div6 = float(line1[44:52].replace(' ', '0'))
            bstar = float(line1[53:61].replace(' ', '0'))
            
            # Parse from Line 2
            inclination = float(line2[8:16])
            raan = float(line2[17:25])
            eccentricity_str = line2[26:33].replace(' ', '0')
            eccentricity = float('0.' + eccentricity_str) if eccentricity_str else 0.0
            arg_perigee = float(line2[34:42])
            mean_anomaly = float(line2[43:51])
            mean_motion = float(line2[52:63])
            revolution_number = int(line2[63:68])
            
            # Calculate derived parameters
            semi_major_axis = (EARTH_GRAVITATIONAL_PARAMETER / (mean_motion * 2 * np.pi / 86400)**2)**(1/3)
            altitude_km = semi_major_axis - EARTH_RADIUS
            
            # Orbital period in minutes
            period_minutes = 1440 / mean_motion if mean_motion > 0 else 0
            
            return {
                'inclination': inclination,
                'raan': raan,
                'eccentricity': eccentricity,
                'arg_perigee': arg_perigee,
                'mean_anomaly': mean_anomaly,
                'mean_motion': mean_motion,
                'semi_major_axis': semi_major_axis,
                'altitude_km': altitude_km,
                'period_minutes': period_minutes,
                'epoch_year': epoch_year,
                'epoch_day': epoch_day,
                'ndot_div2': ndot_div2,
                'nddot_div6': nddot_div6,
                'bstar': bstar,
                'revolution_number': revolution_number
            }
            
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing TLE elements: {e}")
            return None
    
    def get_debris_catalog(self, category: str = "debris") -> pd.DataFrame:
        """
        Fetch comprehensive debris catalog from CelesTrak with error handling.
        
        Args:
            category: Debris category (debris, analyst, fengyun, cosmos, iridium)
            
        Returns:
            DataFrame with debris orbital data
        """
        self._rate_limit()
        
        try:
            # CelesTrak debris TLE URLs
            tle_urls = {
                'debris': f"{self.base_url}debris.txt",
                'analyst': f"{self.base_url}analyst.txt",
                'fengyun': f"{self.base_url}fengyun-1c-debris.txt",
                'cosmos': f"{self.base_url}cosmos-2251-debris.txt",
                'iridium': f"{self.base_url}iridium-33-debris.txt",
                'cerise': f"{self.base_url}cerise-debris.txt"
            }
            
            url = tle_urls.get(category, tle_urls['debris'])
            
            logger.info(f"Fetching debris data from: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse TLE data
            debris_data = self._parse_tle_data(response.text)
            
            if not debris_data:
                logger.warning("No valid debris data found")
                return pd.DataFrame()
            
            df = pd.DataFrame(debris_data)
            
            # Add derived classifications
            df['orbital_zone'] = df['altitude_km'].apply(self._classify_orbital_zone)
            df['size_category'] = df.apply(self._estimate_size_category, axis=1)
            df['collision_risk'] = df.apply(self._calculate_initial_risk, axis=1)
            df['category'] = category
            
            logger.info(f"Successfully loaded {len(df)} debris objects")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching CelesTrak data: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error in get_debris_catalog: {e}")
            return pd.DataFrame()
    
    def get_active_satellites(self) -> pd.DataFrame:
        """Fetch active satellite catalog with enhanced metadata."""
        self._rate_limit()
        
        try:
            satellite_urls = [
                f"{self.base_url}active.txt",
                f"{self.base_url}stations.txt",
                f"{self.base_url}visual.txt"
            ]
            
            all_satellites = []
            
            for url in satellite_urls:
                try:
                    response = self.session.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    
                    satellite_data = self._parse_tle_data(response.text)
                    all_satellites.extend(satellite_data)
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to fetch from {url}: {e}")
                    continue
            
            if not all_satellites:
                logger.warning("No satellite data retrieved")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_satellites)
            
            # Remove duplicates based on NORAD ID
            if 'name' in df.columns:
                df = df.drop_duplicates(subset=['name'], keep='first')
            
            # Add classifications
            df['orbital_zone'] = df['altitude_km'].apply(self._classify_orbital_zone)
            df['status'] = 'Active'
            df['object_type'] = 'Satellite'
            
            logger.info(f"Successfully loaded {len(df)} active satellites")
            return df
            
        except Exception as e:
            logger.error(f"Error in get_active_satellites: {e}")
            return pd.DataFrame()
    
    def _classify_orbital_zone(self, altitude_km: float) -> str:
        """Classify orbital zone based on altitude with enhanced categories."""
        if pd.isna(altitude_km):
            return "Unknown"
        
        for zone, (min_alt, max_alt) in ORBITAL_ZONES.items():
            if min_alt <= altitude_km <= max_alt:
                return zone
        
        if altitude_km < ORBITAL_ZONES['VEO'][0]:
            return "Suborbital"
        elif altitude_km > ORBITAL_ZONES['LUNAR'][1]:
            return "Deep Space"
        
        return "Unknown"
    
    def _estimate_size_category(self, row: pd.Series) -> str:
        """Estimate debris size category based on orbital characteristics."""
        # This is a simplified estimation - real implementation would use radar cross-section
        altitude = row.get('altitude_km', 0)
        
        if altitude < 400:
            return 'large'  # Objects in very low orbits likely large
        elif altitude < 1000:
            return 'medium'
        else:
            return 'small'
    
    def _calculate_initial_risk(self, row: pd.Series) -> float:
        """Calculate initial collision risk estimate."""
        altitude = row.get('altitude_km', 0)
        eccentricity = row.get('eccentricity', 0)
        
        # Higher risk for lower altitudes and higher eccentricities
        altitude_factor = max(0, (1000 - altitude) / 1000) if altitude > 0 else 0
        eccentricity_factor = min(eccentricity * 2, 1.0)
        
        base_risk = 0.1 * altitude_factor + 0.1 * eccentricity_factor
        return min(base_risk, 1.0)

class NOAASpaceWeatherSource:
    """Enhanced interface to NOAA Space Weather data with comprehensive indices."""
    
    def __init__(self):
        self.base_url = API_ENDPOINTS['noaa_space_weather']['base_url']
        self.timeout = API_ENDPOINTS['noaa_space_weather']['timeout_seconds']
        self.session = requests.Session()
        self.cache = {}
        self.cache_duration = 1800  # 30 minutes
    
    def _get_cached_data(self, key: str) -> Optional[Dict]:
        """Get cached data if still valid."""
        if key in self.cache:
            cached_time, data = self.cache[key]
            if time.time() - cached_time < self.cache_duration:
                return data
        return None
    
    def _set_cached_data(self, key: str, data: Dict):
        """Cache data with timestamp."""
        self.cache[key] = (time.time(), data)
    
    def get_solar_activity(self) -> Dict[str, Any]:
        """Get comprehensive solar activity indices with caching."""
        cached = self._get_cached_data('solar_activity')
        if cached:
            return cached
        
        try:
            # Solar flux data
            solar_data = self._fetch_solar_indices()
            
            # Geomagnetic data
            geomag_data = self._fetch_geomagnetic_indices()
            
            # Solar wind data
            solar_wind_data = self._fetch_solar_wind_data()
            
            # Combine all data
            activity_data = {
                **solar_data,
                **geomag_data,
                **solar_wind_data,
                'timestamp': datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(solar_data, geomag_data)
            }
            
            # Calculate atmospheric density impact
            activity_data['atmospheric_impact'] = self._calculate_atmospheric_impact(activity_data)
            
            self._set_cached_data('solar_activity', activity_data)
            return activity_data
            
        except Exception as e:
            logger.error(f"Error fetching solar activity data: {e}")
            return self._get_default_solar_activity()
    
    def _fetch_solar_indices(self) -> Dict[str, float]:
        """Fetch solar flux and sunspot data."""
        try:
            # F10.7 solar flux
            flux_url = f"{self.base_url}json/f107_cm_flux.json"
            response = self.session.get(flux_url, timeout=self.timeout)
            
            if response.status_code == 200:
                flux_data = response.json()
                latest_flux = flux_data[-1] if flux_data else {}
                
                f107 = float(latest_flux.get('f107', 150))
                f107_adj = float(latest_flux.get('f107_adj', f107))
                
                # Sunspot number
                ssn_url = f"{self.base_url}json/solar-cycle/observed-solar-cycle-indices.json"
                ssn_response = self.session.get(ssn_url, timeout=self.timeout)
                
                ssn = 80  # Default
                if ssn_response.status_code == 200:
                    ssn_data = ssn_response.json()
                    if ssn_data:
                        ssn = float(ssn_data[-1].get('ssn', 80))
                
                return {
                    'solar_flux_f107': f107,
                    'solar_flux_f107_adj': f107_adj,
                    'sunspot_number': ssn,
                    'solar_activity_level': self._classify_solar_activity(f107, ssn)
                }
            
        except Exception as e:
            logger.warning(f"Error fetching solar indices: {e}")
        
        return {
            'solar_flux_f107': 150,
            'solar_flux_f107_adj': 150,
            'sunspot_number': 80,
            'solar_activity_level': 'Moderate'
        }
    
    def _fetch_geomagnetic_indices(self) -> Dict[str, float]:
        """Fetch geomagnetic activity indices."""
        try:
            # Kp index
            kp_url = f"{self.base_url}json/planetary_k_index_1m.json"
            response = self.session.get(kp_url, timeout=self.timeout)
            
            kp_index = 3.0  # Default moderate activity
            ap_index = 15   # Default
            
            if response.status_code == 200:
                kp_data = response.json()
                if kp_data:
                    latest_kp = kp_data[-1]
                    kp_index = float(latest_kp.get('kp', 3.0))
                    ap_index = float(latest_kp.get('ap', 15))
            
            # Dst index (simplified estimation)
            dst_index = -20 if kp_index > 5 else -10 if kp_index > 3 else 0
            
            return {
                'kp_index': kp_index,
                'ap_index': ap_index,
                'dst_index': dst_index,
                'geomagnetic_activity_level': self._classify_geomagnetic_activity(kp_index),
                'storm_level': self._get_storm_level(kp_index)
            }
            
        except Exception as e:
            logger.warning(f"Error fetching geomagnetic indices: {e}")
        
        return {
            'kp_index': 3.0,
            'ap_index': 15,
            'dst_index': 0,
            'geomagnetic_activity_level': 'Quiet',
            'storm_level': 'None'
        }
    
    def _fetch_solar_wind_data(self) -> Dict[str, float]:
        """Fetch solar wind parameters."""
        try:
            # Solar wind speed and density
            sw_url = f"{self.base_url}json/solar-wind-speed-density.json"
            response = self.session.get(sw_url, timeout=self.timeout)
            
            if response.status_code == 200:
                sw_data = response.json()
                if sw_data:
                    latest_sw = sw_data[-1]
                    return {
                        'solar_wind_speed': float(latest_sw.get('wind_speed', 400)),
                        'solar_wind_density': float(latest_sw.get('density', 5.0)),
                        'solar_wind_pressure': float(latest_sw.get('pressure', 2.0))
                    }
            
        except Exception as e:
            logger.warning(f"Error fetching solar wind data: {e}")
        
        return {
            'solar_wind_speed': 400,
            'solar_wind_density': 5.0,
            'solar_wind_pressure': 2.0
        }
    
    def _classify_solar_activity(self, f107: float, ssn: float) -> str:
        """Classify overall solar activity level."""
        if f107 > 200 or ssn > 150:
            return "Very High"
        elif f107 > 175 or ssn > 100:
            return "High"
        elif f107 > 150 or ssn > 50:
            return "Moderate"
        elif f107 > 125 or ssn > 25:
            return "Low"
        else:
            return "Very Low"
    
    def _classify_geomagnetic_activity(self, kp: float) -> str:
        """Classify geomagnetic activity level."""
        for level, params in GEOMAGNETIC_INDICES.items():
            min_kp, max_kp = params['kp_range']
            if min_kp <= kp < max_kp:
                return params['description']
        return "Unknown"
    
    def _get_storm_level(self, kp: float) -> str:
        """Get geomagnetic storm level."""
        if kp >= 8:
            return "Extreme"
        elif kp >= 7:
            return "Severe"
        elif kp >= 6:
            return "Strong"
        elif kp >= 5:
            return "Moderate"
        elif kp >= 4:
            return "Minor"
        else:
            return "None"
    
    def _assess_data_quality(self, solar_data: Dict, geomag_data: Dict) -> str:
        """Assess overall data quality."""
        # Simple quality assessment based on data availability
        solar_quality = len([v for v in solar_data.values() if v is not None and v != 0])
        geomag_quality = len([v for v in geomag_data.values() if v is not None and v != 0])
        
        total_metrics = len(solar_data) + len(geomag_data)
        available_metrics = solar_quality + geomag_quality
        
        quality_ratio = available_metrics / total_metrics if total_metrics > 0 else 0
        
        if quality_ratio > 0.8:
            return "High"
        elif quality_ratio > 0.6:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_atmospheric_impact(self, activity_data: Dict) -> Dict[str, float]:
        """Calculate atmospheric density impact from space weather."""
        f107 = activity_data.get('solar_flux_f107', 150)
        kp = activity_data.get('kp_index', 3.0)
        
        # Empirical model for atmospheric density enhancement
        f107_factor = 1.0 + (f107 - 150) * 0.005  # 0.5% per flux unit
        kp_factor = 1.0 + kp * 0.15  # 15% per Kp unit
        
        density_enhancement = max(0.5, min(4.0, f107_factor * kp_factor))
        
        return {
            'density_enhancement_factor': density_enhancement,
            'drag_increase_percent': (density_enhancement - 1.0) * 100,
            'orbital_decay_acceleration': density_enhancement ** 1.5
        }
    
    def _get_default_solar_activity(self) -> Dict[str, Any]:
        """Return default solar activity data when APIs fail."""
        return {
            'solar_flux_f107': 150,
            'solar_flux_f107_adj': 150,
            'sunspot_number': 80,
            'solar_activity_level': 'Moderate',
            'kp_index': 3.0,
            'ap_index': 15,
            'dst_index': 0,
            'geomagnetic_activity_level': 'Quiet',
            'storm_level': 'None',
            'solar_wind_speed': 400,
            'solar_wind_density': 5.0,
            'solar_wind_pressure': 2.0,
            'atmospheric_impact': {
                'density_enhancement_factor': 1.0,
                'drag_increase_percent': 0.0,
                'orbital_decay_acceleration': 1.0
            },
            'timestamp': datetime.now().isoformat(),
            'data_quality': 'Low'
        }
    
    def get_space_weather_forecast(self, days: int = 7) -> List[Dict]:
        """Get space weather forecast with enhanced modeling."""
        try:
            current_activity = self.get_solar_activity()
            forecast = []
            
            # Simple autoregressive forecast model
            base_f107 = current_activity['solar_flux_f107']
            base_kp = current_activity['kp_index']
            
            for day in range(days):
                forecast_date = datetime.now() + timedelta(days=day)
                
                # Add trend and noise
                f107_trend = np.random.normal(0, 5)  # ±5 sfu daily variation
                kp_trend = np.random.normal(0, 0.5)  # ±0.5 Kp daily variation
                
                # Persistence with decay
                persistence_factor = 0.9 ** day
                
                f107_forecast = base_f107 + f107_trend * (1 - persistence_factor)
                kp_forecast = max(0, min(9, base_kp + kp_trend * (1 - persistence_factor)))
                
                forecast.append({
                    'date': forecast_date.isoformat(),
                    'solar_flux_f107': max(70, f107_forecast),
                    'kp_index': kp_forecast,
                    'solar_activity_level': self._classify_solar_activity(f107_forecast, 0),
                    'geomagnetic_activity_level': self._classify_geomagnetic_activity(kp_forecast),
                    'confidence': max(0.3, 0.9 - day * 0.1),  # Decreasing confidence
                    'forecast_model': 'Simple Autoregressive'
                })
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return []

class ESADebrisSource:
    """Interface to ESA Space Debris Office data and MASTER model."""
    
    def __init__(self):
        self.base_url = API_ENDPOINTS['esa_debris']['base_url']
        self.timeout = API_ENDPOINTS['esa_debris']['timeout_seconds']
        self.session = requests.Session()
    
    def get_master_catalog(self, altitude_range: Tuple[float, float] = (200, 2000)) -> pd.DataFrame:
        """
        Generate debris population based on ESA MASTER model statistics.
        
        Args:
            altitude_range: Altitude range for debris generation (km)
            
        Returns:
            DataFrame with synthetic debris population based on MASTER model
        """
        try:
            logger.info("Generating ESA MASTER-based debris population")
            
            debris_data = self._generate_master_population(altitude_range)
            
            if not debris_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(debris_data)
            df['source'] = 'ESA_MASTER'
            df['last_updated'] = datetime.now()
            df['data_type'] = 'Statistical_Model'
            
            # Add risk assessments
            df['collision_risk'] = df.apply(self._calculate_collision_risk, axis=1)
            df['orbital_zone'] = df['altitude_km'].apply(self._classify_orbital_zone)
            
            logger.info(f"Generated {len(df)} MASTER model objects")
            return df
            
        except Exception as e:
            logger.error(f"Error generating MASTER catalog: {e}")
            return pd.DataFrame()
    
    def _generate_master_population(self, altitude_range: Tuple[float, float]) -> List[Dict]:
        """Generate debris population based on MASTER model size distribution."""
        debris_data = []
        min_alt, max_alt = altitude_range
        
        # ESA MASTER size distribution (objects per size bin)
        size_distributions = [
            # (size_min_cm, size_max_cm, density_per_km3, total_population)
            (0.1, 1.0, 1e6, 150000),      # Small debris
            (1.0, 10.0, 1e4, 50000),      # Medium debris  
            (10.0, 100.0, 1e2, 5000),     # Large debris
            (100.0, 1000.0, 1e0, 500),    # Very large debris
        ]
        
        object_id = 1
        
        for size_min, size_max, spatial_density, population in size_distributions:
            # Calculate number of objects in altitude range
            altitude_volume = max_alt - min_alt
            objects_in_range = min(population, int(spatial_density * altitude_volume / 1000))
            
            for _ in range(objects_in_range):
                # Random orbital parameters
                altitude = self._sample_altitude_distribution(min_alt, max_alt)
                semi_major_axis = altitude + EARTH_RADIUS
                
                # Size within bin (log-normal distribution)
                size_cm = np.random.lognormal(
                    np.log(np.sqrt(size_min * size_max)), 
                    np.log(size_max / size_min) / 4
                )
                size_cm = max(size_min, min(size_max, size_cm))
                
                # Orbital elements with realistic distributions
                inclination = self._sample_inclination_distribution()
                eccentricity = self._sample_eccentricity_distribution()
                raan = np.random.uniform(0, 360)
                arg_perigee = np.random.uniform(0, 360)
                mean_anomaly = np.random.uniform(0, 360)
                
                # Calculate mean motion and period
                mean_motion = np.sqrt(EARTH_GRAVITATIONAL_PARAMETER / semi_major_axis**3) * 86400 / (2 * np.pi)
                period_minutes = 1440 / mean_motion if mean_motion > 0 else 0
                
                # Physical properties
                mass_kg = self._estimate_mass_from_size(size_cm)
                radar_cross_section = self._estimate_radar_cross_section(size_cm)
                material = self._assign_material_type(size_cm)
                
                debris_data.append({
                    'object_id': f"MASTER_{object_id:06d}",
                    'name': f"ESA MASTER Object {object_id}",
                    'size_cm': size_cm,
                    'altitude_km': altitude,
                    'semi_major_axis': semi_major_axis,
                    'eccentricity': eccentricity,
                    'inclination': inclination,
                    'raan': raan,
                    'arg_perigee': arg_perigee,
                    'mean_anomaly': mean_anomaly,
                    'mean_motion': mean_motion,
                    'period_minutes': period_minutes,
                    'mass_kg': mass_kg,
                    'radar_cross_section': radar_cross_section,
                    'material': material,
                    'size_category': self._get_size_category(size_cm),
                    'creation_mechanism': self._assign_creation_mechanism(),
                    'parent_object': self._assign_parent_object(altitude, inclination)
                })
                
                object_id += 1
        
        return debris_data
    
    def _sample_altitude_distribution(self, min_alt: float, max_alt: float) -> float:
        """Sample altitude with realistic LEO distribution."""
        # Higher density at lower altitudes (exponential decay)
        u = np.random.random()
        # Exponential distribution with scale parameter
        scale = (max_alt - min_alt) / 3
        altitude = min_alt + scale * (-np.log(1 - u))
        return min(max_alt, altitude)
    
    def _sample_inclination_distribution(self) -> float:
        """Sample inclination with realistic distribution."""
        # Bimodal distribution: LEO missions and SSO
        if np.random.random() < 0.6:
            # LEO missions (0-60 degrees)
            return np.random.exponential(15)
        else:
            # Sun-synchronous and polar orbits (80-100 degrees)
            return 90 + np.random.normal(8, 5)
    
    def _sample_eccentricity_distribution(self) -> float:
        """Sample eccentricity with realistic distribution."""
        # Most debris has low eccentricity
        e = np.random.beta(2, 10) * 0.4  # Beta distribution scaled to [0, 0.4]
        return min(0.99, e)
    
    def _estimate_mass_from_size(self, size_cm: float) -> float:
        """Estimate mass from size using material density assumptions."""
        # Assume spherical object with average density
        volume_m3 = (4/3) * np.pi * (size_cm/200)**3  # Convert cm to m
        
        # Average density of space debris (aluminum-dominated)
        avg_density = 2500  # kg/m³
        
        mass = volume_m3 * avg_density
        return max(0.001, mass)  # Minimum 1 gram
    
    def _estimate_radar_cross_section(self, size_cm: float) -> float:
        """Estimate radar cross section from size."""
        # Simplified model: RCS ≈ π * (effective_radius)²
        effective_radius_m = size_cm / 200  # Convert cm to m, assume sphere
        rcs_m2 = np.pi * effective_radius_m**2
        return max(1e-6, rcs_m2)  # Minimum RCS
    
    def _assign_material_type(self, size_cm: float) -> str:
        """Assign material type based on size and probability."""
        if size_cm < 1.0:
            # Small debris likely paint, insulation
            return np.random.choice(['aluminum', 'unknown'], p=[0.7, 0.3])
        elif size_cm < 10.0:
            # Medium debris from spacecraft components
            return np.random.choice(['aluminum', 'steel', 'titanium', 'carbon_fiber'], 
                                  p=[0.6, 0.2, 0.1, 0.1])
        else:
            # Large debris from major components
            return np.random.choice(['aluminum', 'steel', 'titanium'], 
                                  p=[0.5, 0.3, 0.2])
    
    def _get_size_category(self, size_cm: float) -> str:
        """Get size category based on debris classification."""
        for category, (min_size, max_size) in DEBRIS_SIZE_CATEGORIES.items():
            if min_size <= size_cm < max_size:
                return category
        return 'unknown'
    
    def _assign_creation_mechanism(self) -> str:
        """Assign debris creation mechanism."""
        mechanisms = [
            'Collision', 'Explosion', 'Breakup', 'Degradation', 
            'Mission-related', 'Unknown'
        ]
        probabilities = [0.3, 0.25, 0.2, 0.1, 0.1, 0.05]
        return np.random.choice(mechanisms, p=probabilities)
    
    def _assign_parent_object(self, altitude: float, inclination: float) -> str:
        """Assign likely parent object based on orbital characteristics."""
        # Simplified parent assignment based on orbital parameters
        if 400 <= altitude <= 450 and 50 <= inclination <= 55:
            return "ISS-related"
        elif 780 <= altitude <= 850 and 98 <= inclination <= 102:
            return "Fengyun-1C"
        elif 850 <= altitude <= 950 and 72 <= inclination <= 76:
            return "Cosmos-2251"
        elif 700 <= altitude <= 900:
            return "Various LEO missions"
        else:
            return "Unknown"
    
    def _calculate_collision_risk(self, row: pd.Series) -> float:
        """Calculate collision risk based on object characteristics."""
        altitude = row.get('altitude_km', 0)
        size = row.get('size_cm', 0)
        eccentricity = row.get('eccentricity', 0)
        
        # Risk factors
        altitude_risk = max(0, (1000 - altitude) / 1000) if altitude > 0 else 0
        size_risk = min(size / 100, 1.0)  # Larger objects pose higher risk
        eccentricity_risk = eccentricity * 2  # Higher eccentricity increases risk
        
        # Combined risk (weighted average)
        total_risk = (0.5 * altitude_risk + 0.3 * size_risk + 0.2 * eccentricity_risk)
        return min(1.0, total_risk)
    
    def _classify_orbital_zone(self, altitude_km: float) -> str:
        """Classify orbital zone based on altitude."""
        if pd.isna(altitude_km):
            return "Unknown"
        
        for zone, (min_alt, max_alt) in ORBITAL_ZONES.items():
            if min_alt <= altitude_km <= max_alt:
                return zone
        
        return "Unknown"

# Global data source instances for easy access
celestrak_source = CelesTrakDataSource()
noaa_source = NOAASpaceWeatherSource()
esa_source = ESADebrisSource()

def get_all_orbital_data() -> Dict[str, pd.DataFrame]:
    """Fetch data from all sources and return combined dataset."""
    logger.info("Fetching data from all sources...")
    
    results = {
        'debris': pd.DataFrame(),
        'satellites': pd.DataFrame(),
        'esa_model': pd.DataFrame(),
        'space_weather': {}
    }
    
    try:
        # Fetch debris data
        results['debris'] = celestrak_source.get_debris_catalog()
        
        # Fetch satellite data
        results['satellites'] = celestrak_source.get_active_satellites()
        
        # Fetch ESA model data (limited sample)
        results['esa_model'] = esa_source.get_master_catalog((200, 1000))
        
        # Fetch space weather
        results['space_weather'] = noaa_source.get_solar_activity()
        
        logger.info("Successfully fetched data from all sources")
        
    except Exception as e:
        logger.error(f"Error in get_all_orbital_data: {e}")
    
    return results

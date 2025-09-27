# -*- coding: utf-8 -*-
"""
AEGIS-OS v5.0 Real-Time Data Manager
Real-time data aggregation and processing for space debris tracking
Integrates multiple data sources for comprehensive situational awareness
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Any
import threading
import json
from modules.constants import *
from modules.data_sources import CelesTrakDataSource, NOAASpaceWeatherSource, ESADebrisSource

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeDataManager:
    """Real-time data manager for space debris tracking and monitoring."""
    
    def __init__(self):
        """Initialize real-time data manager with data sources."""
        self.celestrak = CelesTrakDataSource()
        self.noaa = NOAASpaceWeatherSource()
        self.esa = ESADebrisSource()
        
        # Data cache
        self.data_cache = {}
        self.cache_timestamps = {}
        self.cache_duration = 300  # 5 minutes default cache
        
        # System status
        self.system_status = {
            'CelesTrak': {'status': 'inactive', 'last_update': None, 'error_count': 0},
            'NOAA Space Weather': {'status': 'inactive', 'last_update': None, 'error_count': 0},
            'ESA MASTER': {'status': 'inactive', 'last_update': None, 'error_count': 0},
            'Real-time Tracking': {'status': 'inactive', 'last_update': None, 'error_count': 0}
        }
        
        # Background update thread
        self.update_thread = None
        self.update_interval = 300  # 5 minutes
        self.running = False
        
        logger.info("Real-time data manager initialized")
    
    def start_real_time_updates(self):
        """Start background thread for real-time data updates."""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._background_update_loop, daemon=True)
            self.update_thread.start()
            logger.info("Real-time updates started")
    
    def stop_real_time_updates(self):
        """Stop background real-time updates."""
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        logger.info("Real-time updates stopped")
    
    def _background_update_loop(self):
        """Background thread loop for continuous data updates."""
        while self.running:
            try:
                self.update_all_data_sources()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in background update loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def update_all_data_sources(self):
        """Update all data sources and system status."""
        logger.info("Updating all data sources")
        
        # Update CelesTrak data
        self._update_celestrak_status()
        
        # Update NOAA space weather
        self._update_noaa_status()
        
        # Update ESA data
        self._update_esa_status()
        
        # Update tracking status
        self._update_tracking_status()
        
        logger.info("All data sources updated")
    
    def _update_celestrak_status(self):
        """Update CelesTrak data source status."""
        try:
            # Try to fetch a small sample of data
            debris_sample = self.celestrak.get_debris_catalog()
            
            if not debris_sample.empty:
                self.system_status['CelesTrak']['status'] = 'active'
                self.system_status['CelesTrak']['error_count'] = 0
                self._cache_data('celestrak_sample', debris_sample.head(10))
            else:
                self.system_status['CelesTrak']['status'] = 'degraded'
            
            self.system_status['CelesTrak']['last_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"CelesTrak update error: {e}")
            self.system_status['CelesTrak']['status'] = 'inactive'
            self.system_status['CelesTrak']['error_count'] += 1
    
    def _update_noaa_status(self):
        """Update NOAA space weather status."""
        try:
            # Get current space weather data
            weather_data = self.noaa.get_solar_activity()
            
            if weather_data and 'solar_flux_f107' in weather_data:
                self.system_status['NOAA Space Weather']['status'] = 'active'
                self.system_status['NOAA Space Weather']['error_count'] = 0
                self._cache_data('space_weather', weather_data)
            else:
                self.system_status['NOAA Space Weather']['status'] = 'degraded'
            
            self.system_status['NOAA Space Weather']['last_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"NOAA update error: {e}")
            self.system_status['NOAA Space Weather']['status'] = 'inactive'
            self.system_status['NOAA Space Weather']['error_count'] += 1
    
    def _update_esa_status(self):
        """Update ESA MASTER data status."""
        try:
            # ESA MASTER is a model-based system, so we simulate status
            # In real implementation, this would check ESA API availability
            master_data = self.esa.get_master_catalog((400, 1000))  # Sample range
            
            if not master_data.empty:
                self.system_status['ESA MASTER']['status'] = 'active'
                self.system_status['ESA MASTER']['error_count'] = 0
                self._cache_data('esa_master_sample', master_data.head(10))
            else:
                self.system_status['ESA MASTER']['status'] = 'degraded'
            
            self.system_status['ESA MASTER']['last_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"ESA update error: {e}")
            self.system_status['ESA MASTER']['status'] = 'inactive'
            self.system_status['ESA MASTER']['error_count'] += 1
    
    def _update_tracking_status(self):
        """Update real-time tracking status."""
        try:
            # Generate simulated tracking data for demonstration
            tracking_data = self._generate_live_tracking_data()
            
            if tracking_data is not None and not tracking_data.empty:
                self.system_status['Real-time Tracking']['status'] = 'active'
                self.system_status['Real-time Tracking']['error_count'] = 0
                self._cache_data('live_tracking', tracking_data)
            else:
                self.system_status['Real-time Tracking']['status'] = 'degraded'
            
            self.system_status['Real-time Tracking']['last_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Tracking update error: {e}")
            self.system_status['Real-time Tracking']['status'] = 'inactive'
            self.system_status['Real-time Tracking']['error_count'] += 1
    
    def _cache_data(self, key: str, data: Any):
        """Cache data with timestamp."""
        self.data_cache[key] = data
        self.cache_timestamps[key] = time.time()
    
    def _get_cached_data(self, key: str, max_age: Optional[float] = None) -> Optional[Any]:
        """Get cached data if still valid."""
        if key not in self.data_cache or key not in self.cache_timestamps:
            return None
        
        max_age = max_age or self.cache_duration
        if time.time() - self.cache_timestamps[key] > max_age:
            return None
        
        return self.data_cache[key]
    
    def get_system_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current system status for all data sources."""
        # Update timestamps for display
        for source, status in self.system_status.items():
            if status['last_update']:
                time_diff = datetime.now() - status['last_update']
                status['minutes_since_update'] = int(time_diff.total_seconds() / 60)
            else:
                status['minutes_since_update'] = None
        
        return self.system_status
    
    def get_live_tracking_data(self) -> Optional[pd.DataFrame]:
        """Get current live tracking data."""
        # Check cache first
        cached_data = self._get_cached_data('live_tracking', max_age=60)  # 1 minute cache
        if cached_data is not None:
            return cached_data
        
        # Generate new tracking data
        tracking_data = self._generate_live_tracking_data()
        if tracking_data is not None:
            self._cache_data('live_tracking', tracking_data)
        
        return tracking_data
    
    def _generate_live_tracking_data(self) -> pd.DataFrame:
        """Generate simulated live tracking data for demonstration."""
        try:
            # In real implementation, this would come from tracking stations
            num_objects = 50
            current_time = datetime.now()
            
            # Generate random tracking data
            data = []
            for i in range(num_objects):
                # Simulate tracking of various objects
                object_id = f"TRACK_{i+1:04d}"
                
                # Random orbital parameters with realistic constraints
                altitude = np.random.uniform(300, 1500)  # km
                velocity = np.sqrt(EARTH_GRAVITATIONAL_PARAMETER / (altitude + EARTH_RADIUS))  # km/s
                latitude = np.random.uniform(-85, 85)  # degrees
                longitude = np.random.uniform(-180, 180)  # degrees
                
                # Object characteristics
                object_type = np.random.choice(['Debris', 'Satellite', 'Rocket Body'], p=[0.7, 0.2, 0.1])
                size_category = np.random.choice(['small', 'medium', 'large'], p=[0.6, 0.3, 0.1])
                
                # Risk assessment
                collision_risk = np.random.exponential(0.1)  # Exponential distribution
                risk_level = 'High' if collision_risk > 0.3 else 'Medium' if collision_risk > 0.1 else 'Low'
                
                # Tracking quality
                tracking_quality = np.random.uniform(0.7, 1.0)
                last_observation = current_time - timedelta(minutes=np.random.randint(1, 60))
                
                data.append({
                    'object_id': object_id,
                    'object_type': object_type,
                    'altitude_km': round(altitude, 2),
                    'velocity_km_s': round(velocity, 3),
                    'latitude_deg': round(latitude, 4),
                    'longitude_deg': round(longitude, 4),
                    'collision_risk': round(collision_risk, 4),
                    'risk_level': risk_level,
                    'size_category': size_category,
                    'tracking_quality': round(tracking_quality, 3),
                    'last_observation': last_observation,
                    'orbital_zone': self._classify_orbital_zone(altitude),
                    'time_to_reentry_hours': self._estimate_reentry_time(altitude, object_type),
                    'tracking_station': np.random.choice(['Station_A', 'Station_B', 'Station_C']),
                    'data_confidence': round(tracking_quality * 100, 1)
                })
            
            df = pd.DataFrame(data)
            
            # Sort by collision risk (highest first)
            df = df.sort_values('collision_risk', ascending=False)
            
            logger.info(f"Generated {len(df)} live tracking records")
            return df
            
        except Exception as e:
            logger.error(f"Error generating live tracking data: {e}")
            return pd.DataFrame()
    
    def _classify_orbital_zone(self, altitude_km: float) -> str:
        """Classify orbital zone based on altitude."""
        for zone, (min_alt, max_alt) in ORBITAL_ZONES.items():
            if min_alt <= altitude_km <= max_alt:
                return zone
        
        if altitude_km < ORBITAL_ZONES['VEO'][0]:
            return "Suborbital"
        elif altitude_km > ORBITAL_ZONES['LUNAR'][1]:
            return "Deep Space"
        
        return "Unknown"
    
    def _estimate_reentry_time(self, altitude_km: float, object_type: str) -> Optional[float]:
        """Estimate time to reentry in hours."""
        if altitude_km > 600:
            return None  # Stable orbit
        
        # Simplified reentry estimation
        base_time = (altitude_km - 200) * 24  # Hours
        
        # Object type affects drag
        type_factors = {'Debris': 0.8, 'Satellite': 1.0, 'Rocket Body': 1.2}
        factor = type_factors.get(object_type, 1.0)
        
        estimated_time = max(1, base_time * factor)
        return round(estimated_time, 1)
    
    def get_real_time_alerts(self) -> List[Dict[str, Any]]:
        """Get current real-time alerts and warnings."""
        alerts = []
        
        try:
            # Get live tracking data
            tracking_data = self.get_live_tracking_data()
            
            if tracking_data is not None and not tracking_data.empty:
                # High collision risk alerts
                high_risk_objects = tracking_data[tracking_data['collision_risk'] > 0.3]
                for _, obj in high_risk_objects.head(5).iterrows():  # Top 5 risks
                    alerts.append({
                        'type': 'collision_risk',
                        'severity': 'high',
                        'object_id': obj['object_id'],
                        'message': f"High collision risk detected for {obj['object_id']}",
                        'risk_value': obj['collision_risk'],
                        'timestamp': datetime.now()
                    })
                
                # Imminent reentry alerts
                reentry_objects = tracking_data[
                    (tracking_data['time_to_reentry_hours'].notna()) & 
                    (tracking_data['time_to_reentry_hours'] < 24)
                ]
                for _, obj in reentry_objects.head(3).iterrows():
                    alerts.append({
                        'type': 'reentry_warning',
                        'severity': 'medium',
                        'object_id': obj['object_id'],
                        'message': f"Object {obj['object_id']} expected to reenter in {obj['time_to_reentry_hours']:.1f} hours",
                        'reentry_time': obj['time_to_reentry_hours'],
                        'timestamp': datetime.now()
                    })
            
            # Space weather alerts
            weather_data = self._get_cached_data('space_weather')
            if weather_data and weather_data.get('kp_index', 0) > 5:
                alerts.append({
                    'type': 'space_weather',
                    'severity': 'medium',
                    'message': f"Elevated geomagnetic activity (Kp={weather_data['kp_index']:.1f})",
                    'kp_index': weather_data['kp_index'],
                    'timestamp': datetime.now()
                })
            
            # System status alerts
            for source, status in self.system_status.items():
                if status['status'] == 'inactive' and status['error_count'] > 3:
                    alerts.append({
                        'type': 'system_error',
                        'severity': 'low',
                        'message': f"{source} data source is experiencing issues",
                        'error_count': status['error_count'],
                        'timestamp': datetime.now()
                    })
        
        except Exception as e:
            logger.error(f"Error generating real-time alerts: {e}")
        
        return alerts
    
    def get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance and health metrics."""
        try:
            metrics = {
                'data_sources_active': sum(1 for status in self.system_status.values() if status['status'] == 'active'),
                'total_data_sources': len(self.system_status),
                'cache_size': len(self.data_cache),
                'update_frequency_minutes': self.update_interval / 60,
                'system_uptime_hours': 0,  # Would track actual uptime
                'data_quality_score': self._calculate_data_quality_score(),
                'last_full_update': max(
                    (status['last_update'] for status in self.system_status.values() if status['last_update']),
                    default=None
                ),
                'processing_status': 'Running' if self.running else 'Stopped',
                'memory_usage_mb': self._estimate_memory_usage()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score (0-1)."""
        active_sources = sum(1 for status in self.system_status.values() if status['status'] == 'active')
        total_sources = len(self.system_status)
        
        base_score = active_sources / total_sources if total_sources > 0 else 0
        
        # Adjust for recent updates
        recent_updates = sum(
            1 for status in self.system_status.values()
            if status['last_update'] and (datetime.now() - status['last_update']).total_seconds() < 600
        )
        
        recency_bonus = (recent_updates / total_sources) * 0.2 if total_sources > 0 else 0
        
        return min(1.0, base_score + recency_bonus)
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Simple estimation based on cached data
        total_objects = 0
        for data in self.data_cache.values():
            if isinstance(data, pd.DataFrame):
                total_objects += len(data)
            elif isinstance(data, (list, dict)):
                total_objects += len(data) if hasattr(data, '__len__') else 1
        
        # Rough estimate: 1KB per tracked object
        estimated_mb = (total_objects * 1024) / (1024 * 1024)
        return round(estimated_mb, 2)
    
    def cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items()
            if current_time - timestamp > self.cache_duration * 2
        ]
        
        for key in expired_keys:
            self.data_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_real_time_updates()
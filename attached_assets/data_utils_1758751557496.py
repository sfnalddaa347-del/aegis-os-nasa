# -*- coding: utf-8 -*-
"""
Data processing utilities for space debris analysis
TLE parsing, data validation, filtering, and preprocessing functions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import re
import json
import logging
import warnings
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_tle_line(line1: str, line2: str) -> Dict[str, float]:
    """
    Parse Two-Line Element (TLE) data with comprehensive error checking.
    
    Args:
        line1: First line of TLE
        line2: Second line of TLE
        
    Returns:
        Dictionary of orbital parameters
    """
    try:
        # Clean and validate input lines
        line1 = line1.strip()
        line2 = line2.strip()
        
        if len(line1) != 69 or len(line2) != 69:
            raise ValueError(f"Invalid TLE line length: {len(line1)}, {len(line2)}")
        
        if line1[0] != '1' or line2[0] != '2':
            raise ValueError("Invalid TLE line indicators")
        
        # Verify checksums
        if not _verify_tle_checksum(line1) or not _verify_tle_checksum(line2):
            logger.warning("TLE checksum verification failed")
        
        # Parse Line 1
        catalog_number = int(line1[2:7])
        classification = line1[7]
        intl_designator = line1[9:17].strip()
        
        # Epoch
        epoch_year = int(line1[18:20])
        epoch_year = epoch_year + 2000 if epoch_year < 57 else epoch_year + 1900
        epoch_day = float(line1[20:32])
        
        # Mean motion derivatives
        ndot_div2 = float(line1[33:43])
        nddot_div6_str = line1[44:52]
        nddot_div6 = _parse_exponential_notation(nddot_div6_str)
        
        # BSTAR drag term
        bstar_str = line1[53:61]
        bstar = _parse_exponential_notation(bstar_str)
        
        ephemeris_type = int(line1[62])
        element_number = int(line1[64:68])
        
        # Parse Line 2
        inclination = float(line2[8:16])
        raan = float(line2[17:25])
        
        # Eccentricity (implied decimal point)
        eccentricity_str = line2[26:33]
        eccentricity = float('0.' + eccentricity_str)
        
        arg_perigee = float(line2[34:42])
        mean_anomaly = float(line2[43:51])
        mean_motion = float(line2[52:63])
        revolution_number = int(line2[63:68])
        
        # Calculate derived parameters
        epoch_datetime = _tle_epoch_to_datetime(epoch_year, epoch_day)
        
        # Semi-major axis calculation
        mu = 398600.4418  # km³/s²
        n_rad_per_sec = mean_motion * 2 * np.pi / 86400  # Convert rev/day to rad/s
        semi_major_axis = (mu / n_rad_per_sec**2)**(1/3) if n_rad_per_sec > 0 else 0
        
        # Orbital period
        period_minutes = 1440 / mean_motion if mean_motion > 0 else 0
        
        # Perigee and apogee altitudes
        earth_radius = 6378.137  # km
        perigee_altitude = semi_major_axis * (1 - eccentricity) - earth_radius
        apogee_altitude = semi_major_axis * (1 + eccentricity) - earth_radius
        
        return {
            'catalog_number': catalog_number,
            'classification': classification,
            'intl_designator': intl_designator,
            'epoch_year': epoch_year,
            'epoch_day': epoch_day,
            'epoch_datetime': epoch_datetime.isoformat(),
            'ndot_div2': ndot_div2,
            'nddot_div6': nddot_div6,
            'bstar': bstar,
            'ephemeris_type': ephemeris_type,
            'element_number': element_number,
            'inclination': inclination,
            'raan': raan,
            'eccentricity': eccentricity,
            'arg_perigee': arg_perigee,
            'mean_anomaly': mean_anomaly,
            'mean_motion': mean_motion,
            'revolution_number': revolution_number,
            'semi_major_axis': semi_major_axis,
            'period_minutes': period_minutes,
            'perigee_altitude': perigee_altitude,
            'apogee_altitude': apogee_altitude,
            'altitude_km': semi_major_axis - earth_radius
        }
        
    except Exception as e:
        logger.error(f"Error parsing TLE: {e}")
        raise ValueError(f"TLE parsing failed: {e}")

def _verify_tle_checksum(line: str) -> bool:
    """Verify TLE line checksum."""
    try:
        if len(line) != 69:
            return False
        
        # Calculate checksum for first 68 characters
        checksum = 0
        for char in line[:68]:
            if char.isdigit():
                checksum += int(char)
            elif char == '-':
                checksum += 1
        
        calculated_checksum = checksum % 10
        provided_checksum = int(line[68])
        
        return calculated_checksum == provided_checksum
        
    except (ValueError, IndexError):
        return False

def _parse_exponential_notation(exp_str: str) -> float:
    """Parse TLE exponential notation (e.g., ' 12345-3' = 0.12345e-3)."""
    try:
        exp_str = exp_str.strip()
        if not exp_str or exp_str.isspace():
            return 0.0
        
        # Handle space-filled fields
        if all(c == ' ' for c in exp_str):
            return 0.0
        
        # Parse sign
        sign = 1
        if exp_str[0] == '-':
            sign = -1
            exp_str = exp_str[1:]
        elif exp_str[0] == '+':
            exp_str = exp_str[1:]
        elif exp_str[0] == ' ':
            exp_str = exp_str[1:]
        
        # Find exponent part
        if len(exp_str) >= 2:
            mantissa_str = exp_str[:-2]
            exponent_str = exp_str[-2:]
            
            # Parse exponent
            exp_sign = 1
            if exponent_str[0] == '-':
                exp_sign = -1
                exponent = int(exponent_str[1:])
            elif exponent_str[0] == '+':
                exponent = int(exponent_str[1:])
            else:
                exponent = int(exponent_str)
            
            # Parse mantissa
            mantissa = float('0.' + mantissa_str) if mantissa_str else 0.0
            
            return sign * mantissa * (10 ** (exp_sign * exponent))
        
        return 0.0
        
    except (ValueError, IndexError):
        return 0.0

def _tle_epoch_to_datetime(year: int, day_of_year: float) -> datetime:
    """Convert TLE epoch to datetime object."""
    try:
        # Create datetime for beginning of year
        base_date = datetime(year, 1, 1)
        
        # Add fractional days
        days_to_add = int(day_of_year) - 1  # Day 1 = January 1
        fraction = day_of_year - int(day_of_year)
        
        result_date = base_date + timedelta(days=days_to_add, seconds=fraction * 86400)
        return result_date
        
    except (ValueError, OverflowError):
        logger.error(f"Error converting TLE epoch: year={year}, day={day_of_year}")
        return datetime.now()

def validate_orbital_elements(elements: Dict[str, float], 
                            strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate orbital elements for physical consistency.
    
    Args:
        elements: Dictionary of orbital elements
        strict: Whether to apply strict validation rules
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    try:
        issues = []
        
        # Required elements
        required_elements = [
            'semi_major_axis', 'eccentricity', 'inclination',
            'raan', 'arg_perigee', 'mean_anomaly'
        ]
        
        for element in required_elements:
            if element not in elements:
                issues.append(f"Missing required element: {element}")
        
        if issues:
            return False, issues
        
        # Physical constraints
        sma = elements['semi_major_axis']
        ecc = elements['eccentricity']
        inc = elements['inclination']
        
        # Semi-major axis
        earth_radius = 6378.137  # km
        if sma < earth_radius + 100:  # Minimum 100 km altitude
            issues.append(f"Semi-major axis too small: {sma:.1f} km")
        
        if strict and sma > 100000:  # 100,000 km limit for strict validation
            issues.append(f"Semi-major axis exceptionally large: {sma:.1f} km")
        
        # Eccentricity
        if ecc < 0 or ecc >= 1:
            issues.append(f"Invalid eccentricity: {ecc:.6f}")
        
        if strict and ecc > 0.9:
            issues.append(f"Highly eccentric orbit: {ecc:.6f}")
        
        # Inclination
        if inc < 0 or inc > 180:
            issues.append(f"Invalid inclination: {inc:.2f}°")
        
        # RAAN and argument of perigee
        for angle_name in ['raan', 'arg_perigee', 'mean_anomaly']:
            if angle_name in elements:
                angle = elements[angle_name]
                if angle < 0 or angle >= 360:
                    issues.append(f"Invalid {angle_name}: {angle:.2f}°")
        
        # Perigee altitude check
        perigee_altitude = sma * (1 - ecc) - earth_radius
        if perigee_altitude < 0:
            issues.append(f"Perigee below Earth surface: {perigee_altitude:.1f} km")
        
        # Physical orbit check
        if 'mean_motion' in elements:
            n = elements['mean_motion']  # rev/day
            if n <= 0:
                issues.append(f"Invalid mean motion: {n:.6f} rev/day")
            elif strict and n > 20:  # Very high frequency
                issues.append(f"Exceptionally high mean motion: {n:.2f} rev/day")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        logger.error(f"Error validating orbital elements: {e}")
        return False, [f"Validation error: {e}"]

def filter_orbital_data(data: pd.DataFrame, 
                       filters: Dict[str, Any],
                       remove_invalid: bool = True) -> pd.DataFrame:
    """
    Filter orbital data based on specified criteria.
    
    Args:
        data: DataFrame with orbital data
        filters: Dictionary of filter criteria
        remove_invalid: Whether to remove physically invalid orbits
        
    Returns:
        Filtered DataFrame
    """
    try:
        if data.empty:
            return data
        
        filtered_data = data.copy()
        
        # Remove invalid orbits if requested
        if remove_invalid:
            valid_indices = []
            for idx, row in filtered_data.iterrows():
                elements = {
                    'semi_major_axis': row.get('semi_major_axis', 0),
                    'eccentricity': row.get('eccentricity', 0),
                    'inclination': row.get('inclination', 0),
                    'raan': row.get('raan', 0),
                    'arg_perigee': row.get('arg_perigee', 0),
                    'mean_anomaly': row.get('mean_anomaly', 0)
                }
                is_valid, _ = validate_orbital_elements(elements, strict=False)
                if is_valid:
                    valid_indices.append(idx)
            
            filtered_data = filtered_data.loc[valid_indices]
            logger.info(f"Removed {len(data) - len(filtered_data)} invalid orbits")
        
        # Apply filters
        for filter_name, filter_value in filters.items():
            if filter_name == 'altitude_range' and 'altitude_km' in filtered_data.columns:
                min_alt, max_alt = filter_value
                filtered_data = filtered_data[
                    (filtered_data['altitude_km'] >= min_alt) &
                    (filtered_data['altitude_km'] <= max_alt)
                ]
            
            elif filter_name == 'inclination_range' and 'inclination' in filtered_data.columns:
                min_inc, max_inc = filter_value
                filtered_data = filtered_data[
                    (filtered_data['inclination'] >= min_inc) &
                    (filtered_data['inclination'] <= max_inc)
                ]
            
            elif filter_name == 'eccentricity_max' and 'eccentricity' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['eccentricity'] <= filter_value]
            
            elif filter_name == 'object_type' and 'object_type' in filtered_data.columns:
                if isinstance(filter_value, list):
                    filtered_data = filtered_data[filtered_data['object_type'].isin(filter_value)]
                else:
                    filtered_data = filtered_data[filtered_data['object_type'] == filter_value]
            
            elif filter_name == 'age_days' and 'epoch_datetime' in filtered_data.columns:
                cutoff_date = datetime.now() - timedelta(days=filter_value)
                filtered_data['epoch_dt'] = pd.to_datetime(filtered_data['epoch_datetime'])
                filtered_data = filtered_data[filtered_data['epoch_dt'] >= cutoff_date]
                filtered_data = filtered_data.drop('epoch_dt', axis=1)
            
            elif filter_name == 'size_range' and 'size_cm' in filtered_data.columns:
                min_size, max_size = filter_value
                filtered_data = filtered_data[
                    (filtered_data['size_cm'] >= min_size) &
                    (filtered_data['size_cm'] <= max_size)
                ]
            
            elif filter_name == 'collision_risk_min' and 'collision_risk' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['collision_risk'] >= filter_value]
        
        logger.info(f"Filtered data: {len(data)} -> {len(filtered_data)} objects")
        return filtered_data
        
    except Exception as e:
        logger.error(f"Error filtering orbital data: {e}")
        return data

def standardize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names across different data sources.
    
    Args:
        data: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    try:
        # Define column mapping
        column_mapping = {
            # Common variations
            'NORAD_CAT_ID': 'catalog_number',
            'norad_id': 'catalog_number',
            'OBJECT_ID': 'catalog_number',
            'sat_id': 'catalog_number',
            
            # Orbital elements
            'INCLINATION': 'inclination',
            'inc': 'inclination',
            'RA_OF_ASC_NODE': 'raan',
            'raan_deg': 'raan',
            'ECCENTRICITY': 'eccentricity',
            'ecc': 'eccentricity',
            'ARG_OF_PERICENTER': 'arg_perigee',
            'arg_perigee_deg': 'arg_perigee',
            'MEAN_ANOMALY': 'mean_anomaly',
            'mean_anom': 'mean_anomaly',
            'MEAN_MOTION': 'mean_motion',
            'mean_motion_rev_per_day': 'mean_motion',
            
            # Physical properties
            'RCS_SIZE': 'radar_cross_section',
            'rcs': 'radar_cross_section',
            'SIZE': 'size_cm',
            'diameter': 'size_cm',
            'MASS': 'mass_kg',
            'mass': 'mass_kg',
            
            # Position and velocity
            'X': 'position_x_km',
            'Y': 'position_y_km', 
            'Z': 'position_z_km',
            'VX': 'velocity_x_km_s',
            'VY': 'velocity_y_km_s',
            'VZ': 'velocity_z_km_s',
            
            # Time information
            'EPOCH': 'epoch_datetime',
            'epoch': 'epoch_datetime',
            'observation_time': 'epoch_datetime',
            
            # Object information
            'OBJECT_NAME': 'name',
            'sat_name': 'name',
            'OBJECT_TYPE': 'object_type',
            'type': 'object_type',
            'COUNTRY_CODE': 'country',
            'country': 'country',
            'LAUNCH_DATE': 'launch_date',
            'launch': 'launch_date'
        }
        
        # Apply mapping
        standardized_data = data.rename(columns=column_mapping)
        
        # Ensure required columns exist with default values
        required_columns = {
            'catalog_number': 0,
            'name': 'Unknown',
            'object_type': 'Unknown',
            'inclination': 0.0,
            'raan': 0.0,
            'eccentricity': 0.0,
            'arg_perigee': 0.0,
            'mean_anomaly': 0.0,
            'mean_motion': 15.0,  # Typical LEO value
            'semi_major_axis': 7000.0,
            'altitude_km': 500.0,
            'epoch_datetime': datetime.now().isoformat()
        }
        
        for col, default_value in required_columns.items():
            if col not in standardized_data.columns:
                standardized_data[col] = default_value
        
        logger.info(f"Standardized {len(column_mapping)} column names")
        return standardized_data
        
    except Exception as e:
        logger.error(f"Error standardizing column names: {e}")
        return data

def detect_data_quality_issues(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect data quality issues in orbital dataset.
    
    Args:
        data: DataFrame to analyze
        
    Returns:
        Dictionary of quality assessment results
    """
    try:
        if data.empty:
            return {'status': 'empty', 'issues': ['Dataset is empty']}
        
        issues = []
        warnings = []
        statistics = {}
        
        # Missing data analysis
        missing_data = data.isnull().sum()
        total_cells = len(data) * len(data.columns)
        missing_percentage = (missing_data.sum() / total_cells) * 100
        
        if missing_percentage > 20:
            issues.append(f"High missing data percentage: {missing_percentage:.1f}%")
        elif missing_percentage > 5:
            warnings.append(f"Moderate missing data: {missing_percentage:.1f}%")
        
        statistics['missing_percentage'] = missing_percentage
        statistics['missing_by_column'] = missing_data.to_dict()
        
        # Duplicate detection
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate records")
        
        statistics['duplicate_count'] = duplicate_count
        
        # Orbital elements validation
        if 'eccentricity' in data.columns:
            invalid_ecc = data[
                (data['eccentricity'] < 0) | (data['eccentricity'] >= 1)
            ]
            if len(invalid_ecc) > 0:
                issues.append(f"Invalid eccentricity values: {len(invalid_ecc)} objects")
        
        if 'inclination' in data.columns:
            invalid_inc = data[
                (data['inclination'] < 0) | (data['inclination'] > 180)
            ]
            if len(invalid_inc) > 0:
                issues.append(f"Invalid inclination values: {len(invalid_inc)} objects")
        
        if 'altitude_km' in data.columns:
            below_earth = data[data['altitude_km'] < 0]
            if len(below_earth) > 0:
                issues.append(f"Objects below Earth surface: {len(below_earth)}")
            
            very_high = data[data['altitude_km'] > 100000]
            if len(very_high) > 0:
                warnings.append(f"Very high altitude objects: {len(very_high)}")
        
        # Timestamp validation
        if 'epoch_datetime' in data.columns:
            try:
                epochs = pd.to_datetime(data['epoch_datetime'])
                current_time = datetime.now()
                
                # Check for future dates
                future_epochs = epochs[epochs > current_time]
                if len(future_epochs) > 0:
                    warnings.append(f"Future epoch dates: {len(future_epochs)}")
                
                # Check for very old data
                old_threshold = current_time - timedelta(days=30)
                old_epochs = epochs[epochs < old_threshold]
                if len(old_epochs) > len(data) * 0.5:
                    warnings.append(f"Large portion of old data: {len(old_epochs)} objects")
                
                statistics['epoch_range'] = {
                    'min': epochs.min().isoformat(),
                    'max': epochs.max().isoformat(),
                    'span_days': (epochs.max() - epochs.min()).days
                }
                
            except Exception as e:
                issues.append(f"Invalid epoch datetime format: {e}")
        
        # Data consistency checks
        if 'semi_major_axis' in data.columns and 'altitude_km' in data.columns:
            earth_radius = 6378.137
            calculated_altitude = data['semi_major_axis'] - earth_radius
            altitude_diff = abs(calculated_altitude - data['altitude_km'])
            
            large_differences = altitude_diff > 10  # 10 km tolerance
            if large_differences.sum() > 0:
                warnings.append(f"SMA/altitude inconsistency: {large_differences.sum()} objects")
        
        # Statistical outlier detection
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_columns:
            if col in data.columns and not data[col].empty:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                    outlier_info[col] = len(outliers)
        
        statistics['outliers_by_column'] = outlier_info
        
        # Overall quality score
        issue_weight = len(issues) * 3
        warning_weight = len(warnings) * 1
        quality_score = max(0, 100 - issue_weight - warning_weight)
        
        return {
            'status': 'analyzed',
            'quality_score': quality_score,
            'total_records': len(data),
            'issues': issues,
            'warnings': warnings,
            'statistics': statistics,
            'recommendation': _generate_quality_recommendation(quality_score, issues, warnings)
        }
        
    except Exception as e:
        logger.error(f"Error detecting data quality issues: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'recommendation': 'Unable to assess data quality due to analysis error'
        }

def _generate_quality_recommendation(score: float, issues: List[str], warnings: List[str]) -> str:
    """Generate data quality recommendation."""
    if score >= 90:
        return "Excellent data quality. Suitable for high-precision analysis."
    elif score >= 75:
        return "Good data quality. Minor issues should be addressed."
    elif score >= 60:
        return "Moderate data quality. Significant preprocessing recommended."
    elif score >= 40:
        return "Poor data quality. Extensive cleaning required before analysis."
    else:
        return "Very poor data quality. Data may not be suitable for analysis."

def export_data(data: pd.DataFrame, 
               filename: str,
               format_type: str = 'csv',
               include_metadata: bool = True) -> bool:
    """
    Export data to various formats with metadata.
    
    Args:
        data: DataFrame to export
        filename: Output filename
        format_type: Export format ('csv', 'json', 'excel', 'parquet')
        include_metadata: Whether to include metadata
        
    Returns:
        Success status
    """
    try:
        if data.empty:
            logger.warning("Cannot export empty dataset")
            return False
        
        # Add metadata if requested
        if include_metadata:
            metadata = {
                'export_timestamp': datetime.now().isoformat(),
                'record_count': len(data),
                'columns': list(data.columns),
                'data_types': data.dtypes.astype(str).to_dict(),
                'export_format': format_type,
                'aegis_version': '5.0'
            }
        
        # Export based on format
        if format_type.lower() == 'csv':
            data.to_csv(filename, index=False)
            
            if include_metadata:
                metadata_filename = filename.replace('.csv', '_metadata.json')
                with open(metadata_filename, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        elif format_type.lower() == 'json':
            if include_metadata:
                export_data = {
                    'metadata': metadata,
                    'data': data.to_dict('records')
                }
            else:
                export_data = data.to_dict('records')
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format_type.lower() == 'excel':
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                data.to_excel(writer, sheet_name='Data', index=False)
                
                if include_metadata:
                    metadata_df = pd.DataFrame(list(metadata.items()), 
                                             columns=['Field', 'Value'])
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        elif format_type.lower() == 'parquet':
            data.to_parquet(filename, index=False)
            
            if include_metadata:
                metadata_filename = filename.replace('.parquet', '_metadata.json')
                with open(metadata_filename, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        logger.info(f"Successfully exported {len(data)} records to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return False

def create_data_hash(data: pd.DataFrame) -> str:
    """
    Create a hash of the dataset for integrity checking.
    
    Args:
        data: DataFrame to hash
        
    Returns:
        SHA-256 hash string
    """
    try:
        if data.empty:
            return hashlib.sha256(b'empty').hexdigest()
        
        # Create a consistent string representation
        data_string = data.to_csv(index=False).encode('utf-8')
        return hashlib.sha256(data_string).hexdigest()
        
    except Exception as e:
        logger.error(f"Error creating data hash: {e}")
        return "error"

def merge_orbital_datasets(datasets: List[pd.DataFrame], 
                          merge_strategy: str = 'union',
                          conflict_resolution: str = 'latest') -> pd.DataFrame:
    """
    Merge multiple orbital datasets with conflict resolution.
    
    Args:
        datasets: List of DataFrames to merge
        merge_strategy: 'union', 'intersection', or 'primary'
        conflict_resolution: 'latest', 'primary', 'average'
        
    Returns:
        Merged DataFrame
    """
    try:
        if not datasets:
            return pd.DataFrame()
        
        if len(datasets) == 1:
            return datasets[0].copy()
        
        # Standardize all datasets
        standardized_datasets = [standardize_column_names(df) for df in datasets]
        
        if merge_strategy == 'union':
            # Combine all datasets
            merged = pd.concat(standardized_datasets, ignore_index=True)
            
            # Handle duplicates based on catalog_number
            if 'catalog_number' in merged.columns:
                if conflict_resolution == 'latest':
                    # Keep the latest epoch for each object
                    if 'epoch_datetime' in merged.columns:
                        merged['epoch_dt'] = pd.to_datetime(merged['epoch_datetime'])
                        merged = merged.sort_values('epoch_dt').groupby('catalog_number').tail(1)
                        merged = merged.drop('epoch_dt', axis=1)
                    else:
                        merged = merged.drop_duplicates('catalog_number', keep='last')
                
                elif conflict_resolution == 'primary':
                    # Keep first occurrence (primary dataset)
                    merged = merged.drop_duplicates('catalog_number', keep='first')
        
        elif merge_strategy == 'intersection':
            # Only keep objects present in all datasets
            if 'catalog_number' in standardized_datasets[0].columns:
                common_objects = set(standardized_datasets[0]['catalog_number'])
                for df in standardized_datasets[1:]:
                    if 'catalog_number' in df.columns:
                        common_objects &= set(df['catalog_number'])
                
                # Filter datasets to common objects
                filtered_datasets = []
                for df in standardized_datasets:
                    if 'catalog_number' in df.columns:
                        filtered = df[df['catalog_number'].isin(common_objects)]
                        filtered_datasets.append(filtered)
                
                merged = pd.concat(filtered_datasets, ignore_index=True)
                
                if conflict_resolution == 'latest' and 'epoch_datetime' in merged.columns:
                    merged['epoch_dt'] = pd.to_datetime(merged['epoch_datetime'])
                    merged = merged.sort_values('epoch_dt').groupby('catalog_number').tail(1)
                    merged = merged.drop('epoch_dt', axis=1)
            else:
                merged = standardized_datasets[0].copy()
        
        elif merge_strategy == 'primary':
            # Use first dataset as primary, supplement with others
            merged = standardized_datasets[0].copy()
            
            if 'catalog_number' in merged.columns:
                primary_objects = set(merged['catalog_number'])
                
                for df in standardized_datasets[1:]:
                    if 'catalog_number' in df.columns:
                        # Add objects not in primary dataset
                        additional = df[~df['catalog_number'].isin(primary_objects)]
                        merged = pd.concat([merged, additional], ignore_index=True)
                        primary_objects.update(additional['catalog_number'])
        
        logger.info(f"Merged {len(datasets)} datasets into {len(merged)} records")
        return merged
        
    except Exception as e:
        logger.error(f"Error merging datasets: {e}")
        return datasets[0] if datasets else pd.DataFrame()

def sample_data(data: pd.DataFrame, 
               sample_method: str = 'random',
               sample_size: Union[int, float] = 1000,
               stratify_column: Optional[str] = None) -> pd.DataFrame:
    """
    Sample data using various methods.
    
    Args:
        data: DataFrame to sample
        sample_method: 'random', 'systematic', 'stratified'
        sample_size: Number of samples (int) or fraction (float)
        stratify_column: Column for stratified sampling
        
    Returns:
        Sampled DataFrame
    """
    try:
        if data.empty:
            return data
        
        # Determine actual sample size
        if isinstance(sample_size, float) and 0 < sample_size <= 1:
            actual_size = int(len(data) * sample_size)
        else:
            actual_size = min(int(sample_size), len(data))
        
        if actual_size >= len(data):
            return data.copy()
        
        if sample_method == 'random':
            return data.sample(n=actual_size, random_state=42)
        
        elif sample_method == 'systematic':
            # Systematic sampling
            step = len(data) // actual_size
            indices = range(0, len(data), step)[:actual_size]
            return data.iloc[indices]
        
        elif sample_method == 'stratified' and stratify_column:
            # Stratified sampling
            if stratify_column not in data.columns:
                logger.warning(f"Stratify column {stratify_column} not found, using random sampling")
                return data.sample(n=actual_size, random_state=42)
            
            # Calculate samples per stratum
            strata = data[stratify_column].value_counts()
            sampled_dfs = []
            
            for stratum, count in strata.items():
                stratum_data = data[data[stratify_column] == stratum]
                stratum_sample_size = max(1, int(actual_size * count / len(data)))
                
                if len(stratum_data) <= stratum_sample_size:
                    sampled_dfs.append(stratum_data)
                else:
                    sampled_dfs.append(stratum_data.sample(n=stratum_sample_size, random_state=42))
            
            result = pd.concat(sampled_dfs, ignore_index=True)
            
            # Adjust if we have too many samples
            if len(result) > actual_size:
                result = result.sample(n=actual_size, random_state=42)
            
            return result
        
        else:
            logger.warning(f"Unknown sampling method {sample_method}, using random")
            return data.sample(n=actual_size, random_state=42)
        
    except Exception as e:
        logger.error(f"Error sampling data: {e}")
        return data


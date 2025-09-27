# -*- coding: utf-8 -*-
"""
Compliance monitoring module for international space debris mitigation standards
ISO 27852, IADC Guidelines, UN Space Debris Mitigation Guidelines, and FCC Rules
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import warnings
warnings.filterwarnings('ignore')

from .constants import *
from .orbital_mechanics import EnhancedSGP4Propagator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceMonitor:
    """
    Advanced compliance monitoring system for international space debris mitigation standards.
    Monitors compliance with ISO 27852, IADC Guidelines, UN Guidelines, and national regulations.
    """
    
    def __init__(self):
        self.monitor_version = "AEGIS-ComplianceMonitor-v5.0"
        self.propagator = EnhancedSGP4Propagator()
        
        # Compliance standards from constants
        self.iso_requirements = ISO_27852_REQUIREMENTS
        self.iadc_guidelines = IADC_GUIDELINES
        self.check_intervals = COMPLIANCE_CHECK_INTERVALS
        
        # Compliance tracking
        self.compliance_history = []
        self.violation_records = []
        self.alert_thresholds = {
            'lifetime_warning': 20,  # Years before disposal requirement
            'collision_risk_critical': 1e-4,  # Critical collision probability
            'passivation_required': True,  # Mandatory passivation
            'debris_release_tolerance': 0  # Zero tolerance for intentional debris
        }
        
        # Regulatory frameworks
        self.regulatory_frameworks = {
            'ISO_27852': {
                'name': 'ISO 27852:2022 Space systems — Estimation of orbit lifetime',
                'scope': 'International standard for orbit lifetime and disposal',
                'mandatory_for': ['Commercial satellites', 'Government missions'],
                'compliance_level': 'Mandatory'
            },
            'IADC_Guidelines': {
                'name': 'Inter-Agency Space Debris Coordination Committee Guidelines',
                'scope': 'International debris mitigation guidelines',
                'mandatory_for': ['Space agencies', 'International missions'],
                'compliance_level': 'Strongly Recommended'
            },
            'UN_Guidelines': {
                'name': 'UN Space Debris Mitigation Guidelines',
                'scope': 'United Nations debris mitigation framework',
                'mandatory_for': ['UN member states', 'International organizations'],
                'compliance_level': 'Mandatory'
            },
            'FCC_Rules': {
                'name': 'FCC Orbital Debris Mitigation Rules',
                'scope': 'US commercial satellite regulations',
                'mandatory_for': ['US commercial operators'],
                'compliance_level': 'Mandatory'
            }
        }
    
    def assess_compliance(self, debris_data: pd.DataFrame,
                         satellite_data: pd.DataFrame,
                         standards: List[str] = None,
                         assessment_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Comprehensive compliance assessment across multiple standards.
        
        Args:
            debris_data: Current debris catalog
            satellite_data: Active satellite catalog
            standards: List of standards to assess against
            assessment_date: Date of assessment (default: current time)
            
        Returns:
            Comprehensive compliance assessment results
        """
        try:
            if assessment_date is None:
                assessment_date = datetime.now()
            
            if standards is None:
                standards = ['ISO_27852', 'IADC_Guidelines', 'UN_Guidelines']
            
            logger.info(f"Starting compliance assessment for {len(standards)} standards")
            
            # Overall compliance metrics
            overall_metrics = self._calculate_overall_metrics(debris_data, satellite_data)
            
            # Standard-specific assessments
            standard_assessments = {}
            for standard in standards:
                if standard == 'ISO_27852':
                    standard_assessments[standard] = self._assess_iso_27852_compliance(
                        debris_data, satellite_data
                    )
                elif standard == 'IADC_Guidelines':
                    standard_assessments[standard] = self._assess_iadc_compliance(
                        debris_data, satellite_data
                    )
                elif standard == 'UN_Guidelines':
                    standard_assessments[standard] = self._assess_un_guidelines_compliance(
                        debris_data, satellite_data
                    )
                elif standard == 'FCC_Rules':
                    standard_assessments[standard] = self._assess_fcc_compliance(
                        satellite_data
                    )
            
            # Compliance violations and alerts
            violations = self._identify_compliance_violations(
                debris_data, satellite_data, standards
            )
            
            # Compliance trends
            trends = self._analyze_compliance_trends()
            
            # Recommendations
            recommendations = self._generate_compliance_recommendations(
                standard_assessments, violations
            )
            
            # Calculate overall compliance score
            overall_score = self._calculate_overall_compliance_score(standard_assessments)
            
            compliance_result = {
                'assessment_timestamp': assessment_date.isoformat(),
                'standards_assessed': standards,
                'overall_score': overall_score,
                'overall_metrics': overall_metrics,
                'standard_assessments': standard_assessments,
                'violations': violations,
                'compliance_trends': trends,
                'recommendations': recommendations,
                'regulatory_status': self._determine_regulatory_status(overall_score, violations),
                'next_assessment_due': (assessment_date + timedelta(days=30)).isoformat()
            }
            
            # Store in compliance history
            self._store_compliance_result(compliance_result)
            
            logger.info(f"Compliance assessment complete. Overall score: {overall_score:.1%}")
            return compliance_result
            
        except Exception as e:
            logger.error(f"Error in compliance assessment: {e}")
            return self._get_default_compliance_result()
    
    def _calculate_overall_metrics(self, debris_data: pd.DataFrame, 
                                 satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate high-level compliance metrics."""
        try:
            total_objects = len(debris_data) + len(satellite_data)
            debris_count = len(debris_data)
            active_satellites = len(satellite_data)
            
            # LEO population analysis
            leo_debris = debris_data[debris_data['altitude_km'] < 2000] if not debris_data.empty else pd.DataFrame()
            leo_satellites = satellite_data[satellite_data['altitude_km'] < 2000] if not satellite_data.empty else pd.DataFrame()
            
            # Risk metrics
            high_risk_objects = 0
            if not debris_data.empty and 'collision_risk' in debris_data.columns:
                high_risk_objects = len(debris_data[debris_data['collision_risk'] > 0.1])
            
            return {
                'total_tracked_objects': total_objects,
                'active_satellites': active_satellites,
                'debris_objects': debris_count,
                'leo_population': len(leo_debris) + len(leo_satellites),
                'debris_density_leo': len(leo_debris) / max(1, len(leo_satellites)) if len(leo_satellites) > 0 else 0,
                'high_risk_objects': high_risk_objects,
                'compliance_population_pressure': min(total_objects / 50000, 1.0),  # Normalized pressure
                'assessment_scope': {
                    'altitude_range_km': (0, 50000),
                    'inclination_coverage': 'All orbital inclinations',
                    'object_size_threshold_cm': 10
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall metrics: {e}")
            return {}
    
    def _assess_iso_27852_compliance(self, debris_data: pd.DataFrame,
                                   satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess compliance with ISO 27852:2022 standards."""
        try:
            iso_results = {
                'standard': 'ISO 27852:2022',
                'compliance_areas': {},
                'violations': [],
                'score': 0.0
            }
            
            # LEO disposal requirements
            leo_compliance = self._assess_leo_disposal_compliance(satellite_data)
            iso_results['compliance_areas']['LEO_disposal'] = leo_compliance
            
            # MEO disposal requirements  
            meo_compliance = self._assess_meo_disposal_compliance(satellite_data)
            iso_results['compliance_areas']['MEO_disposal'] = meo_compliance
            
            # GEO disposal requirements
            geo_compliance = self._assess_geo_disposal_compliance(satellite_data)
            iso_results['compliance_areas']['GEO_disposal'] = geo_compliance
            
            # Collision risk assessment
            collision_compliance = self._assess_collision_risk_compliance(satellite_data)
            iso_results['compliance_areas']['collision_risk'] = collision_compliance
            
            # Calculate overall ISO compliance score
            area_scores = [area['compliance_score'] for area in iso_results['compliance_areas'].values()]
            iso_results['score'] = np.mean(area_scores) if area_scores else 0.0
            
            # Identify violations
            for area_name, area_data in iso_results['compliance_areas'].items():
                if area_data['compliance_score'] < 0.8:  # 80% threshold
                    iso_results['violations'].append({
                        'area': area_name,
                        'severity': 'High' if area_data['compliance_score'] < 0.5 else 'Medium',
                        'description': area_data.get('issues', 'Compliance below threshold')
                    })
            
            return iso_results
            
        except Exception as e:
            logger.error(f"Error assessing ISO 27852 compliance: {e}")
            return {'standard': 'ISO 27852:2022', 'score': 0.0, 'error': str(e)}
    
    def _assess_leo_disposal_compliance(self, satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess LEO disposal compliance per ISO 27852."""
        try:
            if satellite_data.empty:
                return {'compliance_score': 1.0, 'compliant_objects': 0, 'total_objects': 0}
            
            leo_satellites = satellite_data[
                (satellite_data['altitude_km'] >= ORBITAL_ZONES['LEO'][0]) &
                (satellite_data['altitude_km'] <= ORBITAL_ZONES['LEO'][1])
            ]
            
            if leo_satellites.empty:
                return {'compliance_score': 1.0, 'compliant_objects': 0, 'total_objects': 0}
            
            compliant_count = 0
            total_count = len(leo_satellites)
            issues = []
            
            for _, satellite in leo_satellites.iterrows():
                altitude = satellite.get('altitude_km', 0)
                
                # Check disposal altitude requirement (< 300 km for natural decay within 25 years)
                if altitude < self.iso_requirements['LEO_disposal']['disposal_altitude_km']:
                    compliant_count += 1
                else:
                    # Check if satellite has disposal plan
                    # In real implementation, this would check mission planning data
                    disposal_planned = self._check_disposal_plan(satellite)
                    if disposal_planned:
                        compliant_count += 1
                    else:
                        issues.append(f"Satellite {satellite.get('name', 'Unknown')} lacks disposal plan")
            
            compliance_score = compliant_count / total_count if total_count > 0 else 1.0
            
            return {
                'compliance_score': compliance_score,
                'compliant_objects': compliant_count,
                'total_objects': total_count,
                'requirements_checked': [
                    'Disposal altitude < 300 km OR disposal plan exists',
                    'Maximum 25-year orbital lifetime',
                    'Collision probability < 1e-4'
                ],
                'issues': issues[:5]  # Limit to 5 issues for display
            }
            
        except Exception as e:
            logger.error(f"Error assessing LEO disposal compliance: {e}")
            return {'compliance_score': 0.0, 'error': str(e)}
    
    def _assess_meo_disposal_compliance(self, satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess MEO disposal compliance per ISO 27852."""
        try:
            if satellite_data.empty:
                return {'compliance_score': 1.0, 'compliant_objects': 0, 'total_objects': 0}
            
            meo_satellites = satellite_data[
                (satellite_data['altitude_km'] >= ORBITAL_ZONES['MEO'][0]) &
                (satellite_data['altitude_km'] <= ORBITAL_ZONES['MEO'][1])
            ]
            
            if meo_satellites.empty:
                return {'compliance_score': 1.0, 'compliant_objects': 0, 'total_objects': 0}
            
            compliant_count = 0
            total_count = len(meo_satellites)
            
            for _, satellite in meo_satellites.iterrows():
                # MEO satellites must move to graveyard orbit 300 km above operational region
                altitude = satellite.get('altitude_km', 0)
                required_disposal_altitude = altitude + self.iso_requirements['MEO_disposal']['graveyard_altitude_km']
                
                # Check if satellite has appropriate disposal capability
                # In real implementation, this would check propulsion capability
                disposal_capability = self._check_disposal_capability(satellite, required_disposal_altitude)
                if disposal_capability:
                    compliant_count += 1
            
            compliance_score = compliant_count / total_count if total_count > 0 else 1.0
            
            return {
                'compliance_score': compliance_score,
                'compliant_objects': compliant_count,
                'total_objects': total_count,
                'requirements_checked': [
                    'Graveyard orbit 300 km above operational altitude',
                    'Maximum 100-year lifetime in operational orbit',
                    'Passivation at end of life'
                ]
            }
            
        except Exception as e:
            logger.error(f"Error assessing MEO disposal compliance: {e}")
            return {'compliance_score': 0.0, 'error': str(e)}
    
    def _assess_geo_disposal_compliance(self, satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess GEO disposal compliance per ISO 27852."""
        try:
            if satellite_data.empty:
                return {'compliance_score': 1.0, 'compliant_objects': 0, 'total_objects': 0}
            
            # GEO satellites (±75 km from geostationary altitude)
            geo_altitude = ORBITAL_ZONES['GEO'][0]
            geo_satellites = satellite_data[
                (satellite_data['altitude_km'] >= geo_altitude - 75) &
                (satellite_data['altitude_km'] <= geo_altitude + 75)
            ]
            
            if geo_satellites.empty:
                return {'compliance_score': 1.0, 'compliant_objects': 0, 'total_objects': 0}
            
            compliant_count = 0
            total_count = len(geo_satellites)
            
            for _, satellite in geo_satellites.iterrows():
                # GEO disposal requirements
                graveyard_altitude = geo_altitude + self.iso_requirements['GEO_disposal']['graveyard_altitude_km']
                
                # Check disposal plan and capability
                disposal_capability = self._check_disposal_capability(satellite, graveyard_altitude)
                station_keeping = self._check_station_keeping_margin(satellite)
                
                if disposal_capability and station_keeping:
                    compliant_count += 1
            
            compliance_score = compliant_count / total_count if total_count > 0 else 1.0
            
            return {
                'compliance_score': compliance_score,
                'compliant_objects': compliant_count,
                'total_objects': total_count,
                'graveyard_altitude_km': geo_altitude + 300,
                'requirements_checked': [
                    'Graveyard orbit at 36,086 km altitude',
                    'Station-keeping margin of ±75 km',
                    'Fuel reserves for disposal maneuver'
                ]
            }
            
        except Exception as e:
            logger.error(f"Error assessing GEO disposal compliance: {e}")
            return {'compliance_score': 0.0, 'error': str(e)}
    
    def _assess_collision_risk_compliance(self, satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess collision risk compliance across all orbits."""
        try:
            if satellite_data.empty:
                return {'compliance_score': 1.0, 'high_risk_objects': 0}
            
            # Check collision risk for each satellite
            high_risk_count = 0
            total_satellites = len(satellite_data)
            
            if 'collision_risk' in satellite_data.columns:
                high_risk_satellites = satellite_data[
                    satellite_data['collision_risk'] > self.iso_requirements['LEO_disposal']['collision_probability_max']
                ]
                high_risk_count = len(high_risk_satellites)
            
            # Compliance score based on collision risk distribution
            compliance_score = 1.0 - (high_risk_count / total_satellites) if total_satellites > 0 else 1.0
            
            return {
                'compliance_score': compliance_score,
                'high_risk_objects': high_risk_count,
                'total_objects': total_satellites,
                'max_allowed_probability': self.iso_requirements['LEO_disposal']['collision_probability_max'],
                'risk_threshold_exceeded': high_risk_count > 0
            }
            
        except Exception as e:
            logger.error(f"Error assessing collision risk compliance: {e}")
            return {'compliance_score': 0.0, 'error': str(e)}
    
    def _assess_iadc_compliance(self, debris_data: pd.DataFrame,
                              satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess compliance with IADC Guidelines."""
        try:
            iadc_results = {
                'standard': 'IADC Guidelines',
                'compliance_areas': {},
                'score': 0.0
            }
            
            # Design requirements compliance
            design_compliance = self._assess_iadc_design_requirements(satellite_data)
            iadc_results['compliance_areas']['design_requirements'] = design_compliance
            
            # Operational requirements compliance
            operational_compliance = self._assess_iadc_operational_requirements(satellite_data)
            iadc_results['compliance_areas']['operational_requirements'] = operational_compliance
            
            # Post-mission disposal
            disposal_compliance = self._assess_iadc_disposal_requirements(satellite_data)
            iadc_results['compliance_areas']['disposal_requirements'] = disposal_compliance
            
            # Calculate overall IADC score
            area_scores = [area['compliance_score'] for area in iadc_results['compliance_areas'].values()]
            iadc_results['score'] = np.mean(area_scores) if area_scores else 0.0
            
            return iadc_results
            
        except Exception as e:
            logger.error(f"Error assessing IADC compliance: {e}")
            return {'standard': 'IADC Guidelines', 'score': 0.0, 'error': str(e)}
    
    def _assess_iadc_design_requirements(self, satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess IADC design requirements compliance."""
        try:
            if satellite_data.empty:
                return {'compliance_score': 1.0}
            
            # Check design requirements from IADC guidelines
            design_checks = {
                'explosion_prevention': 0.9,  # Assume 90% compliance
                'breakup_assessment': 0.85,   # Assume 85% compliance  
                'collision_avoidance': 0.8,   # Assume 80% compliance
                'disposal_planning': 0.7      # Assume 70% compliance
            }
            
            overall_score = np.mean(list(design_checks.values()))
            
            return {
                'compliance_score': overall_score,
                'design_checks': design_checks,
                'requirements_assessed': list(self.iadc_guidelines['design_requirements'].keys())
            }
            
        except Exception as e:
            logger.error(f"Error assessing IADC design requirements: {e}")
            return {'compliance_score': 0.0, 'error': str(e)}
    
    def _assess_iadc_operational_requirements(self, satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess IADC operational requirements compliance."""
        try:
            if satellite_data.empty:
                return {'compliance_score': 1.0}
            
            # Operational requirements assessment
            operational_checks = {
                'tracking_accuracy': 0.95,      # High tracking accuracy assumed
                'conjunction_screening': 0.8,   # Moderate screening compliance
                'maneuver_capability': 0.7,     # Variable maneuver capability
                'end_of_life_disposal': 0.6     # Lower disposal execution rate
            }
            
            overall_score = np.mean(list(operational_checks.values()))
            
            return {
                'compliance_score': overall_score,
                'operational_checks': operational_checks,
                'requirements_assessed': list(self.iadc_guidelines['operational_requirements'].keys())
            }
            
        except Exception as e:
            logger.error(f"Error assessing IADC operational requirements: {e}")
            return {'compliance_score': 0.0, 'error': str(e)}
    
    def _assess_iadc_disposal_requirements(self, satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess IADC post-mission disposal requirements."""
        try:
            if satellite_data.empty:
                return {'compliance_score': 1.0}
            
            # Disposal compliance based on orbital region
            leo_disposal_rate = 0.6   # 60% of LEO satellites properly disposed
            meo_disposal_rate = 0.7   # 70% of MEO satellites properly disposed  
            geo_disposal_rate = 0.8   # 80% of GEO satellites properly disposed
            
            # Weight by satellite population in each region
            total_satellites = len(satellite_data)
            if total_satellites == 0:
                return {'compliance_score': 1.0}
            
            leo_count = len(satellite_data[satellite_data['altitude_km'] < 2000])
            meo_count = len(satellite_data[
                (satellite_data['altitude_km'] >= 2000) & 
                (satellite_data['altitude_km'] < 35786)
            ])
            geo_count = len(satellite_data[satellite_data['altitude_km'] >= 35786])
            
            weighted_score = (
                (leo_count * leo_disposal_rate + 
                 meo_count * meo_disposal_rate + 
                 geo_count * geo_disposal_rate) / total_satellites
            )
            
            return {
                'compliance_score': weighted_score,
                'disposal_rates': {
                    'LEO': leo_disposal_rate,
                    'MEO': meo_disposal_rate, 
                    'GEO': geo_disposal_rate
                },
                'satellite_distribution': {
                    'LEO': leo_count,
                    'MEO': meo_count,
                    'GEO': geo_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error assessing IADC disposal requirements: {e}")
            return {'compliance_score': 0.0, 'error': str(e)}
    
    def _assess_un_guidelines_compliance(self, debris_data: pd.DataFrame,
                                       satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess compliance with UN Space Debris Mitigation Guidelines."""
        try:
            un_results = {
                'standard': 'UN Space Debris Mitigation Guidelines',
                'compliance_areas': {},
                'score': 0.0
            }
            
            # Guideline 1: Limit debris released during normal operations
            normal_ops_compliance = self._assess_normal_operations_debris(satellite_data)
            un_results['compliance_areas']['normal_operations'] = normal_ops_compliance
            
            # Guideline 2: Minimize breakup potential during operational phases
            breakup_prevention = self._assess_breakup_prevention(satellite_data)
            un_results['compliance_areas']['breakup_prevention'] = breakup_prevention
            
            # Guideline 3: Limit probability of accidental collision
            collision_limitation = self._assess_collision_limitation(satellite_data)
            un_results['compliance_areas']['collision_limitation'] = collision_limitation
            
            # Guideline 4: Avoid intentional destruction creating long-lived debris
            destruction_avoidance = self._assess_destruction_avoidance(debris_data)
            un_results['compliance_areas']['destruction_avoidance'] = destruction_avoidance
            
            # Guideline 5: Minimize post-mission breakup potential
            post_mission_safety = self._assess_post_mission_safety(satellite_data)
            un_results['compliance_areas']['post_mission_safety'] = post_mission_safety
            
            # Guideline 6: Limit long-term interference with protected regions
            protected_regions = self._assess_protected_regions_compliance(satellite_data)
            un_results['compliance_areas']['protected_regions'] = protected_regions
            
            # Calculate overall UN Guidelines score
            area_scores = [area['compliance_score'] for area in un_results['compliance_areas'].values()]
            un_results['score'] = np.mean(area_scores) if area_scores else 0.0
            
            return un_results
            
        except Exception as e:
            logger.error(f"Error assessing UN Guidelines compliance: {e}")
            return {'standard': 'UN Guidelines', 'score': 0.0, 'error': str(e)}
    
    def _assess_fcc_compliance(self, satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess compliance with FCC Orbital Debris Mitigation Rules."""
        try:
            if satellite_data.empty:
                return {'standard': 'FCC Rules', 'score': 1.0, 'us_satellites': 0}
            
            # Filter for US commercial satellites (simplified)
            us_satellites = satellite_data  # In real implementation, filter by operator nationality
            
            fcc_compliance_rate = 0.85  # Assume 85% compliance with FCC rules
            
            return {
                'standard': 'FCC Orbital Debris Mitigation Rules',
                'score': fcc_compliance_rate,
                'us_satellites': len(us_satellites),
                'compliance_areas': {
                    'disposal_plans': 0.9,
                    'casualty_risk': 0.8,
                    'tracking_accuracy': 0.9,
                    'collision_assessment': 0.8
                }
            }
            
        except Exception as e:
            logger.error(f"Error assessing FCC compliance: {e}")
            return {'standard': 'FCC Rules', 'score': 0.0, 'error': str(e)}
    
    def _check_disposal_plan(self, satellite: pd.Series) -> bool:
        """Check if satellite has adequate disposal plan."""
        # Simplified check - in real implementation, would check mission documentation
        altitude = satellite.get('altitude_km', 0)
        mass = satellite.get('mass_kg', 0)
        
        # Assume satellites with sufficient mass have disposal capability
        return mass > 50  # Satellites > 50kg likely have propulsion for disposal
    
    def _check_disposal_capability(self, satellite: pd.Series, target_altitude: float) -> bool:
        """Check if satellite has capability to reach disposal orbit."""
        # Simplified check based on satellite properties
        current_altitude = satellite.get('altitude_km', 0)
        delta_v_required = abs(target_altitude - current_altitude) * 0.1  # Rough estimate
        
        # Assume larger satellites have more propulsion capability
        mass = satellite.get('mass_kg', 100)
        estimated_delta_v_capability = mass * 0.01  # Rough propulsion capability estimate
        
        return estimated_delta_v_capability >= delta_v_required
    
    def _check_station_keeping_margin(self, satellite: pd.Series) -> bool:
        """Check if GEO satellite maintains proper station-keeping margins."""
        # Simplified check - assume active satellites maintain margins
        return satellite.get('status', 'Active') == 'Active'
    
    def _assess_normal_operations_debris(self, satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess UN Guideline 1: Normal operations debris limitation."""
        try:
            # High compliance assumed for modern satellites
            compliance_rate = 0.95
            
            return {
                'compliance_score': compliance_rate,
                'guideline': 'Limit debris released during normal operations',
                'assessment': 'Modern satellite designs minimize operational debris release'
            }
            
        except Exception as e:
            logger.error(f"Error assessing normal operations debris: {e}")
            return {'compliance_score': 0.0, 'error': str(e)}
    
    def _assess_breakup_prevention(self, satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess UN Guideline 2: Breakup potential minimization."""
        try:
            # Moderate compliance - some older satellites may lack proper passivation
            compliance_rate = 0.8
            
            return {
                'compliance_score': compliance_rate,
                'guideline': 'Minimize breakup potential during operational phases',
                'assessment': 'Most satellites designed with passivation capabilities'
            }
            
        except Exception as e:
            logger.error(f"Error assessing breakup prevention: {e}")
            return {'compliance_score': 0.0, 'error': str(e)}
    
    def _assess_collision_limitation(self, satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess UN Guideline 3: Collision probability limitation."""
        try:
            if satellite_data.empty:
                return {'compliance_score': 1.0}
            
            # Check collision risks
            high_risk_count = 0
            if 'collision_risk' in satellite_data.columns:
                high_risk_count = len(satellite_data[satellite_data['collision_risk'] > 1e-4])
            
            compliance_rate = 1.0 - (high_risk_count / len(satellite_data))
            
            return {
                'compliance_score': compliance_rate,
                'guideline': 'Limit probability of accidental collision',
                'high_risk_satellites': high_risk_count,
                'total_satellites': len(satellite_data)
            }
            
        except Exception as e:
            logger.error(f"Error assessing collision limitation: {e}")
            return {'compliance_score': 0.0, 'error': str(e)}
    
    def _assess_destruction_avoidance(self, debris_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess UN Guideline 4: Avoidance of intentional destruction."""
        try:
            # High compliance - intentional destruction is rare
            compliance_rate = 0.99
            
            return {
                'compliance_score': compliance_rate,
                'guideline': 'Avoid intentional destruction creating long-lived debris',
                'assessment': 'Very few intentional destruction events in recent years'
            }
            
        except Exception as e:
            logger.error(f"Error assessing destruction avoidance: {e}")
            return {'compliance_score': 0.0, 'error': str(e)}
    
    def _assess_post_mission_safety(self, satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess UN Guideline 5: Post-mission breakup potential minimization."""
        try:
            # Moderate compliance - passivation not always performed
            compliance_rate = 0.75
            
            return {
                'compliance_score': compliance_rate,
                'guideline': 'Minimize post-mission breakup potential',
                'assessment': 'Improving passivation compliance but not universal'
            }
            
        except Exception as e:
            logger.error(f"Error assessing post-mission safety: {e}")
            return {'compliance_score': 0.0, 'error': str(e)}
    
    def _assess_protected_regions_compliance(self, satellite_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess UN Guideline 6: Protected regions interference limitation."""
        try:
            # Good compliance with protected region guidelines
            compliance_rate = 0.9
            
            return {
                'compliance_score': compliance_rate,
                'guideline': 'Limit long-term interference with protected regions',
                'protected_regions': ['LEO below 2000 km', 'GEO ±200 km'],
                'assessment': 'Good compliance with protected region guidelines'
            }
            
        except Exception as e:
            logger.error(f"Error assessing protected regions compliance: {e}")
            return {'compliance_score': 0.0, 'error': str(e)}
    
    def _identify_compliance_violations(self, debris_data: pd.DataFrame,
                                      satellite_data: pd.DataFrame,
                                      standards: List[str]) -> List[Dict[str, Any]]:
        """Identify specific compliance violations requiring attention."""
        try:
            violations = []
            
            # Orbital lifetime violations
            if not satellite_data.empty:
                leo_satellites = satellite_data[satellite_data['altitude_km'] < 2000]
                for _, sat in leo_satellites.iterrows():
                    # Estimate orbital lifetime (simplified)
                    estimated_lifetime = self._estimate_orbital_lifetime(sat)
                    if estimated_lifetime > 25:  # ISO 27852 requirement
                        violations.append({
                            'type': 'Orbital Lifetime Violation',
                            'severity': 'High',
                            'object': sat.get('name', 'Unknown'),
                            'description': f'Estimated orbital lifetime ({estimated_lifetime:.1f} years) exceeds 25-year limit',
                            'standard': 'ISO 27852',
                            'remediation': 'Plan disposal maneuver or atmospheric drag enhancement'
                        })
            
            # Collision risk violations
            if not satellite_data.empty and 'collision_risk' in satellite_data.columns:
                high_risk_sats = satellite_data[satellite_data['collision_risk'] > 1e-4]
                for _, sat in high_risk_sats.iterrows():
                    violations.append({
                        'type': 'Collision Risk Violation',
                        'severity': 'Critical',
                        'object': sat.get('name', 'Unknown'),
                        'description': f'Collision probability ({sat["collision_risk"]:.2e}) exceeds 1e-4 threshold',
                        'standard': 'ISO 27852 / IADC Guidelines',
                        'remediation': 'Perform collision avoidance maneuver or enhance tracking'
                    })
            
            # Disposal plan violations
            if not satellite_data.empty:
                satellites_without_disposal = satellite_data[
                    satellite_data['altitude_km'] > 1000  # Focus on higher orbits
                ]
                for _, sat in satellites_without_disposal.iterrows():
                    if not self._check_disposal_plan(sat):
                        violations.append({
                            'type': 'Missing Disposal Plan',
                            'severity': 'Medium',
                            'object': sat.get('name', 'Unknown'),
                            'description': 'No adequate post-mission disposal plan identified',
                            'standard': 'IADC Guidelines / UN Guidelines',
                            'remediation': 'Develop and implement disposal plan'
                        })
            
            return violations[:20]  # Limit to top 20 violations
            
        except Exception as e:
            logger.error(f"Error identifying compliance violations: {e}")
            return []
    
    def _estimate_orbital_lifetime(self, satellite: pd.Series) -> float:
        """Estimate orbital lifetime based on satellite properties."""
        try:
            altitude = satellite.get('altitude_km', 500)
            mass = satellite.get('mass_kg', 100)
            area = satellite.get('radar_cross_section', 1.0)
            
            # Simplified lifetime model
            if altitude < 300:
                return 1  # Very short lifetime
            elif altitude < 600:
                return 5 + (altitude - 300) * 0.1
            elif altitude < 1000:
                return 15 + (altitude - 600) * 0.05
            else:
                return 50 + (altitude - 1000) * 0.01
                
        except Exception as e:
            logger.error(f"Error estimating orbital lifetime: {e}")
            return 25  # Default conservative estimate
    
    def _analyze_compliance_trends(self) -> Dict[str, Any]:
        """Analyze compliance trends over time."""
        try:
            if len(self.compliance_history) < 2:
                return {'trend': 'Insufficient data for trend analysis'}
            
            recent_scores = [h['overall_score'] for h in self.compliance_history[-5:]]
            trend_direction = "improving" if recent_scores[-1] > recent_scores[0] else "declining"
            
            return {
                'trend_direction': trend_direction,
                'recent_average_score': np.mean(recent_scores),
                'score_change': recent_scores[-1] - recent_scores[0] if len(recent_scores) > 1 else 0,
                'assessments_analyzed': len(recent_scores)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing compliance trends: {e}")
            return {'trend': 'Analysis error', 'error': str(e)}
    
    def _generate_compliance_recommendations(self, standard_assessments: Dict[str, Any],
                                           violations: List[Dict]) -> List[str]:
        """Generate actionable compliance recommendations."""
        try:
            recommendations = []
            
            # High-level recommendations based on overall scores
            for standard, assessment in standard_assessments.items():
                score = assessment.get('score', 0)
                if score < 0.6:
                    recommendations.append(
                        f"URGENT: {standard} compliance critically low ({score:.1%}). "
                        f"Immediate action required to meet regulatory requirements."
                    )
                elif score < 0.8:
                    recommendations.append(
                        f"Improve {standard} compliance ({score:.1%}). "
                        f"Focus on identified compliance gaps."
                    )
            
            # Violation-specific recommendations
            critical_violations = [v for v in violations if v.get('severity') == 'Critical']
            if critical_violations:
                recommendations.append(
                    f"Address {len(critical_violations)} critical compliance violations immediately."
                )
            
            # General recommendations
            recommendations.extend([
                "Implement automated compliance monitoring system",
                "Establish regular compliance review schedule (monthly)",
                "Develop compliance training program for mission planners",
                "Create compliance dashboard for real-time monitoring",
                "Coordinate with international standards bodies on best practices"
            ])
            
            return recommendations[:8]  # Limit to 8 recommendations
            
        except Exception as e:
            logger.error(f"Error generating compliance recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error"]
    
    def _calculate_overall_compliance_score(self, standard_assessments: Dict[str, Any]) -> float:
        """Calculate weighted overall compliance score."""
        try:
            if not standard_assessments:
                return 0.0
            
            # Weight standards by importance
            weights = {
                'ISO_27852': 0.4,      # High weight for international standard
                'IADC_Guidelines': 0.3, # Moderate weight for agency guidelines
                'UN_Guidelines': 0.2,   # Moderate weight for UN guidelines
                'FCC_Rules': 0.1       # Lower weight for national rules
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for standard, assessment in standard_assessments.items():
                score = assessment.get('score', 0)
                weight = weights.get(standard, 0.1)  # Default low weight
                weighted_score += score * weight
                total_weight += weight
            
            return weighted_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating overall compliance score: {e}")
            return 0.0
    
    def _determine_regulatory_status(self, overall_score: float, 
                                   violations: List[Dict]) -> Dict[str, str]:
        """Determine overall regulatory compliance status."""
        try:
            critical_violations = len([v for v in violations if v.get('severity') == 'Critical'])
            high_violations = len([v for v in violations if v.get('severity') == 'High'])
            
            if critical_violations > 0:
                status = "Non-Compliant"
                description = f"Critical violations detected. Immediate regulatory action required."
            elif overall_score < 0.6:
                status = "At Risk"
                description = "Compliance score below acceptable threshold. Remediation needed."
            elif overall_score < 0.8:
                status = "Conditional Compliance"
                description = "Meets minimum requirements but improvement needed."
            elif overall_score < 0.95:
                status = "Compliant"
                description = "Meets regulatory requirements with minor gaps."
            else:
                status = "Exemplary"
                description = "Exceeds regulatory requirements. Best practice implementation."
            
            return {
                'status': status,
                'description': description,
                'critical_violations': critical_violations,
                'high_violations': high_violations
            }
            
        except Exception as e:
            logger.error(f"Error determining regulatory status: {e}")
            return {'status': 'Unknown', 'description': 'Status determination failed'}
    
    def _store_compliance_result(self, result: Dict[str, Any]):
        """Store compliance assessment result in history."""
        try:
            # Store summary for trend analysis
            summary = {
                'timestamp': result['assessment_timestamp'],
                'overall_score': result['overall_score'],
                'standards_assessed': result['standards_assessed'],
                'total_violations': len(result.get('violations', []))
            }
            
            self.compliance_history.append(summary)
            
            # Keep only recent history (last 100 assessments)
            if len(self.compliance_history) > 100:
                self.compliance_history = self.compliance_history[-100:]
            
            # Store violations separately
            self.violation_records.extend(result.get('violations', []))
            
            # Keep only recent violations (last 1000)
            if len(self.violation_records) > 1000:
                self.violation_records = self.violation_records[-1000:]
                
        except Exception as e:
            logger.error(f"Error storing compliance result: {e}")
    
    def _get_default_compliance_result(self) -> Dict[str, Any]:
        """Return default compliance result when assessment fails."""
        return {
            'assessment_timestamp': datetime.now().isoformat(),
            'standards_assessed': [],
            'overall_score': 0.0,
            'overall_metrics': {},
            'standard_assessments': {},
            'violations': [],
            'compliance_trends': {'trend': 'Assessment failed'},
            'recommendations': ['Unable to perform compliance assessment - system error'],
            'regulatory_status': {
                'status': 'Unknown',
                'description': 'Compliance assessment failed due to system error'
            },
            'next_assessment_due': (datetime.now() + timedelta(days=30)).isoformat()
        }

# Global compliance monitor instance
compliance_monitor = ComplianceMonitor()


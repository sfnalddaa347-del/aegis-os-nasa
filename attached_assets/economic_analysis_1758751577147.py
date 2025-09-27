# -*- coding: utf-8 -*-
"""
Economic analysis module for space debris removal and sustainability
Advanced cost-benefit analysis, real options valuation, and market modeling
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from .constants import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EconomicAnalyzer:
    """
    Advanced economic analysis for space debris removal operations.
    Includes cost-benefit analysis, real options valuation, and market dynamics.
    """
    
    def __init__(self):
        self.analyzer_version = "AEGIS-EconomicAnalyzer-v5.0"
        
        # Economic parameters from constants
        self.material_values = MATERIAL_PROPERTIES
        self.recycling_efficiency = RECYCLING_EFFICIENCY
        self.transportation_cost = TRANSPORTATION_COST_PER_KG
        self.removal_cost = DEBRIS_REMOVAL_COST_PER_KG
        
        # Market parameters
        self.discount_rate = 0.05  # 5% annual discount rate
        self.inflation_rate = 0.02  # 2% annual inflation
        self.risk_free_rate = 0.03  # 3% risk-free rate
        
        # Mission parameters
        self.typical_mission_costs = {
            'small_debris_removal': 50e6,      # $50M for small debris mission
            'medium_debris_removal': 150e6,    # $150M for medium debris mission  
            'large_debris_removal': 500e6,     # $500M for large debris mission
            'active_debris_removal': 1000e6,   # $1B for active removal system
            'orbital_servicing': 200e6         # $200M for orbital servicing mission
        }
        
        # Economic impact factors
        self.collision_cost_factors = {
            'small_satellite': 10e6,     # $10M average small satellite
            'large_satellite': 500e6,    # $500M average large satellite
            'space_station': 10000e6,    # $10B space station
            'human_mission': 50000e6,    # $50B human mission cost impact
            'insurance_premium': 0.05    # 5% insurance cost increase per collision
        }
    
    def analyze_debris_removal_mission(self, mission_cost_musd: float,
                                     target_debris_mass_kg: float,
                                     success_probability: float = 0.85,
                                     mission_timeline_years: int = 3,
                                     debris_composition: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Comprehensive economic analysis of debris removal mission.
        
        Args:
            mission_cost_musd: Total mission cost in millions USD
            target_debris_mass_kg: Mass of debris to be removed
            success_probability: Mission success probability (0-1)
            mission_timeline_years: Mission duration in years
            debris_composition: Material composition of debris
            
        Returns:
            Comprehensive economic analysis results
        """
        try:
            logger.info(f"Analyzing debris removal mission: ${mission_cost_musd}M, {target_debris_mass_kg}kg target")
            
            # Default debris composition if not provided
            if debris_composition is None:
                debris_composition = {
                    'aluminum': 0.6,
                    'steel': 0.2,
                    'titanium': 0.1,
                    'copper': 0.05,
                    'other': 0.05
                }
            
            # Calculate direct benefits
            material_recovery_value = self._calculate_material_recovery_value(
                target_debris_mass_kg, debris_composition
            )
            
            # Calculate collision avoidance benefits
            collision_avoidance_benefits = self._calculate_collision_avoidance_benefits(
                target_debris_mass_kg, mission_timeline_years
            )
            
            # Calculate operational benefits
            operational_benefits = self._calculate_operational_benefits(
                target_debris_mass_kg, mission_timeline_years
            )
            
            # Calculate total costs
            total_costs = self._calculate_total_mission_costs(
                mission_cost_musd, target_debris_mass_kg, mission_timeline_years
            )
            
            # Risk adjustment
            risk_adjusted_benefits = self._apply_risk_adjustment(
                material_recovery_value + collision_avoidance_benefits + operational_benefits,
                success_probability
            )
            
            # Financial metrics
            financial_metrics = self._calculate_financial_metrics(
                risk_adjusted_benefits, total_costs, mission_timeline_years
            )
            
            # Sensitivity analysis
            sensitivity_analysis = self._perform_sensitivity_analysis(
                mission_cost_musd, target_debris_mass_kg, success_probability
            )
            
            # Real options valuation
            real_options_value = self._calculate_real_options_value(
                mission_cost_musd, risk_adjusted_benefits, mission_timeline_years
            )
            
            return {
                'analysis_timestamp': datetime.now().isoformat(),
                'mission_parameters': {
                    'cost_musd': mission_cost_musd,
                    'target_mass_kg': target_debris_mass_kg,
                    'success_probability': success_probability,
                    'timeline_years': mission_timeline_years,
                    'debris_composition': debris_composition
                },
                'cost_breakdown': total_costs,
                'benefit_analysis': {
                    'material_recovery_musd': material_recovery_value / 1e6,
                    'collision_avoidance_musd': collision_avoidance_benefits / 1e6,
                    'operational_benefits_musd': operational_benefits / 1e6,
                    'total_benefits_musd': risk_adjusted_benefits / 1e6
                },
                'financial_metrics': financial_metrics,
                'sensitivity_analysis': sensitivity_analysis,
                'real_options_value_musd': real_options_value / 1e6,
                'economic_recommendation': self._generate_economic_recommendation(financial_metrics),
                'risk_assessment': self._assess_economic_risks(financial_metrics, sensitivity_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error in debris removal mission analysis: {e}")
            return self._get_default_economic_analysis()
    
    def _calculate_material_recovery_value(self, mass_kg: float, 
                                         composition: Dict[str, float]) -> float:
        """Calculate value of recovered materials."""
        try:
            total_value = 0
            
            for material, fraction in composition.items():
                if material in self.material_values:
                    material_mass = mass_kg * fraction
                    value_per_kg = self.material_values[material]['value_per_ton'] / 1000
                    
                    # Apply recycling efficiency
                    recoverable_mass = material_mass * self.recycling_efficiency
                    
                    # Calculate value minus processing costs
                    gross_value = recoverable_mass * value_per_kg
                    processing_cost = gross_value * PROCESSING_COST_FACTOR
                    net_value = gross_value - processing_cost
                    
                    total_value += max(0, net_value)  # Ensure non-negative
            
            # Apply transportation cost penalty
            transportation_penalty = mass_kg * (self.transportation_cost / 1000)  # $/kg to $/g conversion
            net_material_value = total_value - transportation_penalty
            
            return max(0, net_material_value)
            
        except Exception as e:
            logger.error(f"Error calculating material recovery value: {e}")
            return 0
    
    def _calculate_collision_avoidance_benefits(self, mass_kg: float, years: int) -> float:
        """Calculate economic benefits from avoided collisions."""
        try:
            # Collision probability model (simplified)
            base_collision_prob = self._estimate_collision_probability(mass_kg)
            annual_collision_prob = base_collision_prob * years
            
            # Potential collision costs
            average_satellite_value = 100e6  # $100M average satellite
            cascade_multiplier = self._calculate_cascade_multiplier(mass_kg)
            
            # Expected collision cost
            expected_collision_cost = (annual_collision_prob * average_satellite_value * 
                                     cascade_multiplier)
            
            # Insurance premium reduction
            insurance_benefits = self._calculate_insurance_benefits(
                expected_collision_cost, years
            )
            
            # Total collision avoidance benefits
            total_benefits = expected_collision_cost + insurance_benefits
            
            return total_benefits
            
        except Exception as e:
            logger.error(f"Error calculating collision avoidance benefits: {e}")
            return 0
    
    def _calculate_operational_benefits(self, mass_kg: float, years: int) -> float:
        """Calculate operational benefits from debris removal."""
        try:
            # Orbital slot value preservation
            slot_value = self._calculate_orbital_slot_value(mass_kg)
            
            # Launch window availability improvement
            launch_window_benefits = self._calculate_launch_window_benefits(mass_kg, years)
            
            # Space traffic management benefits
            traffic_management_benefits = self._calculate_traffic_management_benefits(mass_kg, years)
            
            # Technology demonstration value
            tech_demo_value = min(50e6, mass_kg * 1000)  # Capped technology value
            
            return slot_value + launch_window_benefits + traffic_management_benefits + tech_demo_value
            
        except Exception as e:
            logger.error(f"Error calculating operational benefits: {e}")
            return 0
    
    def _calculate_total_mission_costs(self, base_cost_musd: float, 
                                     mass_kg: float, years: int) -> Dict[str, float]:
        """Calculate comprehensive mission costs."""
        try:
            base_cost = base_cost_musd * 1e6  # Convert to USD
            
            # Development costs (typically 20-30% of total)
            development_cost = base_cost * 0.25
            
            # Launch costs
            launch_cost = self._estimate_launch_costs(mass_kg)
            
            # Operations costs (annual)
            annual_ops_cost = base_cost * 0.05  # 5% per year
            total_ops_cost = annual_ops_cost * years
            
            # Ground segment costs
            ground_segment_cost = base_cost * 0.15
            
            # Contingency (15%)
            contingency = (development_cost + launch_cost + total_ops_cost + 
                          ground_segment_cost) * 0.15
            
            cost_breakdown = {
                'development_musd': development_cost / 1e6,
                'launch_musd': launch_cost / 1e6,
                'operations_musd': total_ops_cost / 1e6,
                'ground_segment_musd': ground_segment_cost / 1e6,
                'contingency_musd': contingency / 1e6,
                'total_musd': (development_cost + launch_cost + total_ops_cost + 
                              ground_segment_cost + contingency) / 1e6
            }
            
            return cost_breakdown
            
        except Exception as e:
            logger.error(f"Error calculating total mission costs: {e}")
            return {'total_musd': base_cost_musd, 'error': str(e)}
    
    def _estimate_collision_probability(self, mass_kg: float) -> float:
        """Estimate annual collision probability based on debris mass."""
        try:
            # Simplified model: larger debris has higher collision probability
            base_prob = 1e-6  # Base annual collision probability
            mass_factor = np.log10(max(1, mass_kg)) / 10  # Logarithmic scaling
            
            return min(base_prob * (1 + mass_factor), 0.01)  # Cap at 1% annual
            
        except Exception as e:
            logger.error(f"Error estimating collision probability: {e}")
            return 1e-6
    
    def _calculate_cascade_multiplier(self, mass_kg: float) -> float:
        """Calculate cascade effect multiplier for collision costs."""
        try:
            # Kessler syndrome consideration
            if mass_kg > 1000:  # Large debris
                return 5.0  # High cascade potential
            elif mass_kg > 100:  # Medium debris
                return 2.0  # Moderate cascade potential
            else:  # Small debris
                return 1.2  # Limited cascade potential
                
        except Exception as e:
            logger.error(f"Error calculating cascade multiplier: {e}")
            return 1.0
    
    def _calculate_insurance_benefits(self, expected_collision_cost: float, years: int) -> float:
        """Calculate insurance premium reduction benefits."""
        try:
            # Insurance market dynamics
            current_premium_rate = 0.05  # 5% of satellite value
            risk_reduction_factor = min(0.5, expected_collision_cost / 100e6)  # Up to 50% reduction
            
            # Annual insurance savings
            annual_savings = expected_collision_cost * current_premium_rate * risk_reduction_factor
            
            # Present value of savings over mission lifetime
            pv_savings = sum([annual_savings / (1 + self.discount_rate)**year 
                             for year in range(1, years + 1)])
            
            return pv_savings
            
        except Exception as e:
            logger.error(f"Error calculating insurance benefits: {e}")
            return 0
    
    def _calculate_orbital_slot_value(self, mass_kg: float) -> float:
        """Calculate value of preserved orbital slots."""
        try:
            # GEO slot value is highest
            base_slot_value = 100e6  # $100M per valuable orbital slot
            
            # Debris impact on slot usability
            slot_impact_factor = min(1.0, mass_kg / 1000)  # Normalized by 1000kg
            
            return base_slot_value * slot_impact_factor
            
        except Exception as e:
            logger.error(f"Error calculating orbital slot value: {e}")
            return 0
    
    def _calculate_launch_window_benefits(self, mass_kg: float, years: int) -> float:
        """Calculate benefits from improved launch window availability."""
        try:
            # Launch delay costs
            average_delay_cost = 1e6  # $1M per day of delay
            debris_delay_factor = mass_kg / 10000  # Delay days per kg
            annual_delay_reduction = debris_delay_factor * 365
            
            # Annual benefit
            annual_benefit = annual_delay_reduction * average_delay_cost
            
            # Present value over mission lifetime
            pv_benefits = sum([annual_benefit / (1 + self.discount_rate)**year 
                              for year in range(1, years + 1)])
            
            return pv_benefits
            
        except Exception as e:
            logger.error(f"Error calculating launch window benefits: {e}")
            return 0
    
    def _calculate_traffic_management_benefits(self, mass_kg: float, years: int) -> float:
        """Calculate space traffic management operational benefits."""
        try:
            # Reduced tracking and monitoring costs
            annual_tracking_cost_reduction = min(5e6, mass_kg * 1000)  # $5M max per year
            
            # Improved orbital operations efficiency
            efficiency_benefit = annual_tracking_cost_reduction * 0.5
            
            # Total annual benefits
            annual_benefits = annual_tracking_cost_reduction + efficiency_benefit
            
            # Present value
            pv_benefits = sum([annual_benefits / (1 + self.discount_rate)**year 
                              for year in range(1, years + 1)])
            
            return pv_benefits
            
        except Exception as e:
            logger.error(f"Error calculating traffic management benefits: {e}")
            return 0
    
    def _estimate_launch_costs(self, payload_mass_kg: float) -> float:
        """Estimate launch costs for debris removal mission."""
        try:
            # Typical launch costs per kg to various orbits
            launch_cost_per_kg = {
                'LEO': 5000,    # $5K per kg to LEO
                'MEO': 15000,   # $15K per kg to MEO
                'GEO': 25000    # $25K per kg to GEO
            }
            
            # Assume LEO mission (most debris removal missions)
            mission_mass = max(payload_mass_kg, 1000)  # Minimum 1000kg spacecraft
            total_launch_cost = mission_mass * launch_cost_per_kg['LEO']
            
            return total_launch_cost
            
        except Exception as e:
            logger.error(f"Error estimating launch costs: {e}")
            return 50e6  # Default $50M launch cost
    
    def _apply_risk_adjustment(self, benefits: float, success_probability: float) -> float:
        """Apply risk adjustment to expected benefits."""
        try:
            # Expected value with success probability
            expected_benefits = benefits * success_probability
            
            # Additional risk discount for uncertainty
            risk_discount_factor = 0.1  # 10% discount for project risk
            risk_adjusted_benefits = expected_benefits * (1 - risk_discount_factor)
            
            return risk_adjusted_benefits
            
        except Exception as e:
            logger.error(f"Error applying risk adjustment: {e}")
            return benefits * 0.5  # Conservative 50% haircut
    
    def _calculate_financial_metrics(self, benefits: float, costs: Dict[str, float], 
                                   years: int) -> Dict[str, Any]:
        """Calculate comprehensive financial metrics."""
        try:
            total_cost = costs.get('total_musd', 0) * 1e6  # Convert to USD
            
            # Net Present Value
            npv = benefits - total_cost
            
            # Return on Investment
            roi = (benefits - total_cost) / total_cost if total_cost > 0 else 0
            
            # Benefit-Cost Ratio
            bcr = benefits / total_cost if total_cost > 0 else 0
            
            # Payback Period (simplified)
            annual_cash_flow = benefits / years if years > 0 else benefits
            payback_period = total_cost / annual_cash_flow if annual_cash_flow > 0 else float('inf')
            
            # Internal Rate of Return (simplified approximation)
            irr = self._calculate_irr_approximation(benefits, total_cost, years)
            
            return {
                'net_present_value': npv / 1e6,  # Convert to millions
                'return_on_investment': roi,
                'benefit_cost_ratio': bcr,
                'payback_period_years': min(payback_period, 50),  # Cap at 50 years
                'internal_rate_of_return': irr,
                'profitability_index': bcr,  # Same as BCR
                'economic_viability': npv > 0 and bcr > 1.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating financial metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_irr_approximation(self, benefits: float, costs: float, years: int) -> float:
        """Approximate Internal Rate of Return calculation."""
        try:
            if costs <= 0 or years <= 0:
                return 0
            
            annual_cash_flow = benefits / years
            
            # Simplified IRR approximation using average return
            if annual_cash_flow > 0:
                return (annual_cash_flow - costs / years) / costs
            else:
                return -self.discount_rate
                
        except Exception as e:
            logger.error(f"Error calculating IRR approximation: {e}")
            return 0
    
    def _perform_sensitivity_analysis(self, base_cost_musd: float, 
                                    mass_kg: float, success_prob: float) -> Dict[str, Any]:
        """Perform sensitivity analysis on key parameters."""
        try:
            sensitivity_results = {}
            
            # Cost sensitivity (-50% to +100%)
            cost_scenarios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            cost_npvs = []
            
            for cost_factor in cost_scenarios:
                scenario_result = self.analyze_debris_removal_mission(
                    base_cost_musd * cost_factor, mass_kg, success_prob, 3
                )
                cost_npvs.append(scenario_result.get('financial_metrics', {}).get('net_present_value', 0))
            
            sensitivity_results['cost_sensitivity'] = {
                'scenarios': cost_scenarios,
                'npv_results': cost_npvs,
                'elasticity': self._calculate_elasticity(cost_scenarios, cost_npvs)
            }
            
            # Success probability sensitivity
            prob_scenarios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            prob_npvs = []
            
            for prob in prob_scenarios:
                scenario_result = self.analyze_debris_removal_mission(
                    base_cost_musd, mass_kg, prob, 3
                )
                prob_npvs.append(scenario_result.get('financial_metrics', {}).get('net_present_value', 0))
            
            sensitivity_results['probability_sensitivity'] = {
                'scenarios': prob_scenarios,
                'npv_results': prob_npvs,
                'elasticity': self._calculate_elasticity(prob_scenarios, prob_npvs)
            }
            
            return sensitivity_results
            
        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_elasticity(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate elasticity of y with respect to x."""
        try:
            if len(x_values) < 2 or len(y_values) < 2:
                return 0
            
            # Simple elasticity calculation
            dx = (x_values[-1] - x_values[0]) / x_values[0]
            dy = (y_values[-1] - y_values[0]) / abs(y_values[0]) if y_values[0] != 0 else 0
            
            return dy / dx if dx != 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating elasticity: {e}")
            return 0
    
    def _calculate_real_options_value(self, cost_musd: float, benefits: float, 
                                    years: int) -> float:
        """Calculate real options value using Black-Scholes-like approach."""
        try:
            # Real options parameters
            S = benefits  # Current value of underlying asset (benefits)
            K = cost_musd * 1e6  # Strike price (exercise cost)
            T = years  # Time to expiration
            r = self.risk_free_rate  # Risk-free rate
            sigma = 0.3  # Volatility estimate (30%)
            
            # Black-Scholes option value calculation
            if S <= 0 or K <= 0 or T <= 0:
                return 0
            
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            # Call option value (option to proceed with project)
            call_value = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            
            return max(call_value, 0)
            
        except Exception as e:
            logger.error(f"Error calculating real options value: {e}")
            return 0
    
    def _generate_economic_recommendation(self, financial_metrics: Dict[str, Any]) -> str:
        """Generate economic recommendation based on financial metrics."""
        try:
            npv = financial_metrics.get('net_present_value', 0)
            bcr = financial_metrics.get('benefit_cost_ratio', 0)
            roi = financial_metrics.get('return_on_investment', 0)
            
            if npv > 100 and bcr > 2.0:  # $100M+ NPV and 2:1 BCR
                return "STRONGLY RECOMMENDED: Excellent economic returns with high NPV and BCR"
            elif npv > 0 and bcr > 1.2:  # Positive NPV and 1.2:1 BCR
                return "RECOMMENDED: Positive economic returns justify investment"
            elif npv > -50 and bcr > 0.8:  # Close to breakeven
                return "CONDITIONAL: Consider if strategic benefits outweigh modest economic loss"
            else:
                return "NOT RECOMMENDED: Insufficient economic returns for investment"
                
        except Exception as e:
            logger.error(f"Error generating economic recommendation: {e}")
            return "Unable to generate recommendation due to analysis error"
    
    def _assess_economic_risks(self, financial_metrics: Dict[str, Any], 
                              sensitivity_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Assess economic risks based on metrics and sensitivity."""
        try:
            risks = {}
            
            # NPV risk
            npv = financial_metrics.get('net_present_value', 0)
            if npv < 0:
                risks['npv_risk'] = "HIGH: Negative NPV indicates economic loss"
            elif npv < 50:
                risks['npv_risk'] = "MEDIUM: Low positive NPV with limited margin"
            else:
                risks['npv_risk'] = "LOW: Strong positive NPV provides good margin"
            
            # Cost overrun risk
            cost_elasticity = (sensitivity_analysis.get('cost_sensitivity', {})
                             .get('elasticity', 0))
            if abs(cost_elasticity) > 2.0:
                risks['cost_risk'] = "HIGH: Highly sensitive to cost changes"
            elif abs(cost_elasticity) > 1.0:
                risks['cost_risk'] = "MEDIUM: Moderately sensitive to cost changes"
            else:
                risks['cost_risk'] = "LOW: Limited sensitivity to cost changes"
            
            # Technical risk
            success_elasticity = (sensitivity_analysis.get('probability_sensitivity', {})
                                .get('elasticity', 0))
            if abs(success_elasticity) > 3.0:
                risks['technical_risk'] = "HIGH: Highly dependent on mission success"
            elif abs(success_elasticity) > 1.5:
                risks['technical_risk'] = "MEDIUM: Moderately dependent on success"
            else:
                risks['technical_risk'] = "LOW: Robust to success probability variations"
            
            return risks
            
        except Exception as e:
            logger.error(f"Error assessing economic risks: {e}")
            return {'assessment_error': str(e)}
    
    def _get_default_economic_analysis(self) -> Dict[str, Any]:
        """Return default economic analysis when calculation fails."""
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'mission_parameters': {},
            'cost_breakdown': {'total_musd': 0, 'error': 'Analysis failed'},
            'benefit_analysis': {'total_benefits_musd': 0},
            'financial_metrics': {
                'net_present_value': 0,
                'return_on_investment': 0,
                'benefit_cost_ratio': 0,
                'economic_viability': False
            },
            'economic_recommendation': "Unable to analyze - insufficient data",
            'risk_assessment': {'analysis_risk': 'HIGH: Analysis failed to complete'}
        }

class SpaceEconomyMarketModel:
    """Model for space economy market dynamics and debris impact."""
    
    def __init__(self):
        self.model_version = "SpaceEconomy-v5.0"
        
        # Market size parameters (2024 estimates)
        self.global_space_economy = 500e9  # $500B global space economy
        self.satellite_services = 150e9    # $150B satellite services
        self.launch_industry = 15e9        # $15B launch industry
        self.debris_impact_factor = 0.02   # 2% economy impact from debris
    
    def model_debris_economic_impact(self, debris_growth_rate: float,
                                   analysis_years: int = 10) -> Dict[str, Any]:
        """Model economic impact of debris growth on space economy."""
        try:
            logger.info(f"Modeling debris economic impact over {analysis_years} years")
            
            # Current debris impact
            current_impact = self.global_space_economy * self.debris_impact_factor
            
            # Project future impacts
            yearly_impacts = []
            cumulative_impact = 0
            
            for year in range(1, analysis_years + 1):
                # Compound debris growth impact
                debris_factor = (1 + debris_growth_rate)**year
                annual_impact = current_impact * debris_factor
                cumulative_impact += annual_impact
                
                yearly_impacts.append({
                    'year': year,
                    'annual_impact_musd': annual_impact / 1e6,
                    'cumulative_impact_musd': cumulative_impact / 1e6,
                    'economy_impact_percent': (annual_impact / self.global_space_economy) * 100
                })
            
            # Calculate industry-specific impacts
            industry_impacts = self._calculate_industry_impacts(debris_growth_rate, analysis_years)
            
            return {
                'analysis_timestamp': datetime.now().isoformat(),
                'model_parameters': {
                    'debris_growth_rate': debris_growth_rate,
                    'analysis_years': analysis_years,
                    'base_economy_size_musd': self.global_space_economy / 1e6
                },
                'aggregate_impact': {
                    'current_annual_impact_musd': current_impact / 1e6,
                    'total_projected_impact_musd': cumulative_impact / 1e6,
                    'average_annual_impact_musd': (cumulative_impact / analysis_years) / 1e6
                },
                'yearly_projections': yearly_impacts,
                'industry_impacts': industry_impacts,
                'economic_scenarios': self._generate_economic_scenarios(debris_growth_rate, analysis_years)
            }
            
        except Exception as e:
            logger.error(f"Error modeling debris economic impact: {e}")
            return {'error': str(e)}
    
    def _calculate_industry_impacts(self, growth_rate: float, years: int) -> Dict[str, Any]:
        """Calculate debris impact on specific space industry sectors."""
        try:
            sectors = {
                'satellite_services': {
                    'base_value': self.satellite_services,
                    'vulnerability': 0.8  # High vulnerability to debris
                },
                'launch_services': {
                    'base_value': self.launch_industry, 
                    'vulnerability': 0.6  # Medium vulnerability
                },
                'manufacturing': {
                    'base_value': 50e9,    # $50B satellite manufacturing
                    'vulnerability': 0.4   # Lower direct vulnerability
                },
                'insurance': {
                    'base_value': 5e9,     # $5B space insurance
                    'vulnerability': 1.0   # Highest vulnerability
                }
            }
            
            industry_impacts = {}
            
            for sector, params in sectors.items():
                base_impact = params['base_value'] * self.debris_impact_factor * params['vulnerability']
                total_impact = sum([base_impact * (1 + growth_rate)**year for year in range(1, years + 1)])
                
                industry_impacts[sector] = {
                    'base_value_musd': params['base_value'] / 1e6,
                    'vulnerability_factor': params['vulnerability'],
                    'total_impact_musd': total_impact / 1e6,
                    'annual_average_impact_musd': (total_impact / years) / 1e6
                }
            
            return industry_impacts
            
        except Exception as e:
            logger.error(f"Error calculating industry impacts: {e}")
            return {}
    
    def _generate_economic_scenarios(self, base_growth_rate: float, 
                                   years: int) -> Dict[str, Any]:
        """Generate optimistic, baseline, and pessimistic economic scenarios."""
        try:
            scenarios = {
                'optimistic': base_growth_rate * 0.5,      # Half the debris growth
                'baseline': base_growth_rate,              # Current trend
                'pessimistic': base_growth_rate * 1.5      # 50% higher growth
            }
            
            scenario_results = {}
            
            for scenario_name, growth_rate in scenarios.items():
                total_impact = sum([
                    self.global_space_economy * self.debris_impact_factor * (1 + growth_rate)**year 
                    for year in range(1, years + 1)
                ])
                
                scenario_results[scenario_name] = {
                    'growth_rate': growth_rate,
                    'total_impact_musd': total_impact / 1e6,
                    'final_year_impact_musd': (self.global_space_economy * self.debris_impact_factor * 
                                             (1 + growth_rate)**years) / 1e6
                }
            
            return scenario_results
            
        except Exception as e:
            logger.error(f"Error generating economic scenarios: {e}")
            return {}

# Global instances
economic_analyzer = EconomicAnalyzer()
market_model = SpaceEconomyMarketModel()


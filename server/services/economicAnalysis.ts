import { SpaceObject, EconomicAnalysis } from "@shared/schema";
import { storage } from "../storage";
import { aiAnalysisService } from "./aiAnalysis";

export interface EconomicMetrics {
  totalInsuredValue: number; // USD
  annualLossEstimate: number; // USD
  removalCostAnalysis: {
    activeRemoval: number; // USD
    naturalDecay: number; // USD (monitoring costs)
    riskMitigation: number; // USD
  };
  sustainabilityMetrics: {
    environmentalImpact: number; // 0-100
    spaceSustainability: number; // 0-100
    complianceScore: number; // 0-100
  };
  marketImpact: {
    launchCostIncrease: number; // percentage
    insurancePremiumIncrease: number; // percentage
    missionDelayRisk: number; // percentage
  };
}

export interface CostBenefitAnalysis {
  scenario: string;
  costs: {
    development: number;
    operational: number;
    maintenance: number;
    insurance: number;
  };
  benefits: {
    riskReduction: number;
    assetProtection: number;
    futureOpportunities: number;
    complianceValue: number;
  };
  netPresentValue: number;
  returnOnInvestment: number;
  paybackPeriod: number; // years
}

export interface RegionalEconomicImpact {
  region: string;
  country: string;
  spaceAssets: number; // USD value
  dependentSectors: string[];
  economicRisk: number; // 0-1
  mitigationInvestment: number; // USD
  potentialGDPImpact: number; // USD
}

export class EconomicAnalysisService {
  private readonly spaceEconomySize = 469e9; // $469 billion global space economy (2023)
  private readonly satelliteInsuranceRate = 0.05; // 5% of satellite value
  private readonly launchCostPerKg = 5000; // USD per kg to LEO

  async calculateComprehensiveEconomics(objects: SpaceObject[]): Promise<EconomicMetrics> {
    try {
      const activeObjects = objects.filter(obj => obj.status === "active");
      const debrisObjects = objects.filter(obj => obj.type === "debris");
      
      // Calculate total insured value
      const totalInsuredValue = activeObjects.reduce((sum, obj) => {
        const estimatedValue = this.estimateSatelliteValue(obj);
        return sum + estimatedValue;
      }, 0);

      // Annual loss estimate based on collision probability
      const annualLossEstimate = await this.calculateAnnualLossEstimate(objects);

      // Removal cost analysis
      const removalCostAnalysis = await this.calculateRemovalCosts(debrisObjects);

      // Sustainability metrics
      const sustainabilityMetrics = await this.calculateSustainabilityMetrics(objects);

      // Market impact analysis
      const marketImpact = this.calculateMarketImpact(objects);

      return {
        totalInsuredValue,
        annualLossEstimate,
        removalCostAnalysis,
        sustainabilityMetrics,
        marketImpact,
      };
    } catch (error) {
      console.error("Error calculating comprehensive economics:", error);
      throw new Error("Failed to calculate economic metrics: " + error.message);
    }
  }

  private estimateSatelliteValue(object: SpaceObject): number {
    const baseCost = {
      satellite: 50e6, // $50M base satellite
      rocket_body: 100e6, // $100M launch vehicle
      debris: 0,
    };

    const base = baseCost[object.type as keyof typeof baseCost] || 50e6;
    
    // Adjust for size and capability
    const sizeFactor = Math.max(0.1, (object.size || 1) / 5);
    const massFactor = Math.max(0.1, (object.mass || 1000) / 5000);
    
    // Adjust for orbit value (GEO more valuable)
    const altitudeFactor = (object.altitude || 400) > 30000 ? 2.0 : 1.0;
    
    return base * sizeFactor * massFactor * altitudeFactor;
  }

  private async calculateAnnualLossEstimate(objects: SpaceObject[]): Promise<number> {
    try {
      const conjunctions = await storage.getActiveConjunctionEvents();
      let totalRisk = 0;

      for (const conjunction of conjunctions) {
        const primaryObj = objects.find(obj => obj.id === conjunction.primaryObjectId);
        const secondaryObj = objects.find(obj => obj.id === conjunction.secondaryObjectId);
        
        if (primaryObj && secondaryObj) {
          const primaryValue = this.estimateSatelliteValue(primaryObj);
          const secondaryValue = this.estimateSatelliteValue(secondaryObj);
          const expectedLoss = (primaryValue + secondaryValue) * conjunction.collisionProbability;
          totalRisk += expectedLoss;
        }
      }

      // Annualize the risk (conjunctions are typically short-term predictions)
      return totalRisk * 365; // Approximate annual risk
    } catch (error) {
      console.error("Error calculating annual loss estimate:", error);
      return 0;
    }
  }

  private async calculateRemovalCosts(debrisObjects: SpaceObject[]): Promise<{
    activeRemoval: number;
    naturalDecay: number;
    riskMitigation: number;
  }> {
    const highRiskDebris = debrisObjects.filter(obj => 
      obj.riskLevel === "high" || obj.riskLevel === "critical"
    );

    // Active removal costs (estimated)
    const activeRemovalCostPerObject = 5e6; // $5M per debris removal mission
    const activeRemoval = highRiskDebris.length * activeRemovalCostPerObject;

    // Natural decay monitoring costs
    const monitoringCostPerYear = 50000; // $50K per object per year
    const averageLifetime = 25; // years
    const naturalDecay = debrisObjects.length * monitoringCostPerYear * averageLifetime;

    // Risk mitigation costs (avoidance maneuvers, tracking)
    const riskMitigation = debrisObjects.length * 10000; // $10K per object for tracking/mitigation

    return {
      activeRemoval,
      naturalDecay,
      riskMitigation,
    };
  }

  private async calculateSustainabilityMetrics(objects: SpaceObject[]): Promise<{
    environmentalImpact: number;
    spaceSustainability: number;
    complianceScore: number;
  }> {
    const totalObjects = objects.length;
    const debrisCount = objects.filter(obj => obj.type === "debris").length;
    const debrisRatio = debrisCount / totalObjects;

    // Environmental impact (higher debris ratio = worse impact)
    const environmentalImpact = Math.max(0, 100 - (debrisRatio * 200));

    // Space sustainability (based on orbital congestion)
    const lowEarthOrbitObjects = objects.filter(obj => (obj.altitude || 0) < 2000).length;
    const congestionFactor = Math.min(1, lowEarthOrbitObjects / 10000);
    const spaceSustainability = Math.max(0, 100 - (congestionFactor * 100));

    // Compliance score (based on tracking and mitigation efforts)
    const trackedObjects = objects.filter(obj => obj.lastUpdate && 
      Date.now() - new Date(obj.lastUpdate).getTime() < 24 * 60 * 60 * 1000
    ).length;
    const trackingRatio = trackedObjects / totalObjects;
    const complianceScore = trackingRatio * 100;

    return {
      environmentalImpact,
      spaceSustainability,
      complianceScore,
    };
  }

  private calculateMarketImpact(objects: SpaceObject[]): {
    launchCostIncrease: number;
    insurancePremiumIncrease: number;
    missionDelayRisk: number;
  } {
    const debrisCount = objects.filter(obj => obj.type === "debris").length;
    const totalObjects = objects.length;
    const debrisRatio = debrisCount / totalObjects;

    // Launch cost increase due to debris avoidance and tracking requirements
    const launchCostIncrease = debrisRatio * 20; // Up to 20% increase

    // Insurance premium increase based on collision risk
    const highRiskRatio = objects.filter(obj => 
      obj.riskLevel === "high" || obj.riskLevel === "critical"
    ).length / totalObjects;
    const insurancePremiumIncrease = highRiskRatio * 50; // Up to 50% increase

    // Mission delay risk due to debris conjunctions
    const missionDelayRisk = Math.min(30, debrisRatio * 40); // Up to 30% chance

    return {
      launchCostIncrease,
      insurancePremiumIncrease,
      missionDelayRisk,
    };
  }

  async performCostBenefitAnalysis(
    scenario: string,
    mitigationStrategy: {
      activeRemovalTargets: number;
      trackingImprovements: boolean;
      launchStandards: boolean;
      internationalCooperation: boolean;
    }
  ): Promise<CostBenefitAnalysis> {
    try {
      // Development costs
      const developmentCosts = {
        activeRemoval: mitigationStrategy.activeRemovalTargets * 10e6, // $10M per removal mission development
        tracking: mitigationStrategy.trackingImprovements ? 500e6 : 0, // $500M tracking upgrade
        standards: mitigationStrategy.launchStandards ? 100e6 : 0, // $100M standards development
        cooperation: mitigationStrategy.internationalCooperation ? 50e6 : 0, // $50M cooperation framework
      };

      // Operational costs (annual)
      const operationalCosts = {
        activeRemoval: mitigationStrategy.activeRemovalTargets * 5e6, // $5M per removal operation
        tracking: mitigationStrategy.trackingImprovements ? 100e6 : 50e6, // Enhanced vs basic tracking
        standards: mitigationStrategy.launchStandards ? 20e6 : 0, // Standards enforcement
        cooperation: mitigationStrategy.internationalCooperation ? 25e6 : 0, // International coordination
      };

      // Benefits calculation
      const currentObjects = await storage.getSpaceObjects();
      const currentEconomics = await this.calculateComprehensiveEconomics(currentObjects);
      
      const riskReduction = this.calculateRiskReduction(mitigationStrategy, currentEconomics);
      const assetProtection = riskReduction * currentEconomics.totalInsuredValue;
      const futureOpportunities = this.calculateFutureOpportunityValue(mitigationStrategy);
      const complianceValue = mitigationStrategy.launchStandards ? 200e6 : 0; // Regulatory compliance value

      // Financial calculations
      const totalCosts = Object.values(developmentCosts).reduce((a, b) => a + b, 0) +
                        Object.values(operationalCosts).reduce((a, b) => a + b, 0) * 10; // 10-year horizon
      
      const totalBenefits = riskReduction + assetProtection + futureOpportunities + complianceValue;
      
      const netPresentValue = this.calculateNPV(totalBenefits, totalCosts, 0.05, 10); // 5% discount rate, 10 years
      const returnOnInvestment = (totalBenefits - totalCosts) / totalCosts * 100;
      const paybackPeriod = totalCosts / (totalBenefits / 10); // Years to payback

      return {
        scenario,
        costs: {
          development: Object.values(developmentCosts).reduce((a, b) => a + b, 0),
          operational: Object.values(operationalCosts).reduce((a, b) => a + b, 0),
          maintenance: operationalCosts.tracking * 0.2, // 20% of operational for maintenance
          insurance: 0, // Assumed covered in operational
        },
        benefits: {
          riskReduction,
          assetProtection,
          futureOpportunities,
          complianceValue,
        },
        netPresentValue,
        returnOnInvestment,
        paybackPeriod,
      };
    } catch (error) {
      console.error("Error performing cost-benefit analysis:", error);
      throw new Error("Failed to perform cost-benefit analysis: " + error.message);
    }
  }

  private calculateRiskReduction(
    strategy: any,
    currentEconomics: EconomicMetrics
  ): number {
    let riskReduction = 0;
    
    if (strategy.activeRemovalTargets > 0) {
      riskReduction += strategy.activeRemovalTargets * 1e6; // $1M risk reduction per removal
    }
    
    if (strategy.trackingImprovements) {
      riskReduction += currentEconomics.annualLossEstimate * 0.3; // 30% reduction in losses
    }
    
    if (strategy.launchStandards) {
      riskReduction += currentEconomics.annualLossEstimate * 0.4; // 40% reduction in future debris
    }
    
    if (strategy.internationalCooperation) {
      riskReduction += currentEconomics.annualLossEstimate * 0.2; // 20% improvement through cooperation
    }
    
    return Math.min(riskReduction, currentEconomics.annualLossEstimate * 0.8); // Max 80% reduction
  }

  private calculateFutureOpportunityValue(strategy: any): number {
    // Future space economy growth enabled by debris mitigation
    const spaceGrowthRate = 0.08; // 8% annual growth
    const enabledGrowth = strategy.activeRemovalTargets > 50 ? 0.02 : 0.01; // Additional growth from mitigation
    
    return this.spaceEconomySize * enabledGrowth * 10; // 10-year benefit
  }

  private calculateNPV(
    benefits: number,
    costs: number,
    discountRate: number,
    years: number
  ): number {
    const annualBenefits = benefits / years;
    let npv = -costs; // Initial investment
    
    for (let year = 1; year <= years; year++) {
      npv += annualBenefits / Math.pow(1 + discountRate, year);
    }
    
    return npv;
  }

  async analyzeRegionalEconomicImpact(): Promise<RegionalEconomicImpact[]> {
    try {
      const regions = [
        {
          region: "North America",
          country: "USA",
          spaceAssets: 200e9,
          dependentSectors: ["Communications", "Navigation", "Earth Observation", "Defense"],
          mitigationInvestment: 5e9,
          gdpDependency: 0.15,
        },
        {
          region: "Europe",
          country: "EU",
          spaceAssets: 80e9,
          dependentSectors: ["Communications", "Earth Observation", "Scientific Research"],
          mitigationInvestment: 2e9,
          gdpDependency: 0.08,
        },
        {
          region: "Asia Pacific",
          country: "China",
          spaceAssets: 60e9,
          dependentSectors: ["Communications", "Navigation", "Remote Sensing"],
          mitigationInvestment: 1.5e9,
          gdpDependency: 0.06,
        },
      ];

      const objects = await storage.getSpaceObjects();
      const debrisRisk = objects.filter(obj => obj.type === "debris").length / objects.length;

      return regions.map(region => ({
        region: region.region,
        country: region.country,
        spaceAssets: region.spaceAssets,
        dependentSectors: region.dependentSectors,
        economicRisk: debrisRisk * region.gdpDependency,
        mitigationInvestment: region.mitigationInvestment,
        potentialGDPImpact: region.spaceAssets * debrisRisk * region.gdpDependency,
      }));
    } catch (error) {
      console.error("Error analyzing regional economic impact:", error);
      throw new Error("Failed to analyze regional economic impact: " + error.message);
    }
  }
}

export const economicAnalysisService = new EconomicAnalysisService();

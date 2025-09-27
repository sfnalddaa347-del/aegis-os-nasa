// Advanced Calculation Validation System
// Comprehensive validation for orbital mechanics and atmospheric calculations

interface ValidationRule {
  name: string;
  description: string;
  validate: (data: any) => ValidationResult;
  severity: 'error' | 'warning' | 'info';
  category: 'orbital' | 'atmospheric' | 'physical' | 'numerical';
}

interface ValidationResult {
  isValid: boolean;
  message: string;
  severity: 'error' | 'warning' | 'info';
  value?: number;
  expected?: number;
  tolerance?: number;
  suggestions?: string[];
}

interface ValidationReport {
  overall: {
    isValid: boolean;
    score: number;
    totalChecks: number;
    passedChecks: number;
    failedChecks: number;
  };
  categories: {
    [key: string]: {
      isValid: boolean;
      score: number;
      checks: ValidationResult[];
    };
  };
  recommendations: string[];
  timestamp: Date;
}

export class CalculationValidationSystem {
  private rules: ValidationRule[] = [];
  private referenceData: Map<string, any> = new Map();
  private validationHistory: Map<string, ValidationReport[]> = new Map();

  constructor() {
    this.initializeValidationRules();
    this.loadReferenceData();
  }

  private initializeValidationRules(): void {
    // Orbital mechanics validation rules
    this.rules.push({
      name: 'orbital_energy_conservation',
      description: 'Check if orbital energy is conserved',
      validate: (data) => this.validateOrbitalEnergyConservation(data),
      severity: 'error',
      category: 'orbital'
    });

    this.rules.push({
      name: 'orbital_angular_momentum_conservation',
      description: 'Check if angular momentum is conserved',
      validate: (data) => this.validateAngularMomentumConservation(data),
      severity: 'error',
      category: 'orbital'
    });

    this.rules.push({
      name: 'orbital_elements_range',
      description: 'Check if orbital elements are within valid ranges',
      validate: (data) => this.validateOrbitalElementsRange(data),
      severity: 'error',
      category: 'orbital'
    });

    this.rules.push({
      name: 'orbital_period_consistency',
      description: 'Check if orbital period is consistent with semi-major axis',
      validate: (data) => this.validateOrbitalPeriodConsistency(data),
      severity: 'warning',
      category: 'orbital'
    });

    // Atmospheric model validation rules
    this.rules.push({
      name: 'atmospheric_density_positive',
      description: 'Check if atmospheric density is positive',
      validate: (data) => this.validateAtmosphericDensityPositive(data),
      severity: 'error',
      category: 'atmospheric'
    });

    this.rules.push({
      name: 'atmospheric_temperature_range',
      description: 'Check if atmospheric temperature is within valid range',
      validate: (data) => this.validateAtmosphericTemperatureRange(data),
      severity: 'warning',
      category: 'atmospheric'
    });

    this.rules.push({
      name: 'atmospheric_pressure_consistency',
      description: 'Check if atmospheric pressure is consistent with density and temperature',
      validate: (data) => this.validateAtmosphericPressureConsistency(data),
      severity: 'warning',
      category: 'atmospheric'
    });

    // Physical constraints validation rules
    this.rules.push({
      name: 'velocity_less_than_escape',
      description: 'Check if velocity is less than escape velocity',
      validate: (data) => this.validateVelocityLessThanEscape(data),
      severity: 'error',
      category: 'physical'
    });

    this.rules.push({
      name: 'altitude_above_surface',
      description: 'Check if altitude is above Earth surface',
      validate: (data) => this.validateAltitudeAboveSurface(data),
      severity: 'error',
      category: 'physical'
    });

    this.rules.push({
      name: 'mass_positive',
      description: 'Check if mass is positive',
      validate: (data) => this.validateMassPositive(data),
      severity: 'error',
      category: 'physical'
    });

    // Numerical stability validation rules
    this.rules.push({
      name: 'numerical_stability',
      description: 'Check for numerical stability issues',
      validate: (data) => this.validateNumericalStability(data),
      severity: 'warning',
      category: 'numerical'
    });

    this.rules.push({
      name: 'convergence_criteria',
      description: 'Check if convergence criteria are met',
      validate: (data) => this.validateConvergenceCriteria(data),
      severity: 'warning',
      category: 'numerical'
    });

    console.log(`Initialized ${this.rules.length} validation rules`);
  }

  private loadReferenceData(): void {
    // Load reference data for validation
    this.referenceData.set('earth_radius', 6378.137); // km
    this.referenceData.set('earth_mass', 5.972e24); // kg
    this.referenceData.set('gravitational_constant', 6.67430e-11); // m³/(kg·s²)
    this.referenceData.set('standard_gravity', 9.80665); // m/s²
    this.referenceData.set('escape_velocity_leo', 11.2); // km/s
    this.referenceData.set('min_atmospheric_density', 1e-15); // kg/m³
    this.referenceData.set('max_atmospheric_density', 1.225); // kg/m³
    this.referenceData.set('min_temperature', 50); // K
    this.referenceData.set('max_temperature', 2000); // K
  }

  // Validate orbital energy conservation
  private validateOrbitalEnergyConservation(data: any): ValidationResult {
    if (!data.position || !data.velocity || !data.semiMajorAxis) {
      return {
        isValid: false,
        message: 'Missing required data for energy conservation check',
        severity: 'error'
      };
    }

    const GM = 398600.4418; // km³/s²
    const r = Math.sqrt(
      data.position[0]**2 + data.position[1]**2 + data.position[2]**2
    );
    const v = Math.sqrt(
      data.velocity[0]**2 + data.velocity[1]**2 + data.velocity[2]**2
    );

    const kineticEnergy = 0.5 * v * v;
    const potentialEnergy = -GM / r;
    const totalEnergy = kineticEnergy + potentialEnergy;

    const expectedEnergy = -GM / (2 * data.semiMajorAxis);
    const energyError = Math.abs(totalEnergy - expectedEnergy) / Math.abs(expectedEnergy);

    const tolerance = 0.01; // 1% tolerance
    const isValid = energyError < tolerance;

    return {
      isValid,
      message: isValid 
        ? 'Orbital energy conservation is valid'
        : `Energy conservation error: ${(energyError * 100).toFixed(2)}%`,
      severity: isValid ? 'info' : 'error',
      value: totalEnergy,
      expected: expectedEnergy,
      tolerance: tolerance,
      suggestions: isValid ? [] : [
        'Check orbital elements calculation',
        'Verify velocity magnitude',
        'Review gravitational constant value'
      ]
    };
  }

  // Validate angular momentum conservation
  private validateAngularMomentumConservation(data: any): ValidationResult {
    if (!data.position || !data.velocity) {
      return {
        isValid: false,
        message: 'Missing required data for angular momentum conservation check',
        severity: 'error'
      };
    }

    // Calculate angular momentum vector
    const r = [data.position[0], data.position[1], data.position[2]];
    const v = [data.velocity[0], data.velocity[1], data.velocity[2]];
    
    const h = [
      r[1] * v[2] - r[2] * v[1],
      r[2] * v[0] - r[0] * v[2],
      r[0] * v[1] - r[1] * v[0]
    ];

    const hMagnitude = Math.sqrt(h[0]**2 + h[1]**2 + h[2]**2);

    if (!data.expectedAngularMomentum) {
      return {
        isValid: true,
        message: 'Angular momentum calculated successfully',
        severity: 'info',
        value: hMagnitude
      };
    }

    const error = Math.abs(hMagnitude - data.expectedAngularMomentum) / data.expectedAngularMomentum;
    const tolerance = 0.01; // 1% tolerance
    const isValid = error < tolerance;

    return {
      isValid,
      message: isValid 
        ? 'Angular momentum conservation is valid'
        : `Angular momentum error: ${(error * 100).toFixed(2)}%`,
      severity: isValid ? 'info' : 'error',
      value: hMagnitude,
      expected: data.expectedAngularMomentum,
      tolerance: tolerance,
      suggestions: isValid ? [] : [
        'Check position and velocity vectors',
        'Verify cross product calculation',
        'Review orbital elements'
      ]
    };
  }

  // Validate orbital elements range
  private validateOrbitalElementsRange(data: any): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Check semi-major axis
    if (data.semiMajorAxis) {
      if (data.semiMajorAxis < 6378.137) {
        errors.push('Semi-major axis is below Earth radius');
      } else if (data.semiMajorAxis > 100000) {
        warnings.push('Semi-major axis is very large (>100,000 km)');
      }
    }

    // Check eccentricity
    if (data.eccentricity !== undefined) {
      if (data.eccentricity < 0) {
        errors.push('Eccentricity cannot be negative');
      } else if (data.eccentricity >= 1) {
        errors.push('Eccentricity must be less than 1 for closed orbits');
      } else if (data.eccentricity > 0.9) {
        warnings.push('Eccentricity is very high (>0.9)');
      }
    }

    // Check inclination
    if (data.inclination !== undefined) {
      if (data.inclination < 0 || data.inclination > Math.PI) {
        errors.push('Inclination must be between 0 and π radians');
      }
    }

    // Check right ascension
    if (data.rightAscension !== undefined) {
      if (data.rightAscension < 0 || data.rightAscension > 2 * Math.PI) {
        errors.push('Right ascension must be between 0 and 2π radians');
      }
    }

    // Check argument of periapsis
    if (data.argumentOfPeriapsis !== undefined) {
      if (data.argumentOfPeriapsis < 0 || data.argumentOfPeriapsis > 2 * Math.PI) {
        errors.push('Argument of periapsis must be between 0 and 2π radians');
      }
    }

    // Check mean anomaly
    if (data.meanAnomaly !== undefined) {
      if (data.meanAnomaly < 0 || data.meanAnomaly > 2 * Math.PI) {
        errors.push('Mean anomaly must be between 0 and 2π radians');
      }
    }

    const isValid = errors.length === 0;
    const severity = errors.length > 0 ? 'error' : (warnings.length > 0 ? 'warning' : 'info');
    const message = isValid 
      ? 'All orbital elements are within valid ranges'
      : `${errors.length} errors, ${warnings.length} warnings found`;

    return {
      isValid,
      message,
      severity,
      suggestions: [...errors, ...warnings]
    };
  }

  // Validate orbital period consistency
  private validateOrbitalPeriodConsistency(data: any): ValidationResult {
    if (!data.semiMajorAxis || !data.period) {
      return {
        isValid: false,
        message: 'Missing required data for period consistency check',
        severity: 'error'
      };
    }

    const GM = 398600.4418; // km³/s²
    const expectedPeriod = 2 * Math.PI * Math.sqrt(Math.pow(data.semiMajorAxis, 3) / GM);
    const error = Math.abs(data.period - expectedPeriod) / expectedPeriod;
    const tolerance = 0.05; // 5% tolerance
    const isValid = error < tolerance;

    return {
      isValid,
      message: isValid 
        ? 'Orbital period is consistent with semi-major axis'
        : `Period inconsistency: ${(error * 100).toFixed(2)}%`,
      severity: isValid ? 'info' : 'warning',
      value: data.period,
      expected: expectedPeriod,
      tolerance: tolerance,
      suggestions: isValid ? [] : [
        'Check semi-major axis calculation',
        'Verify period calculation',
        'Review gravitational constant value'
      ]
    };
  }

  // Validate atmospheric density is positive
  private validateAtmosphericDensityPositive(data: any): ValidationResult {
    if (data.density === undefined) {
      return {
        isValid: false,
        message: 'Atmospheric density not provided',
        severity: 'error'
      };
    }

    const isValid = data.density > 0;
    const minDensity = this.referenceData.get('min_atmospheric_density');

    return {
      isValid,
      message: isValid 
        ? 'Atmospheric density is positive'
        : `Atmospheric density is negative or zero: ${data.density}`,
      severity: isValid ? 'info' : 'error',
      value: data.density,
      expected: minDensity,
      suggestions: isValid ? [] : [
        'Check atmospheric model calculation',
        'Verify altitude input',
        'Review atmospheric constants'
      ]
    };
  }

  // Validate atmospheric temperature range
  private validateAtmosphericTemperatureRange(data: any): ValidationResult {
    if (!data.temperature) {
      return {
        isValid: false,
        message: 'Atmospheric temperature not provided',
        severity: 'error'
      };
    }

    const minTemp = this.referenceData.get('min_temperature');
    const maxTemp = this.referenceData.get('max_temperature');
    const isValid = data.temperature >= minTemp && data.temperature <= maxTemp;

    return {
      isValid,
      message: isValid 
        ? 'Atmospheric temperature is within valid range'
        : `Temperature out of range: ${data.temperature}K (expected: ${minTemp}-${maxTemp}K)`,
      severity: isValid ? 'info' : 'warning',
      value: data.temperature,
      expected: (minTemp + maxTemp) / 2,
      tolerance: (maxTemp - minTemp) / 2,
      suggestions: isValid ? [] : [
        'Check atmospheric model parameters',
        'Verify altitude and solar activity inputs',
        'Review temperature calculation'
      ]
    };
  }

  // Validate atmospheric pressure consistency
  private validateAtmosphericPressureConsistency(data: any): ValidationResult {
    if (!data.pressure || !data.density || !data.temperature) {
      return {
        isValid: false,
        message: 'Missing required data for pressure consistency check',
        severity: 'error'
      };
    }

    const R = 287; // Specific gas constant for air J/(kg·K)
    const expectedPressure = data.density * R * data.temperature;
    const error = Math.abs(data.pressure - expectedPressure) / expectedPressure;
    const tolerance = 0.1; // 10% tolerance
    const isValid = error < tolerance;

    return {
      isValid,
      message: isValid 
        ? 'Atmospheric pressure is consistent with density and temperature'
        : `Pressure inconsistency: ${(error * 100).toFixed(2)}%`,
      severity: isValid ? 'info' : 'warning',
      value: data.pressure,
      expected: expectedPressure,
      tolerance: tolerance,
      suggestions: isValid ? [] : [
        'Check ideal gas law calculation',
        'Verify gas constant value',
        'Review density and temperature inputs'
      ]
    };
  }

  // Validate velocity is less than escape velocity
  private validateVelocityLessThanEscape(data: any): ValidationResult {
    if (!data.velocity || !data.altitude) {
      return {
        isValid: false,
        message: 'Missing required data for escape velocity check',
        severity: 'error'
      };
    }

    const v = Math.sqrt(
      data.velocity[0]**2 + data.velocity[1]**2 + data.velocity[2]**2
    );
    const GM = 398600.4418; // km³/s²
    const r = data.altitude + 6378.137; // km
    const escapeVelocity = Math.sqrt(2 * GM / r);
    const isValid = v < escapeVelocity;

    return {
      isValid,
      message: isValid 
        ? 'Velocity is less than escape velocity'
        : `Velocity exceeds escape velocity: ${v.toFixed(2)} km/s > ${escapeVelocity.toFixed(2)} km/s`,
      severity: isValid ? 'info' : 'error',
      value: v,
      expected: escapeVelocity,
      suggestions: isValid ? [] : [
        'Check velocity calculation',
        'Verify altitude input',
        'Review orbital mechanics'
      ]
    };
  }

  // Validate altitude is above Earth surface
  private validateAltitudeAboveSurface(data: any): ValidationResult {
    if (data.altitude === undefined) {
      return {
        isValid: false,
        message: 'Altitude not provided',
        severity: 'error'
      };
    }

    const earthRadius = this.referenceData.get('earth_radius');
    const isValid = data.altitude >= 0;

    return {
      isValid,
      message: isValid 
        ? 'Altitude is above Earth surface'
        : `Altitude is below Earth surface: ${data.altitude} km`,
      severity: isValid ? 'info' : 'error',
      value: data.altitude,
      expected: earthRadius,
      suggestions: isValid ? [] : [
        'Check altitude calculation',
        'Verify position vector',
        'Review Earth radius value'
      ]
    };
  }

  // Validate mass is positive
  private validateMassPositive(data: any): ValidationResult {
    if (data.mass === undefined) {
      return {
        isValid: false,
        message: 'Mass not provided',
        severity: 'error'
      };
    }

    const isValid = data.mass > 0;

    return {
      isValid,
      message: isValid 
        ? 'Mass is positive'
        : `Mass is negative or zero: ${data.mass} kg`,
      severity: isValid ? 'info' : 'error',
      value: data.mass,
      suggestions: isValid ? [] : [
        'Check mass input',
        'Verify object properties',
        'Review mass calculation'
      ]
    };
  }

  // Validate numerical stability
  private validateNumericalStability(data: any): ValidationResult {
    const warnings: string[] = [];

    // Check for very large numbers
    if (data.position) {
      const maxPosition = Math.max(...data.position.map(Math.abs));
      if (maxPosition > 1e6) {
        warnings.push('Position values are very large (>1,000,000 km)');
      }
    }

    if (data.velocity) {
      const maxVelocity = Math.max(...data.velocity.map(Math.abs));
      if (maxVelocity > 100) {
        warnings.push('Velocity values are very large (>100 km/s)');
      }
    }

    // Check for very small numbers
    if (data.semiMajorAxis && data.semiMajorAxis < 1e-6) {
      warnings.push('Semi-major axis is very small (<1e-6 km)');
    }

    if (data.eccentricity && data.eccentricity < 1e-10) {
      warnings.push('Eccentricity is very small (<1e-10)');
    }

    const isValid = warnings.length === 0;

    return {
      isValid,
      message: isValid 
        ? 'No numerical stability issues detected'
        : `${warnings.length} numerical stability warnings`,
      severity: isValid ? 'info' : 'warning',
      suggestions: warnings
    };
  }

  // Validate convergence criteria
  private validateConvergenceCriteria(data: any): ValidationResult {
    if (!data.convergenceData) {
      return {
        isValid: true,
        message: 'No convergence data to validate',
        severity: 'info'
      };
    }

    const { iterations, tolerance, achievedTolerance } = data.convergenceData;
    const maxIterations = 1000;
    const isValid = achievedTolerance <= tolerance && iterations <= maxIterations;

    return {
      isValid,
      message: isValid 
        ? 'Convergence criteria met'
        : `Convergence failed: ${iterations} iterations, tolerance: ${achievedTolerance}`,
      severity: isValid ? 'info' : 'warning',
      value: achievedTolerance,
      expected: tolerance,
      suggestions: isValid ? [] : [
        'Increase maximum iterations',
        'Relax tolerance requirements',
        'Check initial conditions',
        'Review numerical method'
      ]
    };
  }

  // Run comprehensive validation
  validate(data: any, categories?: string[]): ValidationReport {
    const startTime = Date.now();
    const applicableRules = categories 
      ? this.rules.filter(rule => categories.includes(rule.category))
      : this.rules;

    const results: { [key: string]: ValidationResult[] } = {};
    let totalChecks = 0;
    let passedChecks = 0;
    let failedChecks = 0;

    // Initialize categories
    const categoryList = [...new Set(applicableRules.map(rule => rule.category))];
    categoryList.forEach(category => {
      results[category] = [];
    });

    // Run validation rules
    applicableRules.forEach(rule => {
      try {
        const result = rule.validate(data);
        results[rule.category].push(result);
        totalChecks++;
        
        if (result.isValid) {
          passedChecks++;
        } else {
          failedChecks++;
        }
      } catch (error) {
        console.error(`Validation rule ${rule.name} failed:`, error);
        results[rule.category].push({
          isValid: false,
          message: `Validation rule failed: ${error.message}`,
          severity: 'error'
        });
        totalChecks++;
        failedChecks++;
      }
    });

    // Calculate category scores
    const categoryScores: { [key: string]: any } = {};
    categoryList.forEach(category => {
      const categoryResults = results[category];
      const categoryPassed = categoryResults.filter(r => r.isValid).length;
      const categoryTotal = categoryResults.length;
      const categoryScore = categoryTotal > 0 ? categoryPassed / categoryTotal : 1;

      categoryScores[category] = {
        isValid: categoryScore >= 0.8, // 80% threshold
        score: categoryScore,
        checks: categoryResults
      };
    });

    // Calculate overall score
    const overallScore = totalChecks > 0 ? passedChecks / totalChecks : 1;
    const overallValid = overallScore >= 0.8; // 80% threshold

    // Generate recommendations
    const recommendations = this.generateRecommendations(results);

    const report: ValidationReport = {
      overall: {
        isValid: overallValid,
        score: overallScore,
        totalChecks,
        passedChecks,
        failedChecks
      },
      categories: categoryScores,
      recommendations,
      timestamp: new Date()
    };

    // Store in history
    const dataKey = this.generateDataKey(data);
    if (!this.validationHistory.has(dataKey)) {
      this.validationHistory.set(dataKey, []);
    }
    this.validationHistory.get(dataKey)!.push(report);

    const endTime = Date.now();
    console.log(`Validation completed in ${endTime - startTime}ms`);

    return report;
  }

  // Generate recommendations based on validation results
  private generateRecommendations(results: { [key: string]: ValidationResult[] }): string[] {
    const recommendations: string[] = [];

    Object.entries(results).forEach(([category, categoryResults]) => {
      const failedResults = categoryResults.filter(r => !r.isValid);
      
      if (failedResults.length > 0) {
        recommendations.push(`${category} validation failed: ${failedResults.length} issues found`);
        
        failedResults.forEach(result => {
          if (result.suggestions) {
            recommendations.push(...result.suggestions);
          }
        });
      }
    });

    return [...new Set(recommendations)]; // Remove duplicates
  }

  // Generate data key for history
  private generateDataKey(data: any): string {
    const key = JSON.stringify(data, (key, value) => {
      if (typeof value === 'number') {
        return value.toFixed(6);
      }
      return value;
    });
    return Buffer.from(key).toString('base64').substring(0, 32);
  }

  // Get validation history
  getValidationHistory(dataKey?: string): Map<string, ValidationReport[]> | ValidationReport[] {
    if (dataKey) {
      return this.validationHistory.get(dataKey) || [];
    }
    return this.validationHistory;
  }

  // Get validation statistics
  getValidationStats(): any {
    const totalValidations = Array.from(this.validationHistory.values())
      .reduce((sum, reports) => sum + reports.length, 0);
    
    const averageScore = Array.from(this.validationHistory.values())
      .flat()
      .reduce((sum, report) => sum + report.overall.score, 0) / totalValidations;

    return {
      totalRules: this.rules.length,
      totalValidations,
      averageScore: averageScore || 0,
      historySize: this.validationHistory.size
    };
  }

  // Clear validation history
  clearValidationHistory(): void {
    this.validationHistory.clear();
  }
}

export const calculationValidation = new CalculationValidationSystem();

export default CalculationValidationSystem;

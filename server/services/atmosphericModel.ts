import { storage } from "../storage";
import { InsertAtmosphericData } from "@shared/schema";

export interface NRLMSISE00Params {
  altitude: number; // km
  latitude: number; // degrees
  longitude: number; // degrees
  year: number;
  dayOfYear: number;
  secondsOfDay: number;
  solarFlux: number; // F10.7
  solarFluxAvg: number; // F10.7A
  geomagneticIndex: number; // Ap
}

export interface AtmosphericConditions {
  totalDensity: number; // kg/m³
  temperature: number; // K
  pressure: number; // Pa
  composition: {
    n2: number; // kg/m³
    o2: number; // kg/m³
    o: number; // kg/m³
    ar: number; // kg/m³
    he: number; // kg/m³
    h: number; // kg/m³
  };
  windSpeed: number; // m/s
  windDirection: number; // degrees
}

export interface DragCoefficients {
  cd: number; // drag coefficient
  area: number; // cross-sectional area (m²)
  dragForce: number; // N
  accelerationDrag: number; // m/s²
}

export class AtmosphericModelService {
  private readonly earthRadius = 6371000; // meters
  private readonly gravitationalParameter = 3.986004418e14; // m³/s²

  async calculateNRLMSISE00(params: NRLMSISE00Params): Promise<AtmosphericConditions> {
    try {
      // Simplified NRLMSISE-00 model implementation
      // In production, this would interface with the actual NRLMSISE-00 Fortran code
      
      const { altitude, latitude, solarFlux, geomagneticIndex } = params;
      
      // Base atmospheric parameters at sea level
      const seaLevelDensity = 1.225; // kg/m³
      const seaLevelTemp = 288.15; // K
      const seaLevelPressure = 101325; // Pa
      
      // Scale heights for different atmospheric layers
      const troposphereHeight = 11; // km
      const stratosphereHeight = 47; // km
      const mesosphereHeight = 86; // km
      
      let temperature: number;
      let pressure: number;
      let density: number;
      
      if (altitude <= troposphereHeight) {
        // Troposphere
        const lapseRate = -6.5; // K/km
        temperature = seaLevelTemp + lapseRate * altitude;
        pressure = seaLevelPressure * Math.pow(temperature / seaLevelTemp, -9.80665 / (lapseRate * 287));
        density = pressure / (287 * temperature);
      } else if (altitude <= stratosphereHeight) {
        // Stratosphere
        const stratTemp = 216.65; // K
        temperature = stratTemp;
        const tropoPressure = seaLevelPressure * Math.pow(216.65 / seaLevelTemp, -9.80665 / (-6.5 * 287));
        pressure = tropoPressure * Math.exp(-9.80665 * (altitude - troposphereHeight) * 1000 / (287 * stratTemp));
        density = pressure / (287 * temperature);
      } else {
        // Thermosphere - where most space objects are
        temperature = this.calculateThermosphereTemperature(altitude, solarFlux, geomagneticIndex);
        density = this.calculateThermosphereDensity(altitude, temperature, solarFlux, geomagneticIndex);
        pressure = density * 287 * temperature;
      }
      
      // Solar and geomagnetic activity corrections
      const solarActivityFactor = 1 + (solarFlux - 150) / 300;
      const geoActivityFactor = 1 + geomagneticIndex / 50;
      
      density *= solarActivityFactor * geoActivityFactor;
      
      // Atmospheric composition (simplified)
      const composition = this.calculateComposition(altitude, density);
      
      // Wind modeling (simplified)
      const windSpeed = Math.random() * 200; // m/s (simplified)
      const windDirection = Math.random() * 360; // degrees
      
      return {
        totalDensity: density,
        temperature,
        pressure,
        composition,
        windSpeed,
        windDirection,
      };
    } catch (error) {
      console.error("Error calculating NRLMSISE-00:", error);
      throw new Error("Failed to calculate atmospheric conditions: " + error.message);
    }
  }

  private calculateThermosphereTemperature(
    altitude: number,
    solarFlux: number,
    geoIndex: number
  ): number {
    // Simplified thermosphere temperature model
    const baseTemp = 1000; // K
    const maxTemp = 2000; // K
    const solarFactor = (solarFlux - 70) / (400 - 70);
    const geoFactor = geoIndex / 400;
    
    const altitudeFactor = Math.min(1, (altitude - 86) / 500);
    
    return baseTemp + (maxTemp - baseTemp) * altitudeFactor * (1 + solarFactor + geoFactor);
  }

  private calculateThermosphereDensity(
    altitude: number,
    temperature: number,
    solarFlux: number,
    geoIndex: number
  ): number {
    // Simplified density calculation for thermosphere
    const h0 = 120; // km reference altitude
    const rho0 = 2e-10; // kg/m³ reference density
    
    const scaleHeight = (287 * temperature) / 9.80665 / 1000; // km
    const densityRatio = Math.exp(-(altitude - h0) / scaleHeight);
    
    const solarFactor = Math.pow(solarFlux / 150, 0.5);
    const geoFactor = Math.pow(1 + geoIndex / 400, 0.3);
    
    return rho0 * densityRatio * solarFactor * geoFactor;
  }

  private calculateComposition(altitude: number, totalDensity: number) {
    // Simplified atmospheric composition model
    if (altitude < 100) {
      return {
        n2: totalDensity * 0.78,
        o2: totalDensity * 0.21,
        o: totalDensity * 0.005,
        ar: totalDensity * 0.0093,
        he: totalDensity * 0.0001,
        h: totalDensity * 0.00001,
      };
    } else if (altitude < 200) {
      return {
        n2: totalDensity * 0.60,
        o2: totalDensity * 0.15,
        o: totalDensity * 0.20,
        ar: totalDensity * 0.03,
        he: totalDensity * 0.01,
        h: totalDensity * 0.01,
      };
    } else {
      return {
        n2: totalDensity * 0.30,
        o2: totalDensity * 0.05,
        o: totalDensity * 0.40,
        ar: totalDensity * 0.05,
        he: totalDensity * 0.15,
        h: totalDensity * 0.05,
      };
    }
  }

  async calculateDragCoefficients(
    objectMass: number,
    objectArea: number,
    altitude: number,
    velocity: number,
    solarFlux: number,
    geoIndex: number
  ): Promise<DragCoefficients> {
    try {
      const atmospheric = await this.calculateNRLMSISE00({
        altitude,
        latitude: 0, // Simplified
        longitude: 0,
        year: new Date().getFullYear(),
        dayOfYear: Math.floor((Date.now() - new Date(new Date().getFullYear(), 0, 0).getTime()) / 86400000),
        secondsOfDay: (Date.now() % 86400000) / 1000,
        solarFlux,
        solarFluxAvg: solarFlux,
        geomagneticIndex: geoIndex,
      });

      // Drag coefficient calculation
      const cd = 2.2; // Typical value for satellites
      const dragForce = 0.5 * atmospheric.totalDensity * velocity * velocity * cd * objectArea;
      const accelerationDrag = dragForce / objectMass;

      return {
        cd,
        area: objectArea,
        dragForce,
        accelerationDrag,
      };
    } catch (error) {
      console.error("Error calculating drag coefficients:", error);
      throw new Error("Failed to calculate drag coefficients: " + error.message);
    }
  }

  async calculateOrbitalDecay(
    objectMass: number,
    objectArea: number,
    initialAltitude: number,
    initialVelocity: number,
    timespan: number // days
  ): Promise<{
    finalAltitude: number;
    decayRate: number; // km/day
    timeToReentry: number; // days
    reentryPrediction: {
      probability: number;
      uncertaintyWindow: number; // hours
    };
  }> {
    try {
      const latestAtmospheric = await storage.getLatestAtmosphericData();
      const solarFlux = latestAtmospheric?.solarFlux || 150;
      const geoIndex = latestAtmospheric?.geomagneticIndex || 15;

      let currentAltitude = initialAltitude;
      let currentVelocity = initialVelocity;
      const timestep = 1; // day
      let totalDecay = 0;

      for (let day = 0; day < timespan; day += timestep) {
        const dragCoeffs = await this.calculateDragCoefficients(
          objectMass,
          objectArea,
          currentAltitude,
          currentVelocity,
          solarFlux,
          geoIndex
        );

        // Simplified orbital decay calculation
        const orbitalRadius = currentAltitude * 1000 + this.earthRadius;
        const orbitalVelocity = Math.sqrt(this.gravitationalParameter / orbitalRadius);
        
        // Energy loss due to drag
        const energyLoss = dragCoeffs.dragForce * orbitalVelocity * 86400; // J/day
        const specificEnergy = -this.gravitationalParameter / (2 * orbitalRadius);
        const energyChange = energyLoss / objectMass;
        
        // New orbital energy and altitude
        const newSpecificEnergy = specificEnergy - energyChange;
        const newOrbitalRadius = -this.gravitationalParameter / (2 * newSpecificEnergy);
        const newAltitude = (newOrbitalRadius - this.earthRadius) / 1000;
        
        const dailyDecay = currentAltitude - newAltitude;
        totalDecay += dailyDecay;
        
        currentAltitude = newAltitude;
        currentVelocity = Math.sqrt(this.gravitationalParameter / (currentAltitude * 1000 + this.earthRadius));
        
        // Check for reentry
        if (currentAltitude <= 80) {
          const timeToReentry = day;
          const uncertaintyFactor = Math.min(0.3, timeToReentry / 100); // 30% max uncertainty
          
          return {
            finalAltitude: currentAltitude,
            decayRate: totalDecay / day,
            timeToReentry,
            reentryPrediction: {
              probability: 0.95,
              uncertaintyWindow: timeToReentry * uncertaintyFactor * 24, // hours
            },
          };
        }
      }

      return {
        finalAltitude: currentAltitude,
        decayRate: totalDecay / timespan,
        timeToReentry: -1, // No reentry predicted
        reentryPrediction: {
          probability: 0,
          uncertaintyWindow: 0,
        },
      };
    } catch (error) {
      console.error("Error calculating orbital decay:", error);
      throw new Error("Failed to calculate orbital decay: " + error.message);
    }
  }
}

export const atmosphericModelService = new AtmosphericModelService();

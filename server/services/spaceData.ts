import axios from "axios";
import { storage } from "../storage";
import { InsertSpaceObject, InsertOrbitalPrediction } from "@shared/schema";

export interface TLEData {
  noradId: string;
  name: string;
  line1: string;
  line2: string;
}

export interface SpaceTrackData {
  NORAD_CAT_ID: string;
  OBJECT_NAME: string;
  OBJECT_TYPE: string;
  COUNTRY_CODE: string;
  LAUNCH_DATE: string;
  MEAN_MOTION: string;
  ECCENTRICITY: string;
  INCLINATION: string;
  RA_OF_ASC_NODE: string;
  ARG_OF_PERICENTER: string;
  MEAN_ANOMALY: string;
  EPHEMERIS_TYPE: string;
  CLASSIFICATION_TYPE: string;
  ELEMENT_SET_NO: string;
  REV_AT_EPOCH: string;
  BSTAR: string;
  MEAN_MOTION_DOT: string;
  MEAN_MOTION_DDOT: string;
}

export class SpaceDataService {
  private readonly celestrakBaseUrl = "https://celestrak.org";
  private readonly spaceTrackBaseUrl = "https://www.space-track.org";
  private readonly noaaBaseUrl = "https://services.swpc.noaa.gov";
  private readonly esaBaseUrl = "https://discos.esa.int";

  async fetchCelestrakTLE(catalog: string = "active"): Promise<TLEData[]> {
    try {
      const response = await axios.get(
        `${this.celestrakBaseUrl}/NORAD/elements/gp.php?GROUP=${catalog}&FORMAT=3le`,
        {
          timeout: 10000,
          headers: {
            'User-Agent': 'AEGIS-Space-Debris-Monitor/5.0'
          }
        }
      );

      await storage.updateApiSourceStatus("celestrak", "operational", Date.now() - performance.now());

      const lines = response.data.split('\n').filter((line: string) => line.trim());
      const tleData: TLEData[] = [];

      for (let i = 0; i < lines.length; i += 3) {
        if (i + 2 < lines.length) {
          const name = lines[i].trim();
          const line1 = lines[i + 1].trim();
          const line2 = lines[i + 2].trim();
          
          // Extract NORAD ID from line 1
          const noradId = line1.substring(2, 7).trim();
          
          tleData.push({
            noradId,
            name,
            line1,
            line2,
          });
        }
      }

      return tleData;
    } catch (error) {
      console.error("Error fetching CelesTrak data:", error);
      await storage.updateApiSourceStatus("celestrak", "critical");
      throw new Error("Failed to fetch CelesTrak TLE data: " + error.message);
    }
  }

  async fetchSpaceTrackData(): Promise<SpaceTrackData[]> {
    try {
      // Note: Space-Track.org requires authentication
      // This is a simplified example - in production, implement proper OAuth flow
      const response = await axios.get(
        `${this.spaceTrackBaseUrl}/basicspacedata/query/class/gp/EPOCH/%3Enow-1/format/json`,
        {
          timeout: 15000,
          headers: {
            'User-Agent': 'AEGIS-Space-Debris-Monitor/5.0'
          }
        }
      );

      await storage.updateApiSourceStatus("space-track", "operational", Date.now() - performance.now());
      return response.data;
    } catch (error) {
      console.error("Error fetching Space-Track data:", error);
      await storage.updateApiSourceStatus("space-track", "warning");
      return [];
    }
  }

  async fetchNOAASpaceWeather(): Promise<{
    solarFlux: number;
    geomagneticIndex: number;
    temperature: number;
  }> {
    try {
      const [solarResponse, geoResponse] = await Promise.all([
        axios.get(`${this.noaaBaseUrl}/products/solar-cycle/F107_obs.txt`, { timeout: 10000 }),
        axios.get(`${this.noaaBaseUrl}/products/geomag/ap.json`, { timeout: 10000 })
      ]);

      await storage.updateApiSourceStatus("noaa", "operational", Date.now() - performance.now());

      // Parse solar flux data (simplified)
      const solarLines = solarResponse.data.split('\n');
      const latestSolarLine = solarLines.find((line: string) => line.includes(new Date().getFullYear().toString()));
      const solarFlux = latestSolarLine ? parseFloat(latestSolarLine.split(/\s+/)[1]) : 150.0;

      // Parse geomagnetic data
      const geoData = Array.isArray(geoResponse.data) ? geoResponse.data[0] : geoResponse.data;
      const geomagneticIndex = geoData?.ap || 15;

      return {
        solarFlux,
        geomagneticIndex,
        temperature: 1000 + Math.random() * 500, // Simplified atmospheric temperature
      };
    } catch (error) {
      console.error("Error fetching NOAA data:", error);
      await storage.updateApiSourceStatus("noaa", "warning");
      
      // Return default values
      return {
        solarFlux: 150.0,
        geomagneticIndex: 15,
        temperature: 1200,
      };
    }
  }

  async fetchESADiscos(): Promise<any[]> {
    try {
      const response = await axios.get(
        `${this.esaBaseUrl}/api/objects`,
        {
          timeout: 20000,
          headers: {
            'Accept': 'application/json',
            'User-Agent': 'AEGIS-Space-Debris-Monitor/5.0'
          }
        }
      );

      await storage.updateApiSourceStatus("esa", "operational", Date.now() - performance.now());
      return response.data.data || [];
    } catch (error) {
      console.error("Error fetching ESA DISCOS data:", error);
      await storage.updateApiSourceStatus("esa", "warning");
      return [];
    }
  }

  parseTLEToSpaceObject(tle: TLEData): InsertSpaceObject {
    const line1 = tle.line1;
    const line2 = tle.line2;

    // Parse orbital elements from TLE
    const inclination = parseFloat(line2.substring(8, 16));
    const eccentricity = parseFloat("0." + line2.substring(26, 33));
    const meanMotion = parseFloat(line2.substring(52, 63));
    const period = 1440 / meanMotion; // Convert to minutes

    // Calculate approximate altitude (simplified)
    const semiMajorAxis = Math.pow(398600.4418 / Math.pow(meanMotion * 2 * Math.PI / 86400, 2), 1/3);
    const altitude = semiMajorAxis - 6371; // Earth radius

    // Determine object type based on name
    let type = "unknown";
    if (tle.name.includes("DEB") || tle.name.includes("DEBRIS")) {
      type = "debris";
    } else if (tle.name.includes("R/B") || tle.name.includes("ROCKET")) {
      type = "rocket_body";
    } else {
      type = "satellite";
    }

    // Assess risk level based on altitude and type
    let riskLevel: "low" | "medium" | "high" | "critical" = "low";
    if (type === "debris" && altitude < 800) {
      riskLevel = "high";
    } else if (altitude < 400) {
      riskLevel = "medium";
    }

    return {
      noradId: tle.noradId,
      name: tle.name,
      type,
      altitude,
      inclination,
      eccentricity,
      period,
      rcs: Math.random() * 10, // Placeholder - would need actual RCS data
      mass: type === "debris" ? Math.random() * 100 : Math.random() * 1000,
      size: Math.random() * 5,
      status: "active",
      riskLevel,
    };
  }

  async updateSpaceObjectDatabase(): Promise<{
    updated: number;
    created: number;
    errors: number;
  }> {
    const stats = { updated: 0, created: 0, errors: 0 };

    try {
      console.log("Fetching TLE data from CelesTrak...");
      const tleData = await this.fetchCelestrakTLE("active");
      
      console.log(`Processing ${tleData.length} TLE records...`);
      
      for (const tle of tleData.slice(0, 100)) { // Limit to first 100 for demo
        try {
          const spaceObject = this.parseTLEToSpaceObject(tle);
          const existing = await storage.getSpaceObjectById(tle.noradId);
          
          if (existing) {
            await storage.updateSpaceObject(tle.noradId, spaceObject);
            stats.updated++;
          } else {
            await storage.createSpaceObject(spaceObject);
            stats.created++;
          }
        } catch (error) {
          console.error(`Error processing object ${tle.noradId}:`, error);
          stats.errors++;
        }
      }

      console.log(`Database update complete: ${stats.created} created, ${stats.updated} updated, ${stats.errors} errors`);
      return stats;
    } catch (error) {
      console.error("Error updating space object database:", error);
      throw error;
    }
  }

  async updateAtmosphericData(): Promise<void> {
    try {
      const weatherData = await this.fetchNOAASpaceWeather();
      
      await storage.createAtmosphericData({
        solarFlux: weatherData.solarFlux,
        geomagneticIndex: weatherData.geomagneticIndex,
        densityAt400km: this.calculateAtmosphericDensity(400, weatherData.solarFlux, weatherData.geomagneticIndex),
        temperature: weatherData.temperature,
        modelVersion: "NRLMSISE-00",
        dataSource: "NOAA",
      });

      console.log("Atmospheric data updated successfully");
    } catch (error) {
      console.error("Error updating atmospheric data:", error);
      throw error;
    }
  }

  private calculateAtmosphericDensity(
    altitude: number,
    solarFlux: number,
    geoIndex: number
  ): number {
    // Simplified NRLMSISE-00 model approximation
    const baseAltitude = 400;
    const baseDensity = 2.1e-12; // kg/m³ at 400km
    const scaleHeight = 60; // km
    
    const altitudeFactor = Math.exp(-(altitude - baseAltitude) / scaleHeight);
    const solarFactor = 1 + (solarFlux - 150) / 300;
    const geoFactor = 1 + geoIndex / 100;
    
    return baseDensity * altitudeFactor * solarFactor * geoFactor;
  }
}

export const spaceDataService = new SpaceDataService();

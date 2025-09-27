import {
  spaceObjects,
  orbitalPredictions,
  conjunctionEvents,
  atmosphericData,
  economicAnalysis,
  systemAlerts,
  apiSources,
  type SpaceObject,
  type InsertSpaceObject,
  type OrbitalPrediction,
  type InsertOrbitalPrediction,
  type ConjunctionEvent,
  type InsertConjunctionEvent,
  type AtmosphericData,
  type InsertAtmosphericData,
  type EconomicAnalysis,
  type SystemAlert,
  type InsertSystemAlert,
  type ApiSource,
} from "@shared/schema";
import { db } from "./db";
import { eq, desc, gte, lte, and, or, count, avg } from "drizzle-orm";

export interface IStorage {
  // Space Objects
  getSpaceObjects(limit?: number): Promise<SpaceObject[]>;
  getSpaceObjectById(id: string): Promise<SpaceObject | undefined>;
  getSpaceObjectsByRiskLevel(riskLevel: string): Promise<SpaceObject[]>;
  createSpaceObject(object: InsertSpaceObject): Promise<SpaceObject>;
  updateSpaceObject(id: string, updates: Partial<SpaceObject>): Promise<SpaceObject | undefined>;
  
  // Orbital Predictions
  getOrbitalPredictions(objectId?: string, limit?: number): Promise<OrbitalPrediction[]>;
  createOrbitalPrediction(prediction: InsertOrbitalPrediction): Promise<OrbitalPrediction>;
  
  // Conjunction Events
  getActiveConjunctionEvents(): Promise<ConjunctionEvent[]>;
  getConjunctionEventsByRisk(riskLevel: string): Promise<ConjunctionEvent[]>;
  createConjunctionEvent(event: InsertConjunctionEvent): Promise<ConjunctionEvent>;
  updateConjunctionEvent(id: string, updates: Partial<ConjunctionEvent>): Promise<ConjunctionEvent | undefined>;
  
  // Atmospheric Data
  getLatestAtmosphericData(): Promise<AtmosphericData | undefined>;
  createAtmosphericData(data: InsertAtmosphericData): Promise<AtmosphericData>;
  
  // Economic Analysis
  getEconomicAnalysis(objectId?: string): Promise<EconomicAnalysis[]>;
  createEconomicAnalysis(analysis: Omit<EconomicAnalysis, 'id' | 'analysisDate'>): Promise<EconomicAnalysis>;
  
  // System Alerts
  getActiveAlerts(): Promise<SystemAlert[]>;
  createSystemAlert(alert: InsertSystemAlert): Promise<SystemAlert>;
  acknowledgeAlert(id: string, userId: string): Promise<SystemAlert | undefined>;
  
  // API Sources
  getApiSources(): Promise<ApiSource[]>;
  updateApiSourceStatus(id: string, status: string, lastResponse?: number): Promise<void>;
  
  // Dashboard Metrics
  getDashboardMetrics(): Promise<{
    totalObjects: number;
    highRiskObjects: number;
    activeConjunctions: number;
    criticalAlerts: number;
  }>;
}

export class DatabaseStorage implements IStorage {
  async getSpaceObjects(limit = 1000): Promise<SpaceObject[]> {
    return await db.select().from(spaceObjects).limit(limit).orderBy(desc(spaceObjects.lastUpdate));
  }

  async getSpaceObjectById(id: string): Promise<SpaceObject | undefined> {
    const [object] = await db.select().from(spaceObjects).where(eq(spaceObjects.id, id));
    return object;
  }

  async getSpaceObjectsByRiskLevel(riskLevel: string): Promise<SpaceObject[]> {
    return await db.select().from(spaceObjects).where(eq(spaceObjects.riskLevel, riskLevel));
  }

  async createSpaceObject(object: InsertSpaceObject): Promise<SpaceObject> {
    const [created] = await db.insert(spaceObjects).values({
      ...object,
      id: object.noradId, // Use NORAD ID as primary key
    }).returning();
    return created;
  }

  async updateSpaceObject(id: string, updates: Partial<SpaceObject>): Promise<SpaceObject | undefined> {
    const [updated] = await db
      .update(spaceObjects)
      .set({ ...updates, lastUpdate: new Date() })
      .where(eq(spaceObjects.id, id))
      .returning();
    return updated;
  }

  async getOrbitalPredictions(objectId?: string, limit = 100): Promise<OrbitalPrediction[]> {
    const query = db.select().from(orbitalPredictions).limit(limit).orderBy(desc(orbitalPredictions.predictedTime));
    
    if (objectId) {
      return await query.where(eq(orbitalPredictions.objectId, objectId));
    }
    
    return await query;
  }

  async createOrbitalPrediction(prediction: InsertOrbitalPrediction): Promise<OrbitalPrediction> {
    const [created] = await db.insert(orbitalPredictions).values(prediction).returning();
    return created;
  }

  async getActiveConjunctionEvents(): Promise<ConjunctionEvent[]> {
    return await db
      .select()
      .from(conjunctionEvents)
      .where(
        and(
          eq(conjunctionEvents.status, "active"),
          gte(conjunctionEvents.predictedTime, new Date())
        )
      )
      .orderBy(conjunctionEvents.predictedTime);
  }

  async getConjunctionEventsByRisk(riskLevel: string): Promise<ConjunctionEvent[]> {
    return await db
      .select()
      .from(conjunctionEvents)
      .where(
        and(
          eq(conjunctionEvents.riskLevel, riskLevel),
          eq(conjunctionEvents.status, "active")
        )
      )
      .orderBy(conjunctionEvents.predictedTime);
  }

  async createConjunctionEvent(event: InsertConjunctionEvent): Promise<ConjunctionEvent> {
    const [created] = await db.insert(conjunctionEvents).values(event).returning();
    return created;
  }

  async updateConjunctionEvent(id: string, updates: Partial<ConjunctionEvent>): Promise<ConjunctionEvent | undefined> {
    const [updated] = await db
      .update(conjunctionEvents)
      .set(updates)
      .where(eq(conjunctionEvents.id, id))
      .returning();
    return updated;
  }

  async getLatestAtmosphericData(): Promise<AtmosphericData | undefined> {
    const [data] = await db
      .select()
      .from(atmosphericData)
      .orderBy(desc(atmosphericData.timestamp))
      .limit(1);
    return data;
  }

  async createAtmosphericData(data: InsertAtmosphericData): Promise<AtmosphericData> {
    const [created] = await db.insert(atmosphericData).values(data).returning();
    return created;
  }

  async getEconomicAnalysis(objectId?: string): Promise<EconomicAnalysis[]> {
    const query = db.select().from(economicAnalysis).orderBy(desc(economicAnalysis.analysisDate));
    
    if (objectId) {
      return await query.where(eq(economicAnalysis.objectId, objectId));
    }
    
    return await query.limit(100);
  }

  async createEconomicAnalysis(analysis: Omit<EconomicAnalysis, 'id' | 'analysisDate'>): Promise<EconomicAnalysis> {
    const [created] = await db.insert(economicAnalysis).values({
      ...analysis,
      analysisDate: new Date(),
    }).returning();
    return created;
  }

  async getActiveAlerts(): Promise<SystemAlert[]> {
    return await db
      .select()
      .from(systemAlerts)
      .where(
        and(
          eq(systemAlerts.acknowledged, false),
          or(
            eq(systemAlerts.expiresAt, null),
            gte(systemAlerts.expiresAt, new Date())
          )
        )
      )
      .orderBy(desc(systemAlerts.createdAt));
  }

  async createSystemAlert(alert: InsertSystemAlert): Promise<SystemAlert> {
    const [created] = await db.insert(systemAlerts).values(alert).returning();
    return created;
  }

  async acknowledgeAlert(id: string, userId: string): Promise<SystemAlert | undefined> {
    const [updated] = await db
      .update(systemAlerts)
      .set({
        acknowledged: true,
        acknowledgedBy: userId,
        acknowledgedAt: new Date(),
      })
      .where(eq(systemAlerts.id, id))
      .returning();
    return updated;
  }

  async getApiSources(): Promise<ApiSource[]> {
    return await db.select().from(apiSources).orderBy(apiSources.name);
  }

  async updateApiSourceStatus(id: string, status: string, lastResponse?: number): Promise<void> {
    await db
      .update(apiSources)
      .set({
        status,
        lastResponse,
        lastUpdate: new Date(),
      })
      .where(eq(apiSources.id, id));
  }

  async getDashboardMetrics(): Promise<{
    totalObjects: number;
    highRiskObjects: number;
    activeConjunctions: number;
    criticalAlerts: number;
  }> {
    const [totalObjectsResult] = await db.select({ count: count() }).from(spaceObjects);
    const [highRiskResult] = await db
      .select({ count: count() })
      .from(spaceObjects)
      .where(or(eq(spaceObjects.riskLevel, "high"), eq(spaceObjects.riskLevel, "critical")));
    
    const [activeConjunctionsResult] = await db
      .select({ count: count() })
      .from(conjunctionEvents)
      .where(eq(conjunctionEvents.status, "active"));
    
    const [criticalAlertsResult] = await db
      .select({ count: count() })
      .from(systemAlerts)
      .where(
        and(
          eq(systemAlerts.severity, "critical"),
          eq(systemAlerts.acknowledged, false)
        )
      );

    return {
      totalObjects: totalObjectsResult.count,
      highRiskObjects: highRiskResult.count,
      activeConjunctions: activeConjunctionsResult.count,
      criticalAlerts: criticalAlertsResult.count,
    };
  }
}

export const storage = new DatabaseStorage();

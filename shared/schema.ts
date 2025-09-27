import { sql } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, real, integer, boolean, jsonb, index } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";
import { relations } from "drizzle-orm";

// Space objects table
export const spaceObjects = pgTable("space_objects", {
  id: varchar("id").primaryKey(),
  noradId: varchar("norad_id").notNull().unique(),
  name: text("name").notNull(),
  type: varchar("type").notNull(), // satellite, debris, rocket_body
  country: varchar("country"),
  launchDate: timestamp("launch_date"),
  altitude: real("altitude"), // km
  inclination: real("inclination"), // degrees
  eccentricity: real("eccentricity"),
  period: real("period"), // minutes
  rcs: real("rcs"), // radar cross section
  mass: real("mass"), // kg
  size: real("size"), // meters
  status: varchar("status").notNull().default("active"), // active, inactive, decayed
  riskLevel: varchar("risk_level").notNull().default("low"), // low, medium, high, critical
  lastUpdate: timestamp("last_update").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

// Orbital predictions table
export const orbitalPredictions = pgTable("orbital_predictions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  objectId: varchar("object_id").notNull().references(() => spaceObjects.id),
  predictedTime: timestamp("predicted_time").notNull(),
  altitude: real("altitude").notNull(),
  latitude: real("latitude").notNull(),
  longitude: real("longitude").notNull(),
  velocity: real("velocity").notNull(),
  confidence: real("confidence").notNull(), // 0-1
  aiModel: varchar("ai_model"), // openai, claude
  createdAt: timestamp("created_at").defaultNow(),
});

// Conjunction events table
export const conjunctionEvents = pgTable("conjunction_events", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  primaryObjectId: varchar("primary_object_id").notNull().references(() => spaceObjects.id),
  secondaryObjectId: varchar("secondary_object_id").notNull().references(() => spaceObjects.id),
  predictedTime: timestamp("predicted_time").notNull(),
  missDistance: real("miss_distance").notNull(), // km
  collisionProbability: real("collision_probability").notNull(), // 0-1
  riskLevel: varchar("risk_level").notNull(), // low, medium, high, critical
  status: varchar("status").notNull().default("active"), // active, resolved, false_alarm
  alertSent: boolean("alert_sent").notNull().default(false),
  createdAt: timestamp("created_at").defaultNow(),
});

// Atmospheric data table
export const atmosphericData = pgTable("atmospheric_data", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  timestamp: timestamp("timestamp").notNull().defaultNow(),
  solarFlux: real("solar_flux"), // F10.7 solar flux
  geomagneticIndex: real("geomagnetic_index"), // Ap index
  densityAt400km: real("density_at_400km"), // kg/m³
  temperature: real("temperature"), // K
  modelVersion: varchar("model_version"), // NRLMSISE-00
  dataSource: varchar("data_source"), // NOAA, ESA
});

// Economic analysis table
export const economicAnalysis = pgTable("economic_analysis", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  objectId: varchar("object_id").references(() => spaceObjects.id),
  removalCost: real("removal_cost"), // USD
  insuranceValue: real("insurance_value"), // USD
  potentialDamage: real("potential_damage"), // USD
  sustainabilityScore: real("sustainability_score"), // 0-100
  analysisDate: timestamp("analysis_date").defaultNow(),
  aiInsights: jsonb("ai_insights"),
});

// System alerts table
export const systemAlerts = pgTable("system_alerts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  type: varchar("type").notNull(), // collision, debris, system
  severity: varchar("severity").notNull(), // info, warning, critical
  title: text("title").notNull(),
  message: text("message").notNull(),
  objectIds: text("object_ids").array(), // related object IDs
  acknowledged: boolean("acknowledged").notNull().default(false),
  acknowledgedBy: varchar("acknowledged_by"),
  acknowledgedAt: timestamp("acknowledged_at"),
  createdAt: timestamp("created_at").defaultNow(),
  expiresAt: timestamp("expires_at"),
});

// API sources status table
export const apiSources = pgTable("api_sources", {
  id: varchar("id").primaryKey(),
  name: varchar("name").notNull(),
  url: text("url").notNull(),
  status: varchar("status").notNull().default("operational"), // operational, warning, critical, offline
  lastResponse: integer("last_response"), // ms
  lastUpdate: timestamp("last_update").defaultNow(),
  successRate: real("success_rate").notNull().default(1.0), // 0-1
  errorCount: integer("error_count").notNull().default(0),
});

// Relations
export const spaceObjectsRelations = relations(spaceObjects, ({ many }) => ({
  predictions: many(orbitalPredictions),
  primaryConjunctions: many(conjunctionEvents, { relationName: "primary" }),
  secondaryConjunctions: many(conjunctionEvents, { relationName: "secondary" }),
  economicAnalysis: many(economicAnalysis),
}));

export const orbitalPredictionsRelations = relations(orbitalPredictions, ({ one }) => ({
  object: one(spaceObjects, {
    fields: [orbitalPredictions.objectId],
    references: [spaceObjects.id],
  }),
}));

export const conjunctionEventsRelations = relations(conjunctionEvents, ({ one }) => ({
  primaryObject: one(spaceObjects, {
    fields: [conjunctionEvents.primaryObjectId],
    references: [spaceObjects.id],
    relationName: "primary",
  }),
  secondaryObject: one(spaceObjects, {
    fields: [conjunctionEvents.secondaryObjectId],
    references: [spaceObjects.id],
    relationName: "secondary",
  }),
}));

export const economicAnalysisRelations = relations(economicAnalysis, ({ one }) => ({
  object: one(spaceObjects, {
    fields: [economicAnalysis.objectId],
    references: [spaceObjects.id],
  }),
}));

// Insert schemas
export const insertSpaceObjectSchema = createInsertSchema(spaceObjects).omit({
  id: true,
  lastUpdate: true,
  createdAt: true,
});

export const insertOrbitalPredictionSchema = createInsertSchema(orbitalPredictions).omit({
  id: true,
  createdAt: true,
});

export const insertConjunctionEventSchema = createInsertSchema(conjunctionEvents).omit({
  id: true,
  createdAt: true,
});

export const insertAtmosphericDataSchema = createInsertSchema(atmosphericData).omit({
  id: true,
  timestamp: true,
});

export const insertSystemAlertSchema = createInsertSchema(systemAlerts).omit({
  id: true,
  createdAt: true,
});

// Types
export type SpaceObject = typeof spaceObjects.$inferSelect;
export type InsertSpaceObject = z.infer<typeof insertSpaceObjectSchema>;
export type OrbitalPrediction = typeof orbitalPredictions.$inferSelect;
export type InsertOrbitalPrediction = z.infer<typeof insertOrbitalPredictionSchema>;
export type ConjunctionEvent = typeof conjunctionEvents.$inferSelect;
export type InsertConjunctionEvent = z.infer<typeof insertConjunctionEventSchema>;
export type AtmosphericData = typeof atmosphericData.$inferSelect;
export type InsertAtmosphericData = z.infer<typeof insertAtmosphericDataSchema>;
export type EconomicAnalysis = typeof economicAnalysis.$inferSelect;
export type SystemAlert = typeof systemAlerts.$inferSelect;
export type InsertSystemAlert = z.infer<typeof insertSystemAlertSchema>;
export type ApiSource = typeof apiSources.$inferSelect;

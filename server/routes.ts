import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { storage } from "./storage";
import { spaceDataService } from "./services/spaceData";
import { aiAnalysisService } from "./services/aiAnalysis";
import { sgp4Propagator } from "./services/enhancedSGP4Propagator";
import { nrlmsise00Model } from "./services/enhancedNRLMSISE00Model";
import { monteCarloSimulations } from "./services/enhancedMonteCarloSimulations";
import { calculationValidation } from "./services/calculationValidation";
import { atmosphericModelService } from "./services/atmosphericModel";
import { economicAnalysisService } from "./services/economicAnalysis";
import { cacheService } from "./services/cacheService";
import { performanceMonitor } from "./services/performanceMonitor";
import { healthMonitor } from "./services/healthMonitor";
import { realTimeDataManager } from "./services/realTimeDataManager";
import { kesslerSimulation } from "./services/kesslerSimulation";
import { plottingService } from "./services/plottingService";
import { insertSystemAlertSchema } from "@shared/schema";

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);

  // WebSocket server for real-time updates
  const wss = new WebSocketServer({ server: httpServer, path: '/ws' });
  
  // Store connected clients
  const clients = new Set<WebSocket>();
  
  wss.on('connection', (ws) => {
    clients.add(ws);
    console.log('WebSocket client connected. Total clients:', clients.size);
    
    // Send initial system status
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'system_status',
        data: {
          status: 'connected',
          timestamp: new Date().toISOString(),
        }
      }));
    }
    
    ws.on('close', () => {
      clients.delete(ws);
      console.log('WebSocket client disconnected. Total clients:', clients.size);
    });
    
    ws.on('error', (error) => {
      console.error('WebSocket error:', error);
      clients.delete(ws);
    });
  });

  // Broadcast to all connected clients
  const broadcast = (type: string, data: any) => {
    const message = JSON.stringify({ type, data, timestamp: new Date().toISOString() });
    clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    });
  };

  // Dashboard metrics endpoint
  app.get('/api/dashboard/metrics', async (req, res) => {
    try {
      const metrics = await storage.getDashboardMetrics();
      res.json(metrics);
    } catch (error) {
      console.error('Error fetching dashboard metrics:', error);
      res.status(500).json({ message: 'Failed to fetch dashboard metrics' });
    }
  });

  // Space objects endpoints
  app.get('/api/space-objects', async (req, res) => {
    try {
      const limit = parseInt(req.query.limit as string) || 1000;
      const riskLevel = req.query.risk_level as string;
      
      let objects;
      if (riskLevel) {
        objects = await storage.getSpaceObjectsByRiskLevel(riskLevel);
      } else {
        objects = await storage.getSpaceObjects(limit);
      }
      
      res.json(objects);
    } catch (error) {
      console.error('Error fetching space objects:', error);
      res.status(500).json({ message: 'Failed to fetch space objects' });
    }
  });

  app.get('/api/space-objects/:id', async (req, res) => {
    try {
      const object = await storage.getSpaceObjectById(req.params.id);
      if (!object) {
        return res.status(404).json({ message: 'Space object not found' });
      }
      res.json(object);
    } catch (error) {
      console.error('Error fetching space object:', error);
      res.status(500).json({ message: 'Failed to fetch space object' });
    }
  });

  // Orbital predictions endpoints
  app.get('/api/orbital-predictions', async (req, res) => {
    try {
      const objectId = req.query.object_id as string;
      const limit = parseInt(req.query.limit as string) || 100;
      
      const predictions = await storage.getOrbitalPredictions(objectId, limit);
      res.json(predictions);
    } catch (error) {
      console.error('Error fetching orbital predictions:', error);
      res.status(500).json({ message: 'Failed to fetch orbital predictions' });
    }
  });

  // Conjunction events endpoints
  app.get('/api/conjunction-events', async (req, res) => {
    try {
      const riskLevel = req.query.risk_level as string;
      
      let events;
      if (riskLevel) {
        events = await storage.getConjunctionEventsByRisk(riskLevel);
      } else {
        events = await storage.getActiveConjunctionEvents();
      }
      
      res.json(events);
    } catch (error) {
      console.error('Error fetching conjunction events:', error);
      res.status(500).json({ message: 'Failed to fetch conjunction events' });
    }
  });

  // Atmospheric data endpoints
  app.get('/api/atmospheric-data', async (req, res) => {
    try {
      const data = await storage.getLatestAtmosphericData();
      res.json(data);
    } catch (error) {
      console.error('Error fetching atmospheric data:', error);
      res.status(500).json({ message: 'Failed to fetch atmospheric data' });
    }
  });

  // AI Analysis endpoints
  app.post('/api/ai/risk-assessment/:objectId', async (req, res) => {
    try {
      const object = await storage.getSpaceObjectById(req.params.objectId);
      if (!object) {
        return res.status(404).json({ message: 'Space object not found' });
      }
      
      const assessment = await aiAnalysisService.assessSpaceObjectRisk(object);
      res.json(assessment);
    } catch (error) {
      console.error('Error performing AI risk assessment:', error);
      res.status(500).json({ message: 'Failed to perform AI risk assessment' });
    }
  });

  app.post('/api/ai/collision-prediction', async (req, res) => {
    try {
      const { primaryObjectId, secondaryObjectId, timeHorizon } = req.body;
      
      const primaryObject = await storage.getSpaceObjectById(primaryObjectId);
      const secondaryObject = await storage.getSpaceObjectById(secondaryObjectId);
      
      if (!primaryObject || !secondaryObject) {
        return res.status(404).json({ message: 'One or both space objects not found' });
      }
      
      const prediction = await aiAnalysisService.predictCollisionProbability(
        primaryObject,
        secondaryObject,
        timeHorizon || 24
      );
      
      res.json(prediction);
    } catch (error) {
      console.error('Error performing collision prediction:', error);
      res.status(500).json({ message: 'Failed to perform collision prediction' });
    }
  });

  app.post('/api/ai/kessler-analysis', async (req, res) => {
    try {
      const { region } = req.body;
      const objects = await storage.getSpaceObjects();
      
      const analysis = await aiAnalysisService.analyzeKesslerSyndrome(objects, region || 'LEO');
      res.json(analysis);
    } catch (error) {
      console.error('Error performing Kessler analysis:', error);
      res.status(500).json({ message: 'Failed to perform Kessler analysis' });
    }
  });

  // New AI endpoints for enhanced features
  app.post('/api/ai/space-traffic-management', async (req, res) => {
    try {
      const { region } = req.body;
      const objects = await storage.getSpaceObjects();
      
      const analysis = await aiAnalysisService.analyzeSpaceTrafficManagement(objects, region || 'LEO');
      res.json(analysis);
    } catch (error) {
      console.error('Error performing space traffic management analysis:', error);
      res.status(500).json({ message: 'Failed to perform space traffic management analysis' });
    }
  });

  app.post('/api/ai/sustainability-assessment', async (req, res) => {
    try {
      const { timeHorizon } = req.body;
      const objects = await storage.getSpaceObjects();
      
      const assessment = await aiAnalysisService.assessSpaceSustainability(objects, timeHorizon || 50);
      res.json(assessment);
    } catch (error) {
      console.error('Error performing sustainability assessment:', error);
      res.status(500).json({ message: 'Failed to perform sustainability assessment' });
    }
  });

  app.post('/api/ai/anomaly-detection', async (req, res) => {
    try {
      const objects = await storage.getSpaceObjects();
      const historicalData = []; // TODO: Implement historical data retrieval
      
      const anomalies = await aiAnalysisService.detectOrbitalAnomalies(objects, historicalData);
      res.json(anomalies);
    } catch (error) {
      console.error('Error performing anomaly detection:', error);
      res.status(500).json({ message: 'Failed to perform anomaly detection' });
    }
  });

  app.post('/api/ai/debris-evolution', async (req, res) => {
    try {
      const { timeHorizon } = req.body;
      const objects = await storage.getSpaceObjects();
      
      const evolution = await aiAnalysisService.generateDebrisEvolutionPrediction(objects, timeHorizon || 10);
      res.json(evolution);
    } catch (error) {
      console.error('Error generating debris evolution prediction:', error);
      res.status(500).json({ message: 'Failed to generate debris evolution prediction' });
    }
  });

  // AI Chat Assistant endpoint
  app.post('/api/ai/chat', async (req, res) => {
    try {
      const { message, context } = req.body;
      
      if (!message) {
        return res.status(400).json({ message: 'Message is required' });
      }

      const response = await aiAnalysisService.processChatMessage(message, context);
      res.json(response);
    } catch (error) {
      console.error('Error processing chat message:', error);
      res.status(500).json({ message: 'Failed to process chat message' });
    }
  });

  // SGP4 Propagation endpoint
  app.post('/api/sgp4/propagate', async (req, res) => {
    try {
      const { tle, timeOffset, options } = req.body;
      
      if (!tle || !timeOffset) {
        return res.status(400).json({ message: 'TLE and timeOffset are required' });
      }

      const result = sgp4Propagator.propagate(tle, timeOffset, options);
      res.json(result);
    } catch (error) {
      console.error('Error in SGP4 propagation:', error);
      res.status(500).json({ message: 'Failed to propagate orbit' });
    }
  });

  // Atmospheric density calculation endpoint
  app.post('/api/atmospheric/density', async (req, res) => {
    try {
      const { conditions } = req.body;
      
      if (!conditions) {
        return res.status(400).json({ message: 'Atmospheric conditions are required' });
      }

      const result = nrlmsise00Model.calculateAtmosphericDensity(conditions);
      res.json(result);
    } catch (error) {
      console.error('Error in atmospheric density calculation:', error);
      res.status(500).json({ message: 'Failed to calculate atmospheric density' });
    }
  });

  // Monte Carlo simulation endpoint
  app.post('/api/monte-carlo/simulate', async (req, res) => {
    try {
      const { parameters } = req.body;
      
      if (!parameters) {
        return res.status(400).json({ message: 'Simulation parameters are required' });
      }

      const result = await monteCarloSimulations.runSimulation(parameters);
      res.json(result);
    } catch (error) {
      console.error('Error in Monte Carlo simulation:', error);
      res.status(500).json({ message: 'Failed to run simulation' });
    }
  });

  // Calculation validation endpoint
  app.post('/api/validation/validate', async (req, res) => {
    try {
      const { data, categories } = req.body;
      
      if (!data) {
        return res.status(400).json({ message: 'Data to validate is required' });
      }

      const result = calculationValidation.validate(data, categories);
      res.json(result);
    } catch (error) {
      console.error('Error in calculation validation:', error);
      res.status(500).json({ message: 'Failed to validate calculations' });
    }
  });

  // Economic analysis endpoints
  app.get('/api/economic/metrics', async (req, res) => {
    try {
      const objects = await storage.getSpaceObjects();
      const metrics = await economicAnalysisService.calculateComprehensiveEconomics(objects);
      res.json(metrics);
    } catch (error) {
      console.error('Error calculating economic metrics:', error);
      res.status(500).json({ message: 'Failed to calculate economic metrics' });
    }
  });

  app.post('/api/economic/cost-benefit', async (req, res) => {
    try {
      const { scenario, mitigationStrategy } = req.body;
      const analysis = await economicAnalysisService.performCostBenefitAnalysis(scenario, mitigationStrategy);
      res.json(analysis);
    } catch (error) {
      console.error('Error performing cost-benefit analysis:', error);
      res.status(500).json({ message: 'Failed to perform cost-benefit analysis' });
    }
  });

  app.get('/api/economic/regional-impact', async (req, res) => {
    try {
      const impact = await economicAnalysisService.analyzeRegionalEconomicImpact();
      res.json(impact);
    } catch (error) {
      console.error('Error analyzing regional economic impact:', error);
      res.status(500).json({ message: 'Failed to analyze regional economic impact' });
    }
  });

  // Atmospheric modeling endpoints
  app.post('/api/atmospheric/conditions', async (req, res) => {
    try {
      const params = req.body;
      const conditions = await atmosphericModelService.calculateNRLMSISE00(params);
      res.json(conditions);
    } catch (error) {
      console.error('Error calculating atmospheric conditions:', error);
      res.status(500).json({ message: 'Failed to calculate atmospheric conditions' });
    }
  });

  app.post('/api/atmospheric/drag-analysis', async (req, res) => {
    try {
      const { objectMass, objectArea, altitude, velocity, solarFlux, geoIndex } = req.body;
      
      const dragCoeffs = await atmosphericModelService.calculateDragCoefficients(
        objectMass, objectArea, altitude, velocity, solarFlux, geoIndex
      );
      
      res.json(dragCoeffs);
    } catch (error) {
      console.error('Error calculating drag coefficients:', error);
      res.status(500).json({ message: 'Failed to calculate drag coefficients' });
    }
  });

  app.post('/api/atmospheric/orbital-decay', async (req, res) => {
    try {
      const { objectMass, objectArea, initialAltitude, initialVelocity, timespan } = req.body;
      
      const decay = await atmosphericModelService.calculateOrbitalDecay(
        objectMass, objectArea, initialAltitude, initialVelocity, timespan
      );
      
      res.json(decay);
    } catch (error) {
      console.error('Error calculating orbital decay:', error);
      res.status(500).json({ message: 'Failed to calculate orbital decay' });
    }
  });

  // System alerts endpoints
  app.get('/api/alerts', async (req, res) => {
    try {
      const alerts = await storage.getActiveAlerts();
      res.json(alerts);
    } catch (error) {
      console.error('Error fetching alerts:', error);
      res.status(500).json({ message: 'Failed to fetch alerts' });
    }
  });

  app.post('/api/alerts', async (req, res) => {
    try {
      const alertData = insertSystemAlertSchema.parse(req.body);
      const alert = await storage.createSystemAlert(alertData);
      
      // Broadcast alert to all connected clients
      broadcast('new_alert', alert);
      
      res.status(201).json(alert);
    } catch (error) {
      console.error('Error creating alert:', error);
      res.status(500).json({ message: 'Failed to create alert' });
    }
  });

  app.patch('/api/alerts/:id/acknowledge', async (req, res) => {
    try {
      const { userId } = req.body;
      const alert = await storage.acknowledgeAlert(req.params.id, userId || 'system');
      
      if (!alert) {
        return res.status(404).json({ message: 'Alert not found' });
      }
      
      // Broadcast alert acknowledgment
      broadcast('alert_acknowledged', alert);
      
      res.json(alert);
    } catch (error) {
      console.error('Error acknowledging alert:', error);
      res.status(500).json({ message: 'Failed to acknowledge alert' });
    }
  });

  // API sources status
  app.get('/api/data-sources', async (req, res) => {
    try {
      const sources = await storage.getApiSources();
      res.json(sources);
    } catch (error) {
      console.error('Error fetching API sources:', error);
      res.status(500).json({ message: 'Failed to fetch API sources' });
    }
  });

  // Data update endpoints
  app.post('/api/update/space-objects', async (req, res) => {
    try {
      console.log('Starting space objects database update...');
      const stats = await spaceDataService.updateSpaceObjectDatabase();
      
      // Broadcast update completion
      broadcast('database_updated', stats);
      
      res.json({
        message: 'Space objects database updated successfully',
        stats
      });
    } catch (error) {
      console.error('Error updating space objects database:', error);
      res.status(500).json({ message: 'Failed to update space objects database' });
    }
  });

  app.post('/api/update/atmospheric-data', async (req, res) => {
    try {
      console.log('Updating atmospheric data...');
      await spaceDataService.updateAtmosphericData();
      
      const latestData = await storage.getLatestAtmosphericData();
      broadcast('atmospheric_updated', latestData);
      
      res.json({
        message: 'Atmospheric data updated successfully',
        data: latestData
      });
    } catch (error) {
      console.error('Error updating atmospheric data:', error);
      res.status(500).json({ message: 'Failed to update atmospheric data' });
    }
  });

  // Periodic data updates
  setInterval(async () => {
    try {
      // Update metrics every 30 seconds
      const metrics = await storage.getDashboardMetrics();
      broadcast('metrics_update', metrics);
    } catch (error) {
      console.error('Error in periodic metrics update:', error);
    }
  }, 30000);

  setInterval(async () => {
    try {
      // Update atmospheric data every 5 minutes
      await spaceDataService.updateAtmosphericData();
      const latestData = await storage.getLatestAtmosphericData();
      broadcast('atmospheric_updated', latestData);
    } catch (error) {
      console.error('Error in periodic atmospheric update:', error);
    }
  }, 300000);

  // Health and Performance endpoints
  app.get('/api/health', async (req, res) => {
    try {
      const health = await healthMonitor.runHealthChecks();
      res.json(health);
    } catch (error) {
      console.error('Error getting health status:', error);
      res.status(500).json({ message: 'Failed to get health status' });
    }
  });

  app.get('/api/health/:check', async (req, res) => {
    try {
      const check = await healthMonitor.getCheckHealth(req.params.check);
      if (!check) {
        return res.status(404).json({ message: 'Health check not found' });
      }
      res.json(check);
    } catch (error) {
      console.error('Error getting health check:', error);
      res.status(500).json({ message: 'Failed to get health check' });
    }
  });

  app.get('/api/metrics', async (req, res) => {
    try {
      const metrics = healthMonitor.getSystemMetrics();
      res.json(metrics);
    } catch (error) {
      console.error('Error getting system metrics:', error);
      res.status(500).json({ message: 'Failed to get system metrics' });
    }
  });

  app.get('/api/performance', async (req, res) => {
    try {
      const stats = performanceMonitor.getStats();
      const summary = performanceMonitor.getSummary();
      const slowOperations = performanceMonitor.getSlowOperations();
      
      res.json({
        stats,
        summary,
        slowOperations,
        timestamp: new Date(),
      });
    } catch (error) {
      console.error('Error getting performance data:', error);
      res.status(500).json({ message: 'Failed to get performance data' });
    }
  });

  app.get('/api/cache/stats', async (req, res) => {
    try {
      const stats = cacheService.getCacheStats();
      res.json(stats);
    } catch (error) {
      console.error('Error getting cache stats:', error);
      res.status(500).json({ message: 'Failed to get cache stats' });
    }
  });

  app.post('/api/cache/clear', async (req, res) => {
    try {
      const { type } = req.body;
      
      switch (type) {
        case 'space-objects':
          await cacheService.invalidateSpaceObjects();
          break;
        case 'ai-analysis':
          await cacheService.invalidateAIAnalysis();
          break;
        case 'atmospheric-data':
          await cacheService.invalidateAtmosphericData();
          break;
        case 'economic-data':
          await cacheService.invalidateEconomicData();
          break;
        case 'all':
          await cacheService.invalidateSpaceObjects();
          await cacheService.invalidateAIAnalysis();
          await cacheService.invalidateAtmosphericData();
          await cacheService.invalidateEconomicData();
          break;
        default:
          return res.status(400).json({ message: 'Invalid cache type' });
      }
      
      res.json({ message: `Cache cleared for type: ${type}` });
    } catch (error) {
      console.error('Error clearing cache:', error);
      res.status(500).json({ message: 'Failed to clear cache' });
    }
  });

  // Real-time data manager endpoints
  app.get('/api/realtime/status', async (req, res) => {
    try {
      const status = realTimeDataManager.getStatus();
      res.json(status);
    } catch (error) {
      console.error('Error getting real-time status:', error);
      res.status(500).json({ message: 'Failed to get real-time status' });
    }
  });

  app.get('/api/realtime/current', async (req, res) => {
    try {
      const snapshot = realTimeDataManager.getCurrentSnapshot();
      res.json(snapshot);
    } catch (error) {
      console.error('Error getting current snapshot:', error);
      res.status(500).json({ message: 'Failed to get current snapshot' });
    }
  });

  app.get('/api/realtime/history', async (req, res) => {
    try {
      const limit = parseInt(req.query.limit as string) || 100;
      const history = realTimeDataManager.getDataHistory(limit);
      res.json(history);
    } catch (error) {
      console.error('Error getting data history:', error);
      res.status(500).json({ message: 'Failed to get data history' });
    }
  });

  app.get('/api/realtime/trends', async (req, res) => {
    try {
      const timeWindow = parseInt(req.query.timeWindow as string) || 3600000; // 1 hour
      const trends = realTimeDataManager.getMetricsTrend(timeWindow);
      res.json(trends);
    } catch (error) {
      console.error('Error getting metrics trends:', error);
      res.status(500).json({ message: 'Failed to get metrics trends' });
    }
  });

  // Kessler simulation endpoints
  app.post('/api/simulation/kessler/initialize', async (req, res) => {
    try {
      const { objects, config } = req.body;
      kesslerSimulation.initialize(objects, config);
      res.json({ message: 'Kessler simulation initialized' });
    } catch (error) {
      console.error('Error initializing Kessler simulation:', error);
      res.status(500).json({ message: 'Failed to initialize Kessler simulation' });
    }
  });

  app.post('/api/simulation/kessler/start', async (req, res) => {
    try {
      await kesslerSimulation.start();
      res.json({ message: 'Kessler simulation started' });
    } catch (error) {
      console.error('Error starting Kessler simulation:', error);
      res.status(500).json({ message: 'Failed to start Kessler simulation' });
    }
  });

  app.get('/api/simulation/kessler/status', async (req, res) => {
    try {
      const state = kesslerSimulation.getState();
      res.json(state);
    } catch (error) {
      console.error('Error getting Kessler simulation status:', error);
      res.status(500).json({ message: 'Failed to get Kessler simulation status' });
    }
  });

  app.get('/api/simulation/kessler/results', async (req, res) => {
    try {
      const results = kesslerSimulation.getResults();
      res.json(results);
    } catch (error) {
      console.error('Error getting Kessler simulation results:', error);
      res.status(500).json({ message: 'Failed to get Kessler simulation results' });
    }
  });

  app.post('/api/simulation/kessler/stop', async (req, res) => {
    try {
      kesslerSimulation.stop();
      res.json({ message: 'Kessler simulation stopped' });
    } catch (error) {
      console.error('Error stopping Kessler simulation:', error);
      res.status(500).json({ message: 'Failed to stop Kessler simulation' });
    }
  });

  app.post('/api/simulation/kessler/reset', async (req, res) => {
    try {
      kesslerSimulation.reset();
      res.json({ message: 'Kessler simulation reset' });
    } catch (error) {
      console.error('Error resetting Kessler simulation:', error);
      res.status(500).json({ message: 'Failed to reset Kessler simulation' });
    }
  });

  // Plotting service endpoints
  app.post('/api/plots/orbital-distribution', async (req, res) => {
    try {
      const { objects } = req.body;
      const plotPath = await plottingService.createOrbitalDistributionPlot(objects);
      res.json({ plotPath, message: 'Orbital distribution plot created' });
    } catch (error) {
      console.error('Error creating orbital distribution plot:', error);
      res.status(500).json({ message: 'Failed to create orbital distribution plot' });
    }
  });

  app.post('/api/plots/risk-distribution', async (req, res) => {
    try {
      const { objects } = req.body;
      const plotPath = await plottingService.createRiskDistributionPlot(objects);
      res.json({ plotPath, message: 'Risk distribution plot created' });
    } catch (error) {
      console.error('Error creating risk distribution plot:', error);
      res.status(500).json({ message: 'Failed to create risk distribution plot' });
    }
  });

  app.post('/api/plots/timeseries', async (req, res) => {
    try {
      const { data, config } = req.body;
      const plotPath = await plottingService.createTimeSeriesPlot(data, config);
      res.json({ plotPath, message: 'Time series plot created' });
    } catch (error) {
      console.error('Error creating time series plot:', error);
      res.status(500).json({ message: 'Failed to create time series plot' });
    }
  });

  app.get('/api/plots/list', async (req, res) => {
    try {
      const plots = plottingService.getPlotsList();
      res.json(plots);
    } catch (error) {
      console.error('Error getting plots list:', error);
      res.status(500).json({ message: 'Failed to get plots list' });
    }
  });

  app.delete('/api/plots/:filename', async (req, res) => {
    try {
      const { filename } = req.params;
      const success = plottingService.deletePlot(filename);
      if (success) {
        res.json({ message: 'Plot deleted successfully' });
      } else {
        res.status(404).json({ message: 'Plot not found' });
      }
    } catch (error) {
      console.error('Error deleting plot:', error);
      res.status(500).json({ message: 'Failed to delete plot' });
    }
  });

  return httpServer;
}

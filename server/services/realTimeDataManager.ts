import { EventEmitter } from 'events';
import { SpaceObject, ConjunctionEvent, AtmosphericData } from '@shared/schema';
import { cacheService } from './cacheService';
import { performanceMonitor } from './performanceMonitor';

interface RealTimeDataConfig {
  updateInterval: number; // milliseconds
  maxHistorySize: number;
  enableCaching: boolean;
  enablePerformanceMonitoring: boolean;
}

interface DataSnapshot {
  timestamp: Date;
  spaceObjects: SpaceObject[];
  conjunctionEvents: ConjunctionEvent[];
  atmosphericData: AtmosphericData | null;
  metrics: {
    totalObjects: number;
    highRiskObjects: number;
    activeConjunctions: number;
    averageAltitude: number;
  };
}

export class RealTimeDataManager extends EventEmitter {
  private static instance: RealTimeDataManager;
  private config: RealTimeDataConfig;
  private dataHistory: DataSnapshot[] = [];
  private updateTimer: NodeJS.Timeout | null = null;
  private isRunning = false;
  private lastUpdateTime: Date | null = null;

  public static getInstance(): RealTimeDataManager {
    if (!RealTimeDataManager.instance) {
      RealTimeDataManager.instance = new RealTimeDataManager();
    }
    return RealTimeDataManager.instance;
  }

  constructor() {
    super();
    this.config = {
      updateInterval: 30000, // 30 seconds
      maxHistorySize: 1000,
      enableCaching: true,
      enablePerformanceMonitoring: true,
    };
  }

  // Initialize the real-time data manager
  async initialize(): Promise<void> {
    console.log('Initializing Real-Time Data Manager...');
    
    try {
      // Load initial data
      await this.loadInitialData();
      
      // Start real-time updates
      this.startRealTimeUpdates();
      
      console.log('Real-Time Data Manager initialized successfully');
    } catch (error) {
      console.error('Failed to initialize Real-Time Data Manager:', error);
      throw error;
    }
  }

  // Load initial data from storage
  private async loadInitialData(): Promise<void> {
    const endTiming = this.config.enablePerformanceMonitoring 
      ? performanceMonitor.startTiming('load-initial-data')
      : null;

    try {
      // This would typically load from your storage service
      const initialSnapshot: DataSnapshot = {
        timestamp: new Date(),
        spaceObjects: [], // await storage.getSpaceObjects()
        conjunctionEvents: [], // await storage.getActiveConjunctionEvents()
        atmosphericData: null, // await storage.getLatestAtmosphericData()
        metrics: {
          totalObjects: 0,
          highRiskObjects: 0,
          activeConjunctions: 0,
          averageAltitude: 0,
        },
      };

      this.dataHistory.push(initialSnapshot);
      this.lastUpdateTime = initialSnapshot.timestamp;

      if (endTiming) {
        endTiming({ success: true, dataSize: initialSnapshot.spaceObjects.length });
      }
    } catch (error) {
      if (endTiming) {
        endTiming({ success: false, error: error.message });
      }
      throw error;
    }
  }

  // Start real-time data updates
  startRealTimeUpdates(): void {
    if (this.isRunning) {
      console.warn('Real-time updates already running');
      return;
    }

    this.isRunning = true;
    console.log(`Starting real-time updates every ${this.config.updateInterval}ms`);

    this.updateTimer = setInterval(async () => {
      try {
        await this.updateData();
      } catch (error) {
        console.error('Error in real-time data update:', error);
        this.emit('error', error);
      }
    }, this.config.updateInterval);
  }

  // Stop real-time data updates
  stopRealTimeUpdates(): void {
    if (this.updateTimer) {
      clearInterval(this.updateTimer);
      this.updateTimer = null;
    }
    this.isRunning = false;
    console.log('Real-time updates stopped');
  }

  // Update data snapshot
  private async updateData(): Promise<void> {
    const endTiming = this.config.enablePerformanceMonitoring 
      ? performanceMonitor.startTiming('update-real-time-data')
      : null;

    try {
      const snapshot = await this.createDataSnapshot();
      
      // Add to history
      this.dataHistory.push(snapshot);
      
      // Maintain history size limit
      if (this.dataHistory.length > this.config.maxHistorySize) {
        this.dataHistory = this.dataHistory.slice(-this.config.maxHistorySize);
      }

      // Update cache if enabled
      if (this.config.enableCaching) {
        await this.updateCache(snapshot);
      }

      this.lastUpdateTime = snapshot.timestamp;

      // Emit update event
      this.emit('dataUpdate', snapshot);
      this.emit('metricsUpdate', snapshot.metrics);

      if (endTiming) {
        endTiming({ 
          success: true, 
          objectsCount: snapshot.spaceObjects.length,
          conjunctionsCount: snapshot.conjunctionEvents.length 
        });
      }
    } catch (error) {
      if (endTiming) {
        endTiming({ success: false, error: error.message });
      }
      throw error;
    }
  }

  // Create a new data snapshot
  private async createDataSnapshot(): Promise<DataSnapshot> {
    // This would typically fetch from your data sources
    const spaceObjects: SpaceObject[] = [];
    const conjunctionEvents: ConjunctionEvent[] = [];
    const atmosphericData: AtmosphericData | null = null;

    // Calculate metrics
    const metrics = {
      totalObjects: spaceObjects.length,
      highRiskObjects: spaceObjects.filter(obj => 
        obj.riskLevel === 'high' || obj.riskLevel === 'critical'
      ).length,
      activeConjunctions: conjunctionEvents.length,
      averageAltitude: spaceObjects.length > 0 
        ? spaceObjects.reduce((sum, obj) => sum + (obj.altitude || 0), 0) / spaceObjects.length
        : 0,
    };

    return {
      timestamp: new Date(),
      spaceObjects,
      conjunctionEvents,
      atmosphericData,
      metrics,
    };
  }

  // Update cache with new data
  private async updateCache(snapshot: DataSnapshot): Promise<void> {
    try {
      await cacheService.setSpaceObjects({}, snapshot.spaceObjects);
      await cacheService.setDashboardMetrics(snapshot.metrics);
    } catch (error) {
      console.error('Error updating cache:', error);
    }
  }

  // Get current data snapshot
  getCurrentSnapshot(): DataSnapshot | null {
    return this.dataHistory.length > 0 ? this.dataHistory[this.dataHistory.length - 1] : null;
  }

  // Get data history
  getDataHistory(limit?: number): DataSnapshot[] {
    if (limit) {
      return this.dataHistory.slice(-limit);
    }
    return [...this.dataHistory];
  }

  // Get metrics trend
  getMetricsTrend(timeWindow: number = 3600000): { // 1 hour default
    timestamp: Date;
    totalObjects: number;
    highRiskObjects: number;
    activeConjunctions: number;
    averageAltitude: number;
  }[] {
    const cutoffTime = new Date(Date.now() - timeWindow);
    
    return this.dataHistory
      .filter(snapshot => snapshot.timestamp >= cutoffTime)
      .map(snapshot => ({
        timestamp: snapshot.timestamp,
        ...snapshot.metrics,
      }));
  }

  // Get space objects by criteria
  getSpaceObjectsByCriteria(criteria: {
    riskLevel?: string;
    type?: string;
    altitudeRange?: { min: number; max: number };
    country?: string;
  }): SpaceObject[] {
    const currentSnapshot = this.getCurrentSnapshot();
    if (!currentSnapshot) return [];

    return currentSnapshot.spaceObjects.filter(obj => {
      if (criteria.riskLevel && obj.riskLevel !== criteria.riskLevel) return false;
      if (criteria.type && obj.type !== criteria.type) return false;
      if (criteria.country && obj.country !== criteria.country) return false;
      if (criteria.altitudeRange) {
        const altitude = obj.altitude || 0;
        if (altitude < criteria.altitudeRange.min || altitude > criteria.altitudeRange.max) {
          return false;
        }
      }
      return true;
    });
  }

  // Get conjunction events by risk level
  getConjunctionEventsByRisk(riskLevel: string): ConjunctionEvent[] {
    const currentSnapshot = this.getCurrentSnapshot();
    if (!currentSnapshot) return [];

    return currentSnapshot.conjunctionEvents.filter(event => event.riskLevel === riskLevel);
  }

  // Update configuration
  updateConfig(newConfig: Partial<RealTimeDataConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    // Restart updates if interval changed
    if (newConfig.updateInterval && this.isRunning) {
      this.stopRealTimeUpdates();
      this.startRealTimeUpdates();
    }
  }

  // Get current configuration
  getConfig(): RealTimeDataConfig {
    return { ...this.config };
  }

  // Get status information
  getStatus(): {
    isRunning: boolean;
    lastUpdateTime: Date | null;
    historySize: number;
    config: RealTimeDataConfig;
  } {
    return {
      isRunning: this.isRunning,
      lastUpdateTime: this.lastUpdateTime,
      historySize: this.dataHistory.length,
      config: this.getConfig(),
    };
  }

  // Cleanup resources
  destroy(): void {
    this.stopRealTimeUpdates();
    this.dataHistory = [];
    this.removeAllListeners();
    console.log('Real-Time Data Manager destroyed');
  }
}

export const realTimeDataManager = RealTimeDataManager.getInstance();

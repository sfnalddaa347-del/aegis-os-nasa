import { SpaceObject } from '@shared/schema';
import { EventEmitter } from 'events';
import { performanceMonitor } from './performanceMonitor';

interface KesslerSimulationConfig {
  timeStep: number; // years
  totalTime: number; // years
  collisionProbability: number;
  fragmentationMultiplier: number;
  atmosphericDecayRate: number;
  enableRealTimeUpdates: boolean;
}

interface CollisionEvent {
  timestamp: number; // years from start
  primaryObject: SpaceObject;
  secondaryObject: SpaceObject;
  altitude: number;
  fragmentsGenerated: number;
  impact: 'low' | 'medium' | 'high' | 'critical';
}

interface SimulationResult {
  year: number;
  totalObjects: number;
  activeSatellites: number;
  debrisCount: number;
  collisionEvents: number;
  fragmentsGenerated: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  cascadeProbability: number;
  economicImpact: number; // USD
}

interface KesslerSimulationState {
  isRunning: boolean;
  currentYear: number;
  totalYears: number;
  progress: number; // 0-1
  results: SimulationResult[];
  collisionEvents: CollisionEvent[];
  config: KesslerSimulationConfig;
}

export class KesslerSimulation extends EventEmitter {
  private static instance: KesslerSimulation;
  private state: KesslerSimulationState;
  private simulationTimer: NodeJS.Timeout | null = null;
  private initialObjects: SpaceObject[] = [];

  public static getInstance(): KesslerSimulation {
    if (!KesslerSimulation.instance) {
      KesslerSimulation.instance = new KesslerSimulation();
    }
    return KesslerSimulation.instance;
  }

  constructor() {
    super();
    this.state = {
      isRunning: false,
      currentYear: 0,
      totalYears: 0,
      progress: 0,
      results: [],
      collisionEvents: [],
      config: {
        timeStep: 0.1, // 0.1 years
        totalTime: 50, // 50 years
        collisionProbability: 0.001,
        fragmentationMultiplier: 1000,
        atmosphericDecayRate: 0.02,
        enableRealTimeUpdates: true,
      },
    };
  }

  // Initialize simulation with space objects
  initialize(objects: SpaceObject[], config?: Partial<KesslerSimulationConfig>): void {
    this.initialObjects = [...objects];
    this.state.config = { ...this.state.config, ...config };
    this.state.totalYears = this.state.config.totalTime;
    this.state.currentYear = 0;
    this.state.progress = 0;
    this.state.results = [];
    this.state.collisionEvents = [];

    console.log(`Kessler simulation initialized with ${objects.length} objects`);
    this.emit('initialized', {
      objectCount: objects.length,
      config: this.state.config,
    });
  }

  // Start the simulation
  async start(): Promise<void> {
    if (this.state.isRunning) {
      console.warn('Simulation already running');
      return;
    }

    if (this.initialObjects.length === 0) {
      throw new Error('No objects to simulate. Initialize first.');
    }

    this.state.isRunning = true;
    console.log('Starting Kessler syndrome simulation...');

    // Run simulation
    await this.runSimulation();

    this.state.isRunning = false;
    console.log('Kessler simulation completed');
    this.emit('completed', this.state.results);
  }

  // Run the complete simulation
  private async runSimulation(): Promise<void> {
    const endTiming = performanceMonitor.startTiming('kessler-simulation');
    
    try {
      let currentObjects = [...this.initialObjects];
      
      for (let year = 0; year <= this.state.config.totalTime; year += this.state.config.timeStep) {
        this.state.currentYear = year;
        this.state.progress = year / this.state.config.totalTime;

        // Simulate one time step
        const stepResult = await this.simulateTimeStep(currentObjects, year);
        
        // Update object population
        currentObjects = stepResult.updatedObjects;
        
        // Record results
        this.state.results.push(stepResult.result);
        
        // Record collision events
        this.state.collisionEvents.push(...stepResult.collisionEvents);

        // Emit progress update
        if (this.state.config.enableRealTimeUpdates) {
          this.emit('progress', {
            year,
            progress: this.state.progress,
            result: stepResult.result,
          });
        }
      }

      if (endTiming) {
        endTiming({ 
          success: true, 
          totalYears: this.state.config.totalTime,
          finalObjects: currentObjects.length 
        });
      }
    } catch (error) {
      this.state.isRunning = false;
      if (endTiming) {
        endTiming({ success: false, error: error.message });
      }
      throw error;
    }
  }

  // Simulate one time step
  private async simulateTimeStep(
    objects: SpaceObject[], 
    year: number
  ): Promise<{
    updatedObjects: SpaceObject[];
    result: SimulationResult;
    collisionEvents: CollisionEvent[];
  }> {
    const collisionEvents: CollisionEvent[] = [];
    let updatedObjects = [...objects];

    // Simulate collisions
    const collisions = this.simulateCollisions(updatedObjects, year);
    collisionEvents.push(...collisions);

    // Generate fragments from collisions
    const fragments = this.generateFragments(collisions);
    updatedObjects.push(...fragments);

    // Simulate atmospheric decay
    updatedObjects = this.simulateAtmosphericDecay(updatedObjects, year);

    // Calculate metrics
    const result = this.calculateMetrics(updatedObjects, year, collisionEvents);

    return {
      updatedObjects,
      result,
      collisionEvents,
    };
  }

  // Simulate collisions between objects
  private simulateCollisions(objects: SpaceObject[], year: number): CollisionEvent[] {
    const collisions: CollisionEvent[] = [];
    const collisionPairs = this.findCollisionPairs(objects);

    for (const [obj1, obj2] of collisionPairs) {
      if (Math.random() < this.state.config.collisionProbability) {
        const collision: CollisionEvent = {
          timestamp: year,
          primaryObject: obj1,
          secondaryObject: obj2,
          altitude: (obj1.altitude || 0 + obj2.altitude || 0) / 2,
          fragmentsGenerated: this.calculateFragments(obj1, obj2),
          impact: this.calculateCollisionImpact(obj1, obj2),
        };
        collisions.push(collision);
      }
    }

    return collisions;
  }

  // Find potential collision pairs
  private findCollisionPairs(objects: SpaceObject[]): [SpaceObject, SpaceObject][] {
    const pairs: [SpaceObject, SpaceObject][] = [];
    
    for (let i = 0; i < objects.length; i++) {
      for (let j = i + 1; j < objects.length; j++) {
        const obj1 = objects[i];
        const obj2 = objects[j];
        
        // Check if objects are in similar orbits
        if (this.areInCollisionRange(obj1, obj2)) {
          pairs.push([obj1, obj2]);
        }
      }
    }
    
    return pairs;
  }

  // Check if two objects are in collision range
  private areInCollisionRange(obj1: SpaceObject, obj2: SpaceObject): boolean {
    const altitudeDiff = Math.abs((obj1.altitude || 0) - (obj2.altitude || 0));
    const inclinationDiff = Math.abs((obj1.inclination || 0) - (obj2.inclination || 0));
    
    // Objects are in collision range if they're within 10km altitude and 5° inclination
    return altitudeDiff < 10 && inclinationDiff < 5;
  }

  // Calculate number of fragments generated
  private calculateFragments(obj1: SpaceObject, obj2: SpaceObject): number {
    const mass1 = obj1.mass || 1000;
    const mass2 = obj2.mass || 1000;
    const totalMass = mass1 + mass2;
    
    // NASA standard breakup model
    const fragments = Math.floor(totalMass * this.state.config.fragmentationMultiplier / 1000000);
    return Math.max(10, fragments); // Minimum 10 fragments
  }

  // Calculate collision impact level
  private calculateCollisionImpact(obj1: SpaceObject, obj2: SpaceObject): 'low' | 'medium' | 'high' | 'critical' {
    const mass1 = obj1.mass || 1000;
    const mass2 = obj2.mass || 1000;
    const totalMass = mass1 + mass2;
    
    if (totalMass > 10000) return 'critical';
    if (totalMass > 5000) return 'high';
    if (totalMass > 1000) return 'medium';
    return 'low';
  }

  // Generate fragments from collisions
  private generateFragments(collisions: CollisionEvent[]): SpaceObject[] {
    const fragments: SpaceObject[] = [];
    
    for (const collision of collisions) {
      for (let i = 0; i < collision.fragmentsGenerated; i++) {
        const fragment: SpaceObject = {
          id: `fragment-${collision.timestamp}-${i}`,
          noradId: `FRAG-${Date.now()}-${i}`,
          name: `Fragment from ${collision.primaryObject.name}`,
          type: 'debris',
          altitude: collision.altitude + (Math.random() - 0.5) * 20, // ±10km spread
          inclination: (collision.primaryObject.inclination || 0) + (Math.random() - 0.5) * 10,
          eccentricity: Math.random() * 0.1,
          period: 90 + Math.random() * 20, // 90-110 minutes
          rcs: Math.random() * 0.1, // Small RCS
          mass: Math.random() * 10, // Small mass
          size: Math.random() * 0.5, // Small size
          status: 'active',
          riskLevel: 'medium',
          lastUpdate: new Date(),
          createdAt: new Date(),
        };
        fragments.push(fragment);
      }
    }
    
    return fragments;
  }

  // Simulate atmospheric decay
  private simulateAtmosphericDecay(objects: SpaceObject[], year: number): SpaceObject[] {
    return objects.filter(obj => {
      const altitude = obj.altitude || 0;
      const mass = obj.mass || 1000;
      
      // Objects below 200km decay quickly
      if (altitude < 200) {
        return Math.random() > 0.5; // 50% chance of decay
      }
      
      // Objects above 200km have slower decay
      const decayProbability = this.state.config.atmosphericDecayRate * (200 / altitude) * (1000 / mass);
      return Math.random() > decayProbability;
    });
  }

  // Calculate simulation metrics
  private calculateMetrics(
    objects: SpaceObject[], 
    year: number, 
    collisionEvents: CollisionEvent[]
  ): SimulationResult {
    const activeSatellites = objects.filter(obj => obj.type === 'satellite' && obj.status === 'active').length;
    const debrisCount = objects.filter(obj => obj.type === 'debris').length;
    const highRiskObjects = objects.filter(obj => 
      obj.riskLevel === 'high' || obj.riskLevel === 'critical'
    ).length;

    // Calculate cascade probability
    const cascadeProbability = this.calculateCascadeProbability(objects, collisionEvents);

    // Determine risk level
    let riskLevel: 'low' | 'medium' | 'high' | 'critical';
    if (cascadeProbability > 0.8) riskLevel = 'critical';
    else if (cascadeProbability > 0.6) riskLevel = 'high';
    else if (cascadeProbability > 0.3) riskLevel = 'medium';
    else riskLevel = 'low';

    // Calculate economic impact
    const economicImpact = this.calculateEconomicImpact(objects, collisionEvents);

    return {
      year,
      totalObjects: objects.length,
      activeSatellites,
      debrisCount,
      collisionEvents: collisionEvents.length,
      fragmentsGenerated: collisionEvents.reduce((sum, event) => sum + event.fragmentsGenerated, 0),
      riskLevel,
      cascadeProbability,
      economicImpact,
    };
  }

  // Calculate cascade probability
  private calculateCascadeProbability(objects: SpaceObject[], collisionEvents: CollisionEvent[]): number {
    const totalObjects = objects.length;
    const recentCollisions = collisionEvents.length;
    
    if (totalObjects === 0) return 0;
    
    // Simple cascade probability model
    const objectDensity = totalObjects / 1000000; // objects per million km³
    const collisionRate = recentCollisions / Math.max(1, totalObjects);
    
    return Math.min(1, objectDensity * collisionRate * 10);
  }

  // Calculate economic impact
  private calculateEconomicImpact(objects: SpaceObject[], collisionEvents: CollisionEvent[]): number {
    const satelliteValue = 100000000; // $100M per satellite
    const collisionCost = 50000000; // $50M per collision
    
    const lostSatellites = collisionEvents.length * 2; // Assume 2 satellites lost per collision
    const totalLoss = (objects.length * satelliteValue * 0.01) + (collisionEvents.length * collisionCost);
    
    return totalLoss;
  }

  // Get simulation results
  getResults(): SimulationResult[] {
    return [...this.state.results];
  }

  // Get collision events
  getCollisionEvents(): CollisionEvent[] {
    return [...this.state.collisionEvents];
  }

  // Get current state
  getState(): KesslerSimulationState {
    return { ...this.state };
  }

  // Stop simulation
  stop(): void {
    this.state.isRunning = false;
    if (this.simulationTimer) {
      clearTimeout(this.simulationTimer);
      this.simulationTimer = null;
    }
    console.log('Kessler simulation stopped');
  }

  // Reset simulation
  reset(): void {
    this.stop();
    this.state.results = [];
    this.state.collisionEvents = [];
    this.state.currentYear = 0;
    this.state.progress = 0;
    console.log('Kessler simulation reset');
  }
}

export const kesslerSimulation = KesslerSimulation.getInstance();

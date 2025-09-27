// Global type definitions for AEGIS Space Debris Monitoring System

declare module 'node' {
  export = NodeJS;
}

declare module 'vite/client' {
  interface ImportMetaEnv {
    readonly VITE_API_URL: string;
    readonly VITE_WS_URL: string;
    readonly VITE_APP_TITLE: string;
  }

  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

// Custom types for AEGIS
interface SpaceObject {
  id: string;
  name: string;
  type: 'satellite' | 'debris' | 'rocket_body' | 'space_station';
  altitude: number;
  inclination: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  position: [number, number, number];
  velocity: [number, number, number];
}

interface RiskAssessment {
  overall_risk: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  factors: {
    altitude: number;
    size: number;
    age: number;
    traffic_density: number;
  };
  confidence: number;
  recommendations: string[];
}

interface PerformanceMetrics {
  fps: number;
  memory: number;
  cpu: number;
  render_time: number;
}

interface SimulationResult {
  scenario: string;
  results: any[];
  statistics: {
    mean_collision_prob: number;
    std_collision_prob: number;
    mean_debris_count: number;
    std_debris_count: number;
  };
  convergence: boolean;
}

// Extend global Window interface
declare global {
  interface Window {
    AEGIS: {
      version: string;
      api: {
        getSpaceObjects: () => Promise<SpaceObject[]>;
        getRiskAssessment: () => Promise<RiskAssessment[]>;
        getPerformanceMetrics: () => Promise<PerformanceMetrics>;
        runSimulation: (scenario: string) => Promise<SimulationResult>;
      };
    };
  }
}

export {};

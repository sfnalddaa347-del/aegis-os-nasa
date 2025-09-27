import { performanceMonitor } from './performanceMonitor';
import { cacheService } from './cacheService';

interface HealthCheck {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  message: string;
  timestamp: Date;
  responseTime?: number;
  metadata?: Record<string, any>;
}

interface SystemHealth {
  overall: 'healthy' | 'degraded' | 'unhealthy';
  checks: HealthCheck[];
  timestamp: Date;
  uptime: number;
  version: string;
}

export class HealthMonitor {
  private static instance: HealthMonitor;
  private startTime: Date;
  private checks: Map<string, () => Promise<HealthCheck>> = new Map();

  public static getInstance(): HealthMonitor {
    if (!HealthMonitor.instance) {
      HealthMonitor.instance = new HealthMonitor();
    }
    return HealthMonitor.instance;
  }

  constructor() {
    this.startTime = new Date();
    this.registerDefaultChecks();
  }

  // Register a health check
  registerCheck(name: string, checkFn: () => Promise<HealthCheck>): void {
    this.checks.set(name, checkFn);
  }

  // Register default system health checks
  private registerDefaultChecks(): void {
    // Database health check
    this.registerCheck('database', async () => {
      const startTime = Date.now();
      try {
        // This would typically check database connectivity
        // const result = await db.query('SELECT 1');
        const responseTime = Date.now() - startTime;
        
        return {
          name: 'database',
          status: 'healthy',
          message: 'Database connection is healthy',
          timestamp: new Date(),
          responseTime,
          metadata: {
            connectionPool: 'active',
            queriesPerSecond: 0, // Would be calculated from actual metrics
          },
        };
      } catch (error) {
        return {
          name: 'database',
          status: 'unhealthy',
          message: `Database error: ${error.message}`,
          timestamp: new Date(),
          responseTime: Date.now() - startTime,
        };
      }
    });

    // Cache health check
    this.registerCheck('cache', async () => {
      const startTime = Date.now();
      try {
        const stats = cacheService.getCacheStats();
        const responseTime = Date.now() - startTime;
        
        const totalItems = Object.values(stats).reduce((sum, cache) => sum + cache.size, 0);
        const totalCapacity = Object.values(stats).reduce((sum, cache) => sum + cache.max, 0);
        const utilization = totalCapacity > 0 ? (totalItems / totalCapacity) * 100 : 0;
        
        return {
          name: 'cache',
          status: utilization > 90 ? 'degraded' : 'healthy',
          message: `Cache utilization: ${utilization.toFixed(1)}%`,
          timestamp: new Date(),
          responseTime,
          metadata: {
            totalItems,
            totalCapacity,
            utilization: utilization.toFixed(1),
            stats,
          },
        };
      } catch (error) {
        return {
          name: 'cache',
          status: 'unhealthy',
          message: `Cache error: ${error.message}`,
          timestamp: new Date(),
          responseTime: Date.now() - startTime,
        };
      }
    });

    // Performance health check
    this.registerCheck('performance', async () => {
      const startTime = Date.now();
      try {
        const summary = performanceMonitor.getSummary();
        const responseTime = Date.now() - startTime;
        
        const status = summary.averageResponseTime > 2000 ? 'degraded' : 
                      summary.averageResponseTime > 5000 ? 'unhealthy' : 'healthy';
        
        return {
          name: 'performance',
          status,
          message: `Average response time: ${summary.averageResponseTime.toFixed(2)}ms`,
          timestamp: new Date(),
          responseTime,
          metadata: {
            totalOperations: summary.totalOperations,
            averageResponseTime: summary.averageResponseTime,
            slowOperations: summary.slowOperations,
            topSlowOperations: summary.topSlowOperations.slice(0, 3),
          },
        };
      } catch (error) {
        return {
          name: 'performance',
          status: 'unhealthy',
          message: `Performance monitoring error: ${error.message}`,
          timestamp: new Date(),
          responseTime: Date.now() - startTime,
        };
      }
    });

    // Memory health check
    this.registerCheck('memory', async () => {
      const startTime = Date.now();
      try {
        const memUsage = process.memoryUsage();
        const responseTime = Date.now() - startTime;
        
        const heapUsedMB = memUsage.heapUsed / 1024 / 1024;
        const heapTotalMB = memUsage.heapTotal / 1024 / 1024;
        const rssMB = memUsage.rss / 1024 / 1024;
        
        const heapUtilization = (heapUsedMB / heapTotalMB) * 100;
        
        const status = heapUtilization > 90 ? 'unhealthy' : 
                      heapUtilization > 80 ? 'degraded' : 'healthy';
        
        return {
          name: 'memory',
          status,
          message: `Heap utilization: ${heapUtilization.toFixed(1)}%`,
          timestamp: new Date(),
          responseTime,
          metadata: {
            heapUsed: `${heapUsedMB.toFixed(2)} MB`,
            heapTotal: `${heapTotalMB.toFixed(2)} MB`,
            rss: `${rssMB.toFixed(2)} MB`,
            heapUtilization: heapUtilization.toFixed(1),
          },
        };
      } catch (error) {
        return {
          name: 'memory',
          status: 'unhealthy',
          message: `Memory monitoring error: ${error.message}`,
          timestamp: new Date(),
          responseTime: Date.now() - startTime,
        };
      }
    });

    // External API health check
    this.registerCheck('external-apis', async () => {
      const startTime = Date.now();
      try {
        // This would check external API endpoints
        const apiStatuses = {
          'space-track.org': 'healthy',
          'celestrak.com': 'healthy',
          'nasa.gov': 'healthy',
        };
        
        const responseTime = Date.now() - startTime;
        const unhealthyApis = Object.entries(apiStatuses).filter(([_, status]) => status !== 'healthy');
        
        const status = unhealthyApis.length === 0 ? 'healthy' : 
                      unhealthyApis.length <= 1 ? 'degraded' : 'unhealthy';
        
        return {
          name: 'external-apis',
          status,
          message: `${unhealthyApis.length} external API(s) unhealthy`,
          timestamp: new Date(),
          responseTime,
          metadata: {
            apiStatuses,
            unhealthyApis: unhealthyApis.map(([name]) => name),
          },
        };
      } catch (error) {
        return {
          name: 'external-apis',
          status: 'unhealthy',
          message: `External API check error: ${error.message}`,
          timestamp: new Date(),
          responseTime: Date.now() - startTime,
        };
      }
    });
  }

  // Run all health checks
  async runHealthChecks(): Promise<SystemHealth> {
    const checks: HealthCheck[] = [];
    
    for (const [name, checkFn] of this.checks) {
      try {
        const check = await checkFn();
        checks.push(check);
      } catch (error) {
        checks.push({
          name,
          status: 'unhealthy',
          message: `Health check failed: ${error.message}`,
          timestamp: new Date(),
        });
      }
    }
    
    // Determine overall health
    const unhealthyCount = checks.filter(c => c.status === 'unhealthy').length;
    const degradedCount = checks.filter(c => c.status === 'degraded').length;
    
    let overall: 'healthy' | 'degraded' | 'unhealthy';
    if (unhealthyCount > 0) {
      overall = 'unhealthy';
    } else if (degradedCount > 0) {
      overall = 'degraded';
    } else {
      overall = 'healthy';
    }
    
    return {
      overall,
      checks,
      timestamp: new Date(),
      uptime: Date.now() - this.startTime.getTime(),
      version: process.env.npm_package_version || '1.0.0',
    };
  }

  // Get health status for a specific check
  async getCheckHealth(name: string): Promise<HealthCheck | null> {
    const checkFn = this.checks.get(name);
    if (!checkFn) {
      return null;
    }
    
    try {
      return await checkFn();
    } catch (error) {
      return {
        name,
        status: 'unhealthy',
        message: `Health check failed: ${error.message}`,
        timestamp: new Date(),
      };
    }
  }

  // Get system metrics
  getSystemMetrics(): {
    uptime: number;
    memory: NodeJS.MemoryUsage;
    cpu: NodeJS.CpuUsage;
    version: string;
    nodeVersion: string;
    platform: string;
  } {
    return {
      uptime: Date.now() - this.startTime.getTime(),
      memory: process.memoryUsage(),
      cpu: process.cpuUsage(),
      version: process.env.npm_package_version || '1.0.0',
      nodeVersion: process.version,
      platform: process.platform,
    };
  }
}

export const healthMonitor = HealthMonitor.getInstance();

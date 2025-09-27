import { performance } from 'perf_hooks';

interface PerformanceMetric {
  operation: string;
  duration: number;
  timestamp: Date;
  metadata?: Record<string, any>;
}

interface PerformanceStats {
  operation: string;
  count: number;
  totalDuration: number;
  averageDuration: number;
  minDuration: number;
  maxDuration: number;
  lastExecuted: Date;
}

export class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: PerformanceMetric[] = [];
  private stats: Map<string, PerformanceStats> = new Map();
  private maxMetrics = 10000; // Keep last 10k metrics

  public static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  // Start timing an operation
  startTiming(operation: string): () => void {
    const startTime = performance.now();
    
    return (metadata?: Record<string, any>) => {
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      this.recordMetric({
        operation,
        duration,
        timestamp: new Date(),
        metadata,
      });
    };
  }

  // Record a performance metric
  recordMetric(metric: PerformanceMetric): void {
    this.metrics.push(metric);
    
    // Keep only the last maxMetrics
    if (this.metrics.length > this.maxMetrics) {
      this.metrics = this.metrics.slice(-this.maxMetrics);
    }
    
    // Update statistics
    this.updateStats(metric);
  }

  // Update statistics for an operation
  private updateStats(metric: PerformanceMetric): void {
    const existing = this.stats.get(metric.operation);
    
    if (existing) {
      existing.count++;
      existing.totalDuration += metric.duration;
      existing.averageDuration = existing.totalDuration / existing.count;
      existing.minDuration = Math.min(existing.minDuration, metric.duration);
      existing.maxDuration = Math.max(existing.maxDuration, metric.duration);
      existing.lastExecuted = metric.timestamp;
    } else {
      this.stats.set(metric.operation, {
        operation: metric.operation,
        count: 1,
        totalDuration: metric.duration,
        averageDuration: metric.duration,
        minDuration: metric.duration,
        maxDuration: metric.duration,
        lastExecuted: metric.timestamp,
      });
    }
  }

  // Get performance statistics
  getStats(): PerformanceStats[] {
    return Array.from(this.stats.values()).sort((a, b) => b.totalDuration - a.totalDuration);
  }

  // Get metrics for a specific operation
  getMetricsForOperation(operation: string, limit: number = 100): PerformanceMetric[] {
    return this.metrics
      .filter(m => m.operation === operation)
      .slice(-limit);
  }

  // Get recent metrics
  getRecentMetrics(limit: number = 100): PerformanceMetric[] {
    return this.metrics.slice(-limit);
  }

  // Get slow operations (above threshold)
  getSlowOperations(thresholdMs: number = 1000): PerformanceStats[] {
    return this.getStats().filter(stat => stat.averageDuration > thresholdMs);
  }

  // Clear all metrics and stats
  clear(): void {
    this.metrics = [];
    this.stats.clear();
  }

  // Get performance summary
  getSummary(): {
    totalOperations: number;
    averageResponseTime: number;
    slowOperations: number;
    topSlowOperations: PerformanceStats[];
  } {
    const allStats = this.getStats();
    const totalOperations = allStats.reduce((sum, stat) => sum + stat.count, 0);
    const totalDuration = allStats.reduce((sum, stat) => sum + stat.totalDuration, 0);
    const averageResponseTime = totalOperations > 0 ? totalDuration / totalOperations : 0;
    const slowOperations = allStats.filter(stat => stat.averageDuration > 1000).length;
    const topSlowOperations = allStats
      .filter(stat => stat.averageDuration > 1000)
      .slice(0, 5);

    return {
      totalOperations,
      averageResponseTime,
      slowOperations,
      topSlowOperations,
    };
  }

  // Performance decorator for methods
  static measure<T extends (...args: any[]) => any>(
    operation: string,
    fn: T
  ): T {
    return ((...args: any[]) => {
      const monitor = PerformanceMonitor.getInstance();
      const endTiming = monitor.startTiming(operation);
      
      try {
        const result = fn(...args);
        
        // Handle both sync and async functions
        if (result instanceof Promise) {
          return result.finally(() => {
            endTiming({ success: true });
          }).catch((error) => {
            endTiming({ success: false, error: error.message });
            throw error;
          });
        } else {
          endTiming({ success: true });
          return result;
        }
      } catch (error) {
        endTiming({ success: false, error: error.message });
        throw error;
      }
    }) as T;
  }
}

// Middleware for Express to measure request performance
export const performanceMiddleware = (req: any, res: any, next: any) => {
  const monitor = PerformanceMonitor.getInstance();
  const endTiming = monitor.startTiming(`HTTP ${req.method} ${req.path}`);
  
  res.on('finish', () => {
    endTiming({
      method: req.method,
      path: req.path,
      statusCode: res.statusCode,
      userAgent: req.get('User-Agent'),
    });
  });
  
  next();
};

// Database query performance monitoring
export const measureDatabaseQuery = <T>(
  queryName: string,
  queryFn: () => Promise<T>
): Promise<T> => {
  const monitor = PerformanceMonitor.getInstance();
  const endTiming = monitor.startTiming(`DB Query: ${queryName}`);
  
  return queryFn()
    .then(result => {
      endTiming({ success: true, resultCount: Array.isArray(result) ? result.length : 1 });
      return result;
    })
    .catch(error => {
      endTiming({ success: false, error: error.message });
      throw error;
    });
};

// AI API call performance monitoring
export const measureAIAPICall = <T>(
  modelName: string,
  apiCall: () => Promise<T>
): Promise<T> => {
  const monitor = PerformanceMonitor.getInstance();
  const endTiming = monitor.startTiming(`AI API: ${modelName}`);
  
  return apiCall()
    .then(result => {
      endTiming({ success: true, model: modelName });
      return result;
    })
    .catch(error => {
      endTiming({ success: false, error: error.message, model: modelName });
      throw error;
    });
};

export const performanceMonitor = PerformanceMonitor.getInstance();

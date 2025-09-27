import { LRUCache } from 'lru-cache';

// Cache configuration
const cacheConfig = {
  max: 1000, // Maximum number of items
  ttl: 1000 * 60 * 5, // 5 minutes TTL
  updateAgeOnGet: true,
  updateAgeOnHas: true,
};

// Create cache instances for different data types
export const spaceObjectsCache = new LRUCache<string, any>(cacheConfig);
export const aiAnalysisCache = new LRUCache<string, any>(cacheConfig);
export const atmosphericDataCache = new LRUCache<string, any>(cacheConfig);
export const economicDataCache = new LRUCache<string, any>(cacheConfig);

// Cache key generators
export const generateCacheKey = {
  spaceObjects: (params: Record<string, any> = {}) => 
    `space-objects:${JSON.stringify(params)}`,
  
  aiAnalysis: (type: string, params: Record<string, any> = {}) => 
    `ai-analysis:${type}:${JSON.stringify(params)}`,
  
  atmosphericData: (params: Record<string, any> = {}) => 
    `atmospheric-data:${JSON.stringify(params)}`,
  
  economicData: (params: Record<string, any> = {}) => 
    `economic-data:${JSON.stringify(params)}`,
  
  dashboardMetrics: () => 'dashboard-metrics',
  
  conjunctionEvents: (params: Record<string, any> = {}) => 
    `conjunction-events:${JSON.stringify(params)}`,
};

// Cache service class
export class CacheService {
  private static instance: CacheService;
  
  public static getInstance(): CacheService {
    if (!CacheService.instance) {
      CacheService.instance = new CacheService();
    }
    return CacheService.instance;
  }

  // Generic cache operations
  async get<T>(cache: LRUCache<string, T>, key: string): Promise<T | undefined> {
    return cache.get(key);
  }

  async set<T>(cache: LRUCache<string, T>, key: string, value: T, ttl?: number): Promise<void> {
    cache.set(key, value, { ttl });
  }

  async delete(cache: LRUCache<string, any>, key: string): Promise<boolean> {
    return cache.delete(key);
  }

  async clear(cache: LRUCache<string, any>): Promise<void> {
    cache.clear();
  }

  // Specific cache operations for different data types
  async getSpaceObjects(params: Record<string, any> = {}): Promise<any> {
    const key = generateCacheKey.spaceObjects(params);
    return this.get(spaceObjectsCache, key);
  }

  async setSpaceObjects(params: Record<string, any>, data: any): Promise<void> {
    const key = generateCacheKey.spaceObjects(params);
    await this.set(spaceObjectsCache, key, data);
  }

  async getAIAnalysis(type: string, params: Record<string, any> = {}): Promise<any> {
    const key = generateCacheKey.aiAnalysis(type, params);
    return this.get(aiAnalysisCache, key);
  }

  async setAIAnalysis(type: string, params: Record<string, any>, data: any): Promise<void> {
    const key = generateCacheKey.aiAnalysis(type, params);
    await this.set(aiAnalysisCache, key, data, 1000 * 60 * 10); // 10 minutes for AI analysis
  }

  async getAtmosphericData(params: Record<string, any> = {}): Promise<any> {
    const key = generateCacheKey.atmosphericData(params);
    return this.get(atmosphericDataCache, key);
  }

  async setAtmosphericData(params: Record<string, any>, data: any): Promise<void> {
    const key = generateCacheKey.atmosphericData(params);
    await this.set(atmosphericDataCache, key, data, 1000 * 60 * 2); // 2 minutes for atmospheric data
  }

  async getEconomicData(params: Record<string, any> = {}): Promise<any> {
    const key = generateCacheKey.economicData(params);
    return this.get(economicDataCache, key);
  }

  async setEconomicData(params: Record<string, any>, data: any): Promise<void> {
    const key = generateCacheKey.economicData(params);
    await this.set(economicDataCache, key, data, 1000 * 60 * 15); // 15 minutes for economic data
  }

  async getDashboardMetrics(): Promise<any> {
    const key = generateCacheKey.dashboardMetrics();
    return this.get(spaceObjectsCache, key); // Reuse space objects cache for metrics
  }

  async setDashboardMetrics(data: any): Promise<void> {
    const key = generateCacheKey.dashboardMetrics();
    await this.set(spaceObjectsCache, key, data, 1000 * 30); // 30 seconds for dashboard metrics
  }

  // Cache invalidation methods
  async invalidateSpaceObjects(): Promise<void> {
    await this.clear(spaceObjectsCache);
  }

  async invalidateAIAnalysis(): Promise<void> {
    await this.clear(aiAnalysisCache);
  }

  async invalidateAtmosphericData(): Promise<void> {
    await this.clear(atmosphericDataCache);
  }

  async invalidateEconomicData(): Promise<void> {
    await this.clear(economicDataCache);
  }

  // Cache statistics
  getCacheStats() {
    return {
      spaceObjects: {
        size: spaceObjectsCache.size,
        max: spaceObjectsCache.max,
        ttl: spaceObjectsCache.ttl,
      },
      aiAnalysis: {
        size: aiAnalysisCache.size,
        max: aiAnalysisCache.max,
        ttl: aiAnalysisCache.ttl,
      },
      atmosphericData: {
        size: atmosphericDataCache.size,
        max: atmosphericDataCache.max,
        ttl: atmosphericDataCache.ttl,
      },
      economicData: {
        size: economicDataCache.size,
        max: economicDataCache.max,
        ttl: economicDataCache.ttl,
      },
    };
  }

  // Warm up cache with frequently accessed data
  async warmUpCache(): Promise<void> {
    console.log('Warming up cache...');
    
    // Pre-load common queries
    try {
      // This would typically load from your data service
      // await this.setSpaceObjects({}, await spaceDataService.getSpaceObjects());
      // await this.setDashboardMetrics(await storage.getDashboardMetrics());
      
      console.log('Cache warmed up successfully');
    } catch (error) {
      console.error('Error warming up cache:', error);
    }
  }
}

export const cacheService = CacheService.getInstance();

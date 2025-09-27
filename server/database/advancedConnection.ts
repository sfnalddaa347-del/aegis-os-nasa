import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { Pool } from 'pg';
import Redis from 'ioredis';
import { performance } from 'perf_hooks';

// Database connection configurations
interface DatabaseConfig {
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
  ssl?: boolean;
  maxConnections?: number;
  connectionTimeout?: number;
  idleTimeout?: number;
}

interface RedisConfig {
  host: string;
  port: number;
  password?: string;
  db?: number;
  retryDelayOnFailover?: number;
  maxRetriesPerRequest?: number;
}

// Connection pool manager
class ConnectionPoolManager {
  private pools: Map<string, any> = new Map();
  private redisClients: Map<string, Redis> = new Map();
  private healthChecks: Map<string, boolean> = new Map();
  private metrics: Map<string, any> = new Map();

  constructor() {
    this.initializeMetrics();
    this.startHealthChecks();
  }

  private initializeMetrics() {
    this.metrics.set('connections', {
      active: 0,
      idle: 0,
      total: 0,
      errors: 0,
      queries: 0,
      avgQueryTime: 0
    });

    this.metrics.set('redis', {
      connected: false,
      operations: 0,
      errors: 0,
      avgResponseTime: 0
    });
  }

  // PostgreSQL connection with advanced features
  async createPostgresConnection(
    name: string,
    config: DatabaseConfig,
    options: {
      enableLogging?: boolean;
      enableMetrics?: boolean;
      enableConnectionPooling?: boolean;
      enableSSL?: boolean;
    } = {}
  ) {
    const {
      enableLogging = true,
      enableMetrics = true,
      enableConnectionPooling = true,
      enableSSL = false
    } = options;

    try {
      let connection;

      if (enableConnectionPooling) {
        // Use connection pooling
        const pool = new Pool({
          host: config.host,
          port: config.port,
          database: config.database,
          user: config.username,
          password: config.password,
          ssl: enableSSL ? { rejectUnauthorized: false } : false,
          max: config.maxConnections || 20,
          idleTimeoutMillis: config.idleTimeout || 30000,
          connectionTimeoutMillis: config.connectionTimeout || 2000,
          application_name: `aegis-${name}`,
        });

        // Enhanced error handling
        pool.on('error', (err) => {
          console.error(`PostgreSQL pool error for ${name}:`, err);
          this.updateMetrics('connections', { errors: 1 });
        });

        pool.on('connect', () => {
          this.updateMetrics('connections', { active: 1, total: 1 });
        });

        pool.on('remove', () => {
          this.updateMetrics('connections', { active: -1 });
        });

        connection = pool;
      } else {
        // Direct connection
        const connectionString = `postgresql://${config.username}:${config.password}@${config.host}:${config.port}/${config.database}?sslmode=${enableSSL ? 'require' : 'disable'}`;
        
        connection = postgres(connectionString, {
          max: config.maxConnections || 10,
          idle_timeout: config.idleTimeout || 30,
          connect_timeout: config.connectionTimeout || 2,
        });
      }

      // Wrap with Drizzle ORM
      const db = drizzle(connection, {
        logger: enableLogging,
        schema: {}, // Add your schema here
      });

      this.pools.set(name, { db, connection, type: 'postgres' });
      this.healthChecks.set(name, true);

      // Test connection
      await this.testConnection(name);

      console.log(`✅ PostgreSQL connection '${name}' established successfully`);
      return db;

    } catch (error) {
      console.error(`❌ Failed to create PostgreSQL connection '${name}':`, error);
      this.healthChecks.set(name, false);
      throw error;
    }
  }

  // Redis connection with advanced features
  async createRedisConnection(
    name: string,
    config: RedisConfig,
    options: {
      enableCluster?: boolean;
      enableSentinel?: boolean;
      enableMetrics?: boolean;
    } = {}
  ) {
    const {
      enableCluster = false,
      enableSentinel = false,
      enableMetrics = true
    } = options;

    try {
      let redis;

      if (enableCluster) {
        // Redis Cluster
        redis = new Redis.Cluster([
          {
            host: config.host,
            port: config.port,
          }
        ], {
          redisOptions: {
            password: config.password,
            db: config.db || 0,
          },
          retryDelayOnFailover: config.retryDelayOnFailover || 100,
          maxRetriesPerRequest: config.maxRetriesPerRequest || 3,
        });
      } else if (enableSentinel) {
        // Redis Sentinel
        redis = new Redis({
          sentinels: [{ host: config.host, port: config.port }],
          name: 'mymaster',
          password: config.password,
          db: config.db || 0,
        });
      } else {
        // Standard Redis
        redis = new Redis({
          host: config.host,
          port: config.port,
          password: config.password,
          db: config.db || 0,
          retryDelayOnFailover: config.retryDelayOnFailover || 100,
          maxRetriesPerRequest: config.maxRetriesPerRequest || 3,
          lazyConnect: true,
        });
      }

      // Enhanced event handling
      redis.on('connect', () => {
        console.log(`✅ Redis connection '${name}' established`);
        this.updateMetrics('redis', { connected: true });
      });

      redis.on('error', (err) => {
        console.error(`❌ Redis connection '${name}' error:`, err);
        this.updateMetrics('redis', { errors: 1, connected: false });
      });

      redis.on('close', () => {
        console.log(`🔌 Redis connection '${name}' closed`);
        this.updateMetrics('redis', { connected: false });
      });

      // Connect
      await redis.connect();

      this.redisClients.set(name, redis);
      this.healthChecks.set(name, true);

      return redis;

    } catch (error) {
      console.error(`❌ Failed to create Redis connection '${name}':`, error);
      this.healthChecks.set(name, false);
      throw error;
    }
  }

  // Advanced query execution with metrics
  async executeQuery<T>(
    connectionName: string,
    query: string,
    params: any[] = [],
    options: {
      timeout?: number;
      retries?: number;
      cache?: boolean;
      cacheTTL?: number;
    } = {}
  ): Promise<T> {
    const {
      timeout = 30000,
      retries = 3,
      cache = false,
      cacheTTL = 300
    } = options;

    const startTime = performance.now();
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        // Check cache first
        if (cache) {
          const cacheKey = `query:${Buffer.from(query + JSON.stringify(params)).toString('base64')}`;
          const cached = await this.getFromCache(cacheKey);
          if (cached) {
            return cached;
          }
        }

        const connection = this.pools.get(connectionName);
        if (!connection) {
          throw new Error(`Connection '${connectionName}' not found`);
        }

        // Execute query with timeout
        const result = await Promise.race([
          connection.db.execute(query, params),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Query timeout')), timeout)
          )
        ]) as T;

        // Cache result
        if (cache) {
          const cacheKey = `query:${Buffer.from(query + JSON.stringify(params)).toString('base64')}`;
          await this.setCache(cacheKey, result, cacheTTL);
        }

        // Update metrics
        const queryTime = performance.now() - startTime;
        this.updateMetrics('connections', { 
          queries: 1, 
          avgQueryTime: queryTime 
        });

        return result;

      } catch (error) {
        lastError = error as Error;
        console.error(`Query attempt ${attempt} failed:`, error);

        if (attempt < retries) {
          // Exponential backoff
          const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError || new Error('Query execution failed');
  }

  // Transaction management
  async executeTransaction<T>(
    connectionName: string,
    operations: (db: any) => Promise<T>,
    options: {
      isolationLevel?: 'READ_UNCOMMITTED' | 'READ_COMMITTED' | 'REPEATABLE_READ' | 'SERIALIZABLE';
      timeout?: number;
    } = {}
  ): Promise<T> {
    const { isolationLevel = 'READ_COMMITTED', timeout = 30000 } = options;
    const connection = this.pools.get(connectionName);
    
    if (!connection) {
      throw new Error(`Connection '${connectionName}' not found`);
    }

    const startTime = performance.now();

    try {
      const result = await connection.db.transaction(async (tx: any) => {
        // Set isolation level
        await tx.execute(`SET TRANSACTION ISOLATION LEVEL ${isolationLevel}`);
        
        // Execute operations
        return await operations(tx);
      });

      const transactionTime = performance.now() - startTime;
      this.updateMetrics('connections', { 
        queries: 1, 
        avgQueryTime: transactionTime 
      });

      return result;

    } catch (error) {
      console.error('Transaction failed:', error);
      throw error;
    }
  }

  // Cache operations
  async getFromCache(key: string): Promise<any> {
    try {
      const redis = this.redisClients.values().next().value;
      if (!redis) return null;

      const value = await redis.get(key);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      console.error('Cache get error:', error);
      return null;
    }
  }

  async setCache(key: string, value: any, ttl: number = 300): Promise<void> {
    try {
      const redis = this.redisClients.values().next().value;
      if (!redis) return;

      await redis.setex(key, ttl, JSON.stringify(value));
    } catch (error) {
      console.error('Cache set error:', error);
    }
  }

  async deleteCache(key: string): Promise<void> {
    try {
      const redis = this.redisClients.values().next().value;
      if (!redis) return;

      await redis.del(key);
    } catch (error) {
      console.error('Cache delete error:', error);
    }
  }

  // Health checks
  private async testConnection(name: string): Promise<boolean> {
    try {
      const connection = this.pools.get(name);
      if (!connection) return false;

      if (connection.type === 'postgres') {
        await connection.db.execute('SELECT 1');
      }

      this.healthChecks.set(name, true);
      return true;
    } catch (error) {
      console.error(`Health check failed for ${name}:`, error);
      this.healthChecks.set(name, false);
      return false;
    }
  }

  private startHealthChecks(): void {
    setInterval(async () => {
      for (const [name] of this.pools) {
        await this.testConnection(name);
      }
    }, 30000); // Check every 30 seconds
  }

  // Metrics and monitoring
  private updateMetrics(type: string, updates: any): void {
    const current = this.metrics.get(type) || {};
    this.metrics.set(type, { ...current, ...updates });
  }

  getMetrics(): any {
    return Object.fromEntries(this.metrics);
  }

  getHealthStatus(): any {
    return Object.fromEntries(this.healthChecks);
  }

  // Connection management
  async closeConnection(name: string): Promise<void> {
    const connection = this.pools.get(name);
    if (connection) {
      if (connection.type === 'postgres') {
        await connection.connection.end();
      }
      this.pools.delete(name);
    }

    const redis = this.redisClients.get(name);
    if (redis) {
      await redis.disconnect();
      this.redisClients.delete(name);
    }

    this.healthChecks.delete(name);
  }

  async closeAllConnections(): Promise<void> {
    const promises: Promise<void>[] = [];

    for (const [name] of this.pools) {
      promises.push(this.closeConnection(name));
    }

    await Promise.all(promises);
  }
}

// Singleton instance
export const connectionManager = new ConnectionPoolManager();

// Helper functions
export const createDatabaseConnection = async (
  name: string,
  config: DatabaseConfig,
  options?: any
) => {
  return await connectionManager.createPostgresConnection(name, config, options);
};

export const createRedisConnection = async (
  name: string,
  config: RedisConfig,
  options?: any
) => {
  return await connectionManager.createRedisConnection(name, config, options);
};

export const executeQuery = async <T>(
  connectionName: string,
  query: string,
  params?: any[],
  options?: any
): Promise<T> => {
  return await connectionManager.executeQuery<T>(connectionName, query, params, options);
};

export const executeTransaction = async <T>(
  connectionName: string,
  operations: (db: any) => Promise<T>,
  options?: any
): Promise<T> => {
  return await connectionManager.executeTransaction(connectionName, operations, options);
};

export const getCache = async (key: string) => {
  return await connectionManager.getFromCache(key);
};

export const setCache = async (key: string, value: any, ttl?: number) => {
  return await connectionManager.setCache(key, value, ttl);
};

export const deleteCache = async (key: string) => {
  return await connectionManager.deleteCache(key);
};

export const getConnectionMetrics = () => {
  return connectionManager.getMetrics();
};

export const getConnectionHealth = () => {
  return connectionManager.getHealthStatus();
};

export default connectionManager;

/// <reference types="node" />

declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NODE_ENV: 'development' | 'production' | 'test';
      PORT?: string;
      DATABASE_URL?: string;
      OPENAI_API_KEY?: string;
      ANTHROPIC_API_KEY?: string;
    }
  }
}

export {};

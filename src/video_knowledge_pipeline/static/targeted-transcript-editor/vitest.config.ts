import { defineConfig } from 'vitest/config';
export default defineConfig({ test: { environment: 'jsdom', include: ['tests/**/*.test.ts'], maxWorkers: 2, setupFiles: ['tests/setup.ts'] } });

import { build } from 'esbuild';
await build({ entryPoints: ['main.ts'], outfile: 'editor.bundle.js', bundle: true, format: 'iife', target: 'es2022', sourcemap: true, legalComments: 'linked', minify: true });
console.log('Built local transcript editor (no CDN).');

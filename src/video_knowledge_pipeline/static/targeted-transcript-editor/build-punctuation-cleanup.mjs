import { build } from 'esbuild';
await build({entryPoints:['punctuation-cleanup-cli.ts'],outfile:'punctuation-cleanup-cli.mjs',bundle:true,platform:'node',format:'esm',packages:'external',target:'es2022',legalComments:'linked'});

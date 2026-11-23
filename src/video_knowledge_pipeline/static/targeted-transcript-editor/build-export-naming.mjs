import {build} from 'esbuild';
await build({entryPoints:['export-naming-cli.ts'],outfile:'export-naming-cli.mjs',bundle:true,platform:'node',format:'esm',target:'node22'});
await build({entryPoints:['export-naming-browser.ts'],outfile:'export-naming-browser.js',bundle:true,format:'iife',globalName:'VKPProjectReviewExport',target:'es2022',minify:true});

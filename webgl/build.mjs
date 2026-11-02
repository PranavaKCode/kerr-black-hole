// inline shaders, css and js into one self-contained page, dist/index.html (runs from file://)
//   node webgl/build.mjs
import { readFileSync, writeFileSync, readdirSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, 'dist');
mkdirSync(out, { recursive: true });

const shaders = {};
for (const f of readdirSync(join(here, 'shaders'))) {
  const src = readFileSync(join(here, 'shaders', f), 'utf8').replace(/\r\n/g, '\n');
  if (f.endsWith('.glsl')) {
    // multi-shader file split by "//// name" markers
    const parts = src.split(/^\/\/\/\/ (\S+)\s*$/m);
    for (let i = 1; i < parts.length; i += 2) shaders[parts[i]] = parts[i + 1].trim() + '\n';
  } else shaders[f] = src.trim() + '\n';
}
// GLSL ES requires "#version" on the first line
for (const [k, v] of Object.entries(shaders)) if (!v.startsWith('#version')) throw new Error('shader ' + k + ' must start with #version');

const css = readFileSync(join(here, 'style.css'), 'utf8');
const app = readFileSync(join(here, 'app.js'), 'utf8');
let html = readFileSync(join(here, 'index.src.html'), 'utf8');
const safe = s => s.replace(/<\/script/gi, '<\\/script');
html = html.replace('/*@@CSS@@*/', () => css)
  .replace('/*@@SHADERS@@*/', () => 'window.__KERR_SHADERS = ' + safe(JSON.stringify(shaders)) + ';')
  .replace('/*@@APP@@*/', () => safe(app));
writeFileSync(join(out, 'index.html'), html);
console.log('wrote', join(out, 'index.html'), (html.length / 1024).toFixed(1) + ' KB', Object.keys(shaders));

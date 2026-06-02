// E2E from OCI server: Gemini gen -> RunPod (swap+restore+bg). Measures each stage.
import { GoogleGenAI } from '@google/genai';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import sharp from 'sharp';
import { pickRandomVariation, buildPrompt } from './components/wweVariations.js';

const env = Object.fromEntries(readFileSync('.env', 'utf8').split('\n')
  .filter(l => l.includes('=')).map(l => { const i = l.indexOf('='); return [l.slice(0, i).trim(), l.slice(i + 1).trim()]; }));
const apiKey = env['GEMINI_API-KEY'] || env['GEMINI_API_KEY'];
const RUNPOD_KEY = env['RUNPOD_API_KEY'];
const MODEL = env.WWE_GEMINI_MODEL || 'gemini-3.1-flash-image-preview';
const EP = 'ytn770avgv7n1x';
const SOURCE = '/tmp/ashu.jpeg';

const t0 = Date.now(); const el = () => ((Date.now() - t0) / 1000).toFixed(1) + 's';

// 1) Gemini
const ai = new GoogleGenAI({ apiKey });
const refBuf = await sharp(readFileSync(SOURCE)).resize(512, 512, { fit: 'cover' }).png().toBuffer();
const [vkey, outfit, pose] = pickRandomVariation('male');
console.log(`[${el()}] gemini (${vkey})...`);
const tG = Date.now();
let genB64 = null;
for (let a = 1; a <= 4 && !genB64; a++) {
  const r = await ai.models.generateContent({ model: MODEL,
    contents: [{ inlineData: { mimeType: 'image/png', data: refBuf.toString('base64') } }, { text: buildPrompt(outfit, pose, 'male') }],
    config: { responseModalities: ['TEXT', 'IMAGE'], imageConfig: { aspectRatio: '1:1' } } });
  const p = (r.candidates?.[0]?.content?.parts ?? []).find(x => x.inlineData?.data);
  if (p) genB64 = p.inlineData.data;
}
console.log(`[${el()}] gemini done in ${((Date.now() - tG) / 1000).toFixed(1)}s`);

// 2) RunPod swap+restore+bg
const tR = Date.now();
console.log(`[${el()}] runpod /runsync...`);
const resp = await fetch(`https://api.runpod.ai/v2/${EP}/runsync`, {
  method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${RUNPOD_KEY}` },
  body: JSON.stringify({ input: { op: 'both', image: genB64, source_face: readFileSync(SOURCE).toString('base64'), feather: 0.8, erode: 1 } }),
}).then(r => r.json());
console.log(`[${el()}] runpod done in ${((Date.now() - tR) / 1000).toFixed(1)}s | status=${resp.status} delayMs=${resp.delayTime} execMs=${resp.executionTime}`);
const o = resp.output || {};
if (o.error) { console.log('ERROR:', String(o.error).slice(0, 300)); }
else if (o.image) {
  mkdirSync('/tmp/rp_e2e', { recursive: true });
  const out = `/tmp/rp_e2e/rp_${Date.now()}.png`;
  writeFileSync(out, Buffer.from(o.image, 'base64'));
  console.log(`OUTPUT: ${out} (had_alpha=${o.had_alpha})`);
}
console.log(`\n=== TOTAL: ${el()} ===`);

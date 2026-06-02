// FINAL baked output via RunPod: Gemini gen -> RunPod (swap+restore+bg) -> WWE
// composite (scale to refFaceHeight, place at headTopY, onto templateBg + foreground).
import { GoogleGenAI } from '@google/genai';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import sharp from 'sharp';
import { pickRandomVariation, buildPrompt } from './components/wweVariations.js';
import { getFaceBox } from './components/faceDetect.js';
import { compositeFg } from './components/wweVtonManager.js';

const env = Object.fromEntries(readFileSync('.env', 'utf8').split('\n')
  .filter(l => l.includes('=')).map(l => { const i = l.indexOf('='); return [l.slice(0, i).trim(), l.slice(i + 1).trim()]; }));
const apiKey = env['GEMINI_API-KEY'] || env['GEMINI_API_KEY'];
const RUNPOD_KEY = env['RUNPOD_API_KEY'];
const MODEL = env.WWE_GEMINI_MODEL || 'gemini-3.1-flash-image-preview';
const EP = 'ytn770avgv7n1x';
const SOURCE = '/tmp/ashu.jpeg';
const TEMPLATE = process.argv[2] || 'jey';
const cfg = JSON.parse(readFileSync(`generic-wwe-configs/${TEMPLATE}.json`, 'utf8'));

const t0 = Date.now(); const el = () => ((Date.now() - t0) / 1000).toFixed(1) + 's';

// 1) Gemini
const ai = new GoogleGenAI({ apiKey });
const refBuf = await sharp(readFileSync(SOURCE)).resize(512, 512, { fit: 'cover' }).png().toBuffer();
const [vkey, outfit, pose] = pickRandomVariation('male');
console.log(`[${el()}] gemini (${vkey})...`);
let genB64 = null;
for (let a = 1; a <= 4 && !genB64; a++) {
  const r = await ai.models.generateContent({ model: MODEL,
    contents: [{ inlineData: { mimeType: 'image/png', data: refBuf.toString('base64') } }, { text: buildPrompt(outfit, pose, 'male') }],
    config: { responseModalities: ['TEXT', 'IMAGE'], imageConfig: { aspectRatio: '1:1' } } });
  const p = (r.candidates?.[0]?.content?.parts ?? []).find(x => x.inlineData?.data);
  if (p) genB64 = p.inlineData.data;
}
console.log(`[${el()}] gemini done`);

// 2) RunPod swap+restore+bg -> RGBA cutout
const resp = await fetch(`https://api.runpod.ai/v2/${EP}/runsync`, {
  method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${RUNPOD_KEY}` },
  body: JSON.stringify({ input: { op: 'both', image: genB64, source_face: readFileSync(SOURCE).toString('base64'), feather: 0.8, erode: 1 } }),
}).then(r => r.json());
if (!resp.output || !resp.output.image) throw new Error('runpod: ' + JSON.stringify(resp).slice(0, 300));
let personBuf = await sharp(Buffer.from(resp.output.image, 'base64')).trim({ threshold: 10 }).png().toBuffer();
console.log(`[${el()}] runpod done (exec ${resp.executionTime}ms)`);

// 3) WWE composite (same as production post-imgly)
const templateBuf = await sharp(cfg.templateBg).png().toBuffer();
const tm = await sharp(templateBuf).metadata();
const pm = await sharp(personBuf).metadata();
const g = await getFaceBox(personBuf);
let scale = cfg.refFaceHeight / g.height;
const minH = cfg.minPersonHeight ?? 700;
if (pm.height * scale < minH) scale = minH / pm.height;
const pW = Math.round(pm.width * scale), pH = Math.round(pm.height * scale);
const pr = await sharp(personBuf).resize(pW, pH).png().toBuffer();
const left = Math.round(cfg.headPos.x - (g.left + g.width / 2) * scale), top = cfg.headTopY;
const sl = Math.max(0, -left), st = Math.max(0, -top), dl = Math.max(0, left), dt = Math.max(0, top);
const vw = Math.min(pW - sl, tm.width - dl), vh = Math.min(pH - st, tm.height - dt);
let ov = pr, ol = left, ot = top;
if (sl > 0 || st > 0 || pW > tm.width || pH > tm.height) { ov = await sharp(pr).extract({ left: sl, top: st, width: vw, height: vh }).png().toBuffer(); ol = dl; ot = dt; }
let img = await sharp(templateBuf).composite([{ input: ov, left: ol, top: ot }]).png().toBuffer();
if (cfg.foreground) img = await compositeFg(img, cfg.foreground);

mkdirSync('/tmp/rp_final', { recursive: true });
const out = `/tmp/rp_final/final_${TEMPLATE}_${Date.now()}.jpg`;
writeFileSync(out, await sharp(img).jpeg({ quality: 92 }).toBuffer());
console.log(`OUTPUT: ${out}`);
console.log(`\n=== TOTAL: ${el()} ===`);

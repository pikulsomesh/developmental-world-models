/* Interactive companion to "Do AI models learn like humans?".
   Every filter here mirrors dwm/senses.py; every number is read from docs/data/*.json. */
(() => {
"use strict";
const $ = (s) => document.querySelector(s);
const has = (s) => !!document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// ------------------------------------------------------------------ theme + math (chrome lives in site.js)
const root = document.documentElement;
const renderMath = () => window.DWM && window.DWM.renderMath && window.DWM.renderMath();
window.addEventListener("themechange", () => redrawAll());

// ------------------------------------------------------------------ constants
const REGIMES = ["curriculum_only", "grown", "growth_only", "adult_scratch", "shuffled_order", "reversed_immaturity"];
const RLABEL = { curriculum_only: "Immature input", grown: "Immature input + growth", growth_only: "Growth only",
  adult_scratch: "Adult input", shuffled_order: "Shuffled input quality", reversed_immaturity: "Reversed immaturity" };
const RCOLOR = { curriculum_only: "--c-curriculum", grown: "--c-grown", growth_only: "--c-growth",
  adult_scratch: "--c-adult", shuffled_order: "--c-shuffled", reversed_immaturity: "--c-reversed" };
const RNOTE = {
  curriculum_only: "Fixed 12-month-size network. Input quality follows age: newborn blur first, adult-quality last. Our primary regime.",
  grown: "Starts with a quarter of the width and one frame of memory, and grows at each checkpoint (function-preserving). Input follows age.",
  growth_only: "Grows like the previous regime but always sees adult-quality input. Isolates network growth.",
  adult_scratch: "Fixed size, adult-quality input from the start. The standard machine-learning baseline.",
  shuffled_order: "Fixed size. Every batch gets a random age's input quality, so the variety is the same but the order is gone.",
  reversed_immaturity: "Fixed size. Input goes from adult quality to newborn quality: a negative control for order.",
};
const RTABLE = {
  curriculum_only: ["fixed", "0 → 3 → 6 → 12 mo"], grown: ["grows ×1 → ×4", "0 → 3 → 6 → 12 mo"],
  growth_only: ["grows ×1 → ×4", "adult"], adult_scratch: ["fixed", "adult"],
  shuffled_order: ["fixed", "random per batch"], reversed_immaturity: ["fixed", "12 → 6 → 3 → 0 mo"],
};
const STAGES = [0, 3, 6, 12];
const SENSES = { vision: "Vision", audio: "Hearing", touch: "Touch", proprio: "Body sense" };

let EP = null, TABLES = null, MO = null, RES = null;

// ------------------------------------------------------------------ helpers
function b64(s, type) {
  const bin = atob(s), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return type === "f32" ? new Float32Array(u8.buffer) : u8;
}
function interp(table, a) {
  const ks = Object.keys(table).map(Number).sort((x, y) => x - y);
  if (a <= ks[0]) return table[ks[0]];
  if (a >= ks[ks.length - 1]) return table[ks[ks.length - 1]];
  for (let i = 0; i < ks.length - 1; i++) if (a >= ks[i] && a <= ks[i + 1]) {
    const t = (a - ks[i]) / (ks[i + 1] - ks[i]); return table[ks[i]] * (1 - t) + table[ks[i + 1]] * t;
  }
}
function kernel(sigma) {
  if (sigma < 0.05) return [1];
  const r = Math.max(1, Math.ceil(3 * sigma)), k = [];
  let s = 0;
  for (let x = -r; x <= r; x++) { const v = Math.exp(-0.5 * (x / sigma) ** 2); k.push(v); s += v; }
  return k.map((v) => v / s);
}
// separable blur over an (C,H,W) float array with replicate padding
function blur2d(x, C, H, W, sigma) {
  const k = kernel(sigma); if (k.length === 1) return x;
  const r = (k.length - 1) / 2, tmp = new Float32Array(x.length), out = new Float32Array(x.length);
  for (let c = 0; c < C; c++) for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
    let s = 0; for (let q = -r; q <= r; q++) s += k[q + r] * x[c * H * W + i * W + Math.min(W - 1, Math.max(0, j + q))];
    tmp[c * H * W + i * W + j] = s;
  }
  for (let c = 0; c < C; c++) for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
    let s = 0; for (let q = -r; q <= r; q++) s += k[q + r] * tmp[c * H * W + Math.min(H - 1, Math.max(0, i + q)) * W + j];
    out[c * H * W + i * W + j] = s;
  }
  return out;
}
function rng(seed) { // mulberry32 + Box-Muller
  let a = seed >>> 0;
  const u = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
}
function visionParams(a) {
  const cpd = interp(TABLES.vision.acuity_cpd, a), cs = interp(TABLES.vision.contrast_sens, a);
  return { cpd, sigma: 0.5 * (30 / cpd) * (32 / 224), cs, gain: Math.min(1, Math.max(0.15, Math.log10(cs) / 2)),
    sat: interp(TABLES.vision.color_sat, a) };
}
function filterVision(frame, a) { // frame: Float32Array (3,32,32) in [0,1]
  const p = visionParams(a);
  let x = blur2d(frame, 3, 32, 32, p.sigma);
  let m = 0; for (const v of x) m += v; m /= x.length;
  const y = new Float32Array(x.length), N = 1024;
  for (let i = 0; i < x.length; i++) y[i] = m + (x[i] - m) * p.gain;
  for (let i = 0; i < N; i++) {
    const l = 0.299 * y[i] + 0.587 * y[N + i] + 0.114 * y[2 * N + i];
    for (let c = 0; c < 3; c++) y[c * N + i] = Math.min(1, Math.max(0, l + (y[c * N + i] - l) * p.sat));
  }
  return y;
}
function filterAudio(x, a, seed) { // x: (64) = 2 ears x 32 bins
  const blur = interp(TABLES.audio.freq_blur, a), hf = interp(TABLES.audio.hf_cutoff, a),
    bin = interp(TABLES.audio.binaural, a), noise = interp(TABLES.audio.noise, a);
  const y = blur2d(Float32Array.from(x), 2, 1, 32, blur), g = rng(seed);
  for (let f = 0; f < 32; f++) { const s = 1 / (1 + Math.exp(-((hf * 32 - f) / 1.5))); y[f] *= s; y[32 + f] *= s; }
  for (let f = 0; f < 32; f++) {
    const mono = (y[f] + y[32 + f]) / 2; y[f] = mono + (y[f] - mono) * bin; y[32 + f] = mono + (y[32 + f] - mono) * bin;
  }
  for (let i = 0; i < 64; i++) y[i] += noise * g();
  return y;
}
function filterTouch(x, a, seed) {
  const y = blur2d(Float32Array.from(x), 1, 8, 8, interp(TABLES.touch.blur, a)), g = rng(seed), n = interp(TABLES.touch.noise, a);
  for (let i = 0; i < 64; i++) y[i] += n * g();
  return y;
}
function frameOf(ep, key, t, size) { return ep[key].subarray(t * size, (t + 1) * size); }
function visFrame(ep, t) { const u = ep.vision.subarray(t * 3072, (t + 1) * 3072), f = new Float32Array(3072); for (let i = 0; i < 3072; i++) f[i] = u[i] / 255; return f; }
function putRGB(canvas, x) {
  const ctx = canvas.getContext("2d"), im = ctx.createImageData(32, 32);
  for (let i = 0; i < 1024; i++) { im.data[4 * i] = x[i] * 255; im.data[4 * i + 1] = x[1024 + i] * 255; im.data[4 * i + 2] = x[2048 + i] * 255; im.data[4 * i + 3] = 255; }
  ctx.putImageData(im, 0, 0);
}
function hexRGB(h) { h = h.replace("#", ""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); }

// ------------------------------------------------------------------ 1. senses
let senseAge = 0, senseT = 8, sensesTimer = null;
function drawSenses() {
  if (!EP || !has("#cVision")) return;
  const ep = EP.ordinary, a = senseAge, t = senseT;
  $("#ageLabel").textContent = a === 0 ? "Newborn" : `${+a.toFixed(2)} month${a === 1 ? "" : "s"}`;
  const vp = visionParams(a);
  putRGB($("#cVision"), filterVision(visFrame(ep, t), a));
  $("#pVision").innerHTML = `acuity ${vp.cpd.toFixed(1)} cyc/deg<br>blur σ ${vp.sigma.toFixed(2)} px<br>contrast gain ${vp.gain.toFixed(2)}<br>colour ${vp.sat.toFixed(2)}`;
  // hearing: spectrogram of both ears over the episode
  const cA = $("#cAudio"), ctx = cA.getContext("2d"), im = ctx.createImageData(40, 32), [r, g, b] = hexRGB(css("--c-audio") || "#eb6834");
  const bg = hexRGB(css("--surface-2").startsWith("#") ? css("--surface-2") : "#f2f0eb");
  for (let tt = 0; tt < 20; tt++) {
    const y = filterAudio(frameOf(ep, "audio", tt, 64), a, 1000 + tt);
    for (let ear = 0; ear < 2; ear++) for (let f = 0; f < 32; f++) {
      const v = Math.max(0, Math.min(1, y[ear * 32 + f] / 0.6)), px = (31 - f) * 40 + ear * 20 + tt, o = 4 * px;
      const hi = tt === t ? 1 : 0.8;
      im.data[o] = bg[0] + (r - bg[0]) * v * hi; im.data[o + 1] = bg[1] + (g - bg[1]) * v * hi; im.data[o + 2] = bg[2] + (b - bg[2]) * v * hi; im.data[o + 3] = 255;
    }
  }
  ctx.putImageData(im, 0, 0);
  ctx.fillStyle = css("--text"); ctx.globalAlpha = 0.5; ctx.fillRect(t, 0, 1, 1); ctx.fillRect(20 + t, 0, 1, 1); ctx.globalAlpha = 1;
  $("#pAudio").innerHTML = `left ear | right ear, time →<br>freq. blur ${interp(TABLES.audio.freq_blur, a).toFixed(2)} bins<br>high-freq. pass ${(100 * interp(TABLES.audio.hf_cutoff, a)).toFixed(0)}%<br>left/right difference ${interp(TABLES.audio.binaural, a).toFixed(2)}`;
  // touch
  const tc = filterTouch(frameOf(ep, "touch", t, 64), a, 2000 + t), cT = $("#cTouch").getContext("2d"), it = cT.createImageData(8, 8);
  const [tr, tg, tb] = hexRGB(css("--c-touch") || "#1baf7a");
  for (let i = 0; i < 64; i++) { const v = Math.max(0, Math.min(1, tc[i])); it.data[4 * i] = 20 + (tr - 20) * v; it.data[4 * i + 1] = 20 + (tg - 20) * v; it.data[4 * i + 2] = 22 + (tb - 22) * v; it.data[4 * i + 3] = 255; }
  cT.putImageData(it, 0, 0);
  $("#pTouch").innerHTML = `${EP.ordinary.contact[t] > 0 ? "hand touching toy" : "no contact this frame"}<br>receptive-field blur ${interp(TABLES.touch.blur, a).toFixed(2)}<br>noise ${interp(TABLES.touch.noise, a).toFixed(3)}`;
  // proprioception
  const p = frameOf(ep, "proprio", t, 4), n = interp(TABLES.proprio.noise, a), gp = rng(3000 + t), cP = $("#cProprio"), c2 = cP.getContext("2d");
  c2.clearRect(0, 0, 64, 64);
  c2.strokeStyle = css("--muted"); c2.lineWidth = 1; c2.beginPath(); c2.arc(p[0] * 62 + 1, p[1] * 62 + 1, 4, 0, 7); c2.stroke();
  c2.fillStyle = css("--c-proprio") || "#eda100"; c2.beginPath(); c2.arc((p[0] + n * gp()) * 62 + 1, (p[1] + n * gp()) * 62 + 1, 2.5, 0, 7); c2.fill();
  $("#pProprio").innerHTML = `position noise ${n.toFixed(3)}<br>(fraction of the scene)`;
}
function initSenses() {
  if (!has("#age")) return;
  const q = new URLSearchParams(location.search);
  if (q.has("age")) { senseAge = +q.get("age"); $("#age").value = senseAge; }
  $("#age").oninput = (e) => { senseAge = +e.target.value; drawSenses(); };
  $("#frame").oninput = (e) => { senseT = +e.target.value; drawSenses(); };
  $("#playSenses").onclick = () => {
    if (sensesTimer) { clearInterval(sensesTimer); sensesTimer = null; $("#playSenses").textContent = "▶"; return; }
    $("#playSenses").textContent = "❚❚";
    sensesTimer = setInterval(() => { senseT = (senseT + 1) % 20; $("#frame").value = senseT; drawSenses(); }, 180);
  };
}

// ------------------------------------------------------------------ 2. models
let curRegime = "curriculum_only", curCk = 12;
const spriteCache = {};
function drawModel() {
  if (!has("#sp0")) return;
  const name = `${curRegime}_${curCk}`, img = spriteCache[name] || Object.assign(new Image(), { src: `data/models/${name}.png` });
  spriteCache[name] = img;
  const paint = () => {
    for (let r = 0; r < 3; r++) {
      const c = $(`#sp${r}`); c.width = img.width; c.height = 32;
      c.getContext("2d").drawImage(img, 0, r * 33, img.width, 32, 0, 0, img.width, 32);
      c.style.width = `${img.width * 2}px`; c.style.height = "64px";
    }
  };
  img.complete && img.naturalWidth ? paint() : (img.onload = paint);
  const m = MO && MO[curRegime] && MO[curRegime][curCk];
  if (m) $("#modelInfo").textContent = `checkpoint ${curCk} months · trained on ${m.input_age}-month input · ${m.params.toLocaleString()} parameters · width ×${m.width} · memory ${m.window} frame${m.window > 1 ? "s" : ""}`;
  $("#regimeNote").textContent = RNOTE[curRegime];
  document.querySelectorAll("#ckpt button").forEach((b) => b.classList.toggle("on", +b.dataset.v === curCk));
}
function seg(el, items, cur, on) {
  el.innerHTML = "";
  items.forEach(([v, l]) => { const b = document.createElement("button"); b.textContent = l; b.dataset.v = v; if (String(v) === String(cur)) b.classList.add("on");
    b.onclick = () => { el.querySelectorAll("button").forEach((x) => x.classList.remove("on")); b.classList.add("on"); on(v); }; el.appendChild(b); });
}
function initModels() {
  if (!has("#regime")) return;
  const s = $("#regime");
  REGIMES.forEach((r) => s.add(new Option(RLABEL[r], r)));
  s.onchange = () => { curRegime = s.value; drawModel(); };
  seg($("#ckpt"), STAGES.map((a) => [a, `${a} mo`]), curCk, (v) => { curCk = +v; drawModel(); });
  drawModel();
}
function regimeTable() {
  if (!has("#regimeTable")) return;
  const v = (sense, r, k) => { const rows = RES && RES.senses[sense] && RES.senses[sense][r]; const x = rows && rows[3][k]; return x ? `${x.mean.toFixed(2)}${x.n > 1 ? ` ± ${x.sd.toFixed(2)}` : ""}` : "–"; };
  let h = "<tr><th>Regime</th><th>Network</th><th>Input quality over the four stages</th><th>Vision position R² at 12 mo</th><th>On newborn input</th><th>Hearing position R²</th></tr>";
  REGIMES.forEach((r) => { h += `<tr><td><span class="dot" style="background:var(${RCOLOR[r]})"></span> ${RLABEL[r]}</td><td>${RTABLE[r][0]}</td><td>${RTABLE[r][1]}</td><td>${v("vision", r, "pos_r2")}</td><td>${v("vision", r, "newborn_pos_r2")}</td><td>${v("audio", r, "pos_r2")}</td></tr>`; });
  $("#regimeTable").innerHTML = h;
}

// ------------------------------------------------------------------ charts
function showTip(e, html) { const tip = $("#tip"); if (!tip) return; tip.innerHTML = html; tip.style.opacity = 1; tip.style.left = `${Math.min(innerWidth - 270, e.clientX + 12)}px`; tip.style.top = `${e.clientY + 12}px`; }
function hideTip() { const tip = $("#tip"); if (tip) tip.style.opacity = 0; }
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent && parent.appendChild(e); return e; }
function lineChart(host, { series, xs, xLabel, yLabel, yDomain, height = 300, xFmt = (x) => x, band }) {
  host.innerHTML = "";
  const W = Math.max(300, host.clientWidth || 640), H = height, m = { l: 46, r: 14, t: 12, b: 38 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" }, host);
  const vals = series.flatMap((s) => s.points.flatMap((p) => [p.y - (p.sd || 0), p.y + (p.sd || 0)])).filter(Number.isFinite);
  let [y0, y1] = yDomain || [Math.min(0, ...vals), Math.max(...vals)];
  if (!yDomain) { y0 = Math.min(y0, ...vals); y1 = Math.max(y1, ...vals); }
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const X = (x) => m.l + ((x - xMin) / (xMax - xMin || 1)) * (W - m.l - m.r), Y = (y) => H - m.b - ((y - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  if (band) el("rect", { x: X(band[0]), y: m.t, width: X(band[1]) - X(band[0]), height: H - m.t - m.b, fill: css("--surface-2") }, svg);
  const g = el("g", { class: "grid" }, svg);
  const step = niceStep(y1 - y0);
  for (let v = Math.ceil(y0 / step) * step; v <= y1 + 1e-9; v += step) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g);
    el("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, svg).textContent = +v.toFixed(3);
  }
  if (y0 < 0 && y1 > 0) el("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: css("--faint"), "stroke-width": 1 }, svg);
  xs.forEach((x) => { el("text", { x: X(x), y: H - m.b + 16, "text-anchor": "middle" }, svg).textContent = xFmt(x); });
  if (xLabel) el("text", { x: (W + m.l) / 2, y: H - 4, "text-anchor": "middle" }, svg).textContent = xLabel;
  if (yLabel) el("text", { x: 12, y: (H - m.b + m.t) / 2, transform: `rotate(-90 12 ${(H - m.b + m.t) / 2})`, "text-anchor": "middle" }, svg).textContent = yLabel;
  series.forEach((s) => {
    if (s.hidden) return;
    const pts = s.points.filter((p) => Number.isFinite(p.y));
    if (pts.some((p) => p.sd)) {
      const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(p.y + (p.sd || 0))}`).join("") + pts.slice().reverse().map((p) => `L${X(p.x)},${Y(p.y - (p.sd || 0))}`).join("") + "Z";
      el("path", { d, fill: s.color, opacity: 0.13 }, svg);
    }
    el("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(p.y)}`).join(""), fill: "none", stroke: s.color, "stroke-width": s.width || 2.2, "stroke-dasharray": s.dash || "" }, svg);
    if (!s.noDots) pts.forEach((p) => {
      const c = el("circle", { cx: X(p.x), cy: Y(p.y), r: 4, fill: s.color, stroke: css("--surface"), "stroke-width": 1.5 }, svg);
      c.onmousemove = (e) => showTip(e, `<b>${s.name}</b><br>${xFmt(p.x)}: ${p.y.toFixed(3)}${p.sd ? ` ± ${p.sd.toFixed(3)}` : ""}${p.n ? ` (n=${p.n})` : ""}`);
      c.onmouseleave = hideTip;
    });
  });
}
function niceStep(range) { const r = range / 5, p = 10 ** Math.floor(Math.log10(r)), f = r / p; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p; }
function barChart(host, { groups, keys, colors, labels, yDomain, yLabel, height = 260 }) {
  host.innerHTML = "";
  const W = Math.max(300, host.clientWidth || 640), H = height, m = { l: 46, r: 10, t: 12, b: 40 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" }, host);
  const vals = groups.flatMap((g) => keys.map((k) => g.v[k]).filter(Boolean).flatMap((x) => [x.mean - x.sd, x.mean + x.sd]));
  let [y0, y1] = yDomain || [Math.min(0, ...vals), Math.max(0, ...vals)];
  const Y = (y) => H - m.b - ((y - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const step = niceStep(y1 - y0), gg = el("g", { class: "grid" }, svg);
  for (let v = Math.ceil(y0 / step) * step; v <= y1 + 1e-9; v += step) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, gg);
    el("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, svg).textContent = +v.toFixed(3);
  }
  el("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: css("--faint") }, svg);
  const gw = (W - m.l - m.r) / groups.length, bw = Math.min(46, (gw * 0.8) / keys.length);
  groups.forEach((g, gi) => {
    const x0 = m.l + gi * gw + (gw - bw * keys.length) / 2;
    el("text", { x: m.l + gi * gw + gw / 2, y: H - m.b + 18, "text-anchor": "middle" }, svg).textContent = g.label;
    keys.forEach((k, ki) => {
      const x = g.v[k]; if (!x) return;
      const r = el("rect", { x: x0 + ki * bw + 1, width: bw - 2, y: Math.min(Y(x.mean), Y(0)), height: Math.abs(Y(x.mean) - Y(0)), fill: colors[k], rx: 2 }, svg);
      if (x.sd) el("line", { x1: x0 + ki * bw + bw / 2, x2: x0 + ki * bw + bw / 2, y1: Y(x.mean - x.sd), y2: Y(x.mean + x.sd), stroke: css("--text"), "stroke-width": 1.2 }, svg);
      r.onmousemove = (e) => showTip(e, `<b>${labels[k]}</b> · ${g.label}<br>${x.mean.toFixed(3)}${x.sd ? ` ± ${x.sd.toFixed(3)}` : ""}${x.n ? ` (n=${x.n})` : ""}`);
      r.onmouseleave = hideTip;
    });
  });
  if (yLabel) el("text", { x: 12, y: (H - m.b + m.t) / 2, transform: `rotate(-90 12 ${(H - m.b + m.t) / 2})`, "text-anchor": "middle" }, svg).textContent = yLabel;
}
function legend(host, items, hidden, onToggle) {
  host.innerHTML = "";
  items.forEach(([k, label, color, dash]) => {
    const s = document.createElement("span"); s.innerHTML = `<i style="background:${color};${dash ? "height:2px;background:repeating-linear-gradient(90deg," + color + " 0 4px,transparent 4px 7px)" : ""}"></i>${label}`;
    if (hidden && hidden.has(k)) s.classList.add("off");
    if (onToggle) s.onclick = () => onToggle(k); host.appendChild(s);
  });
}
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const sdv = (a) => a.length > 1 ? Math.sqrt(mean(a.map((x) => (x - mean(a)) ** 2))) : 0;
const agg = (a) => a.length ? { mean: mean(a), sd: sdv(a), n: a.length } : null;

// ------------------------------------------------------------------ 3. trajectories
let tSense = "vision", tMetric = "pos_r2";
const hiddenR = new Set();
const METRICS = { pos_r2: "Position R²", type_acc: "Identity accuracy", newborn_pos_r2: "Position R² on newborn input", own_pos_r2: "Position R², own-age input" };
function drawTraj() {
  if (!RES || !has("#trajChart")) return;
  const rows = RES.senses[tSense] || {}, series = [];
  const own = RES.extra && RES.extra.own_age;
  REGIMES.forEach((r) => {
    let pts;
    if (tMetric === "own_pos_r2") {
      const o = own && own[r] && own[r][tSense]; if (!o) return;
      pts = STAGES.map((a) => { const x = agg((o[a] || []).map((d) => d.pos_r2)); return x ? { x: a, y: x.mean, sd: x.sd, n: x.n } : { x: a, y: NaN }; });
    } else {
      if (!rows[r]) return;
      pts = rows[r].map((row) => { const x = row[tMetric]; return x ? { x: row.age, y: x.mean, sd: x.sd, n: x.n } : { x: row.age, y: NaN }; });
    }
    series.push({ name: RLABEL[r], color: css(RCOLOR[r]), points: pts, hidden: hiddenR.has(r) });
  });
  const ceil = RES.extra && RES.extra.ceiling && RES.extra.ceiling[tSense];
  const cm = tMetric === "type_acc" ? "type_acc" : "pos_r2";
  if (ceil && tMetric !== "newborn_pos_r2") series.unshift({ name: "Information ceiling", color: css("--faint"), width: 1.6, dash: "5 4", noDots: false,
    points: Object.keys(ceil).map(Number).sort((a, b) => a - b).map((a) => ({ x: a, y: ceil[a][cm] })) });
  const xs = [0, 1, 2, 3, 4, 6, 9, 12].filter((a) => series.some((s) => s.points.some((p) => p.x === a)));
  lineChart($("#trajChart"), { series, xs, xLabel: "checkpoint (months)", yLabel: METRICS[tMetric], xFmt: (x) => `${x}`,
    yDomain: tMetric === "type_acc" ? [0, 1] : undefined });
  legend($("#trajLegend"), [...(ceil && tMetric !== "newborn_pos_r2" ? [["ceil", "Information ceiling", css("--faint"), true]] : []),
    ...REGIMES.map((r) => [r, RLABEL[r], css(RCOLOR[r])])], hiddenR, (k) => { if (k === "ceil") return; hiddenR.has(k) ? hiddenR.delete(k) : hiddenR.add(k); drawTraj(); });
  const notes = { vision: "Vision changes most with age, because its front end changes most.", audio: "Hearing localises by comparing the two ears; that difference matures over the year.",
    touch: "Touch is probed on frames where the hand touches the toy. Identity is hard here because the hand rarely covers the whole toy.", proprio: "The body sense is near its ceiling from birth in every regime." };
  $("#trajNote").textContent = notes[tSense] + (tMetric === "type_acc" ? " Chance is 0.25." : "") + (tMetric === "own_pos_r2" ? " Own-age: the probe is fitted and tested on input of the age that checkpoint stands for." : "");
}
function initTraj() {
  if (!has("#senseSeg")) return;
  seg($("#senseSeg"), Object.entries(SENSES), tSense, (v) => { tSense = v; drawTraj(); });
  seg($("#metricSeg"), Object.entries(METRICS), tMetric, (v) => { tMetric = v; drawTraj(); });
}

// ------------------------------------------------------------------ 4. VoE
let voeT = 0, voeTimer = null, voeRegime = "curriculum_only", voeCk = 12, ev = "vanish_all", fvAge = "12";
function drawVoeFrames() {
  if (!EP || !has("#vPos")) return;
  const p = visFrame(EP.voe_possible, voeT), q = visFrame(EP.voe_impossible, voeT), d = new Float32Array(3072);
  for (let i = 0; i < 3072; i++) d[i] = Math.min(1, Math.abs(p[i] - q[i]) * 2);
  putRGB($("#vPos"), p); putRGB($("#vImp"), q); putRGB($("#vDiff"), d);
}
function drawVoeTrace() {
  if (!has("#voeTrace") || !MO || !EP) return;
  const m = MO && MO[voeRegime] && MO[voeRegime][voeCk]; if (!m) return;
  const vis = EP.voe_possible.visible, hid = vis.map((v, i) => (v < 0.5 ? i : -1)).filter((i) => i >= 0);
  lineChart($("#voeTrace"), { height: 220, xs: [...Array(20).keys()].filter((i) => i % 2 === 0), xLabel: "frame", yLabel: "surprise (MSE)",
    band: hid.length ? [hid[0], hid[hid.length - 1]] : null,
    series: [{ name: "Possible", color: css("--c-vision"), points: m.possible.map((y, x) => ({ x, y })).slice(1), noDots: false },
      { name: "Impossible", color: css("--c-adult"), points: m.impossible.map((y, x) => ({ x, y })).slice(1) }] });
}
const EVENTS = { vanish_all: "Toy vanishes from every sense", vanish_visual: "Vanishes from sight, keeps sounding", silent_toy: "Silent toy vanishes" };
const ENOTE = {
  vanish_all: "The original test. The fused model is surprised, but the bars show it is surprised through hearing: the sound stopped.",
  vanish_visual: "The dissociating test the review asked for. The toy disappears from sight and touch but its sound continues behind the screen. Does any channel still notice?",
  silent_toy: "A toy that never makes a sound, so hearing cannot give the game away. Only vision could notice.",
};
function drawFusedVoe() {
  if (!has("#fusedVoe")) return;
  const fv = RES && RES.fusion_voe; const s0 = fv && Object.values(fv)[0];
  if (!fv || !s0 || !Object.values(s0)[0] || !Object.values(s0)[0].vanish_all) { $("#fusedVoe").innerHTML = "<p class='note'>Results pending.</p>"; return; }
  const seeds = Object.keys(fv), chans = ["vision", "audio", "touch", "proprio", "all"];
  const v = {}; chans.forEach((c) => { v[c] = agg(seeds.map((s) => fv[s][fvAge] && fv[s][fvAge][ev] && fv[s][fvAge][ev][c] && fv[s][fvAge][ev][c].voe_index).filter(Number.isFinite)); });
  const colors = { vision: css("--c-vision"), audio: css("--c-audio"), touch: css("--c-touch"), proprio: css("--c-proprio"), all: css("--c-fused") };
  const labels = { vision: "Vision channel", audio: "Hearing channel", touch: "Touch channel", proprio: "Body channel", all: "All channels" };
  barChart($("#fusedVoe"), { groups: chans.map((c) => ({ label: { vision: "Vision", audio: "Hearing", touch: "Touch", proprio: "Body", all: "All" }[c], v: { x: v[c] } })),
    keys: ["x"], colors: { x: null }, labels: { x: "VoE index" }, yDomain: [-0.4, 0.8], yLabel: "VoE index (+ = expects toy back)" });
  // recolour bars per channel
  $("#fusedVoe").querySelectorAll("rect").forEach((r, i) => r.setAttribute("fill", colors[chans[i]]));
  $("#eventNote").textContent = ENOTE[ev];
}
function initVoe() {
  if (!has("#voeFrame")) return;
  $("#voeFrame").oninput = (e) => { voeT = +e.target.value; drawVoeFrames(); };
  $("#playVoe").onclick = () => {
    if (voeTimer) { clearInterval(voeTimer); voeTimer = null; $("#playVoe").textContent = "▶"; return; }
    $("#playVoe").textContent = "❚❚"; voeTimer = setInterval(() => { voeT = (voeT + 1) % 20; $("#voeFrame").value = voeT; drawVoeFrames(); }, 220);
  };
  const s = $("#voeRegime"); REGIMES.forEach((r) => s.add(new Option(RLABEL[r], r))); s.onchange = () => { voeRegime = s.value; drawVoeTrace(); };
  seg($("#voeCkpt"), STAGES.map((a) => [a, `${a} mo`]), voeCk, (v) => { voeCk = +v; drawVoeTrace(); });
  seg($("#eventSeg"), Object.entries(EVENTS), ev, (v) => { ev = v; drawFusedVoe(); });
  seg($("#fvAge"), STAGES.map((a) => [String(a), `${a} mo`]), fvAge, (v) => { fvAge = v; drawFusedVoe(); });
}

// ------------------------------------------------------------------ 5. fusion
function drawFusion() {
  const f = RES && RES.fusion; if (!f || !has("#trackChart")) return;
  const ages = Object.keys(f).map(Number).sort((a, b) => a - b);
  const mk = (grp, key, metric, name, color, dash) => ({ name, color, dash, points: ages.map((a) => { const x = f[a][grp][key] && f[a][grp][key][metric]; return { x: a, y: x ? x.mean : NaN, sd: x ? x.sd : 0, n: x ? x.n : 0 }; }) });
  const tr = [mk("occluded_tracking", "fused", "x_r2", "Fused world model", css("--c-fused")), mk("occluded_tracking", "vision_memory", "x_r2", "Vision memory", css("--c-vision")),
    mk("occluded_tracking", "audio_only", "x_r2", "Hearing alone", css("--c-audio"))];
  lineChart($("#trackChart"), { series: tr, xs: ages, xLabel: "checkpoint (months)", yLabel: "hidden-toy x position R²", height: 260 });
  legend($("#trackLegend"), tr.map((s) => [s.name, s.name, s.color]));
  const fi = [mk("fill_in", "fused_all", "pos_r2", "Fused, all senses", css("--c-fused")), mk("fill_in", "vision_only", "pos_r2", "Vision alone", css("--c-vision")),
    mk("fill_in", "fused_no_vision", "pos_r2", "Fused, vision switched off", css("--c-touch"), "5 4")];
  lineChart($("#fillChart"), { series: fi, xs: ages, xLabel: "checkpoint (months)", yLabel: "toy position R²", height: 260 });
  legend($("#fillLegend"), fi.map((s) => [s.name, s.name, s.color, s.dash]));
}

// ------------------------------------------------------------------ verdicts

// ------------------------------------------------------------------ live numbers (results.json -> text)
const fx = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "–");
const cell = (x, d = 2, pm = true) => (x && Number.isFinite(x.mean) ? `${x.mean.toFixed(d)}${pm && x.n > 1 ? ` ± ${x.sd.toFixed(d)}` : ""}` : "–");
const SK = (s, r, k, st = 3) => RES && RES.senses && RES.senses[s] && RES.senses[s][r] && RES.senses[s][r][st] && RES.senses[s][r][st][k];
function liveValue(spec) { // "m|regime|sense|key|stage" | "ms|..." (mean ± sd) | "f|group|key|metric" | "ce|sense|key|age" | "cp|sense|field" | "st|name|field" | "n|regime|sense"
  const [t, ...a] = spec.split("|");
  if (!RES) return "–";
  if (t === "m") return cell(SK(a[1], a[0], a[2], +(a[3] ?? 3)), 2, false);
  if (t === "ms") return cell(SK(a[1], a[0], a[2], +(a[3] ?? 3)), 2, true);
  if (t === "n") { const x = SK(a[1], a[0], "pos_r2", 3); return x ? String(x.n) : "–"; }
  if (t === "f") { const g = RES.fusion && RES.fusion["12"] && RES.fusion["12"][a[0]]; return cell(g && g[a[1]] && g[a[1]][a[2]], 2, false); }
  if (t === "ce") { const c = RES.extra && RES.extra.ceiling && RES.extra.ceiling[a[0]]; return c && c[a[2]] ? fx(c[a[2]][a[1]]) : "–"; }
  if (t === "cp") { const c = RES.extra && RES.extra.compute && RES.extra.compute[a[0]]; return c ? fx(c[a[1]], 2) : "–"; }
  if (t === "st") { const c = RES.extra && RES.extra.stats && RES.extra.stats.comparisons && RES.extra.stats.comparisons[a[0]]; if (!c) return "–"; const v = a[1] === "g" ? (c.hedges_g ?? c.g) : c[a[1]]; return a[1] === "p" ? (Number.isFinite(v) ? (v < 0.001 ? "<0.001" : v.toFixed(3)) : "–") : fx(v); }
  return "–";
}
function fillLive() { document.querySelectorAll("[data-live]").forEach((e) => { e.textContent = liveValue(e.dataset.live); }); }

function drawVerdicts() { // numeric sentences on the home page cards
  if (!has("[data-card]")) return;
  const V = (s, r, k, a = 3) => { const x = SK(s, r, k, a); return x ? x.mean.toFixed(2) : "–"; };
  const ft = RES && RES.fusion && RES.fusion["12"] && RES.fusion["12"].occluded_tracking;
  const f = (k) => (ft && ft[k] ? ft[k].x_r2.mean.toFixed(2) : "–");
  const t = {
    1: `Vision position R² ${V("vision", "curriculum_only", "pos_r2", 0)} at the newborn checkpoint and ${V("vision", "curriculum_only", "pos_r2")} at 12 months; touch and proprioception are close to their final level from the first checkpoint. The order is set by the input filters, not discovered.`,
    2: `Vision at 12 months: ${V("vision", "curriculum_only", "pos_r2")} with immature input versus ${V("vision", "adult_scratch", "pos_r2")} with adult input. Hearing: ${V("audio", "curriculum_only", "pos_r2")} versus ${V("audio", "adult_scratch", "pos_r2")}. Growing the network on top of immature input slowed learning.`,
    3: "No vision model had a positive violation-of-expectation index. The fused model did, but the effect was carried by the hearing channel and was absent for a silent object.",
    4: `Hidden-object position R² at 12 months: fused ${f("fused")}, vision memory ${f("vision_memory")}, hearing alone ${f("audio_only")} (position decoded from the model state).`,
  };
  document.querySelectorAll("[data-card]").forEach((e) => { e.textContent = t[e.dataset.card] || ""; });
}

// ------------------------------------------------------------------ results page tables
function td(v, cls) { return `<td${cls ? ` class="${cls}"` : ""}>${v}</td>`; }
function drawResultsTables() {
  if (!RES) return;
  const sensesList = Object.entries(SENSES);
  const T = $("#trajTables");
  if (T) {
    T.innerHTML = sensesList.map(([s, label]) => {
      const head = `<tr><th>Regime</th>${STAGES.map((a) => `<th class="num">${a} mo</th>`).join("")}<th class="num">identity acc. 12 mo</th></tr>`;
      const rows = REGIMES.map((r) => RES.senses[s] && RES.senses[s][r] ? `<tr><td><span class="dot" style="background:var(${RCOLOR[r]})"></span> ${RLABEL[r]}</td>${STAGES.map((a, i) => td(cell(SK(s, r, "pos_r2", i), 2, true), "num")).join("")}${td(cell(SK(s, r, "type_acc", 3), 2, true), "num")}</tr>` : "").join("");
      return `<div class="tablewrap"><table class="t"><caption>${label}: toy position R² (probe on adult-quality input; mean ± s.d. over seeds) at each checkpoint</caption>${head}${rows}</table></div>`;
    }).join("");
  }
  const S = $("#statsTable");
  if (S) {
    const c = RES.extra && RES.extra.stats && RES.extra.stats.comparisons;
    if (!c || !Object.keys(c).length) S.innerHTML = "<tr><td>Planned-comparison statistics are generated with the paper and are not yet available in this build of the site.</td></tr>";
    else {
      let h = "<tr><th>Comparison (12-month checkpoint)</th><th class=\"num\">Hedges' g</th><th class=\"num\">approx. 95% CI</th><th class=\"num\">p (Welch)</th><th class=\"num\">n per group</th></tr>";
      for (const [name, d] of Object.entries(c)) {
        const g = d.hedges_g ?? d.g, na = Array.isArray(d.a) ? d.a.length : (d.n && d.n[0]) ?? d.n, nb = Array.isArray(d.b) ? d.b.length : (d.n && d.n[1]) ?? d.n;
        let ci = "–";
        if (Number.isFinite(g) && na && nb) { const se = Math.sqrt((na + nb) / (na * nb) + g * g / (2 * (na + nb))); ci = `[${(g - 1.96 * se).toFixed(2)}, ${(g + 1.96 * se).toFixed(2)}]`; }
        h += `<tr>${td(name)}${td(fx(g), "num")}${td(ci, "num")}${td(Number.isFinite(d.p) ? (d.p < 0.001 ? "<0.001" : d.p.toFixed(3)) : "–", "num")}${td(na && nb ? `${na} / ${nb}` : "–", "num")}</tr>`;
      }
      S.innerHTML = h;
    }
  }
  const F = $("#fusionTable");
  if (F) {
    const f = RES.fusion && RES.fusion["12"];
    if (!f) F.innerHTML = "<tr><td>Fusion results are not available in this build.</td></tr>";
    else {
      const rows = [["Hidden toy, x position R²", "occluded_tracking", "x_r2", [["fused", "Fused"], ["vision_memory", "Vision memory"], ["audio_only", "Hearing alone"]]],
        ["Visible toy, position R²", "fill_in", "pos_r2", [["fused_all", "Fused, all senses"], ["vision_only", "Vision alone"], ["fused_no_vision", "Fused, vision off"]]]];
      let h = "<tr><th>Quantity (12 mo)</th><th>Readout</th><th class=\"num\">mean ± s.d.</th><th class=\"num\">n seeds</th></tr>";
      rows.forEach(([q, g, m, keys]) => keys.forEach(([k, l], i) => { const x = f[g] && f[g][k] && f[g][k][m]; h += `<tr>${td(i ? "" : q)}${td(l)}${td(cell(x), "num")}${td(x ? x.n : "–", "num")}</tr>`; }));
      F.innerHTML = h;
    }
  }
  const Vt = $("#voeTable");
  if (Vt) {
    const fv = RES.fusion_voe, seeds = fv ? Object.keys(fv) : [];
    if (!seeds.length || !fv[seeds[0]]["12"]) Vt.innerHTML = "<tr><td>Fused violation-of-expectation results are not available in this build.</td></tr>";
    else {
      const chans = ["vision", "audio", "touch", "proprio", "all"];
      let h = `<tr><th>Event (12 mo)</th>${chans.map((c) => `<th class="num">${c === "all" ? "All channels" : c === "audio" ? "Hearing" : c === "proprio" ? "Body" : c[0].toUpperCase() + c.slice(1)}</th>`).join("")}</tr>`;
      Object.entries(EVENTS).forEach(([e, l]) => {
        h += `<tr>${td(l)}${chans.map((c) => td(cell(agg(seeds.map((s) => fv[s]["12"] && fv[s]["12"][e] && fv[s]["12"][e][c] && fv[s]["12"][e][c].voe_index).filter(Number.isFinite))), "num")).join("")}</tr>`;
      });
      Vt.innerHTML = h;
    }
  }
  const C = $("#computeTable");
  if (C) {
    const c = RES.extra && RES.extra.compute;
    if (!c) C.innerHTML = "<tr><td>Compute accounting is not available in this build.</td></tr>";
    else C.innerHTML = "<tr><th>Sense</th><th class=\"num\">Parameters by stage (grown)</th><th class=\"num\">Fixed size</th><th class=\"num\">Updates × params, grown : fixed</th></tr>" +
      Object.entries(c).map(([s, d]) => `<tr>${td(SENSES[s] || s)}${td((d.params_by_stage_grown || []).map((x) => x.toLocaleString()).join(" → "), "num")}${td(d.params_fixed ? d.params_fixed.toLocaleString() : "–", "num")}${td(fx(d.ratio_grown_to_fixed), "num")}</tr>`).join("");
  }
  const M = $("#milestoneTable");
  if (M) {
    const ms = RES.extra && RES.extra.milestones;
    if (!ms) M.innerHTML = "<tr><td>Milestone analysis is generated with the paper and is not yet available in this build.</td></tr>";
    else {
      const flat = (o) => (o && typeof o === "object" ? Object.entries(o).map(([k, v]) => `${k}: ${flat(v)}`).join("; ") : typeof o === "number" ? +o.toFixed(2) : String(o));
      M.innerHTML = "<tr><th>Quantity</th><th>Value</th></tr>" + Object.entries(ms).map(([k, v]) => `<tr>${td(k.replace(/_/g, " "))}${td(flat(v))}</tr>`).join("");
    }
  }
}

// ------------------------------------------------------------------ model zoo (wiki/models.html)
const ZOO_PARAMS = { vision: [43507, 152995, 334675, 586499], audio: [13792, 41824, 87264, 149088], touch: [13792, 41824, 87264, 149088], proprio: [8340, 29972, 66452, 117268] };
const ZOO_LATENT = { vision: 32, audio: 16, touch: 16, proprio: 8 };
const ZOO_CAP = [[1, 1], [2, 2], [3, 4], [4, 6]];
const ZOO_AGE = { curriculum_only: [0, 3, 6, 12], grown: [0, 3, 6, 12], growth_only: [12, 12, 12, 12], adult_scratch: [12, 12, 12, 12], shuffled_order: ["random", "random", "random", "random"], reversed_immaturity: [12, 6, 3, 0] };
const ZOO_PUBLISHED = new Set(["curriculum_only", "adult_scratch", "grown"]);
const ZOO_GROWS = new Set(["grown", "growth_only"]);
const REPO = "https://github.com/pikulsomesh/developmental-world-models";
function drawZoo() {
  const host = $("#zooTable"); if (!host) return;
  const fs = $("#zooSense").value, fr = $("#zooRegime").value;
  let h = "<tr><th>Regime</th><th>Sense</th><th class=\"num\">Stage (months)</th><th class=\"num\">Input age</th><th class=\"num\">Width</th><th class=\"num\">Memory</th><th class=\"num\">Latent</th><th class=\"num\">Parameters</th><th>Checkpoint (seed 0)</th></tr>";
  let n = 0;
  REGIMES.forEach((r) => Object.keys(ZOO_PARAMS).forEach((s) => {
    if ((fs !== "all" && fs !== s) || (fr !== "all" && fr !== r)) return;
    STAGES.forEach((a, i) => {
      const grows = ZOO_GROWS.has(r), k = grows ? i : 3, file = `${s}_${r}_stage${i}.pt`;
      const ck = ZOO_PUBLISHED.has(r) ? `<a href="${REPO}/tree/main/models/${r}/seed0"><code>models/${r}/seed0/${file}</code></a>` : "<span class=\"note\">not distributed</span>";
      h += `<tr>${td(RLABEL[r])}${td(SENSES[s])}${td(a, "num")}${td(ZOO_AGE[r][i], "num")}${td(`×${ZOO_CAP[k][0]}`, "num")}${td(ZOO_CAP[k][1], "num")}${td(ZOO_LATENT[s], "num")}${td(ZOO_PARAMS[s][k].toLocaleString(), "num")}${td(ck)}</tr>`; n++;
    });
  }));
  host.innerHTML = h; const c = $("#zooCount"); if (c) c.textContent = `${n} checkpoints shown`;
}
function initZoo() {
  if (!has("#zooTable")) return;
  const rs = $("#zooRegime"), ss = $("#zooSense");
  rs.add(new Option("All regimes", "all")); REGIMES.forEach((r) => rs.add(new Option(RLABEL[r], r)));
  ss.add(new Option("All senses", "all")); Object.entries(SENSES).forEach(([k, v]) => ss.add(new Option(v, k)));
  rs.onchange = ss.onchange = drawZoo; drawZoo();
}

// ------------------------------------------------------------------ references (wiki/references.html)
function initRefs() {
  const host = $("#refList"); if (!host) return;
  fetch("../data/references.json").then((r) => (r.ok ? r.json() : Promise.reject())).then((refs) => {
    if (!Array.isArray(refs) || !refs.length) throw 0;
    host.innerHTML = "<ol>" + refs.map((r) => {
      const d = r.doi ? (/^https?:/.test(r.doi) ? r.doi : `https://doi.org/${r.doi}`) : "";
      return `<li id="ref-${String(r.key).replace(/[^\w-]/g, "")}">${String(r.text).replace(/[<>&]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c]))}${d ? ` <a href="${d}">doi</a>` : ""}</li>`;
    }).join("") + "</ol>";
  }).catch(() => { host.innerHTML = "<p>The reference list is generated with the paper. See the <a href=\"../paper/report.pdf\">PDF</a> or <a href=\"" + REPO + "/blob/main/report/refs.bib\">refs.bib</a>.</p>"; });
}

function redrawAll() { drawSenses(); drawTraj(); drawVoeTrace(); drawFusedVoe(); drawFusion(); }

// ------------------------------------------------------------------ boot (runs after site.js has built the page chrome)
function boot() {
  renderMath();
  initSenses(); initModels(); initTraj(); initVoe(); initZoo(); initRefs();
  const needEP = has("#cVision") || has("#vPos"), needMO = has("#sp0") || has("#voeTrace"),
    needRES = has("#verdicts, [data-card], [data-live], #regimeTable, #trajChart, #fusedVoe, #trackChart, #trajTables, #statsTable, #fusionTable, #voeTable, #computeTable, #milestoneTable");
  const get = (need, u) => (need ? fetch(u).then((r) => (r.ok ? r.json() : null)).catch(() => null) : Promise.resolve(undefined));
  const R = document.body.dataset.root || "";
  Promise.all([get(needEP, R + "data/episodes.json"), get(needMO, R + "data/model_outputs.json"), get(needRES, R + "data/results.json")])
    .then(([eps, mo, res]) => {
      if ((needEP && !eps) || (needMO && !mo) || (needRES && !res)) {
        const s = $("#dataStatus"); if (s) s.textContent = "Some data files could not be loaded. Serve this folder over HTTP (for example python3 -m http.server) rather than opening the file directly.";
      }
      if (eps) {
        TABLES = eps.tables; EP = {};
        for (const [k, e] of Object.entries(eps.episodes)) EP[k] = { ...e, vision: b64(e.vision), audio: b64(e.audio, "f32"), touch: b64(e.touch, "f32"), proprio: b64(e.proprio, "f32") };
      }
      MO = mo || null; RES = res || null;
      fillLive(); drawVerdicts(); regimeTable(); drawModel(); redrawAll(); drawVoeFrames(); drawResultsTables();
    });
}
document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", boot) : boot();
let rz = null, lastW = innerWidth;
window.addEventListener("resize", () => { if (innerWidth === lastW) return; lastW = innerWidth; clearTimeout(rz); rz = setTimeout(redrawAll, 150); });
})();

/* Baby Senses Lab: shared helpers and one init per page (body[data-page]).
   Everything runs in the browser. Nothing a user picks, records or uploads leaves their device. */
(() => {
"use strict";
const L = window.LAB;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const store = { get(k, d) { try { const v = localStorage.getItem("lab-" + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("lab-" + k, JSON.stringify(v)); } catch (e) {} } };

function interp(table, a) {
  const ks = Object.keys(table).map(Number).sort((x, y) => x - y);
  if (a <= ks[0]) return table[ks[0]];
  if (a >= ks[ks.length - 1]) return table[ks[ks.length - 1]];
  for (let i = 0; i < ks.length - 1; i++) if (a >= ks[i] && a <= ks[i + 1]) {
    const t = (a - ks[i]) / (ks[i + 1] - ks[i]); return table[ks[i]] * (1 - t) + table[ks[i + 1]] * t;
  }
}

// ---------------------------------------------------------------- age state
const QUICK = [0, 1, 2, 3, 4, 6, 9, 12, 18, 24];
function ageWords(a) {
  if (a < 0.25) return "Newborn";
  if (a < 1) return `${Math.round(a * 4.35)} weeks`;
  const m = Math.round(a * 2) / 2;
  const s = `${m % 1 ? m.toFixed(1) : m} month${m === 1 ? "" : "s"}`;
  return m >= 12 && m % 12 === 0 ? `${s} (${m / 12} year${m === 12 ? "" : "s"})` : s;
}
function ageAdj(a) { // "newborn", "3-week-old", "6-month-old"
  if (a < 0.25) return "newborn";
  if (a < 1) return `${Math.round(a * 4.35)}-week-old`;
  const m = Math.round(a * 2) / 2; return `${m % 1 ? m.toFixed(1) : m}-month-old`;
}
function getAge() {
  const q = new URLSearchParams(location.search).get("age");
  const a = q != null && q !== "" && !isNaN(+q) ? +q : store.get("age", 6);
  return Math.max(0, Math.min(24, a));
}
let AGE = getAge();
function setAge(a, push = true) {
  AGE = Math.max(0, Math.min(24, +a));
  store.set("age", AGE);
  if (push) { const u = new URL(location.href); u.searchParams.set("age", AGE); history.replaceState(null, "", u); }
  $$("[data-age-link]").forEach((x) => { const u = new URL(x.href); u.searchParams.set("age", AGE); x.href = u.href; });
  window.dispatchEvent(new CustomEvent("agechange", { detail: AGE }));
}
function agePicker(host, { id = "age", label = "Baby's age", onChange } = {}) {
  host.classList.add("agepick");
  host.innerHTML = `<div class="big" aria-live="polite"><small>${esc(label)}</small><span></span></div>
    <div><input type="range" min="0" max="24" step="0.5" aria-label="${esc(label)} in months" id="${id}">
    <div class="ticks" aria-hidden="true"><span>Birth</span><span>6 mo</span><span>12 mo</span><span>18 mo</span><span>24 mo</span></div></div>
    <div class="quick" role="group" aria-label="Quick ages">${QUICK.map((q) => `<button type="button" data-q="${q}">${q === 0 ? "Newborn" : q + " mo"}</button>`).join("")}</div>`;
  const r = $("input", host), big = $(".big span", host);
  const paint = (a) => { r.value = a; big.textContent = ageWords(a); $$(".quick button", host).forEach((b) => b.classList.toggle("on", +b.dataset.q === a)); };
  r.oninput = () => (onChange ? onChange(+r.value) : setAge(+r.value));
  $$(".quick button", host).forEach((b) => (b.onclick = () => (onChange ? onChange(+b.dataset.q) : setAge(+b.dataset.q))));
  return paint;
}

// ---------------------------------------------------------------- vision model
function vision(a, src = "brain") {
  const cpd = interp(L.ACUITY[src], a);
  const cs = interp(L.CONTRAST_SENS, a);
  return { cpd, snellen: Math.round(20 * 30 / cpd / 5) * 5, cs,
    contrast: Math.min(1, Math.max(0.15, Math.log10(cs) / 2)), sat: interp(L.COLOR_SAT, a) };
}
// Gaussian blur that removes detail finer than the acuity limit: the blur's modulation transfer
// falls to 10% at the cut-off frequency, sigma = sqrt(ln 10 / 2) / (pi * cpd) degrees.
function blurPx(cpd, pxPerDeg) { return (Math.sqrt(Math.log(10) / 2) / (Math.PI * cpd)) * pxPerDeg; }
function pxPerDeg(displayPx, sizeCm, distCm) { return displayPx / ((2 * Math.atan(sizeCm / 2 / distCm)) * 180 / Math.PI); }
function cssFilter(v, sigma) { return `blur(${Math.max(0, sigma).toFixed(2)}px) contrast(${v.contrast.toFixed(3)}) saturate(${v.sat.toFixed(3)})`; }
function detailMm(cpd, distCm) { return (distCm * 10 * Math.tan((1 / (2 * cpd)) * Math.PI / 180)); }

// ---------------------------------------------------------------- sample scenes (drawn here, no photos)
function drawSample(kind, W = 800, H = 600, t = 0) {
  const c = document.createElement("canvas"); c.width = W; c.height = H; const g = c.getContext("2d");
  if (kind === "face") {
    g.fillStyle = "#e9dccb"; g.fillRect(0, 0, W, H);
    g.fillStyle = "#3b2a20"; g.beginPath(); g.ellipse(W / 2, H * 0.47, 205, 245, 0, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#f1c7a4"; g.beginPath(); g.ellipse(W / 2, H * 0.53, 170, 210, 0, 0, Math.PI * 2); g.fill();
    for (const s of [-1, 1]) {
      g.fillStyle = "#fff"; g.beginPath(); g.ellipse(W / 2 + s * 68, H * 0.46, 34, 20, 0, 0, Math.PI * 2); g.fill();
      g.fillStyle = "#4b2e1a"; g.beginPath(); g.arc(W / 2 + s * 68, H * 0.46, 15, 0, Math.PI * 2); g.fill();
      g.fillStyle = "#000"; g.beginPath(); g.arc(W / 2 + s * 68, H * 0.46, 7, 0, Math.PI * 2); g.fill();
      g.strokeStyle = "#3b2a20"; g.lineWidth = 7; g.beginPath(); g.arc(W / 2 + s * 68, H * 0.42, 40, Math.PI * 1.15, Math.PI * 1.85); g.stroke();
    }
    g.strokeStyle = "#c98f6b"; g.lineWidth = 5; g.beginPath(); g.moveTo(W / 2, H * 0.5); g.lineTo(W / 2 - 14, H * 0.6); g.lineTo(W / 2 + 6, H * 0.61); g.stroke();
    g.strokeStyle = "#b0413e"; g.lineWidth = 9; g.lineCap = "round"; g.beginPath(); g.arc(W / 2, H * 0.62, 62, 0.2 * Math.PI, 0.8 * Math.PI); g.stroke();
  } else if (kind === "card") {
    g.fillStyle = "#fff"; g.fillRect(0, 0, W, H);
    for (let r = 8; r >= 1; r--) { g.fillStyle = r % 2 ? "#000" : "#fff"; g.beginPath(); g.arc(W * 0.3, H / 2, r * 28, 0, Math.PI * 2); g.fill(); }
    const s = 40; for (let y = 0; y < 6; y++) for (let x = 0; x < 6; x++) if ((x + y) % 2) { g.fillStyle = "#000"; g.fillRect(W * 0.6 + x * s, H / 2 - 3 * s + y * s, s, s); }
    for (let i = 0; i < 24; i++) { g.fillStyle = i % 2 ? "#000" : "#fff"; g.fillRect(W * 0.6 + i * 10, H * 0.84, 10, 50); }
  } else if (kind === "toys") {
    g.fillStyle = "#f4ead9"; g.fillRect(0, 0, W, H);
    g.fillStyle = "#b98d5a"; g.fillRect(0, H * 0.72, W, H * 0.28);
    const cols = ["#d62828", "#f77f00", "#fcbf49", "#2a9d8f", "#264653", "#7b2cbf"];
    cols.forEach((c, i) => { g.fillStyle = c; g.fillRect(70 + i * 70, H * 0.72 - 60 - (i % 2) * 60, 60, 60 + (i % 2) * 60); });
    g.fillStyle = "#e63946"; g.beginPath(); g.arc(W * 0.78, H * 0.62, 70, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#ffd60a"; g.beginPath(); g.arc(W * 0.78, H * 0.62, 70, -0.6, 0.6); g.lineTo(W * 0.78, H * 0.62); g.fill();
    g.fillStyle = "#1d3557"; g.font = "bold 22px Helvetica, Arial, sans-serif"; g.fillText("A B C", 82, H * 0.18); g.font = "12px Helvetica, Arial, sans-serif"; g.fillText("Small print on the toy box: ages 0+", 82, H * 0.24);
    g.fillStyle = "#2a9d8f"; g.beginPath(); g.moveTo(W * 0.55, H * 0.72); g.lineTo(W * 0.62, H * 0.5); g.lineTo(W * 0.69, H * 0.72); g.fill();
  } else if (kind === "room") {
    const sky = g.createLinearGradient(0, 0, 0, H); sky.addColorStop(0, "#cfe3f2"); sky.addColorStop(1, "#f7f2e8"); g.fillStyle = sky; g.fillRect(0, 0, W, H);
    g.fillStyle = "#8d6e63"; g.fillRect(0, H * 0.75, W, H * 0.25);
    g.fillStyle = "#fff"; g.fillRect(W * 0.62, H * 0.12, 220, 170); g.strokeStyle = "#6d4c41"; g.lineWidth = 8; g.strokeRect(W * 0.62, H * 0.12, 220, 170);
    g.beginPath(); g.moveTo(W * 0.62 + 110, H * 0.12); g.lineTo(W * 0.62 + 110, H * 0.12 + 170); g.stroke();
    g.fillStyle = "#43a047"; g.beginPath(); g.arc(W * 0.2, H * 0.55, 60, 0, Math.PI * 2); g.fill(); g.fillStyle = "#795548"; g.fillRect(W * 0.2 - 10, H * 0.55, 20, 110);
    g.fillStyle = "#5c6bc0"; g.fillRect(W * 0.35, H * 0.55, 230, 110); g.fillStyle = "#3949ab"; g.fillRect(W * 0.35, H * 0.5, 230, 30);
    // a person at the far side
    g.fillStyle = "#f1c7a4"; g.beginPath(); g.arc(W * 0.86, H * 0.5, 24, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#c62828"; g.fillRect(W * 0.86 - 26, H * 0.54, 52, 90);
  } else if (kind === "moving") { // animated scene for the video panel: a toy crossing behind a screen
    g.fillStyle = "#ece6da"; g.fillRect(0, 0, W, H); g.fillStyle = "#c8b89a"; g.fillRect(0, H * 0.7, W, H * 0.3);
    const x = W * (0.1 + 0.8 * (0.5 - 0.5 * Math.cos(t * 0.9)));
    g.fillStyle = "#d62828"; g.beginPath(); g.arc(x, H * 0.55, 50, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#fff"; g.beginPath(); g.arc(x - 16, H * 0.53, 10, 0, Math.PI * 2); g.arc(x + 16, H * 0.53, 10, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#264653"; g.fillRect(W * 0.44, H * 0.3, W * 0.12, H * 0.4);
  }
  return c;
}

// ---------------------------------------------------------------- page: age profile
function nearestRow(rows, a) { let r = rows[0]; for (const x of rows) if (x[0] <= a) r = x; return r; }
function initAge() {
  const paint = agePicker($("#picker"));
  const cmpSel = $("#compare");
  cmpSel.innerHTML = `<option value="">No comparison</option>` + QUICK.map((q) => `<option value="${q}">${q === 0 ? "Newborn" : q + " months"}</option>`).join("");
  cmpSel.value = store.get("cmp", "");
  cmpSel.onchange = () => { store.set("cmp", cmpSel.value); render(); };
  function render() {
    const a = AGE, b = cmpSel.value === "" ? null : +cmpSel.value; paint(a);
    $("#ageTitle").textContent = ageAdj(a);
    $("#domains").innerHTML = L.DOMAINS.map((d) => `<div class="card domain"><h3><span class="dot" style="background:${d.color}"></span>${esc(d.name)}</h3><dl>${d.items.map(([name, rows]) => {
      const now = nearestRow(rows, a), then = b == null ? null : nearestRow(rows, b);
      const changed = then && then !== now;
      return `<dt>${esc(name)}</dt><dd>${esc(now[1])}</dd>${changed ? `<dd class="then"><span class="new">At ${esc(ageWords(b))}</span>${esc(then[1])}</dd>` : ""}`;
    }).join("")}</dl></div>`).join("");
    const v = vision(a), vl = vision(a, "look");
    $("#numbers").innerHTML = `<tr><th scope="row">Sharpness (brain-response measures)</th><td>about 20/${v.snellen} · ${v.cpd.toFixed(1)} cycles per degree</td></tr>
      <tr><th scope="row">Sharpness (looking tests)</th><td>about 20/${vl.snellen} · ${vl.cpd.toFixed(1)} cycles per degree (normal range about half to double)</td></tr>
      <tr><th scope="row">Smallest detail at 30 cm</th><td>about ${detailMm(v.cpd, 30).toFixed(1)} to ${detailMm(vl.cpd, 30).toFixed(1)} mm</td></tr>
      <tr><th scope="row">Colour</th><td>${Math.round(v.sat * 100)}% of adult colour strength (simplified)</td></tr>
      <tr><th scope="row">Words understood</th><td>${(() => { const r = vocabRange(a); return r[1] === 0 ? "none yet" : `roughly ${r[0]}-${r[1]}`; })()}</td></tr>`;
    // phase 2: model checkpoint
    const ck = L.CHECKPOINTS.filter((c) => c <= a).pop();
    $("#modelAge").textContent = ck === 0 ? "newborn" : `${ck}-month`;
    $("#modelImg").src = `../data/models/grown_${ck}.png`;
    $("#modelText").textContent = MODEL_TEXT[ck] + (a > 12 ? " The models stop at the 12-month stage; nothing past it has been trained." : "");
    // activities
    const acts = L.ACTIVITIES.map((x) => ({ x, r: readiness(x, a) })).filter((o) => o.x.typical <= a + 0.5 && a <= o.x.full + 6).sort((p, q) => q.x.typical - p.x.typical || q.r - p.r).slice(0, 6);
    $("#acts").innerHTML = acts.length ? acts.map((o) => `<li><a href="play.html?age=${a}#${o.x.id}">${esc(o.x.name)}</a>: ${esc(o.x.description)}</li>`).join("") : "<li>See the full activity list.</li>";
    $("#printDate").textContent = new Date().toLocaleDateString();
  }
  window.addEventListener("agechange", render); render();
  $("#print").onclick = () => window.print();
}
const MODEL_TEXT = {
  0: "At the newborn stage our vision model gets very blurry input and has learned almost nothing yet: it can barely tell where the toy is. Touch and body-position models already work, because their input is nearly mature.",
  3: "By the 3-month stage the vision model's input is sharper and it starts to locate the toy, unevenly across training runs. The hearing model begins to use left-right differences.",
  6: "By the 6-month stage the vision model locates a visible toy well. Combining senses now helps find a toy that is fully hidden.",
  12: "At the 12-month stage every sense model is near its best. Combining senses finds a hidden toy better than any single sense. But no vision model is surprised when a hidden toy vanishes, unlike babies in lab studies.",
};
function vocabRange(a) { const ks = Object.keys(L.VOCAB).map(Number); let k = ks[0]; for (const x of ks) if (x <= a) k = x; return L.VOCAB[k]; }

// ---------------------------------------------------------------- readiness (activities)
function reqState(cap, need, a) { const lv = L.level(cap, a); return { lv, need, ok: lv >= need, part: lv >= need * 0.6 }; }
function readiness(act, a) { const rs = Object.entries(act.requires).map(([c, n]) => Math.min(1, L.level(c, a) / n)); return rs.reduce((s, x) => s + x, 0) / rs.length; }
function verdict(act, a) {
  const rs = Object.entries(act.requires).map(([c, n]) => reqState(c, n, a));
  if (rs.every((r) => r.ok)) return ["s-ready", "Likely ready"];
  if (rs.every((r) => r.part)) return ["s-emerging", "Emerging"];
  return ["s-notyet", "Not yet"];
}

// ---------------------------------------------------------------- page: activities and words
function initPlay() {
  const paint = agePicker($("#picker"));
  const cats = ["All", ...new Set(L.ACTIVITIES.map((x) => x.cat))];
  let cat = "All", open = location.hash.slice(1) || null;
  $("#cats").innerHTML = cats.map((c) => `<button type="button" class="chip" data-c="${c}">${c}</button>`).join("");
  $$("#cats .chip").forEach((b) => (b.onclick = () => { cat = b.dataset.c; render(); }));
  function detail(x, a) {
    return `<div class="actdetail" id="d-${x.id}"><p>${esc(x.description)}</p>
      <p class="note">Most babies enjoy this from about <b>${x.typical === 0 ? "birth" : x.typical + " months"}</b> and get the most out of it by about <b>${x.full} months</b>.</p>
      <h4>What it takes</h4>${Object.entries(x.requires).map(([c, n]) => { const r = reqState(c, n, a);
        return `<div class="req"><span>${esc(L.CAPS[c][2])}</span><span class="bar" title="Typical level at this age ${Math.round(r.lv * 100)}%, needed ${Math.round(n * 100)}%"><i style="width:${(r.lv * 100).toFixed(0)}%"></i><b style="left:${(n * 100).toFixed(0)}%"></b></span><span class="status ${r.ok ? "s-ready" : r.part ? "s-emerging" : "s-notyet"}">${r.ok ? "Typical" : r.part ? "Emerging" : "Later"}</span></div>`; }).join("")}
      <p class="note">Bar: how established this ability typically is at ${esc(ageWords(a))}. Black line: the level the activity needs.</p>
      <p><b>Good for:</b> ${esc(x.benefits.join(", "))}. <b>Tip:</b> ${esc(x.tips)}</p></div>`;
  }
  function render() {
    const a = AGE; paint(a);
    $$("#cats .chip").forEach((b) => b.classList.toggle("on", b.dataset.c === cat));
    const list = L.ACTIVITIES.filter((x) => cat === "All" || x.cat === cat).map((x) => ({ x, r: readiness(x, a), v: verdict(x, a)[0] })).sort((p, q) => { const o = { "s-ready": 0, "s-emerging": 1, "s-notyet": 2 };
      return o[p.v] - o[q.v] || (p.v === "s-notyet" ? p.x.typical - q.x.typical : q.x.typical - p.x.typical); });
    $("#acts").innerHTML = list.map(({ x, r }) => { const [cls, txt] = verdict(x, a);
      return `<div id="${x.id}"><button class="act" aria-expanded="${open === x.id}" data-id="${x.id}"><span><b>${esc(x.name)}</b><small>${esc(x.cat)} · usually from ${x.typical === 0 ? "birth" : x.typical + " mo"}</small></span><span class="bar" aria-hidden="true"><i style="width:${(r * 100).toFixed(0)}%"></i></span><span class="status ${cls}">${txt}</span></button>${open === x.id ? detail(x, a) : ""}</div>`; }).join("");
    $$("#acts .act").forEach((b) => (b.onclick = () => { open = open === b.dataset.id ? null : b.dataset.id; render(); }));
    word();
  }
  function word() {
    const a = AGE, w = $("#word").value.trim().toLowerCase().replace(/\s+/g, "-"), out = $("#wordOut");
    const r = vocabRange(a);
    const known = Object.entries(L.WORD_AGE).filter(([, m]) => m <= a).map(([k]) => k);
    $("#wordCtx").innerHTML = `Typical number of words understood at ${esc(ageWords(a))}: <b>${r[1] === 0 ? "none yet" : r[0] + "-" + r[1]}</b>. ${known.length ? `Words many babies understand by now: ${known.slice(-14).map(esc).join(", ")}.` : "Before about 6 months, babies are learning the sounds and rhythm of language rather than word meanings."}`;
    if (!w) { out.innerHTML = ""; return; }
    const m = L.WORD_AGE[w];
    if (m == null) { out.innerHTML = `<p class="notice">"${esc(w)}" is not in our list of about 100 early words. ${a < 9 ? "At this age very few words are understood." : "Less common words are usually understood later, and every family's words differ."}</p>`; return; }
    const cls = a >= m + 2 ? "s-ready" : a >= m - 1 ? "s-emerging" : "s-notyet";
    const txt = a >= m + 2 ? "Likely understood" : a >= m - 1 ? "May be understood" : "Probably not yet";
    out.innerHTML = `<p><span class="status ${cls}">${txt}</span> Many babies understand "<b>${esc(w)}</b>" by about <b>${m} months</b>. Understanding comes months before saying it, and babies vary widely.</p>`;
  }
  $("#word").oninput = word;
  window.addEventListener("agechange", render); render();
  if (open) setTimeout(() => { const e = document.getElementById(open); e && e.scrollIntoView({ block: "center" }); }, 50);
}

// ---------------------------------------------------------------- page: see like a baby
function initSee() {
  const paint = agePicker($("#picker"));
  const orig = $("#orig"), baby = $("#baby");
  let mode = "image", src = null, anim = null, objURL = null;
  const settings = { src: store.get("acuity", "brain"), preset: "near" };
  const PRESETS = { near: [30, 25, "Close up: a face or toy about 30 cm away"], arm: [60, 50, "Arm's length: a book or toy 60 cm away"], room: [250, 200, "Across the room: about 2.5 m away"] };
  $("#presets").innerHTML = Object.entries(PRESETS).map(([k, p]) => `<button type="button" class="chip" data-p="${k}">${p[2].split(":")[0]}</button>`).join("");
  $$("#presets .chip").forEach((b) => (b.onclick = () => { settings.preset = b.dataset.p; $("#dist").value = PRESETS[b.dataset.p][0]; $("#size").value = PRESETS[b.dataset.p][1]; render(); }));
  $("#acuitySrc").value = settings.src;
  $("#acuitySrc").onchange = () => { settings.src = $("#acuitySrc").value; store.set("acuity", settings.src); render(); };
  $("#dist").oninput = $("#size").oninput = () => { settings.preset = null; render(); };
  const stopAnim = () => { if (anim) cancelAnimationFrame(anim); anim = null; };
  function useCanvasImage(c) { stopAnim(); mode = "image"; src = c.toDataURL("image/png"); show(); }
  function show() {
    const o = $("#origFrame"), bf = $("#babyFrame"); o.innerHTML = bf.innerHTML = "";
    if (mode === "image") { o.innerHTML = `<img id="orig" alt="Original picture">`; bf.innerHTML = `<img id="baby" alt="Simulated view at the chosen age">`; $("#orig").src = $("#baby").src = src; }
    else if (mode === "video") {
      o.innerHTML = `<video id="orig" playsinline muted loop controls></video>`; bf.innerHTML = `<video id="baby" playsinline muted loop></video>`;
      const v1 = $("#orig"), v2 = $("#baby"); v1.src = v2.src = src;
      v1.onplay = () => { v2.currentTime = v1.currentTime; v2.play(); }; v1.onpause = () => v2.pause(); v1.onseeked = () => (v2.currentTime = v1.currentTime);
      v1.play().catch(() => {});
    } else if (mode === "anim") {
      o.innerHTML = `<canvas id="orig" width="800" height="600" role="img" aria-label="Animated toy crossing behind a screen"></canvas>`; bf.innerHTML = `<canvas id="baby" width="800" height="600" role="img" aria-label="The same animation at the chosen age"></canvas>`;
      const g1 = $("#orig").getContext("2d"), g2 = $("#baby").getContext("2d"), t0 = performance.now();
      const step = (t) => { const c = drawSample("moving", 800, 600, (t - t0) / 1000); g1.drawImage(c, 0, 0); g2.drawImage(c, 0, 0); anim = requestAnimationFrame(step); }; anim = requestAnimationFrame(step);
    }
    $("#strip").parentElement.hidden = mode !== "image";
    $("#dl").hidden = mode !== "image";
    render();
  }
  function filterFor(a, el) {
    const v = vision(a, settings.src); const w = (el && el.clientWidth) || 400;
    return { v, f: cssFilter(v, blurPx(v.cpd, pxPerDeg(w, +$("#size").value || 25, +$("#dist").value || 30))) };
  }
  function render() {
    const a = AGE; paint(a);
    $$("#presets .chip").forEach((b) => b.classList.toggle("on", b.dataset.p === settings.preset));
    const el = $("#baby"); if (!el) return;
    const { v, f } = filterFor(a, el); el.style.filter = f;
    $("#babyCap").textContent = a < 0.25 ? "As a newborn might see it" : `As a ${ageAdj(a)} might see it`;
    const dist = +$("#dist").value || 30, mm = detailMm(v.cpd, dist);
    $("#explain").innerHTML = `<p>At <b>${esc(ageWords(a))}</b>, using ${settings.src === "brain" ? "the paper's brain-response numbers" : "looking-test (acuity card) numbers"}, sharpness is about <b>20/${v.snellen}</b>.</p><ul class="checks">
      <li class="${mm < 3 ? "ok" : "warn"}">Details smaller than about <b>${mm < 10 ? mm.toFixed(1) + " mm" : (mm / 10).toFixed(1) + " cm"}</b> blur away at ${dist} cm${mm > 5 ? ", so small print and fine textures are lost" : ""}.</li>
      <li class="${v.contrast > 0.8 ? "ok" : "warn"}">${v.contrast > 0.8 ? "Most contrasts are visible." : v.contrast > 0.5 ? "Faint, low-contrast edges fade; bold edges remain." : "Only strong light-dark edges stand out, like black on white."}</li>
      <li class="${v.sat > 0.8 ? "ok" : "warn"}">${v.sat > 0.8 ? "Colours look close to how an adult sees them." : v.sat > 0.3 ? "Colours are muted; strong reds and greens come through best." : "Colour is faint; the scene is mostly light and dark."}</li>
      ${a < 2 ? `<li class="warn">Young babies focus best at 20-30 cm. Things across the room are far beyond what they can make out.</li>` : ""}
      ${a >= 4 && a < 9 ? `<li class="ok">Faces and their expressions are visible at close range.</li>` : ""}</ul>`;
    if (mode === "image" && src) {
      $("#strip").innerHTML = [0, 1, 2, 3, 6, 12].map((s) => `<figure data-a="${s}"><div class="frame"><img alt="" src="${src}" style="filter:${filterFor(s, { clientWidth: $("#strip").clientWidth / 6 }).f}"></div><figcaption>${s === 0 ? "Newborn" : s + " mo"}</figcaption></figure>`).join("");
      $$("#strip figure").forEach((f) => (f.onclick = () => setAge(+f.dataset.a)));
    }
  }
  $("#samples").innerHTML = [["face", "A face"], ["card", "Black-and-white card"], ["toys", "Toy shelf"], ["room", "A room"], ["anim", "Moving toy (animation)"]].map(([k, t]) => `<button type="button" class="chip" data-s="${k}">${t}</button>`).join("");
  $$("#samples .chip").forEach((b) => (b.onclick = () => { $$("#samples .chip").forEach((x) => x.classList.toggle("on", x === b)); if (b.dataset.s === "anim") { stopAnim(); mode = "anim"; show(); } else useCanvasImage(drawSample(b.dataset.s)); }));
  $("#file").onchange = (e) => {
    const f = e.target.files[0]; if (!f) return; stopAnim(); if (objURL) URL.revokeObjectURL(objURL);
    objURL = URL.createObjectURL(f); src = objURL; mode = f.type.startsWith("video") ? "video" : "image";
    $$("#samples .chip").forEach((x) => x.classList.remove("on")); show();
  };
  $("#dl").onclick = () => {
    const img = $("#baby"); const c = document.createElement("canvas"); c.width = img.naturalWidth; c.height = img.naturalHeight;
    const g = c.getContext("2d"); const scale = img.naturalWidth / img.clientWidth;
    if (!("filter" in g)) { alert("Your browser cannot save filtered images. A screenshot works instead."); return; }
    g.filter = img.style.filter.replace(/blur\(([\d.]+)px\)/, (m, p) => `blur(${(+p * scale).toFixed(2)}px)`); g.drawImage(img, 0, 0);
    const a = document.createElement("a"); a.download = `baby-view-${AGE}mo.png`; a.href = c.toDataURL("image/png"); a.click();
  };
  window.addEventListener("agechange", render); window.addEventListener("resize", render);
  $("#samples .chip").click();
}

// ---------------------------------------------------------------- audio
let AC = null;
const ctx = () => (AC = AC || new (window.AudioContext || window.webkitAudioContext)());
async function synth(kind) {
  const sr = 44100, dur = kind === "music" ? 6 : 5, oc = new OfflineAudioContext(2, sr * dur, sr);
  if (kind === "voice") { // sing-song babble "ma-ma, ba-ba", moving left to right
    const pan = oc.createStereoPanner(); pan.connect(oc.destination); pan.pan.setValueAtTime(-0.9, 0); pan.pan.linearRampToValueAtTime(0.9, dur);
    const syll = [0.2, 0.6, 1.2, 1.6, 2.4, 2.8, 3.4, 3.8];
    syll.forEach((t0, i) => {
      const o = oc.createOscillator(); o.type = "sawtooth"; const f0 = 260 + 60 * Math.sin(i * 1.3);
      o.frequency.setValueAtTime(f0, t0); o.frequency.linearRampToValueAtTime(f0 * (i % 2 ? 0.85 : 1.15), t0 + 0.32);
      const g = oc.createGain(); g.gain.setValueAtTime(0, t0); g.gain.linearRampToValueAtTime(0.25, t0 + 0.06); g.gain.linearRampToValueAtTime(0.2, t0 + 0.25); g.gain.linearRampToValueAtTime(0, t0 + 0.34);
      const sum = oc.createGain(); sum.gain.value = 1;
      [[800, 6, 1], [1250, 8, 0.6], [2600, 10, 0.25]].forEach(([fr, q, gain]) => { const b = oc.createBiquadFilter(); b.type = "bandpass"; b.frequency.value = fr; b.Q.value = q; const bg = oc.createGain(); bg.gain.value = gain * 3; o.connect(b); b.connect(bg); bg.connect(sum); });
      o.connect(g); sum.connect(g); g.connect(pan); o.start(t0); o.stop(t0 + 0.36);
    });
  } else if (kind === "rattle") { // noise bursts on the left, then the right
    const n = oc.createBuffer(1, sr * dur, sr), d = n.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    [0.3, 0.55, 0.8, 1.05, 2.6, 2.85, 3.1, 3.35].forEach((t0, i) => {
      const s = oc.createBufferSource(); s.buffer = n; const b = oc.createBiquadFilter(); b.type = "bandpass"; b.frequency.value = 5000; b.Q.value = 1.2;
      const g = oc.createGain(); g.gain.setValueAtTime(0, t0); g.gain.linearRampToValueAtTime(0.5, t0 + 0.01); g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.18);
      const p = oc.createStereoPanner(); p.pan.value = i < 4 ? -0.9 : 0.9; s.connect(b); b.connect(g); g.connect(p); p.connect(oc.destination); s.start(t0, Math.random()); s.stop(t0 + 0.2);
    });
  } else { // music box: a simple tune with a bass line
    const notes = [0, 0, 7, 7, 9, 9, 7, -1, 5, 5, 4, 4, 2, 2, 0];
    notes.forEach((n, i) => { if (n < 0) return; const t0 = 0.2 + i * 0.38;
      [[523.25 * 2 ** (n / 12), "triangle", 0.25, 0.3], [130.81 * 2 ** ((i % 4 === 0 ? 0 : 7) / 12), "sine", 0.18, -0.3]].forEach(([f, ty, amp, pn]) => {
        const o = oc.createOscillator(); o.type = ty; o.frequency.value = f; const g = oc.createGain(); g.gain.setValueAtTime(0, t0); g.gain.linearRampToValueAtTime(amp, t0 + 0.01); g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.7);
        const p = oc.createStereoPanner(); p.pan.value = pn; o.connect(g); g.connect(p); p.connect(oc.destination); o.start(t0); o.stop(t0 + 0.75); }); });
  }
  return oc.startRendering();
}
function hearing(a) { return { db: interp(L.HEAR.quieter_db, a), stereo: interp(L.HEAR.stereo, a), noise: interp(L.HEAR.noise, a), treble: interp(L.HEAR.treble_cut_db, a) }; }
// Build source -> (mid/side stereo narrowing) -> treble cut -> quieter -> out, plus a soft noise floor.
function babyChain(ac, src, a, dest) {
  const h = hearing(a);
  const split = ac.createChannelSplitter(2), merge = ac.createChannelMerger(2);
  src.connect(split);
  const w = h.stereo, k1 = (1 + w) / 2, k2 = (1 - w) / 2; // L' = k1 L + k2 R, R' = k2 L + k1 R
  const gLL = ac.createGain(), gLR = ac.createGain(), gRL = ac.createGain(), gRR = ac.createGain();
  gLL.gain.value = gRR.gain.value = k1; gLR.gain.value = gRL.gain.value = k2;
  split.connect(gLL, 0); split.connect(gLR, 0); split.connect(gRL, 1); split.connect(gRR, 1);
  gLL.connect(merge, 0, 0); gRL.connect(merge, 0, 0); gLR.connect(merge, 0, 1); gRR.connect(merge, 0, 1);
  const shelf = ac.createBiquadFilter(); shelf.type = "highshelf"; shelf.frequency.value = 3000; shelf.gain.value = -h.treble;
  const vol = ac.createGain(); vol.gain.value = 10 ** (-h.db / 20);
  merge.connect(shelf); shelf.connect(vol); vol.connect(dest);
  const nb = ac.createBuffer(1, ac.sampleRate * 2, ac.sampleRate), nd = nb.getChannelData(0); for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
  const ns = ac.createBufferSource(); ns.buffer = nb; ns.loop = true; const ng = ac.createGain(); ng.gain.value = h.noise * 0.5; const nf = ac.createBiquadFilter(); nf.type = "lowpass"; nf.frequency.value = 2500;
  ns.connect(nf); nf.connect(ng); ng.connect(dest); ns.start();
  return { stop: () => { try { ns.stop(); } catch (e) {} } };
}
function initHear() {
  const paint = agePicker($("#picker"));
  let buf = null, playing = null, rec = null;
  const an = () => { const ac = ctx(); const x = ac.createAnalyser(); x.fftSize = 2048; x.connect(ac.destination); return x; };
  function stop() { if (playing) { try { playing.s.stop(); } catch (e) {} playing.n && playing.n.stop(); playing = null; } $$(".playbtn").forEach((b) => b.classList.remove("on")); }
  function play(asBaby) {
    if (!buf) return; stop(); const ac = ctx(); ac.resume();
    const s = ac.createBufferSource(); s.buffer = buf; const A = an();
    const n = asBaby ? babyChain(ac, s, AGE, A) : (s.connect(A), null);
    s.start(); playing = { s, n }; s.onended = () => { if (playing && playing.s === s) stop(); };
    (asBaby ? $("#playBaby") : $("#playAdult")).classList.add("on");
    drawSpec(A, asBaby);
  }
  function drawSpec(A, asBaby) {
    const c = $("#spec"), g = c.getContext("2d"), d = new Uint8Array(A.frequencyBinCount);
    const loop = () => { if (!playing) return; c.width = c.clientWidth * devicePixelRatio; c.height = c.clientHeight * devicePixelRatio;
      A.getByteFrequencyData(d); g.clearRect(0, 0, c.width, c.height); g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue(asBaby ? "--c-audio" : "--c-vision");
      const nb = 160; for (let i = 0; i < nb; i++) { const f0 = Math.floor(2 ** (i / nb * 9.3) * 1.5); const v = d[Math.min(d.length - 1, f0)] / 255; g.fillRect(i * c.width / nb, c.height * (1 - v), c.width / nb - 1, c.height * v); }
      requestAnimationFrame(loop); }; loop();
  }
  async function useSynth(k) { $("#srcNote").textContent = "Preparing sound…"; buf = await synth(k); $("#srcNote").textContent = SRC_NOTE[k]; }
  const SRC_NOTE = { voice: "A made-up sing-song babble that moves from your left to your right.", rattle: "A rattle shaken on the left, then on the right.", music: "A music-box tune, melody on the right and bass on the left." };
  $("#sources").innerHTML = [["voice", "Sing-song voice"], ["rattle", "Rattle"], ["music", "Music box"]].map(([k, t]) => `<button type="button" class="chip" data-k="${k}">${t}</button>`).join("");
  $$("#sources .chip").forEach((b) => (b.onclick = () => { stop(); $$("#sources .chip").forEach((x) => x.classList.toggle("on", x === b)); useSynth(b.dataset.k); }));
  $("#afile").onchange = async (e) => { const f = e.target.files[0]; if (!f) return; stop(); $$("#sources .chip").forEach((x) => x.classList.remove("on"));
    try { buf = await ctx().decodeAudioData(await f.arrayBuffer()); $("#srcNote").textContent = `Your file: ${f.name}`; } catch (err) { $("#srcNote").textContent = "That file could not be read as audio."; } };
  $("#rec").onclick = async () => {
    if (rec) { rec.stop(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); const chunks = []; rec = new MediaRecorder(stream);
      rec.ondataavailable = (e) => chunks.push(e.data);
      rec.onstop = async () => { stream.getTracks().forEach((t) => t.stop()); rec = null; $("#rec").textContent = "Record your voice (5 s)";
        try { buf = await ctx().decodeAudioData(await new Blob(chunks).arrayBuffer()); $("#srcNote").textContent = "Your recording (kept only in this browser tab)."; } catch (err) { $("#srcNote").textContent = "The recording could not be read."; } };
      rec.start(); $("#rec").textContent = "Stop recording"; $("#srcNote").textContent = "Recording… talk or sing to an imaginary baby."; setTimeout(() => rec && rec.state === "recording" && rec.stop(), 5000);
    } catch (e) { $("#srcNote").textContent = "Microphone not available or permission declined."; }
  };
  $("#playAdult").onclick = () => play(false); $("#playBaby").onclick = () => play(true); $("#stop").onclick = stop;
  function render() {
    const a = AGE; paint(a); const h = hearing(a);
    $("#babyLabel").textContent = a < 0.25 ? "newborn" : ageAdj(a);
    $("#hearOut").innerHTML = `<li><b>Quieter:</b> about ${Math.round(h.db)} dB less loud, standing for higher hearing thresholds at this age.</li>
      <li><b>Duller treble:</b> high sounds turned down by ${Math.round(h.treble)} dB${h.treble < 1 ? " (none)" : ""}.</li>
      <li><b>Narrower left-right:</b> ${Math.round(h.stereo * 100)}% of the adult difference between ears, so sounds are harder to place.</li>
      <li><b>Background hiss:</b> a soft noise floor for less selective listening.</li>`;
    const sp = nearestRow(L.DOMAINS[1].items[1][1], a)[1], lst = nearestRow(L.DOMAINS[1].items[0][1], a)[1];
    $("#langOut").textContent = `${lst} ${sp}`;
    if (playing && playing.n) play(true);
  }
  window.addEventListener("agechange", render); render();
  $("#sources .chip").click();
}

// ---------------------------------------------------------------- page: models (phase 2)
function initModels() {
  const host = $("#ckpts"); let ck = 12;
  host.innerHTML = L.CHECKPOINTS.map((c) => `<button type="button" class="chip" data-c="${c}">${c === 0 ? "Newborn" : c + " months"}</button>`).join("");
  const paint = () => { $$("#ckpts .chip").forEach((b) => b.classList.toggle("on", +b.dataset.c === ck));
    $("#ckImg").src = `../data/models/grown_${ck}.png`; $("#ckImg").alt = `Vision model outputs at the ${ck}-month stage`; $("#ckText").textContent = MODEL_TEXT[ck]; };
  $$("#ckpts .chip").forEach((b) => (b.onclick = () => { ck = +b.dataset.c; paint(); })); paint();
}

// ---------------------------------------------------------------- page: observation session (phase 3)
const TESTS = {
  faces: { name: "Face or scramble?", about: "A simple face pattern and the same parts scrambled appear side by side. Note which one the baby looks at first and longer.", from: 0,
    typical: "Newborns tend to look more at face-like patterns (Johnson & Morton 1991). The preference changes over the first months as babies learn real faces.", trials: 4 },
  track: { name: "Follow the toy", about: "A bright toy moves slowly across the screen and back. Note how the baby's eyes follow it.", from: 0,
    typical: "Smooth following of a slow target usually appears around 2 months; before that following is jerky or brief.", trials: 3 },
  stripes: { name: "Stripes or grey? (sharpness)", about: "Stripes appear on one side and plain grey on the other, getting finer each time. Note which side the baby looks at. The finest stripes the baby still prefers give a rough sharpness estimate.", from: 0,
    typical: "Based on acuity-card methods (Teller 1997). Screens are not calibrated cards, so treat the number as rough.", trials: 0 },
  sound: { name: "Turn to the sound", about: "A gentle rattle sound plays from the left or right speaker. Note which way the baby turns.", from: 0,
    typical: "Newborns turn slowly toward sounds; quick, accurate turns are usual by 4-6 months. Needs speakers on both sides.", trials: 6 },
  hide: { name: "Hide-and-seek (surprise)", about: "A toy rolls behind a screen. When the screen drops, the toy is either there (expected) or gone (surprising). Time how long the baby looks at the end scene.", from: 3,
    typical: "In lab studies, babies from about 3-5 months often look longer at the surprising ending (Baillargeon 1987), though results vary between labs and single sessions are noisy.", trials: 4 },
};
function initSession() {
  const S = { tests: [], results: [], meta: {} };
  const setup = $("#setup"), stage = $("#stage"), cv = $("#scv"), g = cv.getContext("2d"), bar = $("#obs");
  $("#tests").innerHTML = Object.entries(TESTS).map(([k, t]) => `<label><input type="checkbox" value="${k}" ${k !== "hide" ? "checked" : ""}><span><b>${esc(t.name)}</b><small>${esc(t.about)}</small></span></label>`).join("");
  $("#bAge").value = store.get("age", 3);
  $("#start").onclick = () => {
    S.meta = { nickname: $("#bName").value.trim() || "Baby", age_months: +$("#bAge").value, corrected: $("#bPrem").value, screen_cm: +$("#scrW").value, distance_cm: +$("#dist").value,
      observer: $("#obsName").value.trim(), started: new Date().toISOString(), screen_px: screen.width, dpr: devicePixelRatio };
    S.tests = $$("#tests input:checked").map((x) => x.value); S.results = [];
    if (!S.tests.length) return alert("Pick at least one activity.");
    if (!S.meta.screen_cm || !S.meta.distance_cm) return alert("Please enter the screen width and the viewing distance.");
    stage.classList.add("on"); if (stage.requestFullscreen) stage.requestFullscreen().catch(() => {});
    next();
  };
  let cur = null, raf = null, keyH = null;
  const W = () => cv.width, H = () => cv.height;
  function size() { cv.width = innerWidth * devicePixelRatio; cv.height = innerHeight * devicePixelRatio; }
  window.addEventListener("resize", size);
  const ppd = () => pxPerDeg(screen.width * devicePixelRatio, S.meta.screen_cm, S.meta.distance_cm); // device px per degree
  function controls(q, buttons) {
    bar.innerHTML = `<span class="q">${q}</span>` + buttons.map((b, i) => `<button type="button" data-i="${i}" class="${b.hold ? "hold" : ""}">${esc(b.t)}${b.k ? `<kbd>${b.k}</kbd>` : ""}</button>`).join("") + `<button type="button" data-x="skip">Skip</button><button type="button" data-x="end">End session</button>`;
    $$("button[data-i]", bar).forEach((el) => { const b = buttons[+el.dataset.i];
      if (b.hold) { el.onpointerdown = () => { el.classList.add("down"); b.down(); }; el.onpointerup = el.onpointerleave = () => { if (el.classList.contains("down")) { el.classList.remove("down"); b.up(); } }; }
      else el.onclick = b.f; });
    $("button[data-x=skip]", bar).onclick = () => { cur && cur.done(true); };
    $("button[data-x=end]", bar).onclick = finish;
    if (keyH) removeEventListener("keydown", keyH); if (window._keyup) removeEventListener("keyup", window._keyup);
    keyH = (e) => { const b = buttons.find((x) => x.k && x.k.toLowerCase() === (e.key === " " ? "space" : e.key.toLowerCase().replace("arrow", ""))); if (!b) return; e.preventDefault(); if (b.hold) { if (!e.repeat) { const hb = $$("button.hold", bar)[0]; hb && hb.classList.add("down"); b.down(); } } else b.f(); };
    window._keyup = (e) => { const b = buttons.find((x) => x.hold && x.k === "Space"); if (b && e.key === " ") { const hb = $$("button.hold", bar)[0]; hb && hb.classList.remove("down"); b.up(); } };
    addEventListener("keydown", keyH); addEventListener("keyup", window._keyup);
  }
  const clear = (c = "#6f6f6f") => { g.fillStyle = c; g.fillRect(0, 0, W(), H()); };
  function next() {
    cancelAnimationFrame(raf); size();
    const k = S.tests.shift(); if (!k) return finish();
    const runner = { faces: runFaces, track: runTrack, stripes: runStripes, sound: runSound, hide: runHide }[k];
    clear(); controls(`<b>${esc(TESTS[k].name)}</b>: get the baby's attention on the screen, then press Start.`, [{ t: "Start", k: "Enter", f: () => runner(k) }]);
  }
  function record(k, row) { S.results.push({ test: k, time: new Date().toISOString(), ...row }); }
  function mkDone(k, after) { return (skipped) => { cancelAnimationFrame(raf); if (skipped) record(k, { skipped: true }); after && after(); next(); }; }

  // --- faces: face vs scrambled, side randomised
  function face(cx, cy, r, scr) {
    g.fillStyle = "#f2d1b3"; g.beginPath(); g.ellipse(cx, cy, r, r * 1.25, 0, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#111"; const parts = scr ? [[-0.4, 0.55, 0.22], [0.35, -0.1, 0.22], [0, -0.55, 0.3]] : [[-0.38, -0.3, 0.22], [0.38, -0.3, 0.22], [0, 0.5, 0.3]];
    parts.forEach(([x, y, s], i) => { g.beginPath(); if (i < 2) g.arc(cx + x * r, cy + y * r, s * r * 0.6, 0, Math.PI * 2); else g.ellipse(cx + x * r, cy + y * r, s * r, s * r * 0.35, 0, 0, Math.PI * 2); g.fill(); });
  }
  function runFaces(k) {
    let n = 0; const T = TESTS[k].trials;
    const trial = () => { if (n >= T) return cur.done();
      const faceLeft = Math.random() < 0.5; clear(); const r = Math.min(W(), H()) * 0.17;
      face(W() * 0.27, H() * 0.45, r, !faceLeft); face(W() * 0.73, H() * 0.45, r, faceLeft);
      const ans = (side) => () => { record(k, { trial: n + 1, face_side: faceLeft ? "left" : "right", looked: side, chose_face: side === "unsure" ? null : (side === "left") === faceLeft }); n++; trial(); };
      controls(`Trial ${n + 1} of ${T}: where did the baby look first and longest? (Watch the baby, not the screen.)`, [{ t: "Left", k: "Left", f: ans("left") }, { t: "Right", k: "Right", f: ans("right") }, { t: "Can't tell", k: "Down", f: ans("unsure") }]); };
    cur = { done: mkDone(k) }; trial();
  }
  // --- track: slow target across and back
  function runTrack(k) {
    let n = 0; const T = TESTS[k].trials;
    const trial = () => { if (n >= T) return cur.done();
      const degPerSec = 10, speed = degPerSec * ppd(), r = Math.max(30, 2.5 * ppd()), t0 = performance.now(); let x = r * 2, dir = 1, last = t0;
      const draw = (t) => { const dt = (t - last) / 1000; last = t; x += dir * speed * dt; if (x > W() - 2 * r) dir = -1; if (x < 2 * r && dir < 0) dir = 1;
        clear(); g.fillStyle = "#fff"; g.beginPath(); g.arc(x, H() * 0.45, r, 0, Math.PI * 2); g.fill(); g.fillStyle = "#000"; g.beginPath(); g.arc(x, H() * 0.45, r * 0.6, 0, Math.PI * 2); g.fill(); g.fillStyle = "#d62828"; g.beginPath(); g.arc(x, H() * 0.45, r * 0.3, 0, Math.PI * 2); g.fill();
        raf = requestAnimationFrame(draw); };
      raf = requestAnimationFrame(draw);
      const ans = (v) => () => { cancelAnimationFrame(raf); record(k, { trial: n + 1, speed_deg_s: degPerSec, response: v }); n++; trial(); };
      controls(`Trial ${n + 1} of ${T}: how did the baby's eyes follow the toy?`, [{ t: "Smoothly", k: "1", f: ans("smooth") }, { t: "In jumps", k: "2", f: ans("jumpy") }, { t: "Briefly", k: "3", f: ans("brief") }, { t: "Not at all", k: "4", f: ans("none") }]); };
    cur = { done: mkDone(k) }; trial();
  }
  // --- stripes: preferential looking with a simple staircase
  function runStripes(k) {
    const P = ppd(), maxCpd = P / 4; // finest: 4 device pixels per cycle
    const levels = []; for (let c = 0.25; c <= maxCpd; c *= Math.SQRT2) levels.push(c);
    let li = 0, tries = [], side;
    const panel = (x0, w, cpd) => {
      const y0 = H() * 0.15, h = H() * 0.6;
      if (cpd) { const period = Math.max(4, Math.round(P / cpd)); for (let x = 0; x < w; x++) { g.fillStyle = ((x % period) < period / 2) ? "#fff" : "#000"; g.fillRect(x0 + x, y0, 1, h); } }
      else { const img = g.createImageData(Math.floor(w), Math.floor(h)); for (let y = 0; y < img.height; y++) for (let x = 0; x < img.width; x++) { const v = (x + y) % 2 ? 255 : 0, i = (y * img.width + x) * 4; img.data[i] = img.data[i + 1] = img.data[i + 2] = v; img.data[i + 3] = 255; } g.putImageData(img, x0, y0); }
    };
    const trial = () => {
      if (li >= levels.length) { record(k, { result: "finest_shown", cpd: +levels[levels.length - 1].toFixed(2), note: "screen limit reached; true sharpness may be higher" }); return cur.done(); }
      side = Math.random() < 0.5 ? "left" : "right"; const w = W() * 0.32;
      clear(); panel(side === "left" ? W() * 0.1 : W() * 0.58, w, levels[li]); panel(side === "left" ? W() * 0.58 : W() * 0.1, w, 0);
      const ans = (look) => () => {
        const ok = look === side; tries.push(ok);
        record(k, { level_cpd: +levels[li].toFixed(2), stripes_side: side, looked: look, correct: look === "unsure" ? null : ok });
        const c = tries.filter(Boolean).length, wrong = tries.filter((x) => !x).length;
        if (c >= 2) { li++; tries = []; trial(); }
        else if (wrong >= 2 || tries.length >= 3) { const best = li ? levels[li - 1] : null; record(k, { result: "estimate", cpd: best ? +best.toFixed(2) : null, snellen: best ? `20/${Math.round(600 / best)}` : null }); cur.done(); }
        else trial();
      };
      controls(`Level ${li + 1}: ${levels[li].toFixed(1)} cycles/degree. Which side did the baby look at?`, [{ t: "Left", k: "Left", f: ans("left") }, { t: "Right", k: "Right", f: ans("right") }, { t: "Can't tell", k: "Down", f: () => { record(k, { level_cpd: +levels[li].toFixed(2), looked: "unsure" }); trial(); } }]);
    };
    cur = { done: mkDone(k) }; trial();
  }
  // --- sound: rattle from one side
  function runSound(k) {
    let n = 0; const T = TESTS[k].trials; const ac = ctx(); ac.resume();
    const nb = ac.createBuffer(1, ac.sampleRate, ac.sampleRate), d = nb.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    const shake = (side) => { [0, 0.22, 0.44, 0.66, 0.88, 1.1].forEach((dt) => { const t0 = ac.currentTime + 0.05 + dt;
      const s = ac.createBufferSource(); s.buffer = nb; const b = ac.createBiquadFilter(); b.type = "bandpass"; b.frequency.value = 4000; b.Q.value = 1; const gn = ac.createGain();
      gn.gain.setValueAtTime(0, t0); gn.gain.linearRampToValueAtTime(0.35, t0 + 0.01); gn.gain.exponentialRampToValueAtTime(0.001, t0 + 0.15); const p = ac.createStereoPanner(); p.pan.value = side === "left" ? -1 : 1;
      s.connect(b); b.connect(gn); gn.connect(p); p.connect(ac.destination); s.start(t0); s.stop(t0 + 0.2); }); };
    const trial = () => { if (n >= T) return cur.done();
      clear(); const side = Math.random() < 0.5 ? "left" : "right";
      controls(`Trial ${n + 1} of ${T}: wait until the baby looks at the centre, then play.`, [{ t: "Play sound", k: "Enter", f: () => { shake(side);
        controls(`Trial ${n + 1}: which way did the baby turn?`, [{ t: "Left", k: "Left", f: ans("left") }, { t: "Right", k: "Right", f: ans("right") }, { t: "No turn", k: "Down", f: ans("none") }]); } }]);
      const ans = (v) => () => { record(k, { trial: n + 1, sound_side: side, turned: v, correct: v === "none" ? false : v === side }); n++; trial(); }; };
    cur = { done: mkDone(k) }; trial();
  }
  // --- hide: toy rolls behind a screen; screen drops; observer holds while baby looks
  function runHide(k) {
    let n = 0; const T = TESTS[k].trials; const order = ["expected", "surprise", "surprise", "expected"].sort(() => Math.random() - 0.5);
    const trial = () => { if (n >= T) return cur.done();
      const kind = order[n % order.length], t0 = performance.now(); let lookStart = null, looked = 0, phase = "roll";
      const sx = W() * 0.5, sw = W() * 0.16, sh = H() * 0.4, gy = H() * 0.62, r = H() * 0.06;
      const draw = (t) => { const s = (t - t0) / 1000; clear("#77736b");
        g.fillStyle = "#b8a98a"; g.fillRect(0, gy, W(), H() - gy);
        const tx = Math.min(sx, W() * 0.1 + s * W() * 0.18); const toyVisible = kind === "expected" || s < 2.2;
        const drop = Math.max(0, Math.min(1, (s - 3.2) / 0.8)); // screen lowers between 3.2 and 4 s
        if (toyVisible) { g.fillStyle = "#d62828"; g.beginPath(); g.arc(tx, gy - r, r, 0, Math.PI * 2); g.fill(); g.fillStyle = "#fff"; g.beginPath(); g.arc(tx - r * 0.35, gy - r * 1.1, r * 0.18, 0, Math.PI * 2); g.arc(tx + r * 0.35, gy - r * 1.1, r * 0.18, 0, Math.PI * 2); g.fill(); }
        g.fillStyle = "#264653"; g.fillRect(sx - sw / 2, gy - sh * (1 - drop), sw, sh * (1 - drop));
        if (drop >= 1 && phase === "roll") { phase = "look"; controls(`Trial ${n + 1} of ${T}: hold while the baby looks at the scene; release when they look away.`, [{ t: "Baby is looking (hold)", k: "Space", hold: true, down: () => (lookStart = performance.now()), up: () => { if (lookStart) { looked += performance.now() - lookStart; lookStart = null; } } }, { t: "Done", k: "Enter", f: fin }]); }
        raf = requestAnimationFrame(draw); };
      const fin = () => { cancelAnimationFrame(raf); if (lookStart) looked += performance.now() - lookStart; record(k, { trial: n + 1, ending: kind, looking_s: +(looked / 1000).toFixed(2) }); n++; trial(); };
      controls(`Trial ${n + 1} of ${T}: watch the baby.`, [{ t: "Done", k: "Enter", f: fin }]);
      raf = requestAnimationFrame(draw); };
    cur = { done: mkDone(k) }; trial();
  }
  function finish() {
    cancelAnimationFrame(raf); if (keyH) removeEventListener("keydown", keyH); if (window._keyup) removeEventListener("keyup", window._keyup);
    stage.classList.remove("on"); if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    S.meta.ended = new Date().toISOString();
    const hist = store.get("sessions", []); hist.push({ meta: S.meta, results: S.results }); store.set("sessions", hist.slice(-50));
    summary({ meta: S.meta, results: S.results }); renderHistory();
  }
  function summary(sess) {
    const { meta, results } = sess, a = meta.age_months, by = (k) => results.filter((r) => r.test === k && !r.skipped);
    const parts = [];
    const f = by("faces").filter((r) => r.chose_face != null); if (f.length) parts.push(["Face or scramble?", `Looked at the face pattern in ${f.filter((r) => r.chose_face).length} of ${f.length} clear trials.`, TESTS.faces.typical]);
    const t = by("track"); if (t.length) parts.push(["Follow the toy", ["smooth", "jumpy", "brief", "none"].map((v) => `${v}: ${t.filter((r) => r.response === v).length}`).join(", "), TESTS.track.typical + ` (${ageWords(a)}: ${nearestRow(L.DOMAINS[0].items[3][1], a)[1]})`]);
    const s = by("stripes").find((r) => r.result); if (s) { const lo = vision(a, "look").cpd; parts.push(["Stripes or grey?", s.cpd ? `Finest stripes preferred: about ${s.cpd} cycles/degree (roughly ${s.snellen || "20/" + Math.round(600 / s.cpd)}).${s.result === "finest_shown" ? " The screen could not show finer stripes, so the true value may be higher." : ""}` : "No clear preference even for the widest stripes.", `Typical looking-test values at ${ageWords(a)}: about ${(lo / 2).toFixed(1)}-${(lo * 2).toFixed(1)} cycles/degree. ${TESTS.stripes.typical}`]); }
    const so = by("sound"); if (so.length) parts.push(["Turn to the sound", `Turned the correct way in ${so.filter((r) => r.correct).length} of ${so.length} trials; no turn in ${so.filter((r) => r.turned === "none").length}.`, TESTS.sound.typical]);
    const h = by("hide"); if (h.length) { const m = (k) => { const x = h.filter((r) => r.ending === k).map((r) => r.looking_s); return x.length ? (x.reduce((p, q) => p + q, 0) / x.length).toFixed(1) : "–"; }; parts.push(["Hide-and-seek", `Average looking: ${m("expected")} s at the expected ending, ${m("surprise")} s at the surprising one.`, TESTS.hide.typical]); }
    $("#summary").hidden = false;
    $("#sumBody").innerHTML = `<p><b>${esc(meta.nickname)}</b>, ${esc(ageWords(a))}${meta.corrected === "yes" ? " (corrected age)" : ""} · ${new Date(meta.started).toLocaleString()}</p>` +
      (parts.length ? `<div class="tablewrap"><table class="t"><thead><tr><th>Activity</th><th>What you recorded</th><th>For context</th></tr></thead><tbody>${parts.map((p) => `<tr><th scope="row">${esc(p[0])}</th><td>${esc(p[1])}</td><td>${esc(p[2])}</td></tr>`).join("")}</tbody></table></div>` : "<p>No observations were recorded.</p>");
    $("#dlJson").onclick = () => download(`session-${meta.started.slice(0, 10)}.json`, JSON.stringify(sess, null, 2), "application/json");
    $("#dlCsv").onclick = () => download(`session-${meta.started.slice(0, 10)}.csv`, toCsv(sess), "text/csv");
    $("#summary").scrollIntoView();
  }
  function toCsv(sess) {
    const keys = [...new Set(sess.results.flatMap((r) => Object.keys(r)))], m = sess.meta;
    const q = (v) => v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v);
    return ["nickname,age_months,corrected,screen_cm,distance_cm," + keys.join(","), ...sess.results.map((r) => [m.nickname, m.age_months, m.corrected, m.screen_cm, m.distance_cm, ...keys.map((k) => r[k])].map(q).join(","))].join("\n");
  }
  function download(name, text, type) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }
  function renderHistory() {
    const hist = store.get("sessions", []); $("#history").innerHTML = hist.length ? hist.map((h, i) => `<li><button type="button" class="chip" data-i="${i}">${esc(h.meta.nickname)} · ${esc(ageWords(h.meta.age_months))} · ${new Date(h.meta.started).toLocaleString()}</button></li>`).reverse().join("") : "<li>No saved sessions in this browser yet.</li>";
    $$("#history .chip").forEach((b) => (b.onclick = () => summary(hist[+b.dataset.i])));
  }
  $("#clearHist").onclick = () => { if (confirm("Delete all sessions saved in this browser? Downloaded files are not affected.")) { store.set("sessions", []); renderHistory(); } };
  renderHistory();
}

// ---------------------------------------------------------------- home
function initHome() {
  const paint = agePicker($("#picker"), { label: "Pick an age to start" });
  const go = () => { paint(AGE); $("#goAge").href = `app/age.html?age=${AGE}`; $("#goAge").textContent = `See what a ${ageAdj(AGE)} can do →`; };
  window.addEventListener("agechange", go); go();
}

const PAGES = { home: initHome, age: initAge, play: initPlay, see: initSee, hear: initHear, models: initModels, session: initSession };
function boot() { const p = document.body.dataset.page; if (PAGES[p]) PAGES[p](); setAge(AGE, false); }
document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", boot) : boot();
window.LABJS = { ageWords, vision, blurPx, pxPerDeg, detailMm, readiness, interp };
})();

/* Shared site chrome: theme, header, sidebar, breadcrumbs, in-page contents, pager, footer, math.
   Every page declares <body data-root="../" data-path="wiki/overview.html">; content lives in <main id="main">. */
(() => {
"use strict";
const GH = "https://github.com/pikulsomesh/developmental-world-models";
const root = document.documentElement;
try { const t = localStorage.getItem("theme"); if (t) root.dataset.theme = t; } catch (e) {}

const WIKI = [
  ["wiki/overview.html", "Study design"], ["wiki/models.html", "Models"], ["wiki/methods.html", "Methods"],
  ["wiki/theory.html", "Theory"], ["wiki/results.html", "Results"], ["wiki/limitations.html", "Scope and limitations"],
  ["wiki/reproduce.html", "Code and data"], ["wiki/references.html", "References"],
];
const EXPLORE = [
  ["explore/index.html", "Simulations and tools"], ["explore/senses.html", "Infant senses"], ["explore/models.html", "Model viewer"],
  ["explore/development.html", "Development curves"], ["explore/expectation.html", "Hide-and-seek"], ["explore/fusion.html", "Fusion"],
];
const APP = [
  ["index.html", "Start", ""], ["app/age.html", "Age profile", "1"], ["app/see.html", "See like a baby", "1"], ["app/hear.html", "Hear like a baby", "1"],
  ["app/play.html", "Activities and words", "1"], ["app/models.html", "What the models learned", "2"], ["app/session.html", "Try with a baby", "3"], ["app/guide.html", "How to use", ""],
];
const SITE = "Do AI Models Learn Like Humans?";
window.DWM = { GH, WIKI, EXPLORE, APP };

function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function renderMath() {
  if (!window.katex) return false;
  document.querySelectorAll(".eq").forEach((el) => {
    if (el.dataset.done) return;
    const tex = el.dataset.tex || el.textContent;
    try { katex.render(tex, el, { displayMode: true, throwOnError: false }); el.dataset.done = 1; } catch (e) {}
  });
  document.querySelectorAll(".m").forEach((el) => {
    if (el.dataset.done) return;
    try { katex.render(el.textContent, el, { throwOnError: false }); el.dataset.done = 1; } catch (e) {}
  });
  return true;
}
window.DWM.renderMath = renderMath;

function build() {
  const body = document.body, R = body.dataset.root || "", path = body.dataset.path || "index.html";
  const section = path.startsWith("wiki/") ? "wiki" : path.startsWith("explore/") ? "explore" : path.startsWith("app/") ? "app" : "home";
  const list = section === "wiki" ? WIKI : section === "explore" ? EXPLORE : null;
  const main = document.getElementById("main");
  const link = (href, text, cur) => `<a href="${R}${href}"${cur ? ' aria-current="page"' : ""}>${text}</a>`;

  // skip link + header
  const skip = document.createElement("a"); skip.className = "skip"; skip.href = "#main"; skip.textContent = "Skip to content";
  body.prepend(skip);
  const hdr = document.createElement("header"); hdr.className = "site-header";
  hdr.innerHTML = `<div class="wrap"><a class="brand" href="${R}index.html">Developmental World Models</a>
    <nav aria-label="Primary">${link("index.html", "Baby Senses Lab", section === "home" || section === "app")}${link("explore/index.html", "Research tools", section === "explore")}${link("wiki/overview.html", "Wiki", section === "wiki")}
    <a href="${R}paper/report.pdf">Paper</a><a href="${GH}">GitHub</a></nav>
    <button class="icon-btn" id="theme" aria-label="Toggle light or dark theme">Theme</button></div>`;
  skip.after(hdr);
  if (section === "app" || section === "home") { // app tab strip, grouped by phase
    const an = document.createElement("nav"); an.className = "appnav"; an.setAttribute("aria-label", "Baby Senses Lab");
    let ph = null;
    an.innerHTML = `<div class="wrap">${APP.map((p) => { const head = p[2] && p[2] !== ph ? `<span class="ph">Phase ${p[2]}</span>` : ""; ph = p[2] || ph; return head + link(p[0], p[1], p[0] === path); }).join("")}</div>`;
    hdr.after(an);
  }
  document.getElementById("theme").onclick = () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
    window.dispatchEvent(new Event("themechange"));
  };

  // headings: ids + in-page list
  const h2s = [...main.querySelectorAll("h2")];
  h2s.forEach((h) => { if (!h.id) h.id = h.textContent.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""); });
  const idx = list ? list.findIndex((p) => p[0] === path) : -1;
  const title = (list && idx >= 0) ? list[idx][1] : (document.title.split(" · ")[0]);

  // layout (sidebar for wiki + explore)
  const layout = document.createElement("div"); layout.className = "layout" + (list ? "" : " nosidebar");
  main.parentNode.insertBefore(layout, main);
  if (list) {
    const aside = document.createElement("aside"); aside.className = "sidebar"; aside.setAttribute("aria-label", "Section navigation");
    const other = section === "wiki" ? ["Explore", EXPLORE] : ["Wiki", WIKI];
    aside.innerHTML = `<button class="side-toggle" aria-expanded="false" aria-controls="sidebody">Menu: ${esc(section === "wiki" ? "Wiki" : "Explore")} pages and contents</button>
      <div class="sidebody" id="sidebody"><h2>${section === "wiki" ? "Wiki" : "Simulations and tools"}</h2>
      <ul>${list.map((p) => `<li>${link(p[0], p[1], p[0] === path)}</li>`).join("")}</ul>
      ${h2s.length > 1 ? `<h2>In this page</h2><ul class="toc">${h2s.map((h) => `<li><a href="#${h.id}">${esc(h.textContent)}</a></li>`).join("")}</ul>` : ""}
      <h2>${other[0] === "Wiki" ? "Wiki" : "Explore"}</h2><ul>${other[1].slice(0, 1).map((p) => `<li>${link(p[0], other[0] === "Wiki" ? "Wiki: study design" : "Simulations and tools")}</li>`).join("")}</ul></div>`;
    layout.appendChild(aside);
    const tg = aside.querySelector(".side-toggle");
    tg.onclick = () => { const o = aside.classList.toggle("open"); tg.setAttribute("aria-expanded", o); };
  }
  layout.appendChild(main);
  main.classList.add(section === "explore" || section === "app" ? "wide" : "doc");
  if (section === "home") main.classList.remove("doc");

  // breadcrumbs
  if (section !== "home" && section !== "app") {
    const c = document.createElement("nav"); c.className = "crumbs"; c.setAttribute("aria-label", "Breadcrumb");
    c.innerHTML = `<a href="${R}index.html">Home</a><span class="sep">/</span><a href="${R}${section === "wiki" ? "wiki/overview.html" : "explore/index.html"}">${section === "wiki" ? "Wiki" : "Explore"}</a>` +
      (path === "explore/index.html" ? "" : `<span class="sep">/</span>${esc(title)}`);
    main.prepend(c);
  }

  // prev / next
  if (list && idx >= 0) {
    const p = list[idx - 1], n = list[idx + 1];
    const d = document.createElement("nav"); d.className = "pager"; d.setAttribute("aria-label", "Previous and next page");
    d.innerHTML = (p ? `<a class="prev" href="${R}${p[0]}"><small>Previous</small>${esc(p[1])}</a>` : "<span></span>") + (n ? `<a class="next" href="${R}${n[0]}"><small>Next</small>${esc(n[1])}</a>` : "");
    main.appendChild(d);
  }

  // footer
  const f = document.createElement("footer"); f.className = "site-footer";
  f.innerHTML = `<div class="wrap"><div class="cols">
    <div><h2>Baby Senses Lab</h2><p>A simulation and research tool from the Developmental World Models project, by Somesh Mohapatra. It is not a medical or diagnostic tool.</p><p>Code released under the MIT licence. Data and figures are generated from the repository.</p></div>
    <div><h2>Wiki</h2><ul>${WIKI.slice(0, 5).map((p) => `<li>${link(p[0], p[1])}</li>`).join("")}</ul></div>
    <div><h2>More</h2><ul>${WIKI.slice(5).map((p) => `<li>${link(p[0], p[1])}</li>`).join("")}<li>${link("explore/index.html", "Simulations and tools")}</li></ul></div>
    <div><h2>Cite and contribute</h2><ul><li>${link("index.html#cite", "How to cite")}</li><li><a href="${R}paper/report.pdf">Paper (PDF)</a></li><li><a href="${GH}">Source code</a></li><li><a href="${GH}/issues">Report an issue</a></li></ul></div>
  </div></div>`;
  body.appendChild(f);
  if (!document.getElementById("tip")) { const t = document.createElement("div"); t.className = "tip"; t.id = "tip"; t.setAttribute("role", "tooltip"); body.appendChild(t); }

  document.querySelectorAll("[data-copy]").forEach((b) => { b.onclick = () => { const el = document.querySelector(b.dataset.copy); if (!el) return;
    (navigator.clipboard ? navigator.clipboard.writeText(el.textContent) : Promise.reject()).then(() => { b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1500); }).catch(() => { const r = document.createRange(); r.selectNodeContents(el); const s = getSelection(); s.removeAllRanges(); s.addRange(r); b.textContent = "Selected"; }); }; });
  // scroll to hash after layout is built
  if (location.hash) { const t = document.getElementById(location.hash.slice(1)); if (t) t.scrollIntoView(); }
  if (!renderMath()) window.addEventListener("load", renderMath);
  const kx = document.querySelector("script[data-katex]"); if (kx) kx.addEventListener("load", renderMath);
}
document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", build) : build();
})();

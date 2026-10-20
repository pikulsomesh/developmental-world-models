/* Shared site chrome: audience (families or researchers), theme, header, footer.
   Every page declares <body data-root="../" data-path="app/see.html" data-for="family|research|both">;
   content lives in <main id="main">. The audience is remembered on this device and can be switched at any time. */
(() => {
"use strict";
const GH = "https://github.com/pikulsomesh/developmental-world-models";
const root = document.documentElement;
const get = (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } };
const put = (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} };

// theme and audience are applied before the page paints
const t = get("theme"); if (t) root.dataset.theme = t;
const q = new URLSearchParams(location.search).get("for");
if (q === "family" || q === "research") put("persona", q);
const saved = get("persona");
if (saved === "family" || saved === "research") root.dataset.persona = saved;

const NAV = {
  family: [["app/age.html", "This age"], ["app/see.html", "See"], ["app/hear.html", "Hear"], ["app/play.html", "Play"],
    ["app/session.html", "Try together"], ["app/science.html", "How we know"]],
  research: [["research/findings.html", "Findings"], ["research/approach.html", "Approach"], ["research/papers.html", "Papers"],
    ["research/code.html", "Code"], ["app/see.html", "Simulator"]],
};
const PAPERS = {
  p1: { title: "Learning Like Infants: Input Constraints in Predictive World Models", pdf: "paper/report.pdf?v=60ee92cf", src: GH + "/tree/main/papers/world-models" },
  p2: { title: "Simulating Infant Senses with Age-Tuned Networks", pdf: "paper/baby-senses.pdf?v=a303842b", src: GH + "/tree/main/papers/baby-senses" },
};
window.DWM = { GH, NAV, PAPERS };

function persona() { return root.dataset.persona || null; }
function setPersona(p) {
  root.dataset.persona = p; put("persona", p);
  const page = document.body.dataset.for || "both";
  if (page !== "both" && page !== p) { location.href = (document.body.dataset.root || "") + "index.html"; return; }
  paintChrome(); window.dispatchEvent(new CustomEvent("personachange", { detail: p }));
}
window.DWM.setPersona = setPersona;

let hdr, foot;
function paintChrome() {
  const body = document.body, R = body.dataset.root || "", path = body.dataset.path || "index.html";
  const p = persona() || (body.dataset.for === "research" ? "research" : "family");
  const link = (href, text) => `<a href="${R}${href}"${href === path ? ' aria-current="page"' : ""}>${text}</a>`;
  hdr.querySelector(".brand").textContent = p === "research" ? "Developmental World Models" : "Baby Senses Lab";
  hdr.querySelector("nav.primary").innerHTML = NAV[p].map((x) => link(x[0], x[1])).join("");
  hdr.querySelector("nav.primary").setAttribute("aria-label", p === "research" ? "Research pages" : "Family pages");
  hdr.querySelectorAll(".who button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.p === persona())));
  const ban = document.getElementById("whoBanner"); if (ban) ban.hidden = !!persona();
  foot.innerHTML = `<div class="wrap"><div class="cols">
    <div><p><b>${p === "research" ? "Developmental World Models" : "Baby Senses Lab"}</b> by Somesh Mohapatra.</p>
      <p>${p === "research" ? "Code, results, checkpoints and both papers are open on GitHub." : "A simulation built from published averages. It is not a medical or diagnostic test."}</p></div>
    <div><ul><li><a href="${R}${PAPERS.p1.pdf}">Paper 1: Learning Like Infants (PDF)</a></li><li><a href="${R}${PAPERS.p2.pdf}">Paper 2: Simulating Infant Senses (PDF)</a></li>
      <li><a href="${GH}">Code on GitHub</a></li><li><a href="${GH}/issues">Report a problem</a></li></ul></div>
  </div></div>`;
}

function build() {
  const body = document.body, R = body.dataset.root || "";
  const main = document.getElementById("main");
  const skip = document.createElement("a"); skip.className = "skip"; skip.href = "#main"; skip.textContent = "Skip to content";
  body.prepend(skip);
  hdr = document.createElement("header"); hdr.className = "site-header";
  hdr.innerHTML = `<div class="wrap"><a class="brand" href="${R}index.html"></a>
    <div class="who" role="group" aria-label="Show the site for"><button type="button" data-p="family">Families</button><button type="button" data-p="research">Researchers</button></div>
    <button class="icon-btn" id="theme" type="button"></button>
    <nav class="primary"></nav></div>`;
  skip.after(hdr);
  hdr.querySelectorAll(".who button").forEach((b) => (b.onclick = () => setPersona(b.dataset.p)));

  // deep links: a slim banner asks the same question as the home page
  if (body.dataset.path !== "index.html") {
    const ban = document.createElement("div"); ban.id = "whoBanner"; ban.className = "who-banner";
    ban.innerHTML = `<div class="wrap"><span>Are you here as a curious parent or as a researcher?</span>
      <button type="button" class="btn small" data-p="family">Curious parent</button><button type="button" class="btn small" data-p="research">Researcher</button></div>`;
    ban.querySelectorAll("button").forEach((b) => (b.onclick = () => setPersona(b.dataset.p)));
    hdr.after(ban);
  }

  const wrap = document.createElement("div"); wrap.className = "page";
  main.parentNode.insertBefore(wrap, main); wrap.appendChild(main);
  foot = document.createElement("footer"); foot.className = "site-footer"; body.appendChild(foot);

  // light / dark: follows the system until the visitor picks one, then remembers it
  const isDark = () => (root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches);
  const tb = document.getElementById("theme");
  const paintTheme = () => { const d = isDark(); tb.textContent = d ? "☀ Light" : "☾ Dark"; tb.setAttribute("aria-label", d ? "Switch to light mode" : "Switch to dark mode"); };
  tb.onclick = () => { root.dataset.theme = isDark() ? "light" : "dark"; put("theme", root.dataset.theme); paintTheme(); window.dispatchEvent(new Event("themechange")); };
  try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", paintTheme); } catch (e) {}
  paintTheme();
  paintChrome();

  // any element with data-choose="family|research" picks an audience
  document.querySelectorAll("[data-choose]").forEach((b) => (b.onclick = (e) => { e.preventDefault(); setPersona(b.dataset.choose); scrollTo(0, 0); }));
  // tabs: <div class="tabs"><button data-tab="x"> … and panels with data-panel="x"
  document.querySelectorAll(".tabs").forEach((tabs) => {
    const btns = [...tabs.querySelectorAll("[data-tab]")];
    const show = (id) => { btns.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === id)));
      document.querySelectorAll("[data-panel]").forEach((p) => (p.hidden = p.dataset.panel !== id)); };
    btns.forEach((b) => (b.onclick = () => { show(b.dataset.tab); history.replaceState(null, "", "#" + b.dataset.tab); }));
    const h = location.hash.slice(1); show(btns.some((b) => b.dataset.tab === h) ? h : btns[0].dataset.tab);
  });
  document.querySelectorAll("[data-copy]").forEach((b) => { b.onclick = () => { const el = document.querySelector(b.dataset.copy); if (!el) return;
    (navigator.clipboard ? navigator.clipboard.writeText(el.textContent) : Promise.reject()).then(() => { b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1500); })
      .catch(() => { const r = document.createRange(); r.selectNodeContents(el); const s = getSelection(); s.removeAllRanges(); s.addRange(r); b.textContent = "Selected"; }); }; });
}
document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", build) : build();
})();

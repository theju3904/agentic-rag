"""Inline demo UI served at the root of the FastAPI app.

A single self-contained HTML page (no build step, no CDN, no web fonts) where
the visitor asks a question and sees every pipeline's answer, retrieval
latency, routing trace, and retrieved chunks side by side. ``create_app``
mounts it automatically; it talks to ``POST /compare``.
"""

from __future__ import annotations

DEMO_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<title>Agentic RAG Demo</title>
<style>
:root {
  color-scheme: light dark;
  --bg:#f6f7f9; --surface:#ffffff; --surface-2:#eef1f5; --border:#cfd6de;
  --text:#14181f; --muted:#4b5563; --accent:#2a44b8; --accent-tint:#eaeefc;
  --on-accent:#ffffff;
  --success:#17653a; --success-tint:#e1f3e8; --warn:#7a4200; --warn-tint:#fcefd8;
  --danger:#a8231b; --danger-tint:#fde8e6; --focus:#1f4fe0;
  --shadow:0 1px 2px rgb(0 0 0 / .06);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:#0f1217; --surface:#171b22; --surface-2:#1f252e; --border:#343c49;
    --text:#e8ebf0; --muted:#a6b0be; --accent:#93a8ff; --accent-tint:#1b2340;
    --on-accent:#0f1217;
    --success:#5fd08a; --success-tint:#12281c; --warn:#f0b45a; --warn-tint:#2e2210;
    --danger:#ff8f86; --danger-tint:#32181a; --focus:#b9c6ff;
    --shadow:none;
  }
}
:root[data-theme="dark"] {
  --bg:#0f1217; --surface:#171b22; --surface-2:#1f252e; --border:#343c49;
  --text:#e8ebf0; --muted:#a6b0be; --accent:#93a8ff; --accent-tint:#1b2340;
  --on-accent:#0f1217;
  --success:#5fd08a; --success-tint:#12281c; --warn:#f0b45a; --warn-tint:#2e2210;
  --danger:#ff8f86; --danger-tint:#32181a; --focus:#b9c6ff;
  --shadow:none;
}

*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font: 400 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.mono { font-family: ui-monospace, "SF Mono", Consolas, monospace; }
.muted { color: var(--muted); }
button { font: inherit; color: inherit; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }

.page { max-width: 1200px; margin: 0 auto; padding: 32px 16px 64px; }

/* header */
.hdr h1 { margin: 0 0 8px; font-size: 28px; font-weight: 700; line-height: 1.2; }
.hdr .sub { margin: 0 0 4px; font-size: 16px; }
.legend { margin: 0; font-size: 14px; color: var(--muted); }

/* ask form */
.ask { margin: 24px 0 16px; }
.ask label { display: block; font-size: 14px; font-weight: 600; margin-bottom: 6px; }
.ask-row { display: flex; gap: 8px; }
.ask-input {
  flex: 1; min-width: 0; height: 48px; padding: 0 14px; font-size: 16px;
  border: 1px solid var(--border); border-radius: 6px;
  background: var(--surface); color: var(--text);
}
.ask-input::placeholder { color: var(--muted); }
.ask-btn {
  height: 48px; min-width: 170px; padding: 0 20px; font-weight: 650; cursor: pointer;
  border: 1px solid var(--accent); border-radius: 6px;
  background: var(--accent); color: var(--on-accent);
  display: inline-flex; align-items: center; justify-content: center; gap: 8px;
}
.ask-btn:disabled {
  background: var(--surface-2); color: var(--muted); border-color: var(--border);
  cursor: progress;
}
.ask-hint { min-height: 20px; margin: 6px 0 0; font-size: 13px; color: var(--danger); }
.spinner {
  width: 14px; height: 14px; border-radius: 50%;
  border: 2px solid currentColor; border-right-color: transparent;
}
@media (prefers-reduced-motion: no-preference) {
  .spinner { animation: spin .8s linear infinite; }
}
@keyframes spin { to { transform: rotate(360deg); } }

/* examples */
.examples-label { margin: 0 0 8px; font-size: 14px; font-weight: 600; }
.examples {
  display: grid; gap: 8px; margin-bottom: 8px;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
}
.chip {
  display: flex; flex-direction: column; align-items: flex-start; gap: 2px;
  min-width: 0; min-height: 44px; padding: 8px 14px; text-align: left; cursor: pointer;
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  box-shadow: var(--shadow);
}
.chip:hover { border-color: var(--accent); }
.chip[aria-pressed="true"] { background: var(--accent-tint); border-color: var(--accent); }
.chip-tag {
  font-size: 11px; font-weight: 650; letter-spacing: .05em; text-transform: uppercase;
  color: var(--muted);
}
.chip-q {
  font-size: 14px; line-height: 1.35;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}

.status { min-height: 24px; margin: 8px 0 16px; font-size: 14px; color: var(--muted); }

/* results */
#results-h { margin: 8px 0 12px; font-size: 20px; line-height: 1.3; }
#results-h:focus { outline: none; }
.group-h { margin: 24px 0 12px; font-size: 14px; font-weight: 650; color: var(--muted);
  text-transform: uppercase; letter-spacing: .05em; }

.lat {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px; margin-bottom: 16px; box-shadow: var(--shadow);
}
.lat-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }
.lat-top h3 { margin: 0; font-size: 16px; font-weight: 650; }
.lat-top span { font-size: 12px; color: var(--muted); }
.lat-row { display: grid; grid-template-columns: 150px 1fr 72px; gap: 12px; align-items: center; padding: 4px 0; }
.lat-name { font-size: 14px; }
.lat-row--hero .lat-name { font-weight: 650; }
.lat-track { height: 10px; background: var(--surface-2); border-radius: 999px; overflow: hidden; }
.lat-bar {
  display: block; height: 100%; width: var(--w, 0%); border-radius: 999px;
  background: var(--muted); opacity: .55;
}
.lat-row--hero .lat-bar { background: var(--accent); opacity: 1; }
.lat-val { font-size: 12px; font-weight: 500; text-align: right; }
@media (prefers-reduced-motion: no-preference) {
  .lat-bar { transition: width .4s ease; }
}

.grid { display: grid; gap: 16px; grid-template-columns: 1fr; }
@media (min-width: 640px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (min-width: 1100px) { .grid { grid-template-columns: repeat(4, minmax(0, 1fr)); } }

.card {
  min-width: 0; display: flex; flex-direction: column; gap: 12px; padding: 16px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  box-shadow: var(--shadow);
}
.card--hero {
  border-left: 4px solid var(--accent); background: var(--accent-tint);
}
.card-hd { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
.card-hd h3 { margin: 0; font-size: 16px; font-weight: 650; line-height: 1.3; }
.role { margin: 2px 0 0; font-size: 13px; line-height: 1.4; color: var(--muted); }
.hd-right { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; flex: none; }
.lat-badge {
  font-size: 12px; font-weight: 500; padding: 2px 8px; border-radius: 999px;
  background: var(--surface-2); border: 1px solid var(--border);
}
.card--hero .lat-badge { background: var(--surface); }
.pill {
  font-size: 11px; font-weight: 650; padding: 1px 8px; border-radius: 999px;
  color: var(--accent); border: 1px solid var(--accent);
}
.state-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.badge {
  display: inline-flex; align-items: center; gap: 4px; font-size: 12px; font-weight: 650;
  padding: 2px 10px; border-radius: 999px;
}
.badge--ok { color: var(--success); background: var(--success-tint); }
.badge--abst { color: var(--warn); background: var(--warn-tint); }
.badge--err { color: var(--danger); background: var(--danger-tint); }
.route { font-size: 12px; padding: 2px 10px; border-radius: 6px; background: var(--surface);
  border: 1px solid var(--border); overflow-wrap: anywhere; }

.answer { font-size: 16px; font-weight: 450; line-height: 1.55; overflow-wrap: anywhere; }
.answer--abst {
  font-size: 14px; font-weight: 500; padding: 10px 12px; border-radius: 6px;
  color: var(--warn); background: var(--warn-tint); border: 1px solid var(--warn);
}
.answer-raw { margin: 6px 0 0; font-size: 12px; font-weight: 400; color: var(--muted); }
.meta { margin: 0; font-size: 12px; font-weight: 500; color: var(--muted); }

details { border-top: 1px solid var(--border); padding-top: 8px; }
summary { cursor: pointer; font-size: 14px; font-weight: 600; }
.trace ol { margin: 8px 0 0; padding-left: 20px; font-size: 12px; line-height: 1.5; }
.trace li { margin-bottom: 4px; overflow-wrap: anywhere; }
.chunk { margin-top: 10px; padding-top: 10px; border-top: 1px dashed var(--border); }
.chunk-meta { font-size: 12px; font-weight: 500; color: var(--muted); overflow-wrap: anywhere; }
.chunk-text {
  margin: 4px 0 0; font-size: 14px; line-height: 1.5; overflow-wrap: anywhere;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}
.chunk-text.open { display: block; overflow: visible; }
.more {
  margin-top: 4px; padding: 0; background: none; border: none; cursor: pointer;
  font-size: 13px; font-weight: 600; color: var(--accent); text-decoration: underline;
}

/* skeleton / empty / error */
.skel-bar { height: 12px; border-radius: 6px; background: var(--surface-2); margin-bottom: 8px; }
.skel-bar.w60 { width: 60%; } .skel-bar.w80 { width: 80%; }
@media (prefers-reduced-motion: no-preference) {
  .skel-bar, .lat-skel {
    background: linear-gradient(90deg, var(--surface-2) 25%, var(--border) 50%, var(--surface-2) 75%);
    background-size: 200% 100%; animation: shimmer 1.4s ease-in-out infinite;
  }
}
@keyframes shimmer { to { background-position: -200% 0; } }
.lat-skel { height: 10px; border-radius: 999px; background: var(--surface-2); }
.skel-note { margin: 0; font-size: 13px; color: var(--muted); }

.empty {
  padding: 24px; text-align: center; color: var(--muted);
  border: 2px dashed var(--border); border-radius: 10px;
}
.banner {
  display: flex; gap: 12px; align-items: center; justify-content: space-between;
  padding: 12px 16px; border-radius: 10px; margin-bottom: 16px;
  color: var(--danger); background: var(--danger-tint); border: 1px solid var(--danger);
}
.banner button {
  padding: 6px 14px; border-radius: 6px; cursor: pointer; font-weight: 600;
  background: var(--surface); border: 1px solid var(--danger); color: var(--danger);
}

@media (max-width: 639px) {
  .hdr h1 { font-size: 22px; }
  .ask-row { flex-direction: column; }
  .ask-btn { width: 100%; }
  .lat-row { grid-template-columns: 110px 1fr 64px; gap: 8px; }
}
</style>
</head>
<body>
<main class="page">
  <header class="hdr">
    <h1>Agentic RAG: one question, five retrieval pipelines</h1>
    <p class="sub">A per-query router picks the retrieval strategy; fixed pipelines can&rsquo;t.</p>
    <p class="legend">Each pipeline retrieves 5 passages and answers from them. P5 routes each question to the best retriever.</p>
  </header>

  <form id="ask" class="ask" autocomplete="off">
    <label for="q">Your question</label>
    <div class="ask-row">
      <input id="q" class="ask-input" type="text" placeholder="Ask a question, or pick an example" />
      <button id="go" class="ask-btn" type="submit">Compare pipelines</button>
    </div>
    <p id="hint" class="ask-hint" aria-live="polite"></p>
  </form>

  <p class="examples-label">Try an example</p>
  <div id="chips" class="examples"></div>

  <div id="status" class="status" role="status" aria-live="polite"></div>

  <h2 id="results-h" tabindex="-1" hidden>Results</h2>
  <div id="results">
    <div class="empty">No results yet. Choose an example above or type your own question.</div>
  </div>
</main>

<script>
const ORDER = ["sparse_bm25", "dense", "hybrid_rrf", "hybrid_reranked", "agentic_corrective"];
const META = {
  sparse_bm25:        { name: "P1 BM25",            role: "Keyword match" },
  dense:              { name: "P2 Dense",           role: "Embedding similarity" },
  hybrid_rrf:         { name: "P3 Hybrid",          role: "BM25 + dense, fused" },
  hybrid_reranked:    { name: "P4 Hybrid + Rerank", role: "Fused, then cross-encoder rerank" },
  agentic_corrective: { name: "P5 Agentic",         role: "Router + corrective retrieval" },
};
const HERO = "agentic_corrective";
const EXAMPLES = [
  { tag: "SciFact",      q: "Does autophagy decline in aged organisms?" },
  { tag: "NFCorpus",     q: "Do Fruit & Nut Bars Cause Weight Gain?" },
  { tag: "FiQA",         q: "Why naked call writing is risky compare to Covered call?" },
  { tag: "MultiHop-RAG", q: "Who won the 2023 Nobel Prize in chemistry and for what work?" },
  { tag: "MultiHop-RAG", q: "How much is Binance paying in fines, and what is CEO Changpeng Zhao doing?" },
  { tag: "T2-RAGBench",  q: "What was the amount of Available-for-Sale (AFS) investment securities for JPMorgan Chase at the end of 2018?" },
];

const $ = id => document.getElementById(id);
const formEl = $("ask"), qEl = $("q"), goEl = $("go"), hintEl = $("hint");
const chipsEl = $("chips"), statusEl = $("status"), resultsEl = $("results"), headEl = $("results-h");
let controller = null, runId = 0, busy = false, lastQuestion = "";

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}
function fmt(t) { return t < 1 ? t.toFixed(2) + " s" : t.toFixed(1) + " s"; }
function reducedMotion() { return window.matchMedia("(prefers-reduced-motion: reduce)").matches; }

/* example chips */
const chipButtons = EXAMPLES.map(ex => {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "chip";
  b.title = ex.q;
  b.setAttribute("aria-pressed", "false");
  b.innerHTML = `<span class="chip-tag">${esc(ex.tag)}</span><span class="chip-q">${esc(ex.q)}</span>`;
  b.addEventListener("click", () => {
    qEl.value = ex.q;
    chipButtons.forEach(c => c.setAttribute("aria-pressed", String(c === b)));
    run(true);
  });
  chipsEl.appendChild(b);
  return b;
});

formEl.addEventListener("submit", e => { e.preventDefault(); run(false); });
qEl.addEventListener("input", () => {
  hintEl.textContent = "";
  chipButtons.forEach(c => c.setAttribute("aria-pressed", "false"));
});

function setBusy(on) {
  busy = on;
  goEl.disabled = on;
  goEl.setAttribute("aria-busy", String(on));
  goEl.innerHTML = on
    ? '<span class="spinner" aria-hidden="true"></span><span>Running&hellip;</span>'
    : "Compare pipelines";
}

/* skeleton shown while the (slow) request is in flight */
function cardShell(key, bodyHtml) {
  const m = META[key];
  const hero = key === HERO;
  return `<article class="card${hero ? " card--hero" : ""}" aria-busy="true">
    <header class="card-hd"><div><h3>${esc(m.name)}</h3><p class="role">${esc(m.role)}</p></div></header>
    ${bodyHtml}</article>`;
}
function skeletonHtml() {
  const body = '<div><div class="skel-bar"></div><div class="skel-bar w80"></div><div class="skel-bar w60"></div></div>' +
               '<p class="skel-note">Waiting for results&hellip;</p>';
  const rows = ORDER.map(k =>
    `<div class="lat-row${k === HERO ? " lat-row--hero" : ""}"><span class="lat-name">${esc(META[k].name)}</span>` +
    '<span class="lat-skel"></span><span class="lat-val mono muted">&hellip;</span></div>').join("");
  return `<section class="lat" aria-label="Retrieval latency (loading)">
      <div class="lat-top"><h3>Retrieval latency</h3><span>log scale</span></div>${rows}</section>
    ${cardShell(HERO, body)}
    <h3 class="group-h">Classic pipelines</h3>
    <div class="grid">${ORDER.filter(k => k !== HERO).map(k => cardShell(k, body)).join("")}</div>`;
}

async function run(force) {
  const question = qEl.value.trim();
  if (!question) {
    hintEl.textContent = "Enter a question";
    qEl.focus();
    return;
  }
  if (busy && !force) return;
  if (controller) controller.abort();
  controller = new AbortController();
  const myRun = ++runId;
  lastQuestion = question;
  hintEl.textContent = "";
  setBusy(true);
  headEl.hidden = true;
  resultsEl.innerHTML = skeletonHtml();
  statusEl.textContent = "Running 5 pipelines.";
  try {
    const resp = await fetch("/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 5 }),
      signal: controller.signal,
    });
    if (!resp.ok) throw new Error((await resp.text()).slice(0, 300) || ("HTTP " + resp.status));
    const data = await resp.json();
    if (myRun !== runId) return;
    render(data);
  } catch (e) {
    if (myRun !== runId || e.name === "AbortError") return;
    showError(e.message || String(e));
  } finally {
    if (myRun === runId) setBusy(false);
  }
}

function showError(message) {
  headEl.hidden = true;
  resultsEl.innerHTML = `<div class="banner" role="alert">
      <span><strong aria-hidden="true">! </strong>Could not get results: ${esc(message)}.</span>
      <button type="button" id="retry">Try again</button></div>`;
  $("retry").addEventListener("click", () => run(true));
  statusEl.textContent = "";
}

function routeText(trace) {
  for (const line of trace || []) {
    const m = /selected pool '([^']+)'/.exec(line);
    if (m) return "Router chose: " + m[1];
  }
  return "";
}

function render(data) {
  const keys = ORDER.filter(k => data[k]);
  if (!keys.length) { showError("the API returned no pipeline results"); return; }

  const times = keys.map(k => Number(data[k].retrieval_latency_seconds) || 0);
  const tMin = Math.max(Math.min(...times), 0.001), tMax = Math.max(Math.max(...times), tMin);
  const fastest = keys.reduce((a, k) => (data[k].retrieval_latency_seconds < data[a].retrieval_latency_seconds ? k : a), keys[0]);
  const barW = t => tMax === tMin ? 100 :
    Math.max(2, 100 * Math.log10(1 + Math.max(t, tMin) / tMin) / Math.log10(1 + tMax / tMin));

  const latRows = keys.map(k => {
    const t = Number(data[k].retrieval_latency_seconds) || 0;
    return `<div class="lat-row${k === HERO ? " lat-row--hero" : ""}">
      <span class="lat-name">${esc(META[k].name)}</span>
      <span class="lat-track"><span class="lat-bar" style="--w:${barW(t).toFixed(1)}%"></span></span>
      <span class="lat-val mono">${fmt(t)}</span></div>`;
  }).join("");

  function card(k) {
    const r = data[k], m = META[k], hero = k === HERO;
    const abst = !!r.abstained;
    const trace = r.trace || [];
    const chunks = r.retrieved_chunks || [];
    const route = hero ? routeText(trace) : "";
    const answerHtml = abst
      ? `<div class="answer answer--abst">Abstained. The retrieved passages did not support an answer.
           <p class="answer-raw">Model said: ${esc(r.answer || "")}</p></div>`
      : `<div class="answer">${esc(r.answer || "")}</div>`;
    const traceHtml = trace.length
      ? `<details class="trace"${hero ? " open" : ""}><summary>Trace (${trace.length} step${trace.length === 1 ? "" : "s"})</summary>
           <ol class="mono">${trace.map(t => `<li>${esc(t)}</li>`).join("")}</ol></details>` : "";
    const chunksHtml = chunks.length
      ? `<details class="chunks"><summary>Retrieved chunks (${chunks.length})</summary>${chunks.map((c, i) => {
          const long = (c.text || "").length > 180;
          return `<div class="chunk"><div class="chunk-meta mono">#${i + 1} &middot; score ${Number(c.score).toFixed(4)} &middot; ${esc(c.document_id)}</div>
            <p class="chunk-text">${esc(c.text)}</p>${long ? '<button type="button" class="more" aria-expanded="false">Show more</button>' : ""}</div>`;
        }).join("")}</details>` : "";
    return `<article class="card${hero ? " card--hero" : ""}" aria-labelledby="h-${k}">
      <header class="card-hd">
        <div><h3 id="h-${k}">${esc(m.name)}</h3><p class="role">${esc(m.role)}</p></div>
        <div class="hd-right"><span class="lat-badge mono">${fmt(Number(r.retrieval_latency_seconds) || 0)}</span>${k === fastest ? '<span class="pill">Fastest</span>' : ""}</div>
      </header>
      <div class="state-row">
        ${abst ? '<span class="badge badge--abst"><span aria-hidden="true">&#9208;&#xFE0E;</span> Abstained</span>'
               : '<span class="badge badge--ok"><span aria-hidden="true">&#10003;</span> Answered</span>'}
        ${route ? `<span class="route mono">${esc(route)}</span>` : ""}
      </div>
      ${answerHtml}
      <p class="meta mono">retrieval ${fmt(Number(r.retrieval_latency_seconds) || 0)} &middot; generation ${fmt(Number(r.generation_latency_seconds) || 0)}</p>
      ${traceHtml}${chunksHtml}</article>`;
  }

  const classic = keys.filter(k => k !== HERO);
  resultsEl.innerHTML =
    `<section class="lat" aria-labelledby="lat-h">
       <div class="lat-top"><h3 id="lat-h">Retrieval latency</h3><span>log scale</span></div>${latRows}</section>` +
    (data[HERO] ? card(HERO) : "") +
    (classic.length ? `<h3 class="group-h">Classic pipelines</h3><div class="grid">${classic.map(card).join("")}</div>` : "");

  headEl.hidden = false;
  headEl.focus({ preventScroll: true });
  headEl.scrollIntoView({ block: "start", behavior: reducedMotion() ? "auto" : "smooth" });

  const answered = keys.filter(k => !data[k].abstained).length;
  statusEl.textContent = `Done. ${answered} of ${keys.length} pipelines answered. Fastest: ${META[fastest].name} (${fmt(Number(data[fastest].retrieval_latency_seconds) || 0)}).`;
}

resultsEl.addEventListener("click", e => {
  const btn = e.target.closest(".more");
  if (!btn) return;
  const text = btn.previousElementSibling;
  const open = text.classList.toggle("open");
  btn.setAttribute("aria-expanded", String(open));
  btn.textContent = open ? "Show less" : "Show more";
});
</script>
</body>
</html>
"""

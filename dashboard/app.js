/* FastRSI results dashboard. Vanilla JS, inline SVG. Reads dashboard.json (or window.DASHBOARD in the
   standalone export). Every mark comes from that file; nothing here invents a number. */
(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const PHASE = [
    ["Kernel autotune", "#e2673f"],
    ["Weight load", "#2a9d78"],
    ["JIT build and compile", "#3274bf"],
    ["CUDA graph capture", "#dba21f"],
    ["Warm-up steps", "#efc58f"],
    ["Init / state reset", "#df7fa1"],
    ["Live code install", "#6c5bc4"],
    ["The check itself", "#98a1a9"],
    ["Logprob check", "#ccd2d7"],
  ];
  const PHASE_ALIAS = {
    "Engine init": "Init / state reset",
    "Import, config, other init": "Init / state reset",
    "Run workload": "The check itself",
  };
  const phaseColor = (p) => (PHASE.find((x) => x[0] === (PHASE_ALIAS[p] || p)) || [0, "#999"])[1];
  const phaseName = (p) => PHASE_ALIAS[p] || p;

  let D = null;
  const state = { wid: null, arms: { normal: true, fastrsi: true }, dots: true, others: false, pick: {}, zoom: null };

  // ---------- helpers ----------
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, s, attrs) {
    const t = el("text", Object.assign({ x, y }, attrs || {}), parent);
    t.textContent = s;
    return t;
  }
  function svgFor(container, w, h) {
    container.innerHTML = "";
    const s = el("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, role: "img" }, container);
    return s;
  }
  const $ = (id) => document.getElementById(id);
  const widthOf = (c) => Math.max(300, Math.floor(c.getBoundingClientRect().width));
  const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  const median = (a) => { const s = [...a].sort((x, y) => x - y); const n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : NaN; };
  const secs = (t) => (t < 30 ? `${t.toFixed(1)} s` : `${Math.round(t).toLocaleString("en-US")} s`);
  const r1 = (v) => (Math.round(v * 10 + 1e-9) / 10).toFixed(1);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1;
    const raw = span / Math.max(1, n);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) || raw;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  // ---------- tooltip ----------
  const tip = () => $("tip");
  function showTip(html, ev) {
    const t = tip();
    t.innerHTML = html;
    t.hidden = false;
    const pad = 14, r = t.getBoundingClientRect();
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - pad;
    t.style.left = Math.max(8, x) + "px";
    t.style.top = Math.max(8, y) + "px";
  }
  const hideTip = () => { tip().hidden = true; };
  function hover(node, html) {
    node.addEventListener("pointermove", (ev) => showTip(typeof html === "function" ? html() : html, ev));
    node.addEventListener("pointerleave", hideTip);
  }

  // ---------- stacked horizontal bars ----------
  // rows: [{label, segs: [[phase, seconds, derived?]], total, right?, ghost?}]
  function hbars(container, rows, opts) {
    const W = widthOf(container);
    const narrow = W < 560;
    const labelW = opts.labelW !== undefined ? opts.labelW : (narrow ? 0 : 190);
    const rightW = narrow ? 124 : 130;
    const rowH = opts.rowH || 26, gap = opts.gap || 10;
    const labelAbove = narrow && labelW === 0;
    const per = rowH + gap + (labelAbove ? 16 : 0);
    const H = rows.length * per + 26;
    const s = svgFor(container, W, H);
    const x0 = labelW, x1 = W - rightW;
    const max = opts.max || Math.max(...rows.map((r) => r.total));
    const x = lin(0, max, x0, x1);
    const g = el("g", { class: "grid" }, s);
    for (const t of niceTicks(0, max, narrow ? 3 : 5)) {
      el("line", { x1: x(t), x2: x(t), y1: 0, y2: H - 20 }, g);
      text(s, x(t), H - 6, `${t}`, { "text-anchor": "middle", class: "t-small" });
    }
    rows.forEach((r, i) => {
      const y = i * per + (labelAbove ? 16 : 0);
      if (labelAbove) text(s, 0, y - 4, r.label, { class: "t-ink" });
      else text(s, labelW - 10, y + rowH / 2 + 4, r.label, { "text-anchor": "end", class: "t-ink" });
      if (r.ghost) el("rect", { x: x0, y: y + 0.5, width: Math.max(0, x(r.ghost) - x0), height: rowH - 1, class: "ghost" }, s);
      let cx = 0;
      for (const [p, v, derived] of r.segs) {
        if (v <= 0) continue;
        const rect = el("rect", { x: x(cx), y, width: Math.max(0.5, x(cx + v) - x(cx)), height: rowH, fill: phaseColor(p), class: "seg",
          "fill-opacity": derived ? 0.55 : 1 }, s);
        hover(rect, `<b>${esc(phaseName(p))}</b><br>${secs(v)}${derived ? " <span class='k'>(not split further)</span>" : ""}<br><span class="k">${esc(r.label)}, total ${secs(r.total)}</span>`);
        cx += v;
      }
      const tx = x(r.total) + 8;
      text(s, tx, y + rowH / 2 + 4, secs(r.total), { class: "t-strong" });
      if (r.right) text(s, tx + 46, y + rowH / 2 + 4, r.right, { class: "t-small" });
    });
    return s;
  }

  function ladder(container, rows, base) {
    const out = rows.map((r) => {
      const segs = Object.entries(r.seg);
      return { label: r.label + (r.measured ? "" : " (derived)"), segs, total: segs.reduce((a, b) => a + b[1], 0) };
    });
    out.forEach((o, i) => { o.ghost = i ? out[i - 1].total : null; o.right = i ? `${(base / o.total).toFixed(1)}×` : ""; });
    hbars(container, out, { max: out[0].total });
  }

  function phases(container) {
    const P = D.bands.phases;
    const panels = [P.vllm, ...P.others];
    container.innerHTML = "";
    const used = new Set();
    for (const p of panels) {
      const div = document.createElement("div");
      div.style.marginBottom = "8px";
      const h = document.createElement("div");
      h.textContent = p.title.replace("(H3, 1 GB10 GPU)", "(MiniMax H3, 1 GB10)");
      h.style.cssText = "font-weight:600;font-size:14px;margin:6px 0 4px";
      div.appendChild(h);
      const c = document.createElement("div");
      div.appendChild(c);
      container.appendChild(div);
      const rows = p.rows.map((r) => {
        const segs = r.seg.map((s) => [s[0], s[1], s[2]]);
        segs.forEach((s) => used.add(phaseName(s[0])));
        const total = segs.reduce((a, b) => a + b[1], 0);
        return { label: r.label, segs, total };
      });
      const setup = rows[0].segs.filter((s) => phaseName(s[0]) !== "The check itself" && s[0] !== "Logprob check").reduce((a, b) => a + b[1], 0);
      rows[0].right = `${Math.round((100 * setup) / rows[0].total)}% setup`;
      rows[1].right = `${(rows[0].total / rows[1].total).toFixed(1)}× less`;
      hbars(c, rows, { labelW: widthOf(container) < 560 ? 54 : 90, rowH: 22, gap: 8 });
    }
    $("lg-phases").innerHTML = PHASE.filter((p) => used.has(p[0])).map((p) => `<span><i style="background:${p[1]}"></i>${p[0]}</span>`).join("");
  }

  // ---------- strip plot (log or linear) ----------
  function strip(container, rows, opts) {
    const W = widthOf(container), narrow = W < 560;
    const labelW = narrow ? 0 : 130, rightW = narrow ? 10 : 120;
    const sub = opts.sub || 1, subH = 18, per = sub * subH + (narrow ? 30 : 16);
    const H = rows.length * per + 28;
    const s = svgFor(container, W, H);
    const x0 = labelW + 6, x1 = W - rightW;
    const xf = opts.log ? ((v) => lin(Math.log10(opts.lim[0]), Math.log10(opts.lim[1]), x0, x1)(Math.log10(Math.max(opts.lim[0], v))))
      : ((v) => lin(opts.lim[0], opts.lim[1], x0, x1)(Math.max(opts.lim[0], Math.min(opts.lim[1], v))));
    const g = el("g", { class: "grid" }, s);
    if (opts.band) el("rect", { x: xf(-opts.band), y: 0, width: xf(opts.band) - xf(-opts.band), height: H - 22, fill: "#e6ebf0" }, s);
    for (const t of opts.ticks) {
      el("line", { x1: xf(t), x2: xf(t), y1: 0, y2: H - 22 }, g);
      text(s, xf(t), H - 6, opts.tickFmt ? opts.tickFmt(t) : `${t}`, { "text-anchor": "middle", class: "t-small" });
    }
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    rows.forEach((r, i) => {
      const y = i * per + (narrow ? 16 : 4);
      text(s, narrow ? 0 : labelW, narrow ? y - 4 : y + (sub * subH) / 2 + 4, r.label, { "text-anchor": narrow ? "start" : "end", class: "t-ink" });
      r.sets.forEach((set, j) => {
        const yc = y + j * subH + subH / 2;
        const gg = el("g", {}, s);
        for (const v of set.values) {
          el("circle", { cx: xf(v), cy: yc + (rnd() - 0.5) * (subH - 6), r: 2.1, fill: set.color, "fill-opacity": 0.38 }, gg);
        }
        const m = median(set.values);
        const ring = el("circle", { cx: xf(m), cy: yc, r: 5.5, fill: "#fff", stroke: set.color, "stroke-width": 2.2 }, s);
        hover(ring, `<b>${esc(r.label)}, ${esc(set.name)}</b><br>median ${opts.valFmt(m)}<br><span class="k">${set.values.length} checks</span>`);
        const hit = el("rect", { x: x0, y: yc - subH / 2, width: x1 - x0, height: subH, fill: "transparent" }, s);
        hover(hit, () => `<b>${esc(r.label)}, ${esc(set.name)}</b><br>median ${opts.valFmt(m)}, n = ${set.values.length}`);
        hit.parentNode.insertBefore(hit, gg);
      });
      if (r.right) {
        if (narrow) text(s, W, y - 4, r.right, { class: "t-strong", "text-anchor": "end" });
        else text(s, x1 + 10, y + (sub * subH) / 2 + 4, r.right, { class: "t-strong" });
      }
    });
  }

  function editToResult() {
    const A = D.arms;
    const rows = D.bands.edit_to_result.map((r) => ({
      label: r.label,
      sets: [{ name: A.normal.short, color: A.normal.color, values: r.codex }, { name: A.fastrsi.short, color: A.fastrsi.color, values: r.livelab }],
      right: `${Math.round(r.median[0])} → ${Math.round(r.median[1])} s, ${(r.median[0] / r.median[1]).toFixed(1)}×`,
    }));
    strip($("ch-e2r"), rows, { log: true, lim: [0.1, 3000], ticks: [0.1, 1, 10, 100, 1000], sub: 2, valFmt: (v) => secs(v) });
  }

  function fidelity() {
    const f = D.bands.fidelity;
    const rows = f.rows.map((r) => {
      const v = r.values[0];
      const inside = v.filter((x) => Math.abs(x) <= f.band).length;
      return { label: r.label, sets: [{ name: "FastRSI vs " + r.note, color: D.arms.fastrsi.color, values: v }], right: `${inside}/${v.length}` };
    });
    strip($("ch-fid"), rows, { lim: f.lim, ticks: [-10, -5, 0, 5, 10], tickFmt: (t) => (t > 0 ? `+${t}%` : `${t}%`), band: f.band, sub: 1, valFmt: (v) => `${v.toFixed(2)}%` });
  }

  // ---------- paired bars (Fig. 8) ----------
  function checksPerRun() {
    const c = $("ch-checks"), W = widthOf(c), A = D.arms;
    const rows = D.bands.checks.rows;
    const half = W < 520 ? W : W / 2;
    const stacked = W < 520;
    const labelW = 112, bh = 11, per = 2 * bh + 14;
    const H = (stacked ? 2 : 1) * (rows.length * per + 44);
    const s = svgFor(c, W, H);
    const panels = [["Checks per run", 1, 2, (v) => `${v}`], ["Check-hours per run", 4, 5, (v) => r1(v)]];
    panels.forEach(([title, ic, il, f], pi) => {
      const ox = stacked ? 0 : pi * half, oy = stacked ? pi * (rows.length * per + 44) : 0;
      const lw = stacked || pi === 0 ? labelW : 12;
      const x0 = ox + lw, x1 = ox + half - 44;
      const max = Math.max(...rows.map((r) => Math.max(r[ic], r[il])));
      const x = lin(0, max, x0, x1);
      text(s, x0, oy + 14, title, { class: "t-strong" });
      rows.forEach((r, i) => {
        const y = oy + 24 + i * per;
        if (stacked || pi === 0) text(s, x0 - 8, y + bh + 4, r[0], { "text-anchor": "end", class: "t-ink" });
        [[r[ic], A.normal], [r[il], A.fastrsi]].forEach(([v, arm], j) => {
          const rect = el("rect", { x: x0, y: y + j * bh, width: x(v) - x0, height: bh - 1, fill: arm.color }, s);
          hover(rect, `<b>${esc(r[0])}</b>, ${esc(arm.short)}<br>${title.toLowerCase()}: ${f(v)}`);
          text(s, x(v) + 4, y + j * bh + bh - 2, f(v), { class: "t-small" });
        });
      });
    });
  }

  // ---------- replay (Fig. 10) small multiples ----------
  function replay() {
    const c = $("ch-replay"), W = widthOf(c), m = D.bands.replay, A = D.arms;
    const systems = [...new Set(m.rows.map((r) => r.system))];
    const cols = W < 520 ? 1 : 2, pw = W / cols, ph = 150;
    const H = Math.ceil(systems.length / cols) * (ph + 24);
    const s = svgFor(c, W, H);
    systems.forEach((sys, i) => {
      const ox = (i % cols) * pw, oy = Math.floor(i / cols) * (ph + 24);
      const x0 = ox + 30, x1 = ox + pw - 54, y0 = oy + 22, y1 = oy + ph - 8;
      const x = lin(0, m.x.length - 1, x0, x1), y = lin(0, 8, y1, y0);
      text(s, x0, oy + 12, sys, { class: "t-strong" });
      const g = el("g", { class: "grid" }, s);
      for (const t of [0, 2, 4, 6, 8]) { el("line", { x1: x0, x2: x1, y1: y(t), y2: y(t) }, g); text(s, x0 - 6, y(t) + 4, t, { "text-anchor": "end", class: "t-small" }); }
      m.x.forEach((lab, j) => { if (j % 2 === 0 || j === m.x.length - 1) text(s, x(j), y1 + 14, lab, { "text-anchor": "middle", class: "t-small" }); });
      m.rows.filter((r) => r.system === sys).forEach((r) => {
        const arm = r.arm === "Codex" ? A.normal : A.fastrsi;
        const armName = arm.short;
        el("polyline", { points: r.hours.map((h, j) => `${x(j)},${y(h)}`).join(" "), fill: "none", stroke: arm.color, "stroke-width": 2 }, s);
        r.hours.forEach((h, j) => {
          const dot = el("circle", { cx: x(j), cy: y(h), r: 3, fill: arm.color }, s);
          hover(dot, `<b>${esc(sys)}, ${esc(armName)}</b> (run ${esc(r.run)})<br>model ${esc(m.x[j])} faster: ${h.toFixed(2)} h`);
        });
        const last = r.hours[r.hours.length - 1];
        text(s, x1 + 6, y(last) + (r.arm === "Codex" ? -2 : 10), `${last.toFixed(1)} h`, { fill: arm.color, class: "t-small", style: `fill:${arm.color}` });
      });
    });
    text(s, 0, H - 2, "", {});
  }

  // ---------- ablation (Table 2) ----------
  function ablation() {
    const c = $("ch-ablation"), W = widthOf(c), rows = D.bands.ablation.rows;
    const narrow = W < 520, labelW = narrow ? 0 : 230, per = narrow ? 44 : 30;
    const H = rows.length * per + 26;
    const s = svgFor(c, W, H);
    const x0 = labelW, x1 = W - 120;
    const max = Math.max(...rows.map((r) => Math.max(r.s, r.paper_s)));
    const x = lin(0, max, x0, x1);
    const g = el("g", { class: "grid" }, s);
    for (const t of niceTicks(0, max, 4)) { el("line", { x1: x(t), x2: x(t), y1: 0, y2: H - 20 }, g); text(s, x(t), H - 6, `${t} s`, { "text-anchor": "middle", class: "t-small" }); }
    rows.forEach((r, i) => {
      const y = i * per + (narrow ? 16 : 4), bh = 18;
      text(s, narrow ? 0 : labelW - 10, narrow ? y - 4 : y + bh - 5, r.label.replace(" (paper: full Live Lab artifact)", ""), { "text-anchor": narrow ? "start" : "end", class: "t-ink" });
      const last = i === rows.length - 1;
      const rect = el("rect", { x: x0, y, width: x(r.s) - x0, height: bh, fill: last ? D.arms.normal.color : D.arms.fastrsi.color, "fill-opacity": i === 0 ? 1 : 0.72 }, s);
      const mis = Math.round(r.s) !== r.paper_s || +r1(r.h) !== r.paper_h;
      if (mis) el("line", { x1: x(r.paper_s), x2: x(r.paper_s), y1: y - 3, y2: y + bh + 3, stroke: "#8a5a00", "stroke-width": 2, "stroke-dasharray": "3 2" }, s);
      hover(rect, `<b>${esc(r.label)}</b><br>data: ${r.s.toFixed(1)} s mean check, ${r.h.toFixed(2)} checks/h<br>paper: ${r.paper_s} s, ${r.paper_h.toFixed(1)}/h${mis ? "<br><span class='k'>dashed mark: the paper's value</span>" : ""}`);
      text(s, x(r.s) + 6, y + bh - 5, `${Math.round(r.s)} s, ${r1(r.h)}/h`, { class: "t-strong" });
    });
  }

  // ---------- race chart ----------
  function W_() { return D.workloads.find((w) => w.id === state.wid); }

  function buildPicker() {
    const p = $("picker");
    p.innerHTML = "";
    const groups = [];
    for (const w of D.workloads) {
      let g = groups.find((x) => x.name === w.group);
      if (!g) groups.push((g = { name: w.group, items: [] }));
      g.items.push(w);
    }
    for (const g of groups) {
      const div = document.createElement("div");
      div.className = "pick-group";
      div.innerHTML = `<span>${esc(g.name)}</span>`;
      const row = document.createElement("div");
      row.className = "pick-row";
      for (const w of g.items) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = w.label;
        b.dataset.wid = w.id;
        b.setAttribute("aria-pressed", String(w.id === state.wid));
        b.addEventListener("click", () => selectWorkload(w.id));
        row.appendChild(b);
      }
      div.appendChild(row);
      p.appendChild(div);
    }
  }

  function selectWorkload(id) {
    state.wid = id;
    state.zoom = null;
    const w = W_();
    state.pick = {};
    for (const k of ["normal", "fastrsi"]) state.pick[k] = w.series.find((s) => s.arm === k && s.best).name;
    document.querySelectorAll(".pick-row button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.wid === id)));
    try { localStorage.setItem("livelab-dash-wid", id); } catch (e) { /* storage unavailable */ }
    buildControls();
    drawRace();
  }

  function buildControls() {
    const w = W_(), A = D.arms, c = $("controls");
    c.innerHTML = "";
    for (const k of ["normal", "fastrsi"]) {
      const runs = w.series.filter((s) => s.arm === k);
      const lab = document.createElement("label");
      lab.innerHTML = `<input type="checkbox" ${state.arms[k] ? "checked" : ""}><span class="swatch" style="background:${A[k].color}"></span>${esc(A[k].short)}`;
      lab.querySelector("input").addEventListener("change", (e) => { state.arms[k] = e.target.checked; drawRace(); });
      c.appendChild(lab);
      if (runs.length > 1) {
        const sel = document.createElement("select");
        sel.setAttribute("aria-label", `${A[k].short} run`);
        for (const r of runs) {
          const o = document.createElement("option");
          o.value = r.name;
          o.textContent = r.label.replace(/^(Codex|FastRSI) /, "") + (r.best ? " (paper)" : "");
          if (r.name === state.pick[k]) o.selected = true;
          sel.appendChild(o);
        }
        sel.addEventListener("change", () => { state.pick[k] = sel.value; drawRace(); });
        c.appendChild(sel);
      }
    }
    const mk = (label, key) => {
      const l = document.createElement("label");
      l.innerHTML = `<input type="checkbox" ${state[key] ? "checked" : ""}>${label}`;
      l.querySelector("input").addEventListener("change", (e) => { state[key] = e.target.checked; drawRace(); });
      c.appendChild(l);
    };
    mk("Every check", "dots");
    if (w.series.length > 2) mk("Other picked runs", "others");
  }

  function soonerOf(codex, live) {
    const y = codex[codex.length - 1][1];
    const tc = (codex.find((p) => p[1] >= y - 1e-9) || [null])[0];
    const tl = (live.find((p) => p[1] >= y - 1e-9) || [null])[0];
    return { target: y, tc, tl, label: tl !== null && tl < tc ? `${(tc / tl).toFixed(1)}× sooner` : "tie" };
  }

  function drawRace() {
    const w = W_(), A = D.arms, c = $("ch-race");
    $("race-title").textContent = `${w.group === "Kernels" || w.group === "Training" || w.group === "FastVideo" ? w.group + ": " : w.group + " "}${w.label}`;
    $("race-about").textContent = w.about;
    const fmtV = (v) => (w.digits === 3 ? v.toFixed(3) : `${v.toFixed(2)}×`);
    const sel = { normal: w.series.find((s) => s.name === state.pick.normal), fastrsi: w.series.find((s) => s.name === state.pick.fastrsi) };
    const others = w.series.filter((s) => s !== sel.normal && s !== sel.fastrsi && state.arms[s.arm]);
    const Wd = widthOf(c), narrow = Wd < 560;
    const Hh = narrow ? 300 : 400;
    const m = { l: narrow ? 40 : 56, r: narrow ? 64 : 84, t: 30, b: 34 };
    const s = svgFor(c, Wd, Hh);
    const xd = state.zoom || [0, w.hours];
    const visible = [sel.normal, sel.fastrsi].filter((x) => state.arms[x.arm]).concat(state.others ? others : []);
    const inX = (p) => p[0] >= xd[0] - 1e-9 && p[0] <= xd[1] + 1e-9;
    let ys = [];
    for (const v of visible) {
      // the line's value across the window: last point before it plus the points inside
      const before = v.points.filter((p) => p[0] <= xd[0]).pop();
      if (before) ys.push(before[1]);
      ys = ys.concat(v.points.filter(inX).map((p) => p[1]));
    }
    if (!ys.length) ys = [1, 1.1];
    let lo = Math.min(...ys), hi = Math.max(...ys);
    const pad = (hi - lo || 0.05) * 0.08;
    lo -= pad; hi += pad * 1.6;
    const x = lin(xd[0], xd[1], m.l, Wd - m.r), y = lin(lo, hi, Hh - m.b, m.t);
    const clipId = "clip-race";
    const defs = el("defs", {}, s);
    const cp = el("clipPath", { id: clipId }, defs);
    el("rect", { x: m.l, y: m.t - 6, width: Wd - m.r - m.l, height: Hh - m.b - m.t + 6 }, cp);

    const g = el("g", { class: "grid" }, s);
    for (const t of niceTicks(lo, hi, narrow ? 4 : 6)) {
      el("line", { x1: m.l, x2: Wd - m.r, y1: y(t), y2: y(t) }, g);
      text(s, m.l - 6, y(t) + 4, w.digits === 3 ? t.toFixed(t < 1 && (hi - lo) < 0.1 ? 3 : 2) : t.toFixed(2), { "text-anchor": "end", class: "t-small" });
    }
    for (const t of niceTicks(xd[0], xd[1], narrow ? 4 : 8)) {
      el("line", { x1: x(t), x2: x(t), y1: m.t, y2: Hh - m.b }, g);
      text(s, x(t), Hh - m.b + 16, `${+t.toFixed(2)} h`, { "text-anchor": "middle", class: "t-small" });
    }
    text(s, m.l, 14, w.y_label, { class: "t-small" });

    const plot = el("g", { "clip-path": `url(#${clipId})` }, s);
    // check dots
    if (state.dots) {
      for (const v of visible) {
        if (!v.checks || !v.checks.length) continue;
        const gd = el("g", {}, plot);
        for (const p of v.checks) {
          if (!inX(p) || p[1] < lo || p[1] > hi) continue;
          el("circle", { cx: x(p[0]), cy: y(p[1]), r: narrow ? 2 : 2.6, fill: A[v.arm].color, "fill-opacity": v === sel[v.arm] ? 0.3 : 0.14 }, gd);
        }
      }
    }
    const stepPath = (pts) => {
      let d = "";
      pts.forEach((p, i) => {
        const px = x(p[0]), py = y(p[1]);
        d += i ? ` H${px} V${py}` : `M${px},${py}`;
      });
      return d;
    };
    if (state.others) for (const v of others) el("path", { d: stepPath(v.points), fill: "none", stroke: A[v.arm].color, "stroke-width": 1, "stroke-opacity": 0.5, "stroke-dasharray": v.style === "dashed" ? "4 3" : null }, plot);
    // sooner dimension line
    const so = soonerOf(sel.normal.points, sel.fastrsi.points);
    if (state.arms.normal && state.arms.fastrsi) {
      const ty = y(so.target);
      el("line", { x1: m.l, x2: Wd - m.r, y1: ty, y2: ty, stroke: "#7b8794", "stroke-width": 0.8, "stroke-dasharray": "2 3" }, plot);
      if (so.tl !== null && so.tl < so.tc) {
        const ax = x(so.tl), bx = x(so.tc), dy = ty - 14;
        const dim = el("g", {}, s);
        el("line", { x1: ax, x2: bx, y1: dy, y2: dy, stroke: "#16202b", "stroke-width": 1.2 }, dim);
        for (const [px, dir] of [[ax, 1], [bx, -1]]) {
          el("line", { x1: px, x2: px, y1: dy - 6, y2: ty, stroke: "#16202b", "stroke-width": 1 }, dim);
          el("path", { d: `M${px},${dy} l${dir * 7},-3.5 v7 z`, fill: "#16202b" }, dim);
        }
        const lab = text(dim, (ax + bx) / 2, dy - 7, so.label, { "text-anchor": "middle", class: "t-strong" });
        lab.style.fontSize = "14px";
        hover(dim, `<b>${so.label}</b><br>${esc(A.normal.short)} reaches its final ${fmtV(so.target)} at ${so.tc.toFixed(2)} h; ${esc(A.fastrsi.short)} first reaches it at ${so.tl.toFixed(2)} h.`);
      } else {
        text(s, Wd - m.r - 4, ty - 6, "tie: FastRSI does not reach it first", { "text-anchor": "end", class: "t-small" });
      }
    }
    // best lines and step markers
    const markers = [];
    for (const k of ["normal", "fastrsi"]) {
      if (!state.arms[k]) continue;
      const v = sel[k];
      el("path", { d: stepPath(v.points), fill: "none", stroke: A[k].color, "stroke-width": 2.4, "stroke-linejoin": "miter" }, plot);
      v.points.forEach((p, i) => {
        if (i === 0 || i === v.points.length - 1 || !inX(p)) return;
        el("circle", { cx: x(p[0]), cy: y(p[1]), r: 3.4, fill: k === "normal" ? "#fff" : A[k].color, stroke: A[k].color, "stroke-width": 1.6 }, plot);
        markers.push({ px: x(p[0]), py: y(p[1]), p, i, v, k });
      });
      const last = v.points[v.points.length - 1];
      if (last[0] <= xd[1] + 1e-9) {
        const lx = x(last[0]) + 6, ly = y(last[1]) + (k === "normal" ? 14 : -4);
        const t = text(s, lx, ly, fmtV(last[1]), { class: "t-strong" });
        t.style.fill = A[k].color;
      }
    }
    // hover: nearest marker
    const overlay = el("rect", { x: m.l, y: m.t, width: Wd - m.r - m.l, height: Hh - m.b - m.t, fill: "transparent" }, s);
    const hl = el("circle", { r: 7, fill: "none", stroke: "#16202b", "stroke-width": 1.5, visibility: "hidden" }, s);
    const brush = el("rect", { class: "brush", y: m.t, height: Hh - m.b - m.t, visibility: "hidden" }, s);
    const pt = (ev) => { const r = s.getBoundingClientRect(); return [(ev.clientX - r.left) * (Wd / r.width), (ev.clientY - r.top) * (Hh / r.height)]; };
    let drag = null;
    overlay.addEventListener("pointerdown", (ev) => {
      if (ev.pointerType === "touch") return;
      drag = pt(ev)[0];
      overlay.setPointerCapture(ev.pointerId);
    });
    overlay.addEventListener("pointermove", (ev) => {
      const [mx, my] = pt(ev);
      if (drag !== null) {
        const a = Math.max(m.l, Math.min(drag, mx)), b = Math.min(Wd - m.r, Math.max(drag, mx));
        brush.setAttribute("x", a); brush.setAttribute("width", b - a); brush.setAttribute("visibility", "visible");
        hideTip();
        return;
      }
      let best = null, bd = 26 * 26;
      for (const mk of markers) { const d = (mk.px - mx) ** 2 + (mk.py - my) ** 2; if (d < bd) { bd = d; best = mk; } }
      if (!best) { hl.setAttribute("visibility", "hidden"); hideTip(); return; }
      hl.setAttribute("cx", best.px); hl.setAttribute("cy", best.py); hl.setAttribute("visibility", "visible");
      const info = best.p[2] || {};
      const prev = best.v.points[best.i - 1][1];
      const gain = w.digits === 3 ? `+${(best.p[1] - prev).toFixed(3)}` : `+${(100 * (best.p[1] / prev - 1)).toFixed(1)}%`;
      let html = `<span class="sw" style="background:${A[best.k].color}"></span><b>${esc(best.v.label)}</b>, new best at ${best.p[0].toFixed(2)} h<br>${fmtV(best.p[1])} <span class="k">(${gain} over the previous best)</span>`;
      if (info.commit) html += `<br><span class="k">commit</span> <code>${esc(info.commit.slice(0, 10))}</code>`;
      if (info.subject) html += `<br>${esc(info.subject)}`;
      else if (info.commit) html += `<br><span class="k">no commit subject saved for this run</span>`;
      else html += `<br><span class="k">commit not recovered for this step</span>`;
      showTip(html, ev);
    });
    const endDrag = (ev) => {
      if (drag === null) return;
      const mx = pt(ev)[0];
      const a = Math.min(drag, mx), b = Math.max(drag, mx);
      drag = null;
      brush.setAttribute("visibility", "hidden");
      if (b - a > 8) {
        const inv = lin(m.l, Wd - m.r, xd[0], xd[1]);
        state.zoom = [Math.max(0, inv(Math.max(m.l, a))), Math.min(w.hours, inv(Math.min(Wd - m.r, b)))];
        drawRace();
      }
    };
    overlay.addEventListener("pointerup", endDrag);
    overlay.addEventListener("pointercancel", () => { drag = null; brush.setAttribute("visibility", "hidden"); });
    overlay.addEventListener("pointerleave", () => { if (drag === null) { hl.setAttribute("visibility", "hidden"); hideTip(); } });
    overlay.addEventListener("dblclick", () => { state.zoom = null; drawRace(); });

    // note: paper check rows for this atom
    const mis = D.verification.filter((r) => r.source === `atoms/${w.atom}.json` && !r.match);
    const isPaperPick = state.pick.normal === w.series.find((q) => q.arm === "normal" && q.best).name && state.pick.fastrsi === w.series.find((q) => q.arm === "fastrsi" && q.best).name;
    const parts = [];
    if (!isPaperPick) parts.push("You are viewing runs other than the ones the paper draws.");
    for (const r of mis) parts.push(`${r.figure}: paper prints ${r.paper} for "${r.item}", the data gives ${r.dashboard}. ${r.note || ""}`);
    if (w.hidden_runs.length) parts.push(`Runs left out by the paper's selection: ${w.hidden_runs.join(", ")}.`);
    $("race-note").innerHTML = parts.map(esc).join("<br>");
    $("race-note").style.background = mis.length || !isPaperPick ? "" : "var(--live-wash)";
    $("race-note").style.borderLeftColor = mis.length || !isPaperPick ? "" : "var(--live)";
    $("race-note").style.color = mis.length || !isPaperPick ? "" : "var(--ink)";
  }

  // ---------- paper check table ----------
  function checkTable() {
    const rows = D.verification, ok = rows.filter((r) => r.match).length;
    $("check-sum").textContent = `${ok} of ${rows.length} numbers match the PDF; ${rows.length - ok} do not.`;
    const t = $("check-table");
    t.innerHTML = "<thead><tr><th>Figure</th><th>Item</th><th>Paper</th><th>Data</th><th>Match</th><th>Data file and note</th></tr></thead>";
    const tb = document.createElement("tbody");
    for (const r of rows) {
      const tr = document.createElement("tr");
      if (!r.match) tr.className = "mis";
      tr.innerHTML = `<td>${esc(r.figure)}</td><td>${esc(r.item)}</td><td>${esc(r.paper)}</td><td>${esc(r.dashboard)}</td><td class="${r.match ? "" : "no"}">${r.match ? "yes" : "no"}</td><td><code>${esc(r.source)}</code> ${esc(r.note || "")}</td>`;
      tb.appendChild(tr);
    }
    t.appendChild(tb);
  }

  // ---------- band summaries, notes ----------
  function summaries() {
    const lad = D.bands.ladder;
    const tot = (r) => Object.values(r.seg).reduce((a, b) => a + b, 0);
    const a0 = tot(lad.a[0]), a4 = tot(lad.a[4]), b2 = tot(lad.b[2]);
    $("ix-a").textContent = `${secs(a0)} restart to ${secs(a4)} live`;
    $("sum-a").innerHTML = `On vLLM, a restart spends <b>${secs(a0 - lad.a[0].seg["Run workload"] - lad.a[0].seg["Logprob check"])}</b> before the check starts; keeping the process alive brings the whole check to <b>${secs(a4)}</b> (${(a0 / a4).toFixed(1)}×).`;
    const e = D.bands.edit_to_result;
    const ratios = e.map((r) => r.median[0] / r.median[1]);
    $("ix-b").textContent = `${Math.min(...ratios).toFixed(1)}-${Math.max(...ratios).toFixed(1)}× faster checks`;
    $("sum-b").innerHTML = `A probe gets the live check to <b>${secs(b2)}</b>; inside the races a live check returns <b>${Math.min(...ratios).toFixed(1)}-${Math.max(...ratios).toFixed(1)}×</b> sooner than a restart, and agrees with trusted cold runs.`;
    const main = D.workloads.filter((w) => ["main-vllm", "main-systems"].includes(w.selection));
    const so = main.map((w) => w.sooner).filter((s) => s.label !== "tie").map((s) => s.t_codex / s.t_livelab);
    const ties = main.length - so.length;
    $("ix-c").textContent = `${Math.min(...so).toFixed(1)}-${Math.max(...so).toFixed(1)}× sooner on ${so.length} of ${main.length} panels`;
    $("sum-c").innerHTML = `FastRSI reaches Codex's final result <b>${Math.min(...so).toFixed(1)}-${Math.max(...so).toFixed(1)}× sooner</b> on ${so.length} of ${main.length} paper panels (${ties} ties). Pick a workload; kernels and nanoGPT come from the Oct 29 draft.`;

    const note = (id, pred) => {
      const r = D.verification.filter(pred);
      $(id).innerHTML = r.map((x) => `<b>${esc(x.item)}</b>: paper ${esc(x.paper)}, data ${esc(x.dashboard)}.${x.note ? " " + esc(x.note.replace(/\.?$/, ".")) : ""}`).join("<br>");
    };
    note("note-fid", (r) => r.figure.startsWith("Fig. 9") && !r.match);
    note("note-replay", (r) => r.figure.startsWith("Fig. 10") && !r.match);
    note("note-ablation", (r) => r.figure.startsWith("Table 2") && !r.match);
    note("note-phases", (r) => r.figure.startsWith("Fig. 3") && !r.match);
    $("built").textContent = D.built_at.replace("T", " ").replace("Z", " UTC");
  }

  // ---------- bands open/close ----------
  function bandToggles() {
    document.querySelectorAll(".band-head").forEach((b) => {
      b.addEventListener("click", () => {
        const open = b.getAttribute("aria-expanded") !== "true";
        b.setAttribute("aria-expanded", String(open));
        $(b.getAttribute("aria-controls")).hidden = !open;
        if (open) drawAll();
      });
    });
  }

  function drawAll() {
    const base = D.bands.ladder;
    const tot = (r) => Object.values(r.seg).reduce((a, b) => a + b, 0);
    if (!$("body-a").hidden) { ladder($("ch-ladder-a"), base.a, tot(base.a[0])); phases($("ch-phases")); }
    if (!$("body-b").hidden) { ladder($("ch-ladder-b"), base.b, tot(base.b[0])); editToResult(); checksPerRun(); fidelity(); }
    if (!$("body-c").hidden) { drawRace(); replay(); ablation(); }
  }

  function start(data) {
    D = data;
    let wid = D.workloads[1].id;
    try { const s = localStorage.getItem("livelab-dash-wid"); if (s && D.workloads.some((w) => w.id === s)) wid = s; } catch (e) { /* storage unavailable */ }
    state.wid = wid;
    buildPicker();
    selectWorkload(wid);
    summaries();
    checkTable();
    bandToggles();
    $("zoom-reset").addEventListener("click", () => { state.zoom = null; drawRace(); });
    drawAll();
    let t = null;
    let lastW = window.innerWidth;
    window.addEventListener("resize", () => {
      if (window.innerWidth === lastW) return;
      lastW = window.innerWidth;
      clearTimeout(t);
      t = setTimeout(drawAll, 120);
    });
  }

  if (window.DASHBOARD) start(window.DASHBOARD);
  else fetch("dashboard.json").then((r) => { if (!r.ok) throw new Error(`dashboard.json: HTTP ${r.status}`); return r.json(); }).then(start).catch((e) => {
    document.querySelector("main").innerHTML = `<div class="wrap"><p class="note">Could not load dashboard.json (${esc(e.message)}). Run <code>python3 dashboard/build.py</code>, then <code>python3 dashboard/serve.py</code>.</p></div>`;
  });
})();

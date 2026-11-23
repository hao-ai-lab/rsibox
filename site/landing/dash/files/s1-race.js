// Section 1, #sooner: one animated race per workload.
// Datasets: race_workloads (labels), race_steps (best-so-far steps), race_checks (every check), race_runs (every run per side, for "Show other runs").
// race_why (the agent's commit message and key diff hunk per new best, for the "Why it improved" view),
// race_diffs (the whole commit diff per new best, shown once a point is clicked).
// Calcs: race_sooner (page title), race_targets (chart), race_run_counts (caption), race_best_checks (tooltips).

dash.calc('race_best_checks', {
  inputs: ['race_steps', 'race_checks'],
  title: 'Checks run up to each new best',
  description: 'Per new best: how many checks its side had run by that hour (checks at or before it in race_checks). Empty for workloads without a per-check log.',
  fn: (race_steps, race_checks) => {
    const ck = {};
    race_checks.forEach((c) => (ck[c.workload + '|' + c.arm] = ck[c.workload + '|' + c.arm] || []).push(Number(c.t_h)));
    return race_steps.filter((r) => r.kind === 'best').map((r) => {
      const l = ck[r.workload + '|' + r.arm];
      return { key: r.workload + '.' + r.arm + '.' + r.step, workload: r.workload, arm: r.arm, step: r.step,
        checks_so_far: l ? l.filter((t) => t <= Number(r.t_h) + 1e-9).length : null };
    });
  },
});

dash.calc('race_run_counts', {
  inputs: ['race_runs'],
  title: 'Runs per side',
  description: 'Per workload: how many runs each side has in the race data (the main pair plus any other runs).',
  fn: (race_runs) => {
    const by = {};
    race_runs.forEach((r) => { const o = (by[r.workload] = by[r.workload] || { codex: new Set(), rsibox: new Set() }); if (o[r.arm]) o[r.arm].add(r.run); });
    return Object.keys(by).map((workload) => ({ workload, codex_runs: by[workload].codex.size, rsibox_runs: by[workload].rsibox.size }));
  },
});

dash.calc('race_sooner', {
  inputs: ['race_steps'],
  title: "When rsibox reaches Baseline's final result",
  description: "Per workload: Baseline's final best (target), the hour Baseline first reached it, the hour rsibox's best first matched or beat it, and how many times sooner that is. No hour for rsibox means a tie.",
  fn: (race_steps) => {
    const by = {};
    for (const r of race_steps) {
      const k = r.workload;
      (by[k] = by[k] || { codex: [], rsibox: [] })[r.arm].push({ t: Number(r.t_h), v: Number(r.value), kind: r.kind });
    }
    return Object.keys(by).map((workload) => {
      const c = by[workload].codex.filter((p) => p.kind !== 'end').sort((a, b) => a.t - b.t);
      const f = by[workload].rsibox.filter((p) => p.kind !== 'end').sort((a, b) => a.t - b.t);
      const target = Math.max(...c.map((p) => p.v));
      const t_codex = c.find((p) => p.v >= target).t;
      const hit = f.find((p) => p.v >= target);
      const t_rsibox = hit ? hit.t : null;
      const ratio = hit && hit.t > 0 ? Math.round((t_codex / hit.t) * 100) / 100 : null;
      return { workload, target, t_codex, t_rsibox, ratio };
    });
  },
});

dash.calc('race_targets', {
  inputs: ['race_steps'],
  title: 'Time to reach each result Baseline reached',
  description: "Per workload and each best value Baseline reached: the hour Baseline first got there, the hour rsibox's best first matched or beat it (empty when it never did), how many times sooner that is (or, where rsibox got there later, how many times longer it took: later_ratio), and the time between them in hours and in minutes. is_final marks Baseline's final best.",
  fn: (race_steps) => {
    const by = {};
    for (const r of race_steps) (by[r.workload] = by[r.workload] || { codex: [], rsibox: [] })[r.arm].push({ t: Number(r.t_h), v: Number(r.value), kind: r.kind });
    const out = [];
    Object.keys(by).forEach((workload) => {
      const c = by[workload].codex.filter((p) => p.kind === 'best').sort((a, b) => a.t - b.t);
      const f = by[workload].rsibox.filter((p) => p.kind !== 'end').sort((a, b) => a.t - b.t);
      const top = Math.max(...c.map((p) => p.v));
      c.forEach((p, i) => {
        const hit = f.find((q) => q.v >= p.v);
        const gap = hit ? p.t - hit.t : null;
        out.push({
          key: workload + '.' + i,
          workload,
          target: p.v,
          t_codex: p.t,
          t_rsibox: hit ? hit.t : null,
          ratio: hit && hit.t > 0 ? Math.round((p.t / hit.t) * 100) / 100 : null,
          later_ratio: hit && p.t > 0 && hit.t > p.t ? Math.round((hit.t / p.t) * 100) / 100 : null,
          gap_h: gap == null ? null : Math.round(Math.abs(gap) * 10) / 10,
          gap_min: gap == null ? null : Math.round(Math.abs(gap) * 60),
          rsibox_first: gap == null ? null : gap >= 0 ? 'yes' : 'no',
          is_final: p.v === top ? 'yes' : 'no',
        });
      });
    });
    return out;
  },
});

(function () {
  const FR = window.FR;
  const S = (FR.race = FR.race || { wid: null, u: 1, playing: false, raf: null, render: null, popped: new Set(), others: false, view: 'checks', pinned: false });
  const DURATION = 13.5; // seconds for a whole race

  function workloads() {
    const r = FR.ok('race_workloads'), st = FR.ok('race_steps');
    if (!r) return [];
    const has = st ? new Set(st.data.map((x) => x.workload)) : null;
    return r.data.filter((w) => !has || has.has(w.workload)).sort((a, b) => Number(a.order) - Number(b.order));
  }
  function current() {
    const ws = workloads();
    return ws.find((w) => w.workload === S.wid) || ws[0] || null;
  }

  // Controls: play button and scrub slider live in the HTML; scripts only listen and set allowed attributes.
  const playBtns = () => document.querySelectorAll('#race-play');
  const scrubs = () => document.querySelectorAll('#race-scrub');
  function syncControls() {
    playBtns().forEach((b) => {
      b.setAttribute('aria-pressed', String(S.playing));
      b.querySelectorAll('.fr-lbl-play').forEach((n) => (n.hidden = S.playing));
      b.querySelectorAll('.fr-lbl-pause').forEach((n) => (n.hidden = !S.playing));
    });
    scrubs().forEach((s) => (s.value = String(Math.round(S.u * 1000))));
  }
  function stop() {
    S.playing = false;
    if (S.raf) cancelAnimationFrame(S.raf);
    S.raf = null;
    syncControls();
  }
  function play() {
    if (FR.reduced()) { S.u = 1; stop(); if (S.render) S.render(false); return; }
    if (S.u >= 0.999) { S.u = 0; S.popped = new Set(); }
    S.playing = true;
    syncControls();
    let last = null;
    const tick = (now) => {
      if (last == null) last = now;
      S.u = Math.min(1, S.u + (now - last) / 1000 / DURATION);
      last = now;
      if (S.render) S.render(true);
      syncControls();
      if (S.u < 1 && S.playing) S.raf = requestAnimationFrame(tick);
      else stop();
    };
    S.raf = requestAnimationFrame(tick);
  }
  FR.race.select = (wid) => {
    stop();
    S.wid = wid;
    S.u = 1;
    S.popped = new Set();
    syncControls();
    dash.draw('race-picker');
    dash.draw('race');
  };
  S.tsel = S.tsel || {};
  const views = () => document.querySelectorAll('#race-view-checks, #race-view-why');
  const syncView = () => views().forEach((b) => b.setAttribute('aria-pressed', String((b.id === 'race-view-why') === (S.view === 'why'))));
  views().forEach((b) => b.addEventListener('click', () => { S.view = b.id === 'race-view-why' ? 'why' : 'checks'; S.pinned = false; syncView(); dash.draw('race'); }));
  syncView();
  document.querySelectorAll('#race-others').forEach((c) => {
    c.checked = !!S.others;
    c.addEventListener('change', () => { S.others = c.checked; dash.draw('race'); });
  });
  document.querySelectorAll('#race-target-reset').forEach((b) => b.addEventListener('click', () => { if (S.setTarget) S.setTarget(null); }));
  playBtns().forEach((b) => b.addEventListener('click', () => (S.playing ? stop() : play())));
  scrubs().forEach((s) => s.addEventListener('input', () => {
    const u = Number(s.value) / 1000;
    S.u = u;
    stop();
    if (S.render) S.render(false);
  }));

  // Picker: one button per workload, labels from race_workloads.
  dash.draw('race-picker', (el) => {
    const ws = workloads();
    if (!ws.length) { el.replaceChildren(); return; }
    const cur = current();
    el.replaceChildren(...ws.map((w) => {
      const b = FR.el('button', 'fr-chip', w.label);
      b.type = 'button';
      b.setAttribute('aria-pressed', String(cur && cur.workload === w.workload));
      FR.mark(b, 'race_workloads', 'label', { workload: w.workload });
      b.addEventListener('click', () => FR.race.select(w.workload));
      return b;
    }));
  });

  const fx = (v, d) => Number(v).toFixed(d == null ? 2 : d) + '×';
  const fh = (t) => Number(t).toFixed(1) + ' h';

  function stepPath(pts, x, y, end) {
    if (!pts.length) return '';
    let d = `M${x(pts[0].t).toFixed(1)},${y(pts[0].v).toFixed(1)}`;
    for (let i = 1; i < pts.length; i++) d += `H${x(pts[i].t).toFixed(1)}V${y(pts[i].v).toFixed(1)}`;
    return d + `H${x(end).toFixed(1)}`;
  }

  // "3.1 h" or, under an hour, "45 min", marked against the calc's own field.
  function gapText(parent, tr, cls) {
    const where = { key: tr.key };
    if (Number(tr.gap_h) >= 1) { FR.mark(FR.el('span', cls, Number(tr.gap_h).toFixed(1), parent), 'race_targets', 'gap_h', where); parent.append(' h'); }
    else { FR.mark(FR.el('span', cls, String(tr.gap_min), parent), 'race_targets', 'gap_min', where); parent.append(' min'); }
  }

  function stats(m, w, tr, ends) {
    const box = m.stats;
    box.replaceChildren();
    const s1 = FR.el('div', 'fr-stat', null, box);
    const big = FR.el('div', 'fr-stat-big', null, s1);
    const lbl = FR.el('div', 'fr-stat-lbl', null, s1);
    if (!tr) { big.textContent = '—'; return; }
    const where = { key: tr.key };
    const which = tr.is_final === 'yes' ? " (Baseline's final)" : ' (picked on the chart)';
    if (tr.t_rsibox != null && tr.ratio != null) {
      const later = tr.rsibox_first === 'no' && tr.later_ratio != null;
      FR.mark(FR.el('span', null, Number(later ? tr.later_ratio : tr.ratio).toFixed(1), big), 'race_targets', later ? 'later_ratio' : 'ratio', where);
      big.append('×');
      const sm = FR.el('span', 'fr-stat-mid', null, big);
      sm.append(later ? ' later, ' : ' sooner, ');
      gapText(sm, tr, null);
      sm.append(tr.rsibox_first === 'yes' ? ' earlier' : ' later');
      lbl.append('To reach ');
      FR.mark(FR.el('span', 'fr-num', fx(tr.target), lbl), 'race_targets', 'target', where);
      lbl.append(which + ': rsibox at ');
      FR.mark(FR.el('span', 'fr-num', Number(tr.t_rsibox).toFixed(1), lbl), 'race_targets', 't_rsibox', where);
      lbl.append(' h, Baseline at ');
      FR.mark(FR.el('span', 'fr-num', Number(tr.t_codex).toFixed(1), lbl), 'race_targets', 't_codex', where);
      lbl.append(' h.');
    } else {
      big.textContent = 'Not reached';
      lbl.append('rsibox never reaches ');
      FR.mark(FR.el('span', 'fr-num', fx(tr.target), lbl), 'race_targets', 'target', where);
      lbl.append(which + ' within ');
      FR.mark(FR.el('span', 'fr-num', w.budget_h, lbl), 'race_workloads', 'budget_h', { workload: w.workload });
      lbl.append(' h; Baseline got there at ');
      FR.mark(FR.el('span', 'fr-num', Number(tr.t_codex).toFixed(1), lbl), 'race_targets', 't_codex', where);
      lbl.append(' h.');
    }
    for (const arm of ['codex', 'rsibox']) {
      const e = ends[arm];
      const s = FR.el('div', 'fr-stat ' + (arm === 'codex' ? 'is-codex' : 'is-fast'), null, box);
      const b = FR.el('div', 'fr-stat-big', null, s);
      if (e) FR.mark(FR.el('span', null, fx(e.v), b), 'race_steps', 'value', { workload: w.workload, arm, kind: 'end' });
      else b.textContent = '—';
      const l = FR.el('div', 'fr-stat-lbl', FR.short[arm] + ' final, ', s);
      FR.mark(FR.el('span', 'fr-num', arm === 'codex' ? w.codex_run : w.rsibox_run, l), 'race_workloads', arm === 'codex' ? 'codex_run' : 'rsibox_run', { workload: w.workload });
    }
  }

  // Diff text to colored lines: "FILE name" rows become sticky file headers, @@ hunk headers, + added, - removed.
  function diffLines(box, text) {
    const inner = FR.el('div', 'fr-diff-in', null, box);
    String(text).split('\n').forEach((ln) => {
      const c = ln.startsWith('FILE ') ? 'f' : ln.startsWith('@@') ? 'h' : ln.startsWith('+') ? 'a' : ln.startsWith('-') ? 'd' : ln.startsWith('…') ? 'm' : 'c';
      FR.el('div', 'dl dl-' + c, (c === 'f' ? ln.slice(5) : ln) || ' ', inner);
    });
    return inner;
  }
  document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape' && S.pinned && S.unpin) S.unpin(); });

  function build(m, w, steps, checks, targets, ends0, others, why, bc, diffs) {
    const box = m.box;
    box.replaceChildren(m.tip.node);
    m.tip.hide();
    const W = Math.max(280, box.clientWidth || 640);
    const narrow = W < 520;
    const H = Math.round(Math.max(250, Math.min(380, W * 0.56)));
    const budget = Number(w.budget_h);
    const series = { codex: [], rsibox: [] };
    steps.forEach((r) => series[r.arm] && series[r.arm].push({ t: Number(r.t_h), v: Number(r.value), kind: r.kind, step: r.step, commit: r.commit, subject: r.subject, arm: r.arm }));
    for (const k in series) series[k].sort((a, b) => a.t - b.t || Number(a.step) - Number(b.step));
    const all = [...series.codex, ...series.rsibox, ...(S.others ? others.flatMap((o) => o.pts) : [])].map((p) => p.v);
    const lo = Math.min(1, ...all), hi = Math.max(...all);
    const pad = (hi - lo) * 0.08 || 0.02;
    const yDom = [lo - pad * 0.4, hi + pad];
    const svg = FR.svg('svg', { class: 'fr-svg', role: 'img', width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    box.insertBefore(svg, m.tip.node);
    const yTicks = d3.scaleLinear().domain(yDom).ticks(H < 300 ? 4 : 5);
    const ml = Math.ceil(Math.max(...yTicks.map((v) => FR.textW(svg, fx(v))))) + 12;
    const ends = { codex: series.codex.find((p) => p.kind === 'end'), rsibox: series.rsibox.find((p) => p.kind === 'end') };
    const mr = Math.ceil(Math.max(FR.textW(svg, '0.00×'), 40)) + 14;
    const mt = 36, mb = 26;
    const x = d3.scaleLinear([0, budget], [ml, W - mr]);
    const y = d3.scaleLinear(yDom, [H - mb, mt]);
    svg.setAttribute('aria-label', `${w.label}: best speedup over ${budget} hours. Baseline ends at ${ends.codex ? fx(ends.codex.v) : 'unknown'}, rsibox at ${ends.rsibox ? fx(ends.rsibox.v) : 'unknown'}. Drag the dashed line to pick another result to reach.`);
    // grid and axes
    const g0 = FR.svg('g', {}, svg);
    yTicks.forEach((v) => {
      FR.svg('line', { class: 'gl', x1: ml, x2: W - mr, y1: y(v), y2: y(v) }, g0);
      FR.svg('text', { x: ml - 8, y: y(v) + 4, 'text-anchor': 'end' }, g0).textContent = fx(v);
    });
    const hstep = (W - ml - mr) / budget < 46 ? 2 : 1;
    for (let h = 0; h <= budget + 1e-9; h += hstep) FR.svg('text', { x: x(h), y: H - 8, 'text-anchor': 'middle' }, g0).textContent = h + ' h';
    // checks
    const marks = [];
    const ckG = FR.svg('g', {}, svg);
    checks.forEach((c) => {
      const t = Number(c.t_h), v = Number(c.value);
      if (v < yDom[0] || v > yDom[1] || t > budget) return;
      const n = FR.svg('circle', { class: 'ck ' + c.arm, cx: x(t).toFixed(1), cy: y(v).toFixed(1), r: narrow ? 2 : 2.5 }, ckG);
      marks.push({ t, n, kind: 'ck' });
    });
    // step lines, clipped at the playhead
    const cid = 'fr-race-clip-' + Math.random().toString(36).slice(2, 8);
    const clipRect = FR.svg('rect', { x: 0, y: 0, height: H, width: W }, FR.svg('clipPath', { id: cid }, FR.svg('defs', {}, svg)));
    const lg = FR.svg('g', { 'clip-path': `url(#${cid})` }, svg);
    if (S.others) others.forEach((o) => {
      const pth = FR.svg('path', { class: 'st other ' + o.arm, d: stepPath(o.pts, x, y, budget) }, lg);
      FR.svg('title', {}, pth).textContent = o.run;
    });
    for (const arm of ['codex', 'rsibox']) FR.svg('path', { class: 'st ' + arm, d: stepPath(series[arm], x, y, budget) }, lg);
    // new-best markers
    const nbG = FR.svg('g', {}, svg);
    const pts = [];
    for (const arm of ['codex', 'rsibox']) series[arm].forEach((p) => {
      if (p.kind === 'end') return;
      p.x = x(p.t); p.y = y(p.v);
      pts.push(p);
      if (p.kind !== 'best') return;
      const n = FR.svg('circle', { class: 'nb ' + arm, cx: p.x.toFixed(1), cy: p.y.toFixed(1), r: narrow ? 3 : 3.6 }, nbG);
      marks.push({ t: p.t, n, kind: 'nb', key: arm + p.step });
    });
    // where rsibox reaches the target (drawn by setTarget)
    const reachG = FR.svg('g', {}, svg);
    let reachDot = null, reachBr = null, cur = null;
    // playhead and clock
    const ph = FR.svg('line', { class: 'ph', y1: mt - 6, y2: H - mb }, svg);
    const clock = FR.svg('text', { class: 'clock', x: ml, y: 22 }, svg);
    const clockSub = FR.svg('text', { x: ml + 64, y: 21 }, svg);
    clockSub.textContent = 'of ' + budget + ' h';
    // head labels while playing, end labels at rest
    const heads = {}, endLbls = {};
    for (const arm of ['codex', 'rsibox']) {
      heads[arm] = FR.svg('text', { class: 'lbl-' + arm }, svg);
      endLbls[arm] = FR.svg('text', { class: 'end ' + arm, x: W - mr + 6 }, svg);
    }
    const order = ['codex', 'rsibox'].filter((a) => ends[a]).sort((a, b) => ends[b].v - ends[a].v);
    let prev = -Infinity;
    order.forEach((a) => {
      let yy = y(ends[a].v) + 4;
      if (yy - prev < 14) yy = prev + 14;
      prev = yy;
      endLbls[a].setAttribute('y', yy);
      FR.mark(endLbls[a], 'race_steps', 'value', { workload: w.workload, arm: a, kind: 'end' });
      endLbls[a].textContent = fx(ends[a].v);
    });
    // hover
    const hot = FR.svg('circle', { class: 'hot', r: 7, visibility: 'hidden' }, svg);
    const hit = FR.svg('rect', { x: ml, y: mt - 8, width: W - ml - mr + 8, height: H - mt - mb + 8, fill: 'transparent' }, svg);
    const onMove = (ev) => {
      const r = svg.getBoundingClientRect();
      const sx = (ev.clientX - r.left) * (W / r.width), sy = (ev.clientY - r.top) * (H / r.height);
      const hNow = S.u * budget;
      let best = null, bd = Infinity;
      pts.forEach((p) => {
        if (p.t > hNow + 1e-9) return;
        const d = Math.hypot(p.x - sx, (p.y - sy) * 0.6);
        if (d < bd) { bd = d; best = p; }
      });
      if (S.pinned) return;
      if (!best || bd > 60) { hot.setAttribute('visibility', 'hidden'); m.tip.hide(); return; }
      showTip(best, r);
    };
    const showTip = (best, r) => {
      hot.setAttribute('cx', best.x); hot.setAttribute('cy', best.y); hot.setAttribute('visibility', 'visible');
      const where = { workload: w.workload, arm: best.arm, step: best.step };
      const isWhy = S.view === 'why';
      m.tip.node.classList.toggle('fr-tip-why', isWhy);
      m.tip.show(best.x * (r.width / W), best.y * (r.height / H), (t) => {
        FR.el('b', best.arm, FR.arms[best.arm], t);
        const l1 = FR.el('div', null, null, t);
        FR.mark(FR.el('span', 'tv', fx(best.v, 3), l1), 'race_steps', 'value', where);
        l1.append(best.kind === 'start' ? ' starting code at ' : ' new best at ');
        FR.mark(FR.el('span', 'tv', best.t.toFixed(2), l1), 'race_steps', 't_h', where);
        l1.append(' h');
        const k = bc[best.arm + '.' + best.step];
        if (best.kind === 'best' && k && k.checks_so_far != null) {
          l1.append(', after ');
          FR.mark(FR.el('span', 'tv', String(k.checks_so_far), l1), 'race_best_checks', 'checks_so_far', { key: k.key });
          l1.append(Number(k.checks_so_far) === 1 ? ' check' : ' checks');
        }
        if (!isWhy || best.kind !== 'best') return;
        const y1 = why[best.arm + '.' + best.step];
        if (!y1) {
          if (w.why_missing) FR.mark(FR.el('div', 'tm', w.why_missing, t), 'race_workloads', 'why_missing', { workload: w.workload });
          else FR.el('div', 'tm', 'no description recorded', t);
          return;
        }
        if (S.pinned) {
          const x = FR.el('button', 'fr-tip-close', '×', t);
          x.type = 'button';
          x.setAttribute('aria-label', 'Close');
          x.addEventListener('click', () => unpin());
        }
        FR.mark(FR.el('p', 'fr-why-msg', y1.message, t), 'race_why', 'message', where);
        const full = S.pinned ? diffs[best.arm + '.' + best.step] : null;
        if (full && full.diff) {
          if (y1.stat) FR.mark(FR.el('div', 'tm', y1.stat, t), 'race_why', 'stat', where);
          const bx = FR.mark(FR.el('div', 'fr-diff fr-diff-full', null, t), 'race_diffs', 'diff', where);
          bx.tabIndex = 0;
          diffLines(bx, full.diff);
          if (Number(full.more_lines) > 0) {
            const mo = FR.el('div', 'tm', null, t);
            FR.mark(FR.el('span', 'tv', String(full.more_lines), mo), 'race_diffs', 'more_lines', where);
            mo.append(' more lines not shown (first 400 of ');
            FR.mark(FR.el('span', 'tv', String(full.lines), mo), 'race_diffs', 'lines', where);
            mo.append(').');
          }
          FR.el('div', 'tm', 'Whole commit, every file. Close with ×, Esc, or a click elsewhere on the chart.', t);
          return;
        }
        if (y1.diff) {
          FR.mark(FR.el('div', 'fr-why-file', y1.file, t), 'race_why', 'file', where);
          diffLines(FR.mark(FR.el('div', 'fr-diff', null, t), 'race_why', 'diff', where), y1.diff);
        } else if (y1.diff_note) FR.mark(FR.el('div', 'tm', y1.diff_note, t), 'race_why', 'diff_note', where);
        if (y1.stat) FR.mark(FR.el('div', 'tm', y1.stat, t), 'race_why', 'stat', where);
        FR.el('div', 'tm', S.pinned ? 'Close with ×, Esc, or a click elsewhere on the chart.' : 'Click the point to see the whole diff.', t);
      });
    };
    const unpin = () => { S.pinned = false; m.tip.node.classList.remove('is-pinned'); hot.setAttribute('visibility', 'hidden'); m.tip.hide(); };
    S.unpin = unpin;
    hit.addEventListener('pointermove', onMove);
    hit.addEventListener('pointerdown', onMove);
    hit.addEventListener('pointerleave', () => { if (S.pinned) return; hot.setAttribute('visibility', 'hidden'); m.tip.hide(); });
    // a click on one of Baseline's new bests picks it as the target; in the why view a click pins the tooltip instead
    hit.addEventListener('click', (ev) => {
      const r = svg.getBoundingClientRect();
      const sx = (ev.clientX - r.left) * (W / r.width), sy = (ev.clientY - r.top) * (H / r.height);
      if (S.view === 'why') {
        const hNow = S.u * budget;
        let near = null, nd = 24;
        pts.forEach((p) => { if (p.t > hNow + 1e-9) return; const d = Math.hypot(p.x - sx, p.y - sy); if (d < nd) { nd = d; near = p; } });
        S.pinned = false;
        m.tip.node.classList.remove('is-pinned');
        if (near) { S.pinned = true; m.tip.node.classList.add('is-pinned'); showTip(near, r); }
        else { hot.setAttribute('visibility', 'hidden'); m.tip.hide(); }
        return;
      }
      let best = null, bd = 14;
      targets.forEach((t) => { const d = Math.hypot(x(Number(t.t_codex)) - sx, y(Number(t.target)) - sy); if (d < bd) { bd = d; best = t; } });
      if (best) setTarget(best.key);
    });

    // the target line: drag it, or focus its handle and use the arrow keys; it snaps to values Baseline reached
    const tgtG = FR.svg('g', { class: 'tgt-g' }, svg);
    const tLine = FR.svg('line', { class: 'tgt', x1: ml, x2: W - mr }, tgtG);
    const tHit = FR.svg('line', { class: 'tgt-hit', x1: ml, x2: W - mr }, tgtG);
    const tHandle = FR.svg('g', { class: 'tgt-handle', tabindex: 0, role: 'slider', 'aria-label': 'Result to reach' }, tgtG);
    FR.svg('rect', { x: ml - 2, y: -7, width: 22, height: 14, rx: 7 }, tHandle);
    FR.svg('path', { d: `M${ml + 5},-2h8M${ml + 5},2h8` }, tHandle);
    const nearest = (v) => targets.reduce((a, b) => (Math.abs(Number(b.target) - v) < Math.abs(Number(a.target) - v) ? b : a));
    let dragging = false;
    const dragTo = (ev) => {
      const r = svg.getBoundingClientRect();
      const sy = (ev.clientY - r.top) * (H / r.height);
      const t = nearest(y.invert(sy));
      if (!cur || t.key !== cur.key) setTarget(t.key);
    };
    [tHit, tHandle].forEach((n) => n.addEventListener('pointerdown', (ev) => { dragging = true; ev.preventDefault(); ev.stopPropagation(); try { n.setPointerCapture(ev.pointerId); } catch (e) {} dragTo(ev); }));
    [tHit, tHandle].forEach((n) => n.addEventListener('pointermove', (ev) => { if (dragging) dragTo(ev); }));
    [tHit, tHandle].forEach((n) => n.addEventListener('pointerup', () => { dragging = false; }));
    tHandle.addEventListener('keydown', (ev) => {
      if (!cur) return;
      const i = targets.findIndex((t) => t.key === cur.key);
      const k = ev.key;
      let j = i;
      if (k === 'ArrowUp' || k === 'ArrowRight') j = Math.min(targets.length - 1, i + 1);
      else if (k === 'ArrowDown' || k === 'ArrowLeft') j = Math.max(0, i - 1);
      else if (k === 'Home') j = 0;
      else if (k === 'End') j = targets.length - 1;
      else return;
      ev.preventDefault();
      setTarget(targets[j].key);
      tHandle.focus();
    });

    function drawReach(tr) {
      reachG.replaceChildren();
      reachDot = reachBr = null;
      if (!tr || tr.t_rsibox == null) return;
      const rx = x(Number(tr.t_rsibox)), cx = x(Number(tr.t_codex)), ry = y(Number(tr.target));
      reachDot = FR.svg('g', { class: 'reach' }, reachG);
      FR.svg('line', { x1: rx, x2: rx, y1: ry, y2: H - mb, 'stroke-dasharray': '2 3' }, reachDot);
      FR.svg('circle', { cx: rx, cy: ry, r: 6.5 }, reachDot).setAttribute('style', 'fill:none;stroke:var(--fr-fast);stroke-width:2');
      reachBr = FR.svg('g', { class: 'reach' }, reachG);
      const by = ry - 16;
      FR.svg('path', { class: 'br', d: `M${rx},${by + 4}V${by}H${cx}V${by + 4}` }, reachBr);
      const t = FR.svg('text', { class: 'rt', y: by - 6, 'text-anchor': 'middle' }, reachBr);
      const a = FR.svg('tspan', {}, t);
      const later = tr.rsibox_first === 'no' && tr.later_ratio != null;
      FR.mark(a, 'race_targets', later ? 'later_ratio' : 'ratio', { key: tr.key });
      a.textContent = Number(later ? tr.later_ratio : tr.ratio).toFixed(1);
      t.append('× ' + (later ? 'later' : 'sooner') + ' · ');
      const big = Number(tr.gap_h) >= 1;
      const b = FR.svg('tspan', {}, t);
      FR.mark(b, 'race_targets', big ? 'gap_h' : 'gap_min', { key: tr.key });
      b.textContent = big ? Number(tr.gap_h).toFixed(1) : String(tr.gap_min);
      t.append(big ? ' h' : ' min');
      const tw = (t.getComputedTextLength && t.getComputedTextLength()) || 140;
      let tx = (rx + cx) / 2;
      tx = Math.max(ml + tw / 2 + 2, Math.min(W - mr - tw / 2, tx));
      t.setAttribute('x', tx);
      const bg = FR.svg('rect', { class: 'rbg', x: tx - tw / 2 - 4, y: by - 19, width: tw + 8, height: 17, rx: 3 });
      reachBr.insertBefore(bg, t);
    }
    function setTarget(key) {
      const fin = targets.find((t) => t.is_final === 'yes') || targets[targets.length - 1];
      const tr = (key && targets.find((t) => t.key === key)) || fin;
      if (!tr) return;
      cur = tr;
      S.tsel[w.workload] = tr === fin ? null : tr.key;
      const ty = y(Number(tr.target));
      [tLine, tHit].forEach((n) => { n.setAttribute('y1', ty); n.setAttribute('y2', ty); });
      tHandle.setAttribute('transform', `translate(0,${ty})`);
      tHandle.setAttribute('aria-valuetext', fx(tr.target) + (tr.is_final === 'yes' ? ", Baseline's final" : ''));
      drawReach(tr);
      stats(m, w, tr, ends0);
      document.querySelectorAll('#race-target-reset').forEach((b) => (b.hidden = tr === fin));
      if (S.render) S.render(false);
    }
    S.setTarget = setTarget;

    const val = (arm, h) => { let v = null; for (const p of series[arm]) { if (p.t <= h + 1e-9) v = p.v; else break; } return v; };
    const render = (animating) => {
      const h = S.u * budget, rest = S.u >= 0.999;
      clipRect.setAttribute('width', Math.max(0, x(Math.min(h, budget)) + 1));
      marks.forEach((mk) => {
        const on = mk.t <= h + 1e-9;
        mk.n.style.display = on ? '' : 'none';
        if (mk.kind === 'nb') {
          if (on && animating && !S.popped.has(mk.key)) { S.popped.add(mk.key); mk.n.classList.remove('pop'); void mk.n.getBBox; mk.n.classList.add('pop'); }
          if (!on) S.popped.delete(mk.key);
        }
      });
      if (reachDot) reachDot.style.display = h >= Number(cur.t_rsibox) - 1e-9 ? '' : 'none';
      if (reachBr) reachBr.style.display = h >= Math.max(Number(cur.t_codex), Number(cur.t_rsibox)) - 1e-9 ? '' : 'none';
      ph.setAttribute('x1', x(h)); ph.setAttribute('x2', x(h));
      ph.style.display = rest ? 'none' : '';
      clock.textContent = h.toFixed(1) + ' h';
      for (const arm of ['codex', 'rsibox']) {
        endLbls[arm].style.display = rest ? '' : 'none';
        const v = val(arm, h);
        if (rest || v == null) { heads[arm].style.display = 'none'; continue; }
        heads[arm].style.display = '';
        heads[arm].setAttribute('x', Math.min(x(h) + 6, W - mr + 6));
        heads[arm].setAttribute('y', y(v) + (arm === 'codex' ? 14 : -6));
        heads[arm].textContent = fx(v);
      }
    };
    S.render = render;
    if (targets.length) setTarget(S.tsel[w.workload] || null);
    else stats(m, w, null, ends0);
    return render;
  }

  dash.draw('race', {
    mount(el) {
      const stats = FR.el('div', 'fr-stats', null, el);
      const about = FR.el('p', 'fr-about', null, el);
      const box = FR.el('div', null, null, el);
      box.style.position = 'relative';
      const tip = FR.tip(box);
      return { stats, about, box, tip };
    },
    update(el, m) {
      const st = dash.data('race_steps'), ck = dash.data('race_checks'), wl = dash.data('race_workloads'), so = dash.data('race_targets');
      const rr = dash.data('race_runs'), rc = dash.data('race_run_counts'), ry = dash.data('race_why'), rb = dash.data('race_best_checks'), rd = dash.data('race_diffs');
      const bad = [st, ck, wl, so].find((r) => r.status === 'error' || r.status === 'missing' || r.status === 'declined');
      if (bad || st.status !== 'ok' || wl.status !== 'ok' || so.status !== 'ok') {
        S.render = null;
        m.stats.replaceChildren();
        m.about.textContent = '';
        m.box.replaceChildren(m.tip.node);
        FR.el('p', 'fr-race-empty', bad ? 'Failed to load' : '—', m.box);
        return;
      }
      const w = current();
      if (!w) return;
      S.wid = w.workload;
      const steps = st.data.filter((r) => r.workload === w.workload);
      const checks = ck.status === 'ok' ? ck.data.filter((r) => r.workload === w.workload) : [];
      const targets = (so.data || []).filter((r) => r.workload === w.workload).sort((a, b) => a.target - b.target);
      const ends = {};
      steps.forEach((r) => { if (r.kind === 'end') ends[r.arm] = { v: Number(r.value) }; });
      m.about.replaceChildren();
      FR.mark(FR.el('span', null, w.about, m.about), 'race_workloads', 'about', { workload: w.workload });
      m.about.append('. ');
      if (w.has_checks !== 'yes') m.about.append('No per-check log for this workload, so only new bests are drawn. ');
      const cnt = rc.status === 'ok' ? (rc.data || []).find((r) => r.workload === w.workload) : null;
      if (cnt) {
        m.about.append(' Runs per side: Baseline ');
        FR.mark(FR.el('span', 'fr-num', String(cnt.codex_runs), m.about), 'race_run_counts', 'codex_runs', { workload: w.workload });
        m.about.append(', rsibox ');
        FR.mark(FR.el('span', 'fr-num', String(cnt.rsibox_runs), m.about), 'race_run_counts', 'rsibox_runs', { workload: w.workload });
        m.about.append(Number(cnt.codex_runs) + Number(cnt.rsibox_runs) > 2 ? '; the bold lines are each side\'s main run.' : '.');
      }
      const others = [];
      if (rr.status === 'ok') {
        const g = {};
        rr.data.filter((r) => r.workload === w.workload && r.main !== 'yes').forEach((r) => {
          (g[r.arm + '|' + r.run] = g[r.arm + '|' + r.run] || { arm: r.arm, run: r.run, pts: [] }).pts.push({ t: Number(r.t_h), v: Number(r.value), i: Number(r.point) });
        });
        Object.values(g).forEach((o) => { o.pts.sort((a, b) => a.t - b.t || a.i - b.i); others.push(o); });
      }
      document.querySelectorAll('#race-others').forEach((c) => { c.disabled = !others.length; c.checked = !!S.others; });
      const why = {}, bc = {};
      if (ry.status === 'ok') ry.data.filter((r) => r.workload === w.workload).forEach((r) => (why[r.arm + '.' + r.step] = r));
      if (rb.status === 'ok') (rb.data || []).filter((r) => r.workload === w.workload).forEach((r) => (bc[r.arm + '.' + r.step] = r));
      S.pinned = false;
      m.tip.node.classList.remove('is-pinned');
      const diffs = {};
      if (rd.status === 'ok') rd.data.filter((r) => r.workload === w.workload).forEach((r) => (diffs[r.arm + '.' + r.step] = r));
      S.render = build(m, w, steps, checks, targets, ends, others, why, bc, diffs);
      S.render(false);
      syncControls();
    },
    unmount() { stop(); S.render = null; },
  });

  // Headline range under the page title, from race_sooner.
  function ledeDraw(pick) {
    return (el) => {
      const so = dash.data('race_sooner');
      const rows = so.status === 'ok' ? (so.data || []).filter((r) => r.ratio != null) : [];
      if (!rows.length) { el.textContent = '—'; el.removeAttribute('data-where'); return; }
      const r = rows.reduce((a, b) => (pick(a.ratio, b.ratio) ? a : b));
      el.setAttribute('data-where', FR.where({ workload: r.workload }));
      el.textContent = Number(r.ratio).toFixed(1) + '×';
    };
  }
  dash.draw('lede-min', ledeDraw((a, b) => a <= b));
  dash.draw('lede-max', ledeDraw((a, b) => a >= b));
})();

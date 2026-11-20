// Section 2, #eval-time: per engine, a ladder of evaluation time from restart to live checks.
// Datasets: eval_ladder (seconds per phase per row), ladder_rungs (row labels: the one place to edit them),
// eval_phases (names, order, start-up or check), race_workloads (engine labels, set-up, what can't shrink). Calc: ladder_totals.

dash.calc('ladder_totals', {
  inputs: ['eval_ladder', 'ladder_rungs', 'eval_phases'],
  title: 'Seconds per evaluation, each row',
  description: "Per engine and row: the sum of its phases and of its start-up phases only, the same for the row above, the speedup over the engine's first row (both ways), and for a derived row the phases it removes and their seconds in the row above. Live rows also carry the seconds to install the edit (install_s, empty when not measured) and whether their start-up is that install alone. A start-up speedup is empty when no start-up is left.",
  fn: (eval_ladder, ladder_rungs, eval_phases) => {
    const kind = {};
    eval_phases.forEach((p) => (kind[p.phase] = p.kind));
    const meas = {}, live = {};
    ladder_rungs.forEach((r) => { meas[r.engine + '.' + r.step] = r.measured; live[r.engine + '.' + r.step] = r.live === 'yes'; });
    const by = {};
    eval_ladder.forEach((r) => {
      const k = r.engine + '.' + r.step;
      const o = (by[k] = by[k] || { engine: r.engine, step: Number(r.step), seg: {} });
      o.seg[r.phase] = (o.seg[r.phase] || 0) + Number(r.seconds);
    });
    const rows = Object.values(by).sort((a, b) => (a.engine < b.engine ? -1 : a.engine > b.engine ? 1 : a.step - b.step));
    const r2 = (v) => Math.round(v * 100) / 100;
    const sum = (o, startupOnly) => Object.entries(o.seg).reduce((a, [p, v]) => a + (startupOnly && kind[p] === 'check' ? 0 : v), 0);
    return rows.map((r, i) => {
      const prev = i > 0 && rows[i - 1].engine === r.engine ? rows[i - 1] : null;
      const first = rows.find((x) => x.engine === r.engine);
      const t = sum(r), su = sum(r, true);
      let removed_phase = null, removed_s = null;
      if (prev && meas[r.engine + '.' + r.step] !== 'yes') {
        const gone = Object.keys(prev.seg).filter((p) => !(p in r.seg));
        removed_phase = gone.join(',');
        removed_s = r2(gone.reduce((a, p) => a + prev.seg[p], 0));
      }
      return {
        key: r.engine + '.' + r.step,
        engine: r.engine,
        step: r.step,
        total_s: r2(t),
        prev_total_s: prev ? r2(sum(prev)) : null,
        speedup_vs_start: Math.round((sum(first) / t) * 1000) / 1000,
        startup_s: r2(su),
        prev_startup_s: prev ? r2(sum(prev, true)) : null,
        startup_speedup_vs_start: su > 0 ? Math.round((sum(first, true) / su) * 1000) / 1000 : null,
        removed_phase,
        removed_s,
        live: live[r.engine + '.' + r.step] ? 'yes' : 'no',
        install_s: 'live_install' in r.seg ? r2(r.seg.live_install) : null,
        install_only: su > 0 && Object.keys(r.seg).every((p) => p === 'live_install' || kind[p] === 'check') ? 'yes' : 'no',
      };
    });
  },
});

(function () {
  const FR = window.FR;
  const L = (FR.ladder = FR.ladder || { played: false, play: null, engine: null, visible: false, startupOnly: false });
  const SLOW = 1.5; // animation pace
  const fs = (s) => (s >= 50 ? Math.round(s).toString() : s > 0 && s < 0.1 ? (s < 0.01 ? (s * 1000).toFixed(2) : Math.round(s * 1000)) + ' ms' : (s > 0 && s < 1 ? s.toFixed(2) : (Math.round(s * 10) / 10).toFixed(1)) + ' s');
  const fx = (x) => (x >= 10 ? Math.round(x).toString() : x.toFixed(1)) + '×';

  function engines() {
    const w = FR.ok('race_workloads'), l = FR.ok('eval_ladder');
    if (!w || !l) return [];
    const has = new Set(l.data.map((r) => r.engine));
    return w.data.filter((r) => has.has(r.workload)).sort((a, b) => Number(a.order) - Number(b.order));
  }
  function current() { const es = engines(); return es.find((e) => e.workload === L.engine) || es[0] || null; }
  function phases() {
    const p = FR.ok('eval_phases');
    return p ? p.data.slice().sort((a, b) => Number(a.order) - Number(b.order)) : [];
  }
  const DRAWS = ['ladder-picker', 'ladder-setup', 'ladder-legend', 'ladder', 'ladder-floor'];
  function select(id) {
    L.engine = id;
    DRAWS.forEach((d) => dash.draw(d));
    if (L.play && !FR.reduced()) L.play();
  }

  document.querySelectorAll('#ladder-replay').forEach((b) => b.addEventListener('click', () => { if (L.play) L.play(); }));
  document.querySelectorAll('#ladder-startup').forEach((c) => {
    c.checked = L.startupOnly;
    c.addEventListener('change', () => {
      L.startupOnly = c.checked;
      ['ladder-legend', 'ladder'].forEach((d) => dash.draw(d));
    });
  });

  dash.draw('ladder-picker', (el) => {
    const es = engines(), cur = current();
    el.replaceChildren(...es.map((e) => {
      const b = FR.el('button', 'fr-chip', e.engine_label);
      b.type = 'button';
      b.setAttribute('aria-pressed', String(cur && cur.workload === e.workload));
      FR.mark(b, 'race_workloads', 'engine_label', { workload: e.workload });
      b.addEventListener('click', () => select(e.workload));
      return b;
    }));
  });

  dash.draw('ladder-setup', (el) => {
    const e = current();
    if (!e) { el.textContent = '—'; return; }
    el.replaceChildren();
    FR.mark(FR.el('span', null, e.ladder_setup, el), 'race_workloads', 'ladder_setup', { workload: e.workload });
    el.append('. Source: ');
    FR.mark(FR.el('span', null, e.ladder_source, el), 'race_workloads', 'ladder_source', { workload: e.workload });
    el.append('.');
  });

  // What can't be shrunk, and what was never measured, for this engine.
  dash.draw('ladder-floor', (el) => {
    const e = current();
    el.replaceChildren();
    if (!e) return;
    FR.mark(FR.el('p', 'fr-floor', e.ladder_floor, el), 'race_workloads', 'ladder_floor', { workload: e.workload });
    if (e.ladder_unmeasured) FR.mark(FR.el('p', 'fr-note', e.ladder_unmeasured, el), 'race_workloads', 'ladder_unmeasured', { workload: e.workload });
    const rg = FR.ok('ladder_rungs');
    if (rg) rg.data.filter((r) => r.engine === e.workload && r.install_note).forEach((r) =>
      FR.mark(FR.el('p', 'fr-note', r.install_note, el), 'ladder_rungs', 'install_note', { engine: e.workload, step: r.step }));
  });

  // Legend: only the phases this engine shows, in the shared order and colors.
  dash.draw('ladder-legend', (el) => {
    const e = current(), l = FR.ok('eval_ladder');
    el.replaceChildren();
    if (!e || !l) return;
    const used = new Set(l.data.filter((r) => r.engine === e.workload).map((r) => r.phase));
    phases().filter((p) => used.has(p.phase) && !(L.startupOnly && p.kind === 'check')).forEach((p) => {
      const li = FR.el('li', null, null, el);
      FR.el('span', 'fr-k fr-ph-' + p.phase, null, li);
      FR.mark(FR.el('span', null, p.label, li), 'eval_phases', 'label', { phase: p.phase });
    });
  });

  dash.draw('ladder', {
    mount(el) {
      const box = FR.el('div', null, null, el);
      box.style.position = 'relative';
      const tip = FR.tip(box);
      const m = { box, tip, stopObs: null };
      m.stopObs = FR.onceVisible(el, () => { L.visible = true; if (L.play && !L.played) { L.played = true; L.play(); } }, 0.35);
      return m;
    },
    update(el, m) {
      const lad = dash.data('eval_ladder'), tot = dash.data('ladder_totals'), rg = dash.data('ladder_rungs');
      const box = m.box;
      box.replaceChildren(m.tip.node);
      m.tip.hide();
      const e = current();
      if (lad.status !== 'ok' || tot.status !== 'ok' || rg.status !== 'ok' || !e) {
        L.play = null;
        FR.el('p', 'fr-race-empty', [lad, tot, rg].some((x) => x.status === 'error') ? 'Failed to load' : '—', box);
        return;
      }
      const SO = L.startupOnly;
      const PHS = phases();
      const PN = {}, KIND = {};
      PHS.forEach((p) => { PN[p.phase] = p.label; KIND[p.phase] = p.kind; });
      const PH = PHS.map((p) => p.phase);
      const shown = (p) => !(SO && KIND[p] === 'check');
      const recs = lad.data.filter((r) => r.engine === e.workload);
      const rungs = {};
      rg.data.filter((r) => r.engine === e.workload).forEach((r) => (rungs[Number(r.step)] = r));
      const steps = [...new Set(recs.map((r) => Number(r.step)))].sort((a, b) => a - b);
      const rows = steps.map((s) => {
        const seg = {};
        recs.filter((r) => Number(r.step) === s).forEach((r) => (seg[r.phase] = (seg[r.phase] || 0) + Number(r.seconds)));
        const g = rungs[s] || {};
        return { step: s, label: g.label || 'Row ' + s, group: g.group || '', measured: g.measured || 'yes', live: g.live === 'yes', onlyLbl: g.install_only_label || '', missLbl: g.install_missing_label || '', derived: g.derived_note || '', seg };
      });
      // a group with no measured row yet: its heading and one line, no bar
      const holders = rg.data.filter((r) => r.engine === e.workload && r.placeholder === 'yes').sort((a, b) => Number(a.step) - Number(b.step))
        .map((g) => ({ step: Number(g.step), label: g.label, group: g.group || '' }));
      const tots = {};
      (tot.data || []).forEach((t) => { if (t.engine === e.workload) tots[t.step] = t; });
      const TOT = (t) => Number(SO ? t.startup_s : t.total_s);
      const PREV = (t) => Number(SO ? t.prev_startup_s : t.prev_total_s);
      const W = Math.max(280, box.clientWidth || 640);
      const svg = FR.svg('svg', { class: 'fr-svg', role: 'img', width: W });
      box.insertBefore(svg, m.tip.node);
      const wide = W >= 600;
      const lblW = Math.max(...rows.map((r) => Math.max(FR.textW(svg, r.label, 'row-lbl'), r.derived ? FR.textW(svg, r.derived, 'row-der') : 0)));
      const maxS = Math.max(1, ...rows.map((r) => TOT(tots[r.step])));
      const rightW = Math.ceil(FR.textW(svg, fs(maxS), 'row-tot') + Math.max(FR.textW(svg, '95×', 'row-x'), ...rows.map((r) => FR.textW(svg, r.onlyLbl, 'row-x')))) + 24;
      const lx = wide ? Math.min(Math.ceil(lblW) + 18, W * 0.42) : 0;
      const barH = wide ? 20 : 18;
      const rowH = wide ? 40 : 58;
      const capH = 28;
      const x = d3.scaleLinear([0, maxS], [lx, W - rightW]);
      let yy = 0;
      const pos = rows.map((r, i) => {
        let cap = null;
        if (r.group && (i === 0 || r.group !== rows[i - 1].group)) { yy += i ? 18 : 0; cap = yy + 16; yy += capH; }
        const p = { cap, y: yy };
        yy += rowH;
        return p;
      });
      const holdPos = holders.map((h) => { yy += 18; const p = { cap: yy + 16, y: yy + capH + 14 }; yy += capH + 26; return p; });
      const H = yy + 26;
      svg.setAttribute('height', H);
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      svg.setAttribute('aria-label', `${SO ? 'Start-up seconds' : 'Seconds'} per evaluation for ${e.engine_label}, row by row: ` + rows.map((r) => `${r.label} ${fs(TOT(tots[r.step]))}`).join('; ') + '.');
      const axY = H - 8;
      x.ticks(W < 420 ? 3 : 5).forEach((v) => {
        FR.svg('line', { class: 'gl', x1: x(v), x2: x(v), y1: 4, y2: axY - 12 }, svg);
        FR.svg('text', { x: x(v), y: axY, 'text-anchor': 'middle' }, svg).textContent = v + ' s';
      });
      const MINW = 3; // tiny segments keep a visible sliver
      const geo = (r) => { let px = lx; return PH.map((p) => { const s = shown(p) ? r.seg[p] || 0 : 0; const w = s > 0 ? Math.max(MINW, x(s) - lx) : 0; const g = { p, x0: px, w, s }; px += w; return g; }); };
      const G = rows.map(geo);
      const groups = rows.map((r, i) => {
        const P = pos[i];
        if (P.cap != null) { const c = FR.svg('text', { class: 'row-grp', x: 0, y: P.cap }, svg); FR.mark(c, 'ladder_rungs', 'group', { engine: e.workload, step: r.step }); c.textContent = r.group; }
        const by = wide ? P.y : P.y + 18;
        const g = FR.svg('g', { class: 'lrow' }, svg);
        const t = tots[r.step];
        const lt = FR.svg('text', { class: 'row-lbl', x: 0, y: wide ? by + barH / 2 + 4 : P.y + 12 }, g);
        FR.mark(lt, 'ladder_rungs', 'label', { engine: e.workload, step: r.step });
        lt.textContent = r.label;
        if (r.measured !== 'yes' && r.derived) {
          const dt = FR.svg('text', { class: 'row-der', x: 0, y: wide ? by + barH / 2 + 17 : by + barH + 13 }, g);
          FR.mark(dt, 'ladder_rungs', 'derived_note', { engine: e.workload, step: r.step });
          dt.textContent = r.derived;
          if (wide) lt.setAttribute('y', by + barH / 2 - 2);
        } else if (r.measured !== 'yes' && t.removed_s != null) {
          const dt = FR.svg('text', { class: 'row-der', x: wide ? 0 : FR.textW(svg, r.label, 'row-lbl') + 8, y: wide ? by + barH / 2 + 17 : P.y + 12 }, g);
          dt.append('derived: ');
          const a = FR.svg('tspan', {}, dt);
          FR.mark(a, 'ladder_totals', SO ? 'prev_startup_s' : 'prev_total_s', { key: t.key });
          a.textContent = fs(PREV(t));
          dt.append(' − ');
          const b = FR.svg('tspan', {}, dt); FR.mark(b, 'ladder_totals', 'removed_s', { key: t.key }); b.textContent = fs(Number(t.removed_s));
          dt.append(' ' + String(t.removed_phase).split(',').map((p) => (PN[p] || p).toLowerCase()).join(' + '));
          if (wide) lt.setAttribute('y', by + barH / 2 - 2);
          else { dt.setAttribute('x', 0); dt.setAttribute('y', by + barH + 13); }
        }
        if (i > 0) FR.svg('rect', { class: 'prev', x: lx, y: by - 2, width: Math.max(0, x(PREV(t)) - lx), height: barH + 4, rx: 2 }, g);
        const segs = G[i].map((sg) => {
          const n = FR.svg('rect', { class: 'seg', x: sg.x0, y: by, width: Math.max(0, sg.w), height: barH }, g);
          n.style.fill = `var(--ph-${sg.p})`;
          const show = (ev) => tipFor(ev, r, sg, t);
          n.addEventListener('pointermove', show);
          n.addEventListener('pointerdown', show);
          n.addEventListener('pointerleave', () => m.tip.hide());
          return n;
        });
        const segEnd = G[i].reduce((a, sg) => Math.max(a, sg.x0 + sg.w), lx);
        const endX = Math.max(x(TOT(t)), segEnd, i > 0 ? x(PREV(t)) : 0) + 8;
        const tt = FR.svg('text', { class: 'row-tot', x: endX, y: by + barH / 2 + 5 }, g);
        FR.mark(tt, 'ladder_totals', SO ? 'startup_s' : 'total_s', { key: t.key });
        tt.textContent = r.live && t.install_s == null && TOT(t) === 0 ? '' : fs(TOT(t));
        let xt = null;
        if (i > 0) {
          xt = FR.svg('text', { class: 'row-x', x: endX + FR.textW(svg, tt.textContent, 'row-tot') + 8, y: by + barH / 2 + 5 }, g);
          const sp = SO ? t.startup_speedup_vs_start : t.speedup_vs_start;
          if (SO && t.install_only === 'yes') { FR.mark(xt, 'ladder_rungs', 'install_only_label', { engine: e.workload, step: r.step }); xt.textContent = r.onlyLbl; }
          else if (sp != null) { FR.mark(xt, 'ladder_totals', SO ? 'startup_speedup_vs_start' : 'speedup_vs_start', { key: t.key }); xt.textContent = fx(Number(sp)); }
          if (r.live && t.install_s == null && r.missLbl) {
            const ms = FR.svg('tspan', {}, xt);
            FR.mark(ms, 'ladder_rungs', 'install_missing_label', { engine: e.workload, step: r.step });
            ms.textContent = (xt.textContent ? ' · ' : '') + r.missLbl;
          }
        }
        return { g, segs, tt, xt };
      });
      holders.forEach((h, k) => {
        const c = FR.svg('text', { class: 'row-grp', x: 0, y: holdPos[k].cap }, svg);
        FR.mark(c, 'ladder_rungs', 'group', { engine: e.workload, step: h.step });
        c.textContent = h.group;
        const n = FR.svg('text', { class: 'row-none', x: 0, y: holdPos[k].y }, svg);
        FR.mark(n, 'ladder_rungs', 'label', { engine: e.workload, step: h.step });
        n.textContent = h.label;
      });
      function tipFor(ev, r, sg, t) {
        if (sg.s <= 0) return;
        const br = box.getBoundingClientRect();
        m.tip.show(ev.clientX - br.left, ev.clientY - br.top, (n) => {
          FR.el('b', null, PN[sg.p] || sg.p, n);
          const l = FR.el('div', null, null, n);
          FR.mark(FR.el('span', 'tv', fs(sg.s), l), 'eval_ladder', 'seconds', { engine: e.workload, step: r.step, phase: sg.p });
          l.append(' of ' + fs(TOT(t)) + (SO ? ' of start-up' : '') + ' in this row');
          FR.el('div', 'tm', r.label + (r.measured === 'yes' ? '' : ' (derived)'), n);
        });
      }
      L.play = () => {
        if (FR.reduced()) return;
        groups.forEach((gr, i) => {
          const from = i ? G[i - 1] : G[0].map(() => ({ x0: lx, w: 0 }));
          gr.g.style.opacity = '0';
          gr.segs.forEach((n, k) => { n.setAttribute('x', from[k].x0); n.setAttribute('width', Math.max(0, from[k].w)); });
          [gr.tt, gr.xt].forEach((n) => n && (n.style.opacity = '0'));
          const delay = i * 650 * SLOW;
          d3.select(gr.g).transition().delay(delay).duration(220 * SLOW).style('opacity', 1);
          gr.segs.forEach((n, k) => d3.select(n).transition().delay(delay + 200 * SLOW).duration(750 * SLOW).ease(d3.easeCubicOut)
            .attr('x', G[i][k].x0).attr('width', Math.max(0, G[i][k].w)));
          [gr.tt, gr.xt].forEach((n) => n && d3.select(n).transition().delay(delay + 800 * SLOW).duration(250 * SLOW).style('opacity', 1));
        });
      };
      if (L.visible && !L.played) { L.played = true; L.play(); }
    },
    unmount(el, m) { if (m && m.stopObs) m.stopObs(); L.play = null; },
  });
})();

// Section 3, #model-speed: rsibox's speedup over Baseline as the model gets faster (replay of the logged runs),

// Datasets: replay_hours, race_workloads (labels); fig2_breakdown only for the ≈49 tok/s footnote in the HTML. Calcs: speed_stops, replay_compare.

dash.calc('speed_stops', {
  inputs: ['replay_hours'],
  title: 'Model speed at each replay step',
  description: "The replay's speed-up steps, with tokens per second as 50 tok/s times the speed-up. 50 rounds the measured base of about 49 tok/s; the replay itself gives no tokens per second.",
  fn: (replay_hours) => {
    const base = 50; // tok/s at 1×: the measured ≈49 tok/s, rounded
    const seen = {};
    replay_hours.forEach((r) => {
      if (seen[r.stop]) return;
      const k = r.speedup === '' || r.speedup == null ? null : Number(r.speedup);
      seen[r.stop] = { stop: r.stop, speedup_label: r.speedup_label, speedup: k,
        approx_tok_s: k == null ? null : base * k };
    });
    return Object.values(seen).sort((a, b) => Number(a.stop) - Number(b.stop));
  },
});

dash.calc('replay_compare', {
  inputs: ['replay_hours'],
  title: "rsibox's speedup over Baseline at each model speed",
  description: "Per system and model speed-up: both replayed run lengths and rsibox's speedup, Baseline hours divided by rsibox hours.",
  fn: (replay_hours) => {
    const h = {};
    replay_hours.forEach((r) => { ((h[r.system] = h[r.system] || { codex: {}, rsibox: {} })[r.arm])[r.stop] = Number(r.hours); });
    const out = [];
    Object.keys(h).forEach((system) => {
      const c = h[system].codex, f = h[system].rsibox;
      Object.keys(c).map(Number).sort((a, b) => a - b).forEach((s) => {
        out.push({ key: system + '.' + s, system, stop: s, codex_h: c[s], rsibox_h: f[s], speedup: Math.round((c[s] / f[s]) * 1000) / 1000 });
      });
    });
    return out;
  },
});

(function () {
  const FR = window.FR;
  const SP = (FR.speed = FR.speed || { stop: 0 });
  const FIN_FRAC = 0.86; // where the fastest finite step sits; ∞ is drawn at the far end
  const DRAWS = ['speed-now', 'speed-ticks', 'speed-chart', 'speed-table'];
  const fmtTok = (v) => d3.format(',')(v);

  function stops() {
    const r = FR.ok('speed_stops');
    return r && Array.isArray(r.data) ? r.data : [];
  }
  // Log position of each step on the track; ∞ at 1.
  function fracs() {
    const ss = stops();
    const ks = ss.map((s) => s.speedup).filter((k) => k != null);
    const maxK = Math.max(...ks, 2);
    return ss.map((s) => (s.speedup == null ? 1 : (Math.log(s.speedup) / Math.log(maxK)) * FIN_FRAC));
  }
  function systems() {
    const rc = FR.ok('replay_compare'), wl = FR.ok('race_workloads');
    if (!rc) return [];
    const order = wl ? wl.data.slice().sort((a, b) => Number(a.order) - Number(b.order)) : [];
    const names = [...new Set(rc.data.map((r) => r.system))];
    const pos = (s) => { const i = order.findIndex((w) => w.replay_system === s); return i < 0 ? 99 : i; };
    return names.sort((a, b) => pos(a) - pos(b)).map((s, i) => ({ system: s, idx: i, w: order.find((w) => w.replay_system === s) || null }));
  }
  const sysLabel = (parent, s) => {
    if (s.w) FR.mark(FR.el('span', null, s.w.engine_label, parent), 'race_workloads', 'engine_label', { workload: s.w.workload });
    else parent.append(s.system);
  };

  function setStop(i) {
    const n = stops().length;
    if (!n) return;
    SP.stop = Math.max(0, Math.min(n - 1, i));
    const fr = fracs(), st = stops()[SP.stop];
    document.querySelectorAll('#speed-slider').forEach((s) => {
      s.value = String(Math.round(fr[SP.stop] * 1000));
      s.setAttribute('aria-valuetext', st.approx_tok_s != null ? `about ${st.approx_tok_s} tokens per second, ${st.speedup_label}` : 'infinitely fast generation');
    });
    DRAWS.forEach((d) => dash.draw(d));
  }
  document.querySelectorAll('#speed-slider').forEach((s) => {
    s.addEventListener('input', () => {
      const v = Number(s.value) / 1000, fr = fracs();
      if (!fr.length) return;
      let best = 0;
      fr.forEach((f, i) => { if (Math.abs(f - v) < Math.abs(fr[best] - v)) best = i; });
      if (best !== SP.stop) setStop(best);
      else s.value = String(Math.round(fr[best] * 1000));
    });
    s.addEventListener('change', () => setStop(SP.stop));
    s.addEventListener('keydown', (e) => {
      const k = e.key;
      if (k === 'ArrowRight' || k === 'ArrowUp') { e.preventDefault(); setStop(SP.stop + 1); }
      else if (k === 'ArrowLeft' || k === 'ArrowDown') { e.preventDefault(); setStop(SP.stop - 1); }
      else if (k === 'Home') { e.preventDefault(); setStop(0); }
      else if (k === 'End') { e.preventDefault(); setStop(stops().length - 1); }
    });
  });

  // Current speed readout.
  dash.draw('speed-now', (el) => {
    const st = stops()[SP.stop];
    if (!st) { el.textContent = '—'; return; }
    el.replaceChildren();
    if (st.approx_tok_s != null) {
      el.append('~');
      FR.mark(FR.el('span', null, fmtTok(st.approx_tok_s), el), 'speed_stops', 'approx_tok_s', { stop: st.stop });
      el.append(' tok/s');
      FR.mark(FR.el('span', 'tm', st.speedup_label + (st.speedup === 1 ? ', as measured' : ' faster'), el), 'speed_stops', 'speedup_label', { stop: st.stop });
    } else {
      el.append('∞');
      FR.el('span', 'tm', 'generation takes no time', el);
    }
    const fr = fracs();
    document.querySelectorAll('#speed-slider').forEach((s) => { if (fr.length) s.value = String(Math.round(fr[SP.stop] * 1000)); });
  });

  // Tick labels under the slider, at log positions.
  dash.draw('speed-ticks', (el) => {
    const ss = stops();
    el.replaceChildren();
    if (!ss.length) return;
    const fr = fracs();
    const fin = ss.map((s) => s.speedup != null).lastIndexOf(true);
    const band = FR.el('div', 'fr-gapband', null, el);
    band.style.left = (fr[fin] * 100) + '%';
    band.style.width = ((1 - fr[fin]) * 100) + '%';
    ss.forEach((s, i) => {
      const t = FR.el('div', 'fr-tick' + (i === SP.stop ? ' on' : ''), null, el);
      t.style.left = (fr[i] * 100) + '%';
      if (el.clientWidth < 480 && (i === 3 || i === 5) && i !== SP.stop) t.style.visibility = 'hidden';
      if (s.approx_tok_s != null) {
        FR.mark(FR.el('span', null, fmtTok(s.approx_tok_s), t), 'speed_stops', 'approx_tok_s', { stop: s.stop });
        t.append(FR.el('br'));
        FR.mark(FR.el('span', null, s.speedup_label, t), 'speed_stops', 'speedup_label', { stop: s.stop });
      } else t.append('∞');
    });
  });

  // Headline chart: rsibox's speedup over Baseline vs model speed, one line per system.
  dash.draw('speed-chart', {
    mount(el) {
      const legend = FR.el('ul', 'fr-keys fr-sys-keys', null, el);
      legend.setAttribute('aria-label', 'Systems');
      const box = FR.el('div', null, null, el);
      box.style.position = 'relative';
      return { legend, box, tip: FR.tip(box) };
    },
    update(el, m) {
      const rc = dash.data('replay_compare');
      const ss = stops(), sys = systems();
      const box = m.box;
      box.replaceChildren(m.tip.node);
      m.tip.hide();
      m.legend.replaceChildren();
      if (rc.status !== 'ok' || !ss.length || !sys.length) {
        FR.el('p', 'fr-race-empty', rc.status === 'error' ? 'Failed to load' : '—', box);
        return;
      }
      const cmp = {};
      rc.data.forEach((r) => (cmp[r.key] = r));
      sys.forEach((s) => {
        const li = FR.el('li', null, null, m.legend);
        FR.el('span', 'fr-k fr-sys-' + s.idx, null, li);
        sysLabel(li, s);
      });
      const fr = fracs();
      const fin = ss.map((s) => s.speedup != null).lastIndexOf(true);
      const W = Math.max(280, box.clientWidth || 640);
      const H = Math.round(Math.max(220, Math.min(320, W * 0.48)));
      const svg = FR.svg('svg', { class: 'fr-svg', role: 'img', width: W, height: H, viewBox: `0 0 ${W} ${H}` });
      box.insertBefore(svg, m.tip.node);
      const vals = rc.data.map((r) => Number(r.speedup));
      const y = d3.scaleLinear([Math.min(0.95, d3.min(vals) - 0.02), d3.max(vals) + 0.05], [H - 40, 12]).nice();
      const yt = y.ticks(5);
      const ml = Math.ceil(Math.max(...yt.map((v) => FR.textW(svg, v.toFixed(1) + '×')))) + 14;
      const endW = Math.ceil(FR.textW(svg, '1.52×', 'end')) + 12;
      const mr = endW;
      const x = (f) => ml + f * (W - ml - mr);
      svg.setAttribute('aria-label', "rsibox's speedup over Baseline as model speed grows: " + sys.map((s) => {
        const a = cmp[s.system + '.' + ss[0].stop], b = cmp[s.system + '.' + ss[ss.length - 1].stop];
        return `${s.w ? s.w.engine_label : s.system} ${a ? a.speedup.toFixed(2) : '?'}× to ${b ? b.speedup.toFixed(2) : '?'}×`;
      }).join('; ') + '.');
      FR.svg('rect', { class: 'nodata-band', x: x(fr[fin]), y: 12, width: x(1) - x(fr[fin]), height: H - 52 }, svg);
      yt.forEach((v) => {
        FR.svg('line', { class: v === 1 ? 'one' : 'gl', x1: ml, x2: W - mr, y1: y(v), y2: y(v) }, svg);
        FR.svg('text', { x: ml - 8, y: y(v) + 4, 'text-anchor': 'end' }, svg).textContent = v.toFixed(1) + '×';
      });
      if (y.domain()[0] <= 1) {
        const t = FR.svg('text', { class: 'one-lbl', x: x(fr[fin]) - 6, y: y(1) + 14, 'text-anchor': 'end' }, svg);
        t.textContent = 'same time';
      }
      const narrow = W < 480;
      ss.forEach((s, i) => {
        if (narrow && (i === 3 || i === 5)) return; // keep tick labels apart
        const anchor = i === 0 ? 'start' : i === ss.length - 1 ? 'end' : 'middle';
        const t = FR.svg('text', { x: x(fr[i]) - (i === 0 ? 4 : 0), y: H - 22, 'text-anchor': anchor }, svg);
        t.textContent = s.approx_tok_s != null ? fmtTok(s.approx_tok_s) + (i === 0 ? ' tok/s' : '') : '∞';
        const u = FR.svg('text', { x: x(fr[i]) - (i === 0 ? 4 : 0), y: H - 8, 'text-anchor': anchor }, svg);
        u.textContent = s.approx_tok_s != null ? s.speedup_label : '';
      });
      const sx = x(fr[SP.stop]);
      FR.svg('line', { class: 'sel', x1: sx, x2: sx, y1: 12, y2: H - 40 }, svg);
      const ends = [];
      sys.forEach((s) => {
        const P = ss.map((st, i) => ({ i, r: cmp[s.system + '.' + st.stop] })).filter((p) => p.r);
        const pt = (p) => `${x(fr[p.i]).toFixed(1)},${y(p.r.speedup).toFixed(1)}`;
        const solid = P.filter((p) => p.i <= fin), tail = P.filter((p) => p.i >= fin);
        const ln = FR.svg('path', { class: 'ln', d: 'M' + solid.map(pt).join('L') }, svg);
        ln.style.stroke = `var(--sys-${s.idx})`;
        if (tail.length > 1) { const t = FR.svg('path', { class: 'ln dash', d: 'M' + tail.map(pt).join('L') }, svg); t.style.stroke = `var(--sys-${s.idx})`; }
        P.forEach((p) => { const c = FR.svg('circle', { cx: x(fr[p.i]), cy: y(p.r.speedup), r: p.i === SP.stop ? 4.5 : 2.4 }, svg); c.style.fill = `var(--sys-${s.idx})`; });
        const sel = cmp[s.system + '.' + ss[SP.stop].stop];
        if (sel) ends.push({ s, r: sel });
      });
      // value labels at the selected speed, kept apart
      const right = SP.stop < ss.length - 1 || true;
      ends.sort((a, b) => b.r.speedup - a.r.speedup);
      let prev = -Infinity;
      const lx = SP.stop === ss.length - 1 ? sx + 6 : sx + 8;
      const ys = ends.map((e) => { let yy = y(e.r.speedup) + 4; if (yy - prev < 13) yy = prev + 13; prev = yy; return yy; });
      const over = ys.length ? ys[ys.length - 1] - (H - 44) : 0; // keep the stack above the x-axis labels
      ends.forEach((e, k) => {
        const yy = ys[k] - Math.max(0, over);
        const t = FR.svg('text', { class: 'end halo', x: SP.stop === ss.length - 1 ? sx + 6 : lx, y: yy }, svg);
        t.style.fill = `var(--sys-${e.s.idx})`;
        FR.mark(t, 'replay_compare', 'speedup', { key: e.r.key });
        t.textContent = e.r.speedup.toFixed(2) + '×';
      });
      // tooltip: nearest step, every system with its hours
      const hit = FR.svg('rect', { x: ml, y: 12, width: W - ml - mr, height: H - 52, fill: 'transparent' }, svg);
      const onMove = (ev) => {
        const r = svg.getBoundingClientRect();
        const px = (ev.clientX - r.left) * (W / r.width);
        let bi = 0;
        ss.forEach((s, i) => { if (Math.abs(x(fr[i]) - px) < Math.abs(x(fr[bi]) - px)) bi = i; });
        const st = ss[bi];
        m.tip.show(x(fr[bi]) * (r.width / W), (ev.clientY - r.top), (t) => {
          FR.el('b', null, st.approx_tok_s != null ? `≈${fmtTok(st.approx_tok_s)} tok/s (${st.speedup_label})` : '∞ model speed', t);
          sys.forEach((s) => {
            const k = cmp[s.system + '.' + st.stop];
            if (!k) return;
            const l = FR.el('div', null, null, t);
            sysLabel(l, s);
            l.append(' ');
            FR.mark(FR.el('span', 'tv', k.speedup.toFixed(2) + '×', l), 'replay_compare', 'speedup', { key: k.key });
            const hh = FR.el('span', 'tm', null, l);
            hh.append(' (Baseline ');
            FR.mark(FR.el('span', 'tv', Number(k.codex_h).toFixed(2), hh), 'replay_compare', 'codex_h', { key: k.key });
            hh.append(' h, rsibox ');
            FR.mark(FR.el('span', 'tv', Number(k.rsibox_h).toFixed(2), hh), 'replay_compare', 'rsibox_h', { key: k.key });
            hh.append(' h)');
          });
        });
      };
      hit.addEventListener('pointermove', onMove);
      hit.addEventListener('pointerdown', onMove);
      hit.addEventListener('pointerleave', () => m.tip.hide());
    },
  });

  // Table of hours at the selected speed.
  dash.draw('speed-table', (tb) => {
    const rc = FR.ok('replay_compare'), ss = stops();
    tb.replaceChildren();
    if (!rc || !ss.length) return;
    const st = ss[SP.stop];
    systems().forEach((s) => {
      const r = rc.data.find((x) => x.key === s.system + '.' + st.stop);
      if (!r) return;
      const tr = FR.el('tr', null, null, tb);
      tr.setAttribute('data-key', encodeURIComponent(r.key));
      const th = FR.el('td', null, null, tr);
      FR.el('span', 'fr-k fr-sys-' + s.idx, null, th);
      th.append(' ');
      sysLabel(th, s);
      FR.el('td', 'n codex', Number(r.codex_h).toFixed(2), tr);
      FR.el('td', 'n rsibox', Number(r.rsibox_h).toFixed(2), tr);
      FR.el('td', 'n', r.speedup.toFixed(2) + '×', tr);
    });
  });

  dash.onData(() => {
    const fr = fracs();
    document.querySelectorAll('#speed-slider').forEach((s) => { if (fr.length) s.value = String(Math.round(fr[SP.stop] * 1000)); });
  });
})();

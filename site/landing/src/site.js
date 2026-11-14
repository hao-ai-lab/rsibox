// Landing demo, install prompt, architecture walk-through and the "try one edit" playground.
(function () {
  const $ = (t, c, x, p) => { const n = document.createElement(t); if (c) n.className = c; if (x != null) n.textContent = x; if (p) p.appendChild(n); return n; };
  const reduced = !window.matchMedia || matchMedia('(prefers-reduced-motion: reduce)').matches;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  // Recorded on FastVideo (MiniMax H3 decoding), one GB200: repos/fastvideo/examples/h3-live-edit.
  const REC = { edit_to_run: 0.72, install: 0.06, check: 6.02, undo: 0.014, psnr: 48.1, warm: 72, cold: 500 };

  // ---- landing figures: one evaluation in three bars, and the race. They rotate through the engines, one at a time.
  const hl = document.getElementById('hero-ladder'), hr = document.getElementById('hero-race');
  let heroWl = null;
  if (hl && hr && window.dash) {
    const WLS = dash.data('race_workloads').data.slice().sort((a, b) => +a.order - +b.order);
    const RUNGS = dash.data('ladder_rungs').data, LAD = dash.data('eval_ladder').data;
    const LAST = { 'vllm-d8x32k': 'Targeted probe', megatron: 'Shorter warm-up', ref2va: 'Quick screen' };   // name of each engine's last measured rung
    const PLAY = 4500, HOLD = 2500, TOTAL = PLAY + HOLD;   // ms: the animation, then the pause before the next engine
    let gen = 0, timer = null, cur = 0, paused = false, left = 0, since = 0;
    const later = (g, ms, fn) => setTimeout(() => { if (g === gen) fn(); }, ms);
    const fmtS = (t) => (t >= 100 ? Math.round(t) : t.toFixed(1)) + ' s';

    function ladder(wl, g, anim) {
      hl.replaceChildren();
      const lad = LAD.filter((r) => r.engine === wl.workload), has = (st) => lad.some((r) => +r.step === st);
      const rg = RUNGS.filter((r) => r.engine === wl.workload && has(+r.step)).map((r) => [+r.step, r.group]).sort((a, b) => a[0] - b[0]);
      const g0 = rg[0][1], sys = rg.filter((r) => r[1] === g0).pop()[0], end = rg[rg.length - 1][0];
      const rows = [[rg[0][0], 'Restart for every check'], [sys, 'Cut all system overhead'], end > sys ? [end, LAST[wl.workload] || 'Shorter check'] : [null, 'Shorter check']];
      const phases = [...new Set(lad.map((r) => r.phase))];
      const sec = (st, ph) => { const r = lad.find((x) => +x.step === st && x.phase === ph); return r ? +r.seconds : 0; };
      const tot = (st) => phases.reduce((x, ph) => x + sec(st, ph), 0), base = tot(rows[0][0]);
      rows.forEach(([st, label], i) => {
        const r = $('div', 'hl-row', null, hl); $('span', 'hl-l', label, r);
        const bar = $('span', 'hl-bar', null, r), v = $('span', 'hl-v fr-num', null, r);
        let segs = [];
        if (st == null) { v.classList.add('hl-none'); v.textContent = 'not measured yet'; }
        else {
          const t = tot(st), prev = i ? rows[i - 1][0] : st;
          segs = phases.map((ph) => { const sg = $('i', null, null, bar); sg.style.background = 'var(--ph-' + ph + ')'; sg.style.width = (100 * sec(anim ? prev : st, ph) / base) + '%'; return [sg, 100 * sec(st, ph) / base]; });
          $('b', null, i ? (base / t >= 10 ? Math.round(base / t) : (base / t).toFixed(1)) + '×' : '', v); $('span', null, fmtS(t), v);
        }
        if (anim) {
          r.style.opacity = 0; v.style.opacity = 0;
          const at = 300 + i * 1300;
          later(g, at, () => { r.style.transition = 'opacity 0.25s'; r.style.opacity = 1; });
          later(g, at + (i ? 300 : 0), () => segs.forEach(([sg, w]) => { sg.style.transition = 'width 0.85s cubic-bezier(.2,.7,.2,1)'; sg.style.width = w + '%'; }));
          later(g, at + (i ? 1000 : 200), () => { v.style.transition = 'opacity 0.3s'; v.style.opacity = 1; });
        }
      });
    }

    function race(wl, g, anim, done) {
      hr.replaceChildren();
      const NS = 'http://www.w3.org/2000/svg', el = (t, a, p) => { const n = document.createElementNS(NS, t); for (const k in a) n.setAttribute(k, a[k]); if (p) p.appendChild(n); return n; };
      const W = 480, Hh = 250, m = { l: 46, r: 46, t: 46, b: 36 }, T = +wl.budget_h;
      const pts = (arm) => dash.data('race_steps').data.filter((r) => r.workload === wl.workload && r.arm === arm && r.kind === 'best').map((r) => [+r.t_h, +r.value]).sort((x, y) => x[0] - y[0]);
      const C = pts('codex'), F = pts('fastrsi'), endC = C[C.length - 1][1], endF = F[F.length - 1][1], target = endC, hi = Math.max(endF, endC);
      const tC = C.find((p) => p[1] >= target)[0], hit = F.find((p) => p[1] >= target), tF = hit ? hit[0] : null;
      const stp = hi - 1 < 0.2 ? 0.05 : hi - 1 < 0.5 ? 0.1 : 0.25, dp = stp < 0.1 || stp === 0.25 ? 2 : 1;
      const ticks = []; for (let v = 1; v <= hi + 1e-9; v += stp) ticks.push(v);
      const top = Math.max(hi, ticks[ticks.length - 1]) + (hi - 1) * 0.11;
      const x = (t) => m.l + (W - m.l - m.r) * t / T, y = (v) => Hh - m.b - (Hh - m.t - m.b) * (v - 1) / (top - 1);
      const svg = el('svg', { viewBox: '0 0 ' + W + ' ' + Hh, role: 'img', class: 'hr-svg', 'aria-label': wl.label + ': best speedup over ' + T + ' hours. Baseline ends at ' + endC.toFixed(2) + '×, FastRSI at ' + endF.toFixed(2) + '×.' }, hr);
      ticks.forEach((v) => { el('line', { x1: x(0), x2: x(T), y1: y(v), y2: y(v), class: v === 1 ? 'hr-ax' : 'hr-grid' }, svg); el('text', { x: m.l - 7, y: y(v) + 3.5, class: 'hr-t', 'text-anchor': 'end' }, svg).textContent = v.toFixed(dp) + '×'; });
      el('line', { x1: x(0), x2: x(0), y1: y(1), y2: m.t - 8, class: 'hr-ax' }, svg);
      el('text', { x: 6, y: m.t - 16, class: 'hr-t' }, svg).textContent = 'best speedup so far';
      for (let h = 0; h <= T; h += 1) el('text', { x: x(h), y: Hh - m.b + 15, class: 'hr-t', 'text-anchor': 'middle' }, svg).textContent = h;
      el('text', { x: x(T / 2), y: Hh - 4, class: 'hr-t', 'text-anchor': 'middle' }, svg).textContent = 'hours of agent time';
      const clipId = 'hr-clip-' + g, clip = el('rect', { x: 0, y: 0, width: anim ? x(0) : W, height: Hh }, el('clipPath', { id: clipId }, el('defs', {}, svg)));
      const path = (P) => { let d = 'M' + x(0) + ' ' + y(1); for (const [t, val] of P) d += 'H' + x(t) + 'V' + y(val); return d + 'H' + x(T); };
      const gg = el('g', { 'clip-path': 'url(#' + clipId + ')' }, svg);
      el('path', { d: path(C), class: 'hr-c' }, gg); el('path', { d: path(F), class: 'hr-f' }, gg);
      const dots = [];
      [[C, 'hr-pc'], [F, 'hr-pf']].forEach(([P, cls]) => P.forEach(([t, v]) => { const c = el('circle', { cx: x(t), cy: y(v), r: 2.6, class: cls }, svg); if (anim) c.style.opacity = 0; dots.push([t, c]); }));
      const fin = el('g', { class: 'hr-fin' }, svg);
      el('line', { x1: x(0), x2: x(T), y1: y(target), y2: y(target), class: 'hr-tg' }, fin);
      const close = Math.abs(y(endF) - y(endC)) < 11, up = endF >= endC ? -1 : 1;
      el('text', { x: x(T) + 5, y: y(endC) + (close ? 3.5 - up * 5.5 : 3.5), class: 'hr-t hr-tc' }, fin).textContent = endC.toFixed(2) + '×';
      el('text', { x: x(T) + 5, y: y(endF) + (close ? 3.5 + up * 5.5 : 3.5), class: 'hr-t hr-tf' }, fin).textContent = endF.toFixed(2) + '×';
      if (tF != null && tF > 0 && tF < tC) {
        const by = y(top) - 4;
        el('path', { d: 'M' + x(tF) + ' ' + y(target) + 'V' + by + 'H' + x(tC) + 'V' + y(target), class: 'hr-br' }, fin);
        el('circle', { cx: x(tF), cy: y(target), r: 4.5, class: 'hr-df' }, fin); el('circle', { cx: x(tC), cy: y(target), r: 4.5, class: 'hr-dc' }, fin);
        const lab = el('text', { x: Math.max(150, Math.min(W - 120, (x(tF) + x(tC)) / 2)), y: by - 8, class: 'hr-big', 'text-anchor': 'middle' }, fin);
        lab.textContent = (tC / tF).toFixed(1) + '× sooner';
        el('tspan', { class: 'hr-bigs', dx: 8 }, lab).textContent = (tC - tF).toFixed(1) + ' h earlier';
      }
      if (!anim) { done(); return; }
      fin.style.opacity = 0;
      const t0 = performance.now() + 700, DUR = 3200;
      let ended = false;
      const finish = () => { if (ended || g !== gen) return; ended = true; clip.setAttribute('width', W); dots.forEach(([, c]) => (c.style.opacity = 1)); fin.style.transition = 'opacity 0.5s'; fin.style.opacity = 1; done(); };
      const tick = (now) => {
        if (g !== gen || ended) return;
        const k = Math.max(0, Math.min(1, (now - t0) / DUR)), th = k * T;
        clip.setAttribute('width', x(th));
        dots.forEach(([t, c]) => { if (t <= th) c.style.opacity = 1; });
        if (k < 1) requestAnimationFrame(tick); else finish();
      };
      requestAnimationFrame(tick);
      later(g, 700 + DUR + 1500, finish);   // lands complete even if frames are throttled
    }

    const pager = document.getElementById('hero-pick'), figs = hl.closest('.hero-figs');
    const auto = !reduced && WLS.length > 1;
    const run = () => { clearTimeout(timer); if (auto && !paused) { since = performance.now(); timer = setTimeout(() => show((cur + 1) % WLS.length, true), left); } };
    function show(i, fade) {
      const g = ++gen; clearTimeout(timer); cur = i;
      const wl = WLS[i], anim = !reduced; heroWl = wl;
      if (pager) [...pager.children].forEach((b, k) => { b.setAttribute('aria-pressed', String(k === i)); b.classList.remove('run'); });
      const draw = () => {
        if (g !== gen) return;
        document.getElementById('hero-ladder-sub').textContent = 'Time for one evaluation, ' + wl.engine_label;
        document.getElementById('hero-race-sub').textContent = wl.label + ', best speedup found in ' + (+wl.budget_h) + ' h';
        ladder(wl, g, anim); race(wl, g, anim, () => {});
        if (auto && pager) { const b = pager.children[i]; void b.offsetWidth; b.classList.add('run'); }
        left = TOTAL; run();
        hl.style.opacity = hr.style.opacity = 1;
      };
      if (fade && anim) { hl.style.opacity = hr.style.opacity = 0; later(g, 280, draw); } else draw();
    }
    if (pager) pager.style.setProperty('--hero-total', TOTAL + 'ms');
    if (pager) WLS.forEach((w, i) => { const b = $('button', 'hero-dot', w.engine_label, pager); b.type = 'button'; b.addEventListener('click', () => show(i, true)); });
    const hold = (on) => { if (on === paused) return; paused = on; if (pager) pager.classList.toggle('paused', on); if (on) { clearTimeout(timer); left = Math.max(300, left - (performance.now() - since)); } else run(); };
    figs.addEventListener('mouseenter', () => hold(true)); figs.addEventListener('mouseleave', () => hold(false));
    document.addEventListener('visibilitychange', () => hold(document.hidden));
    show(0, false);
  }

  // ---- click a landing figure: show the full figure in a popup, then put it back in the page
  const fd = document.getElementById('fig-dlg'), slot = document.getElementById('fig-slot');
  if (fd && fd.showModal) {
    let home = null, node = null;
    const back = () => { if (node && home) { home.replaceWith(node); node = home = null; } };
    document.querySelectorAll('[data-pop]').forEach((a) => a.addEventListener('click', (e) => {
      const sec = document.getElementById(a.dataset.pop); if (!sec) return;
      e.preventDefault(); back();
      home = document.createElement('div'); home.style.height = sec.offsetHeight + 'px';
      sec.replaceWith(home); node = sec; slot.appendChild(sec); fd.showModal(); fd.scrollTop = 0;
      if (heroWl) { const chip = [...sec.querySelectorAll('.fr-chip')].find((c) => c.textContent === heroWl.engine_label || c.textContent === heroWl.label); if (chip && chip.getAttribute('aria-pressed') !== 'true') chip.click(); }
      const rp = sec.querySelector('#ladder-replay'); if (rp) setTimeout(() => rp.click(), 150);
    }));
    fd.addEventListener('close', () => { if (!fd.open) back(); });
    document.getElementById('fig-close').addEventListener('click', () => fd.close());
    fd.addEventListener('click', (e) => { if (e.target === fd) fd.close(); });
  }

  // ---- the loop: one vLLM run replayed at faster model speeds. One bar steps through the speeds; evaluation takes over.
  const lsw = document.getElementById('loop-sweep');
  if (lsw && window.dash) {
    const rows = dash.data('fig2_breakdown').data.slice().sort((x, y) => +x.row - +y.row);
    const pick = document.getElementById('sweep-pick'), STEP = 3200;
    const PARTS = [['model_generation_pct', 'fr-sh-gen'], ['overlap_pct', 'fr-sh-ovl'], ['evaluation_pct', 'fr-sh-eval']];
    const bar = $('div', 'sw-bar', null, lsw), segs = PARTS.map(([, cls]) => { const sg = $('i', cls, null, bar); $('span', 'fr-num', null, sg); return sg; });
    const out = $('p', 'sw-out', null, lsw), big = $('b', 'fr-num', null, out), rest = $('span', null, null, out);
    let cur = -1, timer = null, paused = false, live = false, shown = 0, raf = 0;
    pick.style.setProperty('--hero-total', STEP + 'ms');
    const btns = rows.map((r, i) => { const b = $('button', 'hero-dot', r.tok_s_label, pick); b.type = 'button'; b.addEventListener('click', () => show(i)); return b; });
    const run = () => { clearTimeout(timer); if (!reduced && live && !paused) timer = setTimeout(() => show((cur + 1) % rows.length), STEP); };
    function show(i) {
      const r = rows[i], ev = +r.evaluation_pct, from = shown; cur = i;
      btns.forEach((b, k) => { b.setAttribute('aria-pressed', String(k === i)); b.classList.remove('run'); });
      PARTS.forEach(([f], k) => { segs[k].style.width = (+r[f]) + '%'; segs[k].firstChild.textContent = +r[f] >= 6 ? (+r[f]) + '%' : ''; segs[k].title = (+r[f]) + '%'; });
      cancelAnimationFrame(raf);
      const t0 = performance.now(), tw = (now) => { const k = reduced ? 1 : Math.min(1, (now - t0) / 700); shown = from + (ev - from) * k; big.textContent = Math.round(shown) + '%'; if (k < 1) raf = requestAnimationFrame(tw); };
      tw(t0); setTimeout(() => { shown = ev; big.textContent = ev + '%'; }, 900);
      rest.textContent = ' of the run is evaluation alone, with the model already done. Model generating: ' + (+r.model_generation_pct) + '%. Whole run: ' + (+r.total_h) + ' h.';
      lsw.setAttribute('aria-label', 'At ' + r.tok_s_label + ': model generation ' + (+r.model_generation_pct) + '%, generation overlapping evaluation ' + (+r.overlap_pct) + '%, evaluation alone ' + ev + '%, of a ' + (+r.total_h) + ' hour run.');
      if (!reduced && live) { void btns[i].offsetWidth; btns[i].classList.add('run'); }
      run();
    }
    const card = lsw.closest('.fr-card');
    const hold = (on) => { paused = on; pick.classList.toggle('paused', on); if (on) clearTimeout(timer); else run(); };
    card.addEventListener('mouseenter', () => hold(true)); card.addEventListener('mouseleave', () => hold(false));
    show(0);
    if (!reduced && 'IntersectionObserver' in window) new IntersectionObserver((es) => { const on = es[0].isIntersecting; if (on === live) return; live = on; if (on) show(0); else clearTimeout(timer); }, { threshold: 0.5 }).observe(card);
  }

  // ---- the loop: light each stage in turn, with an arrow popping between them. The two evaluation stages hold longest.
  const lp = document.getElementById('lp');
  if (lp && !reduced && 'IntersectionObserver' in window) {
    const parts = [...lp.querySelectorAll('[data-lp]')].sort((x, y) => x.dataset.lp - y.dataset.lp);
    const hold = [900, 450, 2400, 450, 1500, 450, 1100, 900];   // ms per stage; illustrative pacing, not measured
    let i = -1, timer = null, seen = false;
    const step = () => {
      i = (i + 1) % parts.length;
      if (i === 0) parts.forEach((n) => n.classList.remove('on', 'done'));
      parts.forEach((n, k) => { n.classList.toggle('on', k === i); if (k < i) n.classList.add('done'); });
      timer = setTimeout(step, hold[i]);
    };
    lp.classList.add('lp-live');
    new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting && !timer) { seen = true; step(); } else if (!e.isIntersecting && timer) { clearTimeout(timer); timer = null; }
    }), { threshold: 0.4 }).observe(lp);
  }

  // ---- where am I: a numbered label on each section, a side rail on wide screens, and a small contents button everywhere
  const TOC = [['top', 'Start'], ['loop', 'The loop'], ['story', 'Where the overhead comes from'], ['arch', 'How FastRSI works'], ['built', 'Case studies'],
    ['sooner', 'Evaluation: same result sooner'], ['eval-time', 'Evaluation: time per check'], ['model-speed', 'Evaluation: faster models'], ['use', 'Try it']]
    .map(([id, label]) => ({ id, label, el: document.getElementById(id) })).filter((t) => t.el);
  if (TOC.length > 2) {
    TOC.forEach((t, i) => { const h = i && t.el.querySelector('h2'); if (h) h.parentNode.insertBefore($('p', 'sec-k', String(i).padStart(2, '0') + ' / ' + String(TOC.length - 1).padStart(2, '0') + '  ' + t.label, null), h); });
    const go = (t) => { t.el.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' }); };
    const rail = $('nav', 'toc-rail', null, document.body); rail.setAttribute('aria-label', 'Sections');
    const wrap = $('div', 'toc', null, document.body), btn = $('button', 'toc-btn', null, wrap), panel = $('nav', 'toc-panel', null, wrap);
    btn.type = 'button'; btn.setAttribute('aria-expanded', 'false'); btn.setAttribute('aria-label', 'Contents'); panel.hidden = true; panel.setAttribute('aria-label', 'Contents');
    const bIcon = $('span', 'toc-ic', null, btn), bTxt = $('span', 'toc-now', '', btn); bIcon.setAttribute('aria-hidden', 'true'); bIcon.innerHTML = '<i></i><i></i><i></i>';
    const items = TOC.map((t, i) => {
      const r = $('button', 'toc-tick', null, rail); r.type = 'button'; $('span', null, t.label, r); r.addEventListener('click', () => go(t));
      const p = $('button', 'toc-item', null, panel); p.type = 'button'; $('span', 'fr-num', String(i).padStart(2, '0'), p); $('span', null, t.label, p);
      p.addEventListener('click', () => { panel.hidden = true; btn.setAttribute('aria-expanded', 'false'); go(t); });
      return [r, p];
    });
    btn.addEventListener('click', () => { panel.hidden = !panel.hidden; btn.setAttribute('aria-expanded', String(!panel.hidden)); });
    document.addEventListener('click', (e) => { if (!wrap.contains(e.target)) { panel.hidden = true; btn.setAttribute('aria-expanded', 'false'); } });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { panel.hidden = true; btn.setAttribute('aria-expanded', 'false'); } });
    let cur = -1, tick = false;
    const mark = () => {
      tick = false; let k = 0; const line = innerHeight * 0.4;
      TOC.forEach((t, i) => { if (t.el.isConnected && t.el.getBoundingClientRect().top <= line) k = i; });
      if (k === cur) return; cur = k;
      items.forEach(([r, p], i) => { r.classList.toggle('on', i === k); p.classList.toggle('on', i === k); if (i === k) { r.setAttribute('aria-current', 'true'); p.setAttribute('aria-current', 'true'); } else { r.removeAttribute('aria-current'); p.removeAttribute('aria-current'); } });
      bTxt.textContent = k ? String(k).padStart(2, '0') + '  ' + TOC[k].label : 'Contents';
    };
    addEventListener('scroll', () => { if (!tick) { tick = true; requestAnimationFrame(mark); } }, { passive: true });
    addEventListener('resize', mark); mark();
  }

  // ---- "Try it": copy the prompt and show it
  const dlg = document.getElementById('try-dlg'), text = document.getElementById('install-text'), status = document.getElementById('try-status');
  const copyPrompt = async () => {
    try { await navigator.clipboard.writeText(text.textContent); status.textContent = 'Copied. Paste it into your coding agent. It needs Python 3.11, no GPU and no dependencies.'; }
    catch (e) { const r = document.createRange(); r.selectNodeContents(text); const s = getSelection(); s.removeAllRanges(); s.addRange(r); status.textContent = 'Copy the selected prompt, then paste it into your coding agent. It needs Python 3.11, no GPU and no dependencies.'; }
  };
  const openTry = () => { if (dlg.showModal) dlg.showModal(); else dlg.setAttribute('open', ''); copyPrompt(); };
  document.querySelectorAll('#try-open, .try-open').forEach((b) => b.addEventListener('click', openTry));
  const cp = document.getElementById('install-copy'); if (cp) cp.addEventListener('click', copyPrompt);
  const cl = document.getElementById('try-close'); if (cl) cl.addEventListener('click', () => dlg.close());
  if (dlg) dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });

  // ---- cite: copy the BibTeX entry
  const cc = document.getElementById('cite-copy'), cb = document.getElementById('cite-bib');
  if (cc && cb) cc.addEventListener('click', () => {
    const done = () => { cc.textContent = 'Copied'; setTimeout(() => (cc.textContent = 'Copy'), 1600); };
    const pickText = () => { const r = document.createRange(); r.selectNodeContents(cb); const sl = getSelection(); sl.removeAllRanges(); sl.addRange(r); cc.textContent = 'Selected'; setTimeout(() => (cc.textContent = 'Copy'), 1600); };
    try { navigator.clipboard.writeText(cb.textContent).then(done, pickText); } catch (e) { pickText(); }
  });

  // ---- architecture: six steps on one diagram. Plays through once when it comes into view; click a step or a box to jump.
  const ar = document.getElementById('ar');
  if (ar) {
    const N = 6, STEP = 5200, pick = document.getElementById('ar-pick'), btns = [...pick.children];
    const boxes = [...ar.querySelectorAll('.ar-box')], arrows = [...ar.querySelectorAll('.ar-to')];
    let at = 0, timer = null, playing = false, paused = false, played = false;
    pick.style.setProperty('--hero-total', STEP + 'ms');
    const lit = (el, n) => !!el && (el.dataset.on || '').split(' ').includes(String(n));
    const next = () => { clearTimeout(timer); if (playing && !paused) timer = setTimeout(() => { if (at < N) show(at + 1); else { playing = false; btns[at - 1].classList.remove('run'); } }, STEP); };
    function show(n) {
      at = n; ar.dataset.at = n;
      btns.forEach((b, k) => { b.setAttribute('aria-pressed', String(k === n - 1)); b.classList.remove('run'); });
      ar.querySelectorAll('.ar-panel, .ar-text').forEach((p) => { p.hidden = p.dataset.p !== String(n); });
      arrows.forEach((x) => { x.classList.remove('flow'); if (lit(x.previousElementSibling, n) && lit(x.nextElementSibling, n)) { void x.offsetWidth; x.classList.add('flow'); } });
      if (playing) { void btns[n - 1].offsetWidth; btns[n - 1].classList.add('run'); }
      next();
    }
    const play = () => { playing = !reduced; show(1); };
    btns.forEach((b) => b.addEventListener('click', () => { playing = false; clearTimeout(timer); show(+b.dataset.go); }));
    boxes.forEach((bx) => { bx.tabIndex = 0; bx.setAttribute('role', 'button'); const go = () => { const on = bx.dataset.on.split(' ').map(Number); playing = false; clearTimeout(timer); show(on.includes(at) ? on[(on.indexOf(at) + 1) % on.length] : on[0]); }; bx.addEventListener('click', go); bx.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } }); });
    document.getElementById('ar-play').addEventListener('click', play);
    const hold = (on) => { paused = on; pick.classList.toggle('paused', on); if (on) clearTimeout(timer); else next(); };
    ar.addEventListener('mouseenter', () => hold(true)); ar.addEventListener('mouseleave', () => hold(false));
    show(1);
    if (!reduced && 'IntersectionObserver' in window) { const io = new IntersectionObserver((es) => { if (es[0].isIntersecting && !played) { played = true; io.disconnect(); play(); } }, { threshold: 0.45 }); io.observe(ar); }
  }

  // ---- case studies: one replayed upstream change per engine with its diff; a popup lets you switch between all of them
  const ai = document.getElementById('alt-info');
  if (ai && typeof BUILT !== 'undefined') {
    const engs = BUILT.map((e) => ({ engine: e.engine, cases: (e.edits || []).filter((q) => q.files && q.files.length) })).filter((e) => e.cases.length);
    const pick = document.getElementById('alt-pick'), more = document.getElementById('alt-more'), dlg3 = document.getElementById('alt-dlg'), list = document.getElementById('alt-list');
    const times = (x) => (x >= 10 ? Math.round(x) : x.toFixed(1)) + '×', secs = (t) => (t >= 10 ? +t.toFixed(1) : t) + ' s';
    let cur = null;
    const tag = (p, q) => $('span', 'cs-tag' + (q.live ? ' ok' : ''), q.tag || (q.live ? 'installed live' : q.install ? 'restarted from a warm copy' : 'restarted'), p);
    const PH = ['Failed live attempt, stop the old job', 'Process start-up and imports', 'Distributed init, kernels, model build', 'First step and warm-up steps', 'Install the edit live', 'Run the check', 'Other'];
    // two stacked bars (plain restart, FastRSI) on one time axis, then the same phases as a small table
    const phases = (q, info) => {
      const g = $('div', 'alt-gain', null, info), max = Math.max(q.t_rst, q.t_run);
      [['Plain restart', q.t_rst, q.ph[0]], [q.live ? 'FastRSI, installed live' : 'FastRSI, ' + q.tag, q.t_run, q.ph[1]]].forEach(([lab, t, ph], i) => {
        const r = $('div', 'alt-bar', null, g); $('span', 'alt-bl', lab, r); const tr = $('span', 'alt-btr', null, r), st = $('span', 'alt-stk', null, tr); $('span', 'alt-bv fr-num', secs(t), r);
        ph.forEach(([n, s]) => { const sg = $('i', 'ph' + PH.indexOf(n), null, st); sg.style.flex = s + ' 0 0'; sg.title = n + ': ' + s + ' s'; });
        const w = Math.max(0.8, 100 * t / max) + '%'; if (reduced || !i) st.style.width = w; else { st.style.width = '100%'; setTimeout(() => { st.style.transition = 'width 0.9s cubic-bezier(.2,.7,.2,1)'; st.style.width = w; }, 250); }
      });
      const x = $('p', 'alt-x', null, info);
      if (q.x >= 1.05) { $('b', 'fr-num', times(q.x), x); $('span', null, ' sooner from edit to check result', x); } else $('span', null, 'No gain on this one: FastRSI fell back to a full restart.', x);
      const tb = $('table', 'alt-ph', null, info), hd = $('tr', null, null, tb); $('th', null, 'Where the time goes', hd); $('th', null, 'Restart', hd); $('th', null, 'FastRSI', hd);
      const get = (ph, n) => { const f = ph.find((p) => p[0] === n); return f ? f[1].toFixed(1) + ' s' : '–'; };
      PH.filter((n) => q.ph[0].concat(q.ph[1]).some((p) => p[0] === n)).forEach((n) => { const r = $('tr', null, null, tb), c = $('td', null, null, r); $('i', 'ph' + PH.indexOf(n), null, c); c.appendChild(document.createTextNode(n)); $('td', 'fr-num', get(q.ph[0], n), r); $('td', 'fr-num', get(q.ph[1], n), r); });
      (q.notes || []).forEach((n) => $('p', 'alt-note', n, info));
      $('p', 'alt-note', q.setup, info);
    };
    const render = (q, info, df) => {
      info.replaceChildren();
      const top = $('p', 'alt-pr', null, info);
      if (q.agent) $('span', 'alt-agent', 'Our own agent edit · ' + q.repo, top); else { const a = $('a', null, q.repo + ' #' + q.pr + ' ↗', top); a.href = 'https://github.com/' + q.repo + '/pull/' + q.pr; a.target = '_blank'; a.rel = 'noopener'; }
      tag(top, q);
      $('h3', 'alt-t', q.utitle || q.title, info);
      $('p', 'alt-size fr-num', '+' + q.add + ' −' + q.del + ' in ' + q.nfiles + (q.nfiles === 1 ? ' file' : ' files'), info);
      if (q.ph) phases(q, info);
      else if (q.x) {
        const g = $('div', 'alt-gain', null, info);
        [[q.live ? 'Restart, then run' : 'Plain restart', q.t_rst, 1, 'alt-b0'], [q.live ? 'Install live, then run' : 'Restart from a warm copy', q.t_run, q.t_run / q.t_rst, 'alt-b1']].forEach(([lab, t, w, cls], i) => {
          const r = $('div', 'alt-bar', null, g); $('span', 'alt-bl', lab, r); const tr = $('span', 'alt-btr', null, r), f = $('i', cls, null, tr); $('span', 'alt-bv fr-num', secs(t), r);
          const to = 'scaleX(' + Math.max(0.008, w) + ')'; if (reduced || !i) f.style.transform = to; else { f.style.transform = 'scaleX(1)'; setTimeout(() => { f.style.transition = 'transform 0.9s cubic-bezier(.2,.7,.2,1)'; f.style.transform = to; }, 250); }
        });
        const x = $('p', 'alt-x', null, info); $('b', 'fr-num', times(q.x), x); $('span', null, q.live ? ' sooner to a running edit' : ' faster than a plain restart' + (q.pair ? ' (' + q.pair + ')' : ''), x);
      } else $('p', 'alt-pend', 'Timing is being re-measured.', info);
      (q.copy || []).forEach(([label, text]) => {
        if (!text) return;
        const blk = $('div', 'alt-blk', null, info); $('b', null, label, blk); const para = $('p', null, null, blk);
        text.split('`').forEach((part, i) => { if (i % 2) $('code', null, part, para); else para.appendChild(document.createTextNode(part)); });
      });
      df.replaceChildren(); df.scrollTop = 0;
      q.files.forEach((fl) => {
        $('p', 'df-file', fl.path, df); const pre = $('pre', 'df-pre', null, df);
        fl.lines.forEach((ln) => $('span', ln[0] === '+' ? 'df-add' : ln[0] === '-' ? 'df-del' : ln.startsWith('@@') ? 'df-hunk' : null, (ln || ' ') + '\n', pre));
        if (fl.cut) $('p', 'df-more', fl.cut + ' more lines in the pull request', df);
      });
      if (q.more > 0) $('p', 'df-more', q.more + ' more changed ' + (q.more === 1 ? 'file' : 'files') + ' in the pull request', df);
    };
    const show = (e) => {
      cur = e; [...pick.children].forEach((b) => b.setAttribute('aria-pressed', String(b.textContent === e.engine)));
      more.hidden = e.cases.length < 2; more.textContent = 'Show all ' + e.cases.length;
      render(e.cases[0], ai, document.getElementById('alt-diff'));
    };
    engs.forEach((e) => { const b = $('button', 'fr-chip', e.engine, pick); b.type = 'button'; b.addEventListener('click', () => show(e)); });
    const dpick = document.getElementById('alt-dpick');
    let pe = null;   // engine shown in the popup
    const choose = (i) => { [...list.children].forEach((b, k) => b.setAttribute('aria-pressed', String(k === i))); render(pe.cases[i], document.getElementById('alt-pinfo'), document.getElementById('alt-pdiff')); };
    const fill = (e) => {
      pe = e; [...dpick.children].forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.e === e.engine))); list.replaceChildren();
      e.cases.forEach((q, i) => {
        const r = $('button', 'alt-item', null, list); r.type = 'button';
        $('b', null, (q.utitle || q.title).replace(/^(\[[^\]]+\])+\s*/, ''), r);
        const m = $('span', 'alt-im fr-num', q.agent ? 'agent edit' : '#' + q.pr, r); $('span', q.x ? 'alt-ix' : 'alt-ix na', q.x ? times(q.x) : 'n/a', m);
        r.addEventListener('click', () => choose(i));
      });
      choose(0);
    };
    engs.forEach((e) => { const b = $('button', 'fr-chip', null, dpick); b.type = 'button'; b.dataset.e = e.engine; b.textContent = e.engine + ' · ' + e.cases.length; b.addEventListener('click', () => fill(e)); });
    more.addEventListener('click', () => { if (dlg3.showModal) { dlg3.showModal(); dlg3.scrollTop = 0; } fill(cur); });
    document.getElementById('alt-close').addEventListener('click', () => dlg3.close());
    dlg3.addEventListener('click', (ev) => { if (ev.target === dlg3) dlg3.close(); });
    if (engs.length) show(engs[0]); else document.getElementById('built').hidden = true;
  }

  // ---- playground: three recorded cases
  const pl = document.getElementById('pl');
  if (pl) {
    const CASES = [
      { name: 'An edit that works', about: 'A one-line change to the H3 decoder. The probe generates one clip and compares it with the golden output.',
        install: ['{ "ok": true, "outcome": "committed",', '  "mechanisms": ["R0 swap"],', '  "edit_to_run_s": 0.72, "cost_s": 0.06 }'], iok: true,
        check: ['{ "seconds": 6.02, "psnr_vs_golden": 48.1,', '  "proven": true,   // the edited line ran once on rank 0', '}'], cok: true, csec: REC.check,
        undo: ['{ "ok": true, "state": "base", "cost_s": 0.014 }'],
        verdict: 'Result in ' + (REC.edit_to_run + REC.check).toFixed(1) + ' s. A restart to see the same edit takes about ' + REC.warm + ' s warm and about ' + REC.cold + ' s cold.' },
      { name: 'An edit the probe never runs', about: 'A replayed upstream change to methods this workload does not call. The install succeeds, so a naive check would report a pass.',
        install: ['{ "ok": true, "outcome": "committed" }'], iok: true,
        check: ['{ "proven": false,  // zero witness:', '  // no call reached the edited code', '}'], cok: false, csec: REC.check,
        undo: ['{ "ok": true, "state": "base" }'],
        verdict: 'Not counted. The lab injects a marker at the top of every edited function; with zero hits, the check is never a pass.' },
      { name: 'A broken edit', about: 'A patch with a syntax error, one of the faults we inject to test the lab.',
        install: ['{ "ok": false, "outcome": "aborted",', '  // rejected before installing', '}'], iok: false,
        verdict: 'Nothing was installed and the engine never left its base state. Other faults (an exception in the forward pass, out of memory, a NaN loss) fail the check and keep the process.' },
    ];
    const cases = document.getElementById('pl-cases'), about = document.getElementById('pl-about'), out = document.getElementById('pl-out'), state = document.getElementById('pl-state'), hint = document.getElementById('pl-hint');
    const bI = document.getElementById('pl-install'), bC = document.getElementById('pl-check'), bU = document.getElementById('pl-undo');
    let cur = CASES[0], busy = false;
    const setState = (s) => { state.textContent = s; state.className = s === 'edited' ? 'warn' : ''; };
    const block = async (call, linesOut, ok, sec) => {
      const b = $('div', 'pl-blk', null, out); $('div', 'pl-call', '> ' + call, b);
      if (sec && !reduced) { const w = $('span', 'bar', null, b), i = $('i', null, null, w); i.style.transition = 'transform ' + Math.min(1.6, sec / 4) + 's linear'; await sleep(30); i.style.transform = 'scaleX(1)'; await sleep(Math.min(1600, sec * 250)); w.remove(); }
      linesOut.forEach((l) => $('div', 'pl-json ' + (ok ? '' : 'bad'), l, b)); out.scrollTop = out.scrollHeight;
    };
    const pick = (c) => { cur = c; out.replaceChildren(); about.textContent = c.about; setState('base'); bI.disabled = false; bC.disabled = bU.disabled = true; hint.textContent = 'Start with install.';
      [...cases.children].forEach((b, i) => b.setAttribute('aria-pressed', String(CASES[i] === c))); };
    CASES.forEach((c) => { const b = $('button', 'fr-chip', c.name, cases); b.type = 'button'; b.addEventListener('click', () => { if (!busy) pick(c); }); });
    const guard = (fn) => async () => { if (busy) return; busy = true; try { await fn(); } finally { busy = false; } };
    bI.addEventListener('click', guard(async () => { bI.disabled = true; await block('install(patch)', cur.install, cur.iok);
      if (cur.iok) { setState('edited'); bC.disabled = false; bU.disabled = false; hint.textContent = 'Now check it.'; } else { $('div', 'pl-verdict', cur.verdict, out); hint.textContent = 'Pick another case.'; } }));
    bC.addEventListener('click', guard(async () => { bC.disabled = true; await block('check()', cur.check, cur.cok, cur.csec); $('div', 'pl-verdict', cur.verdict, out); out.scrollTop = out.scrollHeight; hint.textContent = 'Uninstall to return to base.'; }));
    bU.addEventListener('click', guard(async () => { bU.disabled = bC.disabled = true; await block('uninstall()', cur.undo, true); setState('base'); bI.disabled = false; hint.textContent = 'Back at base. Install again or pick another case.'; }));
    pick(CASES[0]);
  }
})();

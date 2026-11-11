// The scroll story: one pinned window replays each way of checking an edit. Times come from the ladder data.
(function () {
  const stage = document.getElementById('st-stage'); if (!stage) return;
  const $ = (t, c, x, p) => { const n = document.createElement(t); if (c) n.className = c; if (x != null) n.textContent = x; if (p) p.appendChild(n); return n; };
  const body = document.getElementById('st-body'), title = document.getElementById('st-title'), cap = document.getElementById('st-cap');
  const hTries = document.getElementById('st-tries'), hWait = document.getElementById('st-wait'), hPer = document.getElementById('st-per');
  const reduced = !window.matchMedia || matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Measured numbers: the vLLM ladder (rows 1, 6, 7) from the dashboard's data.
  const lad = dash.data('eval_ladder').data.filter((r) => r.engine === 'vllm-d8x32k');
  const kinds = {}, labels = {}; dash.data('eval_phases').data.forEach((r) => { kinds[r.phase] = r.kind; labels[r.phase] = r.label; });
  const row = (s) => lad.filter((r) => +r.step === s), sum = (rs) => rs.reduce((a, r) => a + +r.seconds, 0);
  const PH = row(1).map((r) => [labels[r.phase] || r.phase, +r.seconds, r.phase]);
  const RESTART = sum(row(1)), STARTUP = sum(row(1).filter((r) => kinds[r.phase] === 'startup'));
  const maxStep = Math.max(...lad.map((r) => +r.step));
  const PROBE = sum(row(maxStep)), FULL = sum(row(maxStep - 1));
  const GRAPH = +(row(1).find((r) => r.phase === 'cuda_graph_capture') || {}).seconds || 0;
  const s1 = (v) => (v >= 100 ? Math.round(v) : v.toFixed(1)) + ' s';
  const put = (k, v) => document.querySelectorAll('[data-st="' + k + '"]').forEach((n) => { n.textContent = v; });
  put('restart_s', Math.round(RESTART)); put('startup_pct', Math.round(100 * STARTUP / RESTART)); put('probe_s', PROBE.toFixed(1));

  const clock = (s) => { s = Math.round(s); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };
  const H = { tries: 0, wait: 0 };
  const hud = (per) => { hTries.textContent = H.tries; hWait.textContent = H.wait < 60 ? H.wait.toFixed(H.wait < 10 && H.wait % 1 ? 1 : 0) + ' s' : clock(H.wait); if (per) hPer.textContent = per; };

  let gen = 0; class Stop extends Error {}
  function ctx(fast) {
    const id = ++gen, alive = () => id === gen;
    const c = {
      fast,
      wait: (ms) => { if (!alive()) throw new Stop(); return fast ? Promise.resolve() : new Promise((ok, no) => setTimeout(() => (alive() ? ok() : no(new Stop())), ms)); },
      frame: () => new Promise((ok, no) => requestAnimationFrame(() => (alive() ? ok() : no(new Stop())))),
      async type(el, text, cps) { if (fast) { el.textContent += text; return; } for (const ch of text) { el.textContent += ch; await c.wait(1000 / (cps || 55)); } },
      async tween(ms, fn) { if (fast) { fn(1); return; } const t0 = performance.now(); for (;;) { await c.frame(); const p = Math.min(1, (performance.now() - t0) / ms); fn(p); if (p >= 1) return; } },
      // Advance the simulated clock by `sec` while a bar fills over `ms` of real time.
      async spend(bar, sec, ms, each) { const w0 = H.wait; await c.tween(ms, (p) => { if (bar) bar.style.transform = 'scaleX(' + p + ')'; if (each) each(p); H.wait = w0 + sec * p; hud(); }); },
    };
    return c;
  }
  const ed = document.getElementById('st-edit'), edFile = document.getElementById('st-efile'), edDiff = document.getElementById('st-ediff'), bar2 = document.getElementById('st-bar2');
  // Top half of the window: the code change as a diff. `rows` are [sign, text] with sign ' ', '-' or '+'.
  const diff = (file, rows) => { ed.hidden = false; bar2.hidden = false; ed.parentNode.classList.add('split'); edFile.textContent = file; edDiff.replaceChildren();
    rows.forEach(([sg, txt], i) => { const l = $('div', 'dl ' + (sg === '+' ? 'add' : sg === '-' ? 'del' : sg === '@' ? 'hunk' : ''), null, edDiff); $('span', 'dg', sg === '@' ? '' : sg, l); $('span', null, txt, l); l.style.animationDelay = (i * 70) + 'ms'; }); };
  const win = (t, capText, top) => { ed.hidden = true; bar2.hidden = true; ed.parentNode.classList.remove('split'); title.textContent = t; body.replaceChildren(); body.classList.toggle('top', !!top); cap.textContent = capText; };
  const line = (cls, txt, parent) => $('div', 'ln ' + (cls || ''), txt, parent || body);
  const bar = (parent) => { const b = $('span', 'bar', null, parent); return $('i', null, null, b); };
  async function cmd(c, text) { const l = line(''); $('span', 'ps', '$ ', l); await c.type($('span', null, '', l), text); await c.wait(260); }
  function cell(n) { const el = $('div', 'cell', null, body); $('span', 'in', '[' + n + ']', el); return { el, code: $('pre', 'code', '', el) }; }
  async function probeCell(c, n, code, verdict, ok, flag) {
    const k = cell(n); await c.type(k.code, code, 70); await c.wait(200);
    const o = $('div', 'o', null, k.el), b = bar(o); await c.spend(b, PROBE, 800);
    o.replaceChildren(); const v = $('span', ok ? 'good' : 'bad', verdict, o); $('span', 't', PROBE.toFixed(1) + ' s', o);
    H.tries++; hud(); if (flag) { await c.wait(700); v.className = 'warn sus'; $('span', 'flag', flag, o); } await c.wait(420); return k;
  }
  const NB_CAP = 'The probe time is measured: ' + PROBE.toFixed(1) + ' s for a small-batch probe on a warm vLLM engine. Speedup values are illustrative.';
  const loaded = (n) => { const k = cell(n); k.code.textContent = 'engine = serve("nemotron-3.5-super")'; const o = $('div', 'o', null, k.el); $('span', 't', 'ready in ' + Math.round(STARTUP) + ' s · start-up paid once', o); };

  const scenes = {
    async restart(c) {
      win('vllm · dev workspace', 'Phase times are measured on vLLM serving Nemotron 3.5 Super on one GB200, shown about 30× faster than real time. The two code edits are illustrative.');
      H.tries = 0; H.wait = 0; hud(Math.round(RESTART) + ' s');
      const EDITS = [
        [['@', '@@ def schedule(self):'], [' ', '    waiting = self.waiting'], ['-', '    budget = self.max_num_batched_tokens'], ['+', '    budget = self.max_num_batched_tokens * 2'], [' ', '    return self._fill(waiting, budget)']],
        [['@', '@@ def schedule(self):'], ['-', '    waiting = self.waiting'], ['+', '    waiting = sorted(self.waiting, key=prompt_len)'], [' ', '    budget = self.max_num_batched_tokens'], [' ', '    return self._fill(waiting, budget)']]];
      diff('scheduler.py', EDITS[0]); await c.wait(700);
      const runs = [['throughput 0.98×   ✗ no gain', 'bad', 30], ['throughput 1.04×   ✓ keep', 'good', 15]];
      for (let i = 0; i < runs.length; i++) {
        await cmd(c, 'python bench.py --model nemotron-3.5-super');
        // one bar for the whole restart; the status text rolls up as each stage starts
        const r = $('div', 'ln ph ph1', null, body), st = $('span', 'ph-st', null, r), track = $('span', 'bar bar-seg', null, r), t = $('span', null, '0 s', r);
        let done = 0;
        for (const [label, sec, ph] of PH) {
          const old = st.firstChild, cur = $('b', 'in', label.toLowerCase(), st); if (old) { old.className = 'out'; setTimeout(() => old.remove(), 260); }
          const sg = $('i', null, null, track); sg.style.flex = '0 0 ' + (100 * sec / RESTART) + '%'; sg.style.background = 'var(--ph-' + ph + ')';
          const d0 = done; await c.spend(null, sec, sec * runs[i][2] , (p) => { sg.style.transform = 'scaleX(' + p + ')'; t.textContent = Math.round(d0 + sec * p) + ' s'; });
          sg.style.transform = 'none'; done += sec;
        }
        t.textContent = s1(done); st.textContent = ''; $('b', 'in', 'restart, then check', st);
        line(runs[i][1], runs[i][0]); H.tries++; hud(); await c.wait(900);
        if (i === 0) { line('dim', '# try another idea'); await c.wait(500); diff('scheduler.py', EDITS[1]); await c.wait(900); }
      }
      line('sum', H.tries + ' tries, ' + clock(H.wait) + ' of waiting. ' + Math.round(100 * STARTUP / RESTART) + '% of it was start-up.');
    },
    async notebook(c) {
      win('notebook · scheduler-debug.ipynb', NB_CAP, true);
      H.tries = 0; H.wait = 0; hud(PROBE.toFixed(1) + ' s'); loaded(1); await c.wait(500);
      await probeCell(c, 2, 'reload(scheduler); probe(engine, steps=10)', '0.98×  ✗ no gain', false);
      await probeCell(c, 3, 'reload(scheduler); probe(engine, steps=10)', '1.04×  ✓ keep', true);
      await probeCell(c, 4, 'reload(scheduler); probe(engine, steps=10)', '1.07×  ✓ keep', true);
      line('sum', H.tries + ' tries in ' + H.wait.toFixed(1) + ' s. One restart would still be loading weights.');
    },
    async catch(c) {
      win('notebook · scheduler-debug.ipynb', 'Three ways a hand reload goes wrong. Speedup values are illustrative; the ' + Math.round(STARTUP) + ' s restart is measured.', true);
      H.tries = 3; H.wait = 3 * PROBE; hud(PROBE.toFixed(1) + ' s');
      await probeCell(c, 5, 'reload(scheduler); probe(engine, steps=10)', '1.31×  ✓', true, 'too good?');
      const notes = $('div', 'notes', null, body);
      for (const [h, t] of [['The edit never ran.', 'The engine still holds objects built from the old class.'], ['The graphs are stale.', 'CUDA graphs captured before the edit replay the old code.'], ['The state is dirty.', 'The last try left its cache and counters behind.']]) {
        const d = $('div', null, null, notes); $('b', null, h + ' ', d); d.appendChild(document.createTextNode(t)); await c.wait(1000);
      }
      await c.wait(400); const v = $('div', 'verdict', null, body); $('span', 'vx', '✗', v); $('b', null, 'Untrusted. ', v); v.appendChild(document.createTextNode('Re-initialising everything by hand is too tedious, and there is no clean way to revert. It has to restart.')); await c.wait(1400);
      const k = cell(6); await c.type(k.code, 'engine = serve("nemotron-3.5-super")  # restart to be sure', 60);
      const o = $('div', 'o', null, k.el), b = bar(o); hud(Math.round(STARTUP) + ' s'); await c.spend(b, STARTUP, 3600);
      o.replaceChildren(); $('span', 't', 'ready in ' + Math.round(STARTUP) + ' s', o); $('span', 'bad', '✗ ' + Math.round(STARTUP) + ' s wasted restarting everything', o);
    },
    async fastrsi(c) {
      win('FastRSI · live engine', 'Install time (0.02 s), probe time and CUDA graph capture (' + Math.round(GRAPH) + ' s) are measured on vLLM. Failed edits behave as in our fault-injection tests: the check fails and the process is kept.', true);
      H.tries = 0; H.wait = 0; hud(PROBE.toFixed(1) + ' s');
      const lab = $('div', 'lab', null, body), left = $('div', 'stack', null, lab), right = $('div', 'txn', null, lab);
      $('h4', null, 'The running engine', left); $('h4', null, 'Each edit is a transaction', right);
      const L = {}; for (const [k, n] of [['proc', 'Process and CUDA context'], ['w', 'Model weights'], ['tune', 'Kernel autotune config'], ['graph', 'CUDA graphs'], ['state', 'Run state: caches, counters']]) { const el = $('div', 'layer', null, left); $('span', null, n, el); L[k] = { el, s: $('span', 'ls', 'warm', el) }; }
      const log = $('div', 'txn-log', null, right);
      const tx = (a, r, cls) => { const el = $('div', 'tx ' + (cls || ''), null, log); const l = $('span', null, a, el); const rr = $('span', 'r', r || '', el); while (log.children.length > 11) log.firstChild.remove(); return { el, l, rr }; };
      const edits = [
        { f: 'scheduler.py', hit: ['state'], cost: 0, v: '✓ 1.04×, kept', ok: true },
        { f: 'attention.py', hit: ['graph', 'state'], cost: GRAPH, v: '✗ 0.97×, reverted', ok: false },
        { f: 'sampler.py', hit: ['state'], cost: 0, v: '✗ exception in forward pass. Check failed, process kept, edit reverted.', ok: false },
      ];
      for (const e of edits) {
        tx('edit  ' + e.f, '', 'head'); await c.wait(450);
        tx('Install into the running engine', '0.02 s'); H.wait += 0.02; hud(); await c.wait(450);
        tx('Touches: ' + e.hit.map((k) => (k === 'graph' ? 'CUDA graphs' : 'run state')).join(', ')); 
        for (const k of e.hit) { L[k].el.classList.add('hit'); L[k].s.textContent = k === 'graph' ? 'recapturing' : 'resetting'; }
        if (e.cost) { const t = tx('Recapture CUDA graphs only', ''); const b = bar(t.rr); t.rr.style.width = '70px'; await c.spend(b, e.cost, 1500); t.rr.style.width = ''; t.rr.textContent = s1(e.cost); } else await c.wait(650);
        for (const k of e.hit) { L[k].el.classList.remove('hit'); L[k].s.textContent = k === 'graph' ? 'fresh' : 'clean'; }
        const p = tx('Probe: 10 steps on the edited path', ''); const pb = bar(p.rr); p.rr.style.width = '70px'; await c.spend(pb, PROBE, 800); p.rr.style.width = ''; p.rr.textContent = PROBE.toFixed(1) + ' s';
        const v = tx(e.v, ''); v.l.className = e.ok ? 'good' : 'bad'; H.tries++; hud(); await c.wait(1100);
        for (const k of e.hit) L[k].s.textContent = 'warm';
      }
    },
    async probe(c) {
      win('FastRSI · probe and confirm', 'Measured on vLLM: ' + PROBE.toFixed(1) + ' s for a small-batch probe, ' + FULL.toFixed(1) + ' s for the full check on the warm engine. Which edits pass is illustrative.', true);
      H.tries = 0; H.wait = 0; hud(PROBE.toFixed(1) + ' s');
      const lanes = $('div', 'lanes', null, body), a = $('div', 'lane', null, lanes), b = $('div', 'lane', null, lanes);
      $('h4', null, 'Targeted probes: the agent steers on these', a); const chips = $('div', 'chips', null, a);
      $('h4', null, 'Full evaluation: confirms the keepers in the background', b); const fulls = $('div', null, null, b);
      const seq = [1, 0, 1, 1, 0, 1, 0, 1]; const pend = [];
      for (let i = 0; i < seq.length; i++) {
        const ch = $('span', 'chip' + (seq[i] ? ' k' : ''), null, chips); $('span', null, 'edit ' + (i + 1), ch); $('span', seq[i] ? 'good' : 'bad', seq[i] ? '✓' : '✗', ch);
        H.tries++; H.wait += PROBE; hud();
        if (seq[i]) {
          const r = $('div', 'full', null, fulls); $('span', null, 'edit ' + (i + 1), r); const fb = bar(r); const rr = $('span', 'r', 'running', r);
          pend.push(c.tween(FULL / PROBE * 620, (p) => { fb.style.transform = 'scaleX(' + p + ')'; }).then(() => { rr.textContent = 'confirmed ✓'; rr.className = 'r good'; ch.classList.add('c'); }));
        }
        await c.wait(620);
      }
      await Promise.all(pend);
      line('sum', H.tries + ' ideas tried in ' + Math.round(H.wait) + ' s. The agent never waited on a full evaluation.', lanes);
    },
  };

  async function play(name, fast) {
    const c = ctx(fast);
    try { await scenes[name](c); } catch (e) { if (!(e instanceof Stop)) throw e; }   // plays once; the replay button runs it again
  }
  // At rest the stage shows the first scene finished; scrolling a step to mid-screen plays its scene.
  const steps = [...document.querySelectorAll('#story .st-step')]; let cur = null;
  play('restart', true); steps[0].classList.add('is-on');
  const seen = new Set();
  // A scene animates the first time its step arrives; after that it shows its finished frame.
  const show = (el) => { const name = el.dataset.scene; if (name === cur) return; cur = name; steps.forEach((s) => s.classList.toggle('is-on', s === el)); play(name, reduced || seen.has(name)); seen.add(name); };
  const rp = document.getElementById('st-replay'); if (rp) rp.addEventListener('click', () => play(cur || 'restart', reduced));
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((es) => { for (const e of es) if (e.isIntersecting) show(e.target); }, { root: document, rootMargin: matchMedia('(max-width: 820px)').matches ? '-78% 0px -21% 0px' : '-45% 0px -45% 0px', threshold: 0 });
    steps.forEach((s) => io.observe(s));
  }
})();

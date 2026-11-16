// FastRSI Audit: renders Supported PRs, Coverage, Architecture & Risk, 4-Panel Atlas, Conventions, and Faults from window.FACTS.
(function () {
  const F = window.FACTS;
  const VIEWS = ['prs', 'coverage', 'boundaries', 'atlas', 'conventions', 'faults'];
  const LEGACY_ENGINES = ['vLLM', 'Megatron-LM', 'FastVideo'];
  const ALL_ENGINES = F.survey.map((e) => e.engine);
  const NR = 'not recorded';

  function h(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const k in attrs || {}) {
      const v = attrs[k];
      if (v == null || v === false) continue;
      if (k === 'class') n.className = v;
      else if (k.slice(0, 2) === 'on') n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? '' : v);
    }
    for (const k of kids.flat(Infinity)) {
      if (k == null || k === false) continue;
      n.append(k.nodeType ? k : document.createTextNode(String(k)));
    }
    return n;
  }

  const rich = (s) => String(s).split('`').map((part, idx) => (idx % 2 ? h('code', null, part) : part));
  const nr = () => h('span', { class: 'au-nr' }, NR);
  const val = (v) => (v == null || v === '' ? nr() : rich(v));
  const fmt = (v, u) => (v == null ? nr() : typeof v === 'number' ? v + ' ' + u : rich(v));
  const num = (v) => (v == null ? nr() : typeof v === 'number' ? v.toLocaleString('en-US') : v);
  const srcText = (s) => String(s).replace(/\bPR #/g, 'FastRSI PR #');
  const src = (s, tag) => h(tag || 'p', { class: 'au-src' }, s ? srcText(s) : NR);
  const badge = (t, kind) => h('span', { class: 'au-b ' + (kind || '') }, t);
  const groups = (g) => (g && g.length ? g.map((x) => h('span', { class: 'au-g' }, x)) : nr());
  const sub = (...k) => h('span', { class: 'sub' }, ...k);
  const pretty = (k) => { const s = k.replace(/_/g, ' '); return s[0].toUpperCase() + s.slice(1); };
  const pct = (a, b) => (b ? (100 * a / b).toFixed(1) : '0.0') + '%';

  function table(cols, rows, cls) {
    const tbl = h('table', { class: 'fr-table au-t ' + (cls || '') },
      h('thead', null, h('tr', null, cols.map((c) => h('th', { class: c.n ? 'n' : null, scope: 'col' }, c.h || c)))),
      h('tbody', null, rows.map((r) => h('tr', null, r.map((cell, i) => (cell && cell.nodeName === 'TD' ? cell : h('td', { class: cols[i].n ? 'n' : cols[i].k ? 'k' : null }, cell)))))));
    return h('div', { class: 'fr-tablewrap', tabindex: '0', role: 'region', 'aria-label': 'Table, scrolls sideways on a narrow screen' }, tbl);
  }
  const kv = (pairs) => h('dl', { class: 'au-kv' }, pairs.map(([k, v]) => [h('dt', null, k), h('dd', null, num(v))]));
  const sec = (title, ...kids) => h('section', { class: 'au-sec' }, title && h('h2', { class: 'fr-h2' }, title), ...kids);
  const p = (...k) => h('p', { class: 'au-p' }, ...k);
  const hint = () => h('p', { class: 'fr-note au-scrollhint' }, 'Wide table: scroll it sideways.');

  function picker(viewState, engineList, counts, redraw) {
    return h('div', { class: 'fr-picker', role: 'group', 'aria-label': 'Engine' }, ['All'].concat(engineList).map((e) =>
      h('button', {
        type: 'button',
        class: 'fr-chip',
        'aria-pressed': String(viewState.engine === e),
        onclick: () => { viewState.engine = e; redraw(); }
      }, counts ? e + ' \u00b7 ' + counts[e] : e)));
  }

  // ================================================================ Shared Atlas State
  const atlas = {
    unit: 'functions',          // 'functions' | 'lines' | 'lines_all'
    widthMode: 'sqrt',          // 'sqrt' | 'linear' | 'equal'
    sim: {
      reinit: true,             // [x] PR #156 Constructor Re-Init
      jit: true,                // [x] Wire kernel/jit.py Triton Rebuild
      frontend: false,          // [ ] API-Server Participant
      core_rebuild: false,      // [ ] Shared Module Rebuild in Core
      native_swap: false        // [ ] Native Shared-Lib Swap
    },
    focusEngine: 'vLLM',
    focusSubsystem: null,       // selected subsystem name in Panel 2
    focusBoundary: null,        // optional boundary filter from Panel 3
    blastFilter: 'all',         // 'all' | 'silent_corruption' | 'rank_hang' | 'orphan_leak' | 'cold_fallback'
    selectedCell: { boundary: 'adapter', engine: 'vLLM' }
  };

  const SIM_TOGGLES = [
    { key: 'reinit', label: 'PR #156 Constructor Re-Init' },
    { key: 'jit', label: 'Wire kernel/jit.py Triton Rebuild' },
    { key: 'frontend', label: 'API-Server Participant' },
    { key: 'core_rebuild', label: 'Shared Module Rebuild in Core' },
    { key: 'native_swap', label: 'Native Shared-Lib Swap' }
  ];

  const MEKKO_SEGS = [
    { key: 'swap', label: 'Swap in place', cls: 'sv-swap', cssVar: 'var(--fr-fast)', live: true },
    { key: 'swap_recapture', label: 'Swap + recapture / Triton', cls: 'sv-swap_recapture', cssVar: 'var(--ph-cuda_graph_capture)', live: true },
    { key: 'reinit', label: 'Constructor re-init (#156)', cls: 'sv-reinit', cssVar: 'var(--good)', live: true },
    { key: 'rebuild', label: 'Module rebuild', cls: 'sv-rebuild', cssVar: 'var(--ph-weight_load)', live: true },
    { key: 'native_live', label: 'Native shared-lib swap', cls: 'sv-native_live', cssVar: 'var(--ph-live_install)', live: true },
    { key: 'respawn', label: 'Zygote warm respawn', cls: 'sv-respawn', cssVar: 'var(--ph-kernel_autotune)', live: false },
    { key: 'restart', label: 'Cold restart (flags / env)', cls: 'sv-restart', cssVar: 'var(--fr-codex)', live: false },
    { key: 'native_restart', label: 'Native C++/CUDA (build)', cls: 'sv-native_restart', cssVar: 'color-mix(in oklab, #9c6b4e 78%, var(--color-fg))', live: false },
    { key: 'none', label: 'Uncovered Python', cls: 'sv-none', cssVar: 'color-mix(in oklab, var(--fr-ink) 16%, var(--color-panel))', live: false, inkLabel: true }
  ];

  // Compute exact segment & firm/uncertain counts for an engine or subsystem under any toggle set `simCfg`
  function computeSlice(item, engineMeta, unitKey, simCfg) {
    const baseKey = unitKey === 'functions' ? 'functions' : 'lines';
    const includeNative = unitKey === 'lines_all';
    const raw = item[baseKey];
    const caps = raw.by_cap; // [total, reached, covered, firm]
    const ac = engineMeta.adapter_caps;
    const reachIdx = simCfg.frontend ? 0 : 1; // 0 = total (all processes), 1 = worker-reached

    const swapCov = ac.swap ? caps.swap[reachIdx] : 0;
    const graphCov = ac.swap_recapture ? caps.swap_recapture[1] : 0;
    const jitCov = simCfg.jit ? caps.jit_rebuild[reachIdx] : 0;
    const recapCov = graphCov + jitCov;

    const reinitOn = simCfg.reinit || simCfg.core_rebuild || ac.rebuild;
    const reinitCov = reinitOn ? caps.constructor_reinit[reachIdx] : 0;

    const rebuildOn = simCfg.core_rebuild || ac.rebuild;
    const rebuildCov = rebuildOn ? caps.rebuild[reachIdx] : 0;

    const respawnCov = ac.respawn ? caps.respawn[1] : 0;
    const restartCount = raw.restart;

    const pyCoveredOrFixed = swapCov + recapCov + reinitCov + rebuildCov + respawnCov + restartCount;
    const uncoveredPy = Math.max(0, raw.total - pyCoveredOrFixed);

    const nativeAmt = includeNative ? (item.native ? item.native.lines : 0) : 0;
    const nativeLive = includeNative && simCfg.native_swap ? nativeAmt : 0;
    const nativeRestart = includeNative && !simCfg.native_swap ? nativeAmt : 0;

    // Firm (syntax-proved) fraction per segment so hatching reflects real syntax vs heuristic share
    const fracFirm = (capName) => {
      const tot = caps[capName][0];
      return tot > 0 ? caps[capName][3] / tot : 0;
    };
    const graphFirmFrac = fracFirm('swap_recapture');
    const recapFirm = Math.round(graphCov * graphFirmFrac) + jitCov; // jit_kernel is 100% syntax-firm
    const rebuildFirm = Math.round(rebuildCov * fracFirm('rebuild'));
    const respawnFirm = Math.round(respawnCov * fracFirm('respawn'));

    const segValues = {
      swap: { v: swapCov, firm: Math.round(swapCov * fracFirm('swap')) },
      swap_recapture: { v: recapCov, firm: Math.min(recapCov, recapFirm) },
      reinit: { v: reinitCov, firm: reinitCov }, // constructor_reinit is 100% syntax-firm
      rebuild: { v: rebuildCov, firm: rebuildFirm },
      native_live: { v: nativeLive, firm: nativeLive }, // native_file is 100% syntax-firm
      respawn: { v: respawnCov, firm: respawnFirm },
      restart: { v: restartCount, firm: 0 },
      native_restart: { v: nativeRestart, firm: nativeRestart },
      none: {
        v: uncoveredPy,
        firm: Math.max(0, Math.min(uncoveredPy, raw.firm - (Math.round(swapCov * fracFirm('swap')) + Math.min(recapCov, recapFirm) + reinitCov + rebuildFirm + respawnFirm)))
      }
    };

    const total = raw.total + nativeAmt;
    const live = swapCov + recapCov + reinitCov + rebuildCov + nativeLive;
    const firmTotal = raw.firm + nativeAmt;
    const uncertainTotal = Math.max(0, total - firmTotal);

    return {
      total,
      pyTotal: raw.total,
      nativeLines: item.native ? item.native.lines : 0,
      nativeFiles: item.native ? item.native.files : 0,
      live,
      respawn: respawnCov,
      restart: restartCount + nativeRestart,
      uncovered: uncoveredPy,
      firm: firmTotal,
      uncertain: uncertainTotal,
      segs: segValues
    };
  }

  const BASELINE_SIM = { reinit: false, jit: false, frontend: false, core_rebuild: false, native_swap: false };

  // Diagnose why a subsystem has uncovered functions/lines and how to unlock them
  function diagnoseSubsystem(subItem, engineMeta) {
    const caps = subItem.functions.by_cap;
    const ac = engineMeta.adapter_caps;
    const unreachedSwap = caps.swap[0] - caps.swap[1];
    const uncovGraph = caps.swap_recapture[0] - (ac.swap_recapture ? caps.swap_recapture[1] : 0);
    const uncovRebuild = caps.rebuild[0] - (ac.rebuild ? caps.rebuild[1] : 0);
    const uncovReinit = caps.constructor_reinit[0] - ((atlas.sim.reinit || atlas.sim.core_rebuild || ac.rebuild) ? caps.constructor_reinit[atlas.sim.frontend ? 0 : 1] : 0);
    const uncovJit = atlas.sim.jit ? 0 : caps.jit_rebuild[0];
    const nativeLines = subItem.native.lines;

    const reasons = [];
    if (nativeLines > 0 && !atlas.sim.native_swap) {
      reasons.push({
        weight: nativeLines,
        why: num(subItem.native.files) + ' native C++/CUDA files (' + num(nativeLines) + ' lines) compile into static shared libraries loaded once via dlopen.',
        unlock: 'Compile edited TU into a versioned .so (~21 s on local NVMe) and rebind torch.ops dispatcher + recapture graphs.'
      });
    }
    if (unreachedSwap > 0 && !atlas.sim.frontend) {
      reasons.push({
        weight: unreachedSwap * 20,
        why: num(unreachedSwap) + ' OP functions run in an unreached frontend, router, or CLI driver process with no kernel Participant.',
        unlock: 'Register a request-gap LocalParticipant in the API server / TokenizerManager process (toggle "API-Server Participant" above).'
      });
    }
    if (uncovGraph > 0) {
      reasons.push({
        weight: uncovGraph * 20,
        why: num(uncovGraph) + ' GRAPH functions sit outside the adapter\'s captured-prefix globs or inside @torch.compile / @jit_fuser regions.',
        unlock: 'Extend adapter captured prefixes and invalidate Dynamo / CudaGraphRunner graph caches on phase-2 commit.'
      });
    }
    if (uncovRebuild > 0 && !atlas.sim.core_rebuild) {
      reasons.push({
        weight: uncovRebuild * 18,
        why: num(uncovRebuild) + ' imperative constructors and STATE/RESOURCE builders allocate buffers or submodules once at startup.',
        unlock: 'Distill FastVideo\'s stage-rebuild protocol (services/rebuild.py) into shared core (toggle "Shared Module Rebuild in Core").'
      });
    }
    if (uncovReinit > 0) {
      reasons.push({
        weight: uncovReinit * 18,
        why: num(uncovReinit) + ' pure fresh-field constructors (__init__ / __post_init__) block without heap field re-initialization.',
        unlock: 'Enable kernel/reinit.py from FastRSI PR #156 (toggle "PR #156 Constructor Re-Init").'
      });
    }
    if (uncovJit > 0) {
      reasons.push({
        weight: uncovJit * 25,
        why: num(uncovJit) + ' @triton.jit kernels keep stale compiled CUBINs on a plain Python function swap.',
        unlock: 'Wire core/livelab/kernel/jit.py evidence in the adapter rules (toggle "Wire kernel/jit.py Triton Rebuild").'
      });
    }
    if (!reasons.length) {
      return {
        why: 'All non-restart Python functions in this subsystem have an active live or warm-respawn path under the current capability selection.',
        unlock: 'Verify runtime reach with a workload probe and check per-rank witness counts > 0.'
      };
    }
    reasons.sort((a, b) => b.weight - a.weight);
    return reasons[0];
  }

  // ================================================================ Supported PRs (#prs)
  const prState = { engine: 'All' };
  const allPrs = F.prs.engines.flatMap((e) => e.prs.map((r) => Object.assign({ engine: e.engine, x: F.extra.prs[r.upstream_repo + '#' + r.pr] || {} }, r)));

  function outcome(r) {
    if (r.installed_live) return badge('Installed live', 'good');
    const rt = r.route || '';
    if (/forked/.test(rt)) return badge('Zygote restart (forked)', 'warn');
    if (/code restart/.test(rt)) return badge('Code restart', 'warn');
    if (/warm restart/.test(rt)) return badge('Warm restart, new process', 'warn');
    return badge(rt || NR, 'mute');
  }
  function diffCell(r) {
    const x = r.x;
    if (x.added != null) return [h('span', { class: 'fr-num' }, '+' + x.added + ' \u2212' + x.removed), sub(x.files + (x.files === 1 ? ' file' : ' files'))];
    return nr();
  }
  function renderPrs() {
    const root = document.querySelector('[data-view="prs"]');
    if (!root) return;
    const list = allPrs.filter((r) => prState.engine === 'All' || r.engine === prState.engine);
    const counts = { All: allPrs.length };
    LEGACY_ENGINES.forEach((e) => { counts[e] = allPrs.filter((r) => r.engine === e).length; });
    const nl = F.prs.not_listed, nlTotal = Object.values(nl).reduce((a, b) => a + b, 0);
    const v = F.prs.engines.find((e) => e.engine === 'vLLM').aggregate, live = allPrs.filter((r) => r.installed_live).length;
    const rows = list.map((r) => [
      [h('a', { href: 'https://github.com/' + r.upstream_repo + '/pull/' + r.pr, target: '_blank', rel: 'noopener' }, '#' + r.pr), sub(r.engine)],
      r.x.gh_title ? [rich(r.x.gh_title), /not recorded/.test(r.title) ? sub('Recorded in the repo as: ', rich(r.title.replace(/ \(title not recorded\)/, ''))) : null] : rich(r.title),
      r.kind ? r.kind : nr(),
      groups(r.function_groups),
      [outcome(r), r.route && /\(/.test(r.route) ? sub('reason: ', h('code', null, r.route.replace(/.*\(|\)/g, ''))) : null, sub(rich(r.measured_effect === 'none recorded' ? 'Effect on the workload: not recorded' : r.measured_effect)), r.note ? sub(rich(r.note)) : null],
      [h('b', null, 'Install: '), val(r.install_time), sub(h('b', null, 'Check: '), val(r.check_time))],
      val(r.restart_time_replaced),
      diffCell(r),
      h('td', { class: 'au-src' }, srcText(r.source))
    ]);
    root.replaceChildren(
      h('p', { class: 'au-intro' }, 'The upstream pull requests that were replayed in a live lab: whether each edit installed into the running engine, how long the install and check took, and the restart it replaces. It does not show speedups, and it leaves out the pull requests the labs could not take.'),
      h('p', { class: 'au-callout' }, 'None of these upstream pull requests showed a measured speedup on the workload. They test whether an edit installs and runs live.'),
      sec(allPrs.length + ' replayed pull requests, ' + live + ' installed live',
        p(nlTotal + ' other pull requests were tried and are not listed here (' + Object.keys(nl).map((e) => nl[e] + ' ' + e).join(', ') + '): they did not apply to the lab\u2019s tree or needed a manual rebase. The vLLM replay covered ' + v.first_pass.live_in_place + ' in place and ' + v.first_pass.forked_restart + ' by forked restart, 19 pull requests in all; the repo names only 6 of them, so the other 13 are not listed either.'),
        picker(prState, LEGACY_ENGINES, counts, renderPrs),
        table([{ h: 'PR', k: 1 }, 'Title', 'Kind', 'Function groups', 'Outcome', 'Time to install and check', 'Restart time it replaces', { h: 'Diff', k: 1 }, 'Source'], rows, 'au-w1'),
        hint(),
        h('p', { class: 'fr-note' }, 'Titles and diff sizes come from GitHub (the pull request\u2019s additions, deletions and changed files, read on 2026-11-11). Everything else comes from the FastRSI repo at the cited lines. The three vLLM rows citing mechanisms.md are examples from an earlier survey, not from the 19-PR replay.')),
      sec('Diff size against time',
        p('One mark per pull request that has a recorded time. The dot is the time FastRSI took by the route in the Outcome column. The red mark is the restart time recorded for the same edit; a red bar is a recorded range. Both axes are logarithmic.'),
        h('div', { class: 'fr-card' },
          h('ul', { class: 'fr-keys', style: 'margin-bottom:10px' },
            h('li', null, h('span', { class: 'au-k-live' }), 'Installed live'),
            h('li', null, h('span', { class: 'au-k-tier' }), 'Went to a restart tier (forked or code restart)'),
            h('li', null, h('span', { class: 'au-k-rs' }), 'Restart time it replaces')),
          h('div', { class: 'au-chart', id: 'pr-chart' }),
          h('p', { class: 'fr-note', id: 'pr-chart-note' }))));
    drawChart(list);
  }

  let chartList = [];
  function drawChart(list) {
    if (list) chartList = list;
    const box = document.getElementById('pr-chart');
    if (!box || !window.d3 || !box.clientWidth) return;
    const pts = chartList.filter((r) => r.x.added != null && (r.x.t != null || r.x.restart != null));
    const missing = chartList.filter((r) => !pts.includes(r));
    document.getElementById('pr-chart-note').textContent = pts.length + ' of ' + chartList.length + ' pull requests plotted. ' + (missing.length ? 'No time recorded for ' + missing.map((r) => '#' + r.pr).join(', ') + '. ' : '') + 'A pull request with only a red mark was refused live and measured with a restart; one with only a dot has no restart time recorded.';
    const W = box.clientWidth, small = W < 520, H = small ? 320 : 380, m = { t: 14, r: small ? 46 : 58, b: 44, l: 46 };
    const x = d3.scaleLog().domain([6, 1500]).range([m.l, W - m.r]), y = d3.scaleLog().domain([0.05, 400]).range([H - m.b, m.t]);
    box.replaceChildren();
    const svg = d3.select(box).append('svg').attr('class', 'fr-svg').attr('viewBox', '0 0 ' + W + ' ' + H).attr('width', W).attr('height', H).attr('role', 'img')
      .attr('aria-label', 'Scatter chart: lines changed against seconds for ' + pts.length + ' pull requests. The table above holds the same values.');
    const xt = [10, 30, 100, 300, 1000], yt = [0.1, 1, 10, 100];
    yt.forEach((v) => { svg.append('line').attr('class', 'gl').attr('x1', m.l).attr('x2', W - m.r).attr('y1', y(v)).attr('y2', y(v)); svg.append('text').attr('x', m.l - 8).attr('y', y(v) + 4).attr('text-anchor', 'end').text(v + ' s'); });
    xt.forEach((v) => { svg.append('line').attr('class', 'gl').attr('y1', m.t).attr('y2', H - m.b).attr('x1', x(v)).attr('x2', x(v)); svg.append('text').attr('x', x(v)).attr('y', H - m.b + 16).attr('text-anchor', 'middle').text(v); });
    svg.append('text').attr('class', 'axt').attr('x', (m.l + W - m.r) / 2).attr('y', H - 6).attr('text-anchor', 'middle').text('Lines changed (added + removed)');
    let tip = null;
    const hide = () => { if (tip) { tip.remove(); tip = null; } };
    const show = (r, px, py) => {
      hide(); const d = r.x;
      tip = h('div', { class: 'fr-tip au-tipbox' },
        h('div', null, h('b', null, r.engine + ' #' + r.pr)), h('div', { class: 'tm' }, rich(d.gh_title)),
        h('div', null, h('span', { class: 'tv' }, '+' + d.added + ' \u2212' + d.removed), ' in ' + d.files + (d.files === 1 ? ' file' : ' files')),
        h('div', null, 'FastRSI: ', d.t == null ? 'no time recorded for the route taken' : [h('span', { class: 'tv' }, d.t + ' s'), ' \u00b7 ' + d.t_what]),
        h('div', null, 'Restart: ', d.restart == null ? NR : [h('span', { class: 'tv' }, (Array.isArray(d.restart) ? d.restart.join('\u2013') : d.restart) + ' s'), ' \u00b7 ' + d.restart_what]));
      box.append(tip);
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      tip.style.left = Math.max(0, Math.min(W - tw, px + 12)) + 'px';
      tip.style.top = (py - th - 10 < 0 ? py + 14 : py - th - 10) + 'px';
    };
    const sorted = pts.slice().sort((a, b) => (a.x.t == null) - (b.x.t == null));
    sorted.forEach((r) => {
      const d = r.x, px = x(d.added + d.removed), g = svg.append('g').attr('class', 'au-pt').attr('tabindex', 0).attr('role', 'img')
        .attr('aria-label', r.engine + ' #' + r.pr + ', ' + (d.added + d.removed) + ' lines changed, FastRSI ' + (d.t == null ? 'no time recorded' : d.t + ' seconds') + ', restart ' + (d.restart == null ? NR : (Array.isArray(d.restart) ? d.restart.join(' to ') : d.restart) + ' seconds'));
      const rs = d.restart == null ? null : Array.isArray(d.restart) ? d.restart : [d.restart, d.restart], rmid = rs && Math.sqrt(rs[0] * rs[1]);
      if (rs && d.t != null) g.append('line').attr('class', 'conn').attr('x1', px).attr('x2', px).attr('y1', y(d.t)).attr('y2', y(rmid));
      if (rs) { if (rs[0] === rs[1]) g.append('line').attr('class', 'rs').attr('x1', px - 7).attr('x2', px + 7).attr('y1', y(rs[0])).attr('y2', y(rs[0])); else g.append('line').attr('class', 'rs').attr('x1', px).attr('x2', px).attr('y1', y(rs[0])).attr('y2', y(rs[1])); }
      if (d.t != null) g.append('circle').attr('class', 'dot ' + (r.installed_live ? 'live' : 'tier')).attr('cx', px).attr('cy', y(d.t)).attr('r', 5.5);
      const ly = d.t != null ? y(d.t) : y(rmid);
      if (d.t != null) g.append('text').attr('class', 'lb halo').attr('x', px + 10).attr('y', ly + 3.5).text('#' + r.pr);
      else g.append('text').attr('class', 'lb halo').attr('x', px).attr('y', ly - 8).attr('text-anchor', 'middle').text('#' + r.pr);
      g.append('rect').attr('class', 'hit').attr('x', px - 12).attr('width', 24).attr('y', (rs ? Math.min(y(rs[1]), ly) : ly) - 10).attr('height', (rs ? Math.abs(ly - y(rs[1])) : 0) + 20);
      const at = () => show(r, px, Math.min(ly, rs ? y(rs[1]) : ly));
      g.on('pointerenter', at).on('pointerleave', hide).on('focus', at).on('blur', hide).on('click', (ev) => { ev.stopPropagation(); at(); });
    });
  }

  // ================================================================ Panel 1 & Panel 2: 6-Engine Marimekko + Simulator + 2-Level Icicle
  const UNIT_LABELS = {
    functions: 'Functions',
    lines: 'Lines of Python',
    lines_all: 'Python + Native C++/CUDA lines'
  };

  function buildPanel1And2(prefixId, engineFilterFn) {
    const enginesShown = F.survey.filter((e) => !engineFilterFn || engineFilterFn(e.engine));
    if (!enginesShown.some((e) => e.engine === atlas.focusEngine) && enginesShown.length) {
      atlas.focusEngine = enginesShown[0].engine;
    }

    const unitButtons = h('div', { class: 'fr-picker', style: 'margin:0', role: 'group', 'aria-label': 'Unit' },
      Object.keys(UNIT_LABELS).map((uk) => h('button', {
        type: 'button',
        class: 'fr-chip',
        'aria-pressed': String(atlas.unit === uk),
        onclick: () => { atlas.unit = uk; refreshAllPanels(true); }
      }, UNIT_LABELS[uk]))
    );

    const scaleButtons = h('div', { class: 'fr-picker', style: 'margin:0', role: 'group', 'aria-label': 'Column width' },
      [['sqrt', 'Sqrt width'], ['linear', 'Linear width'], ['equal', 'Equal width']].map(([sk, sl]) => h('button', {
        type: 'button',
        class: 'fr-chip',
        'aria-pressed': String(atlas.widthMode === sk),
        onclick: () => { atlas.widthMode = sk; refreshAllPanels(true); }
      }, sl))
    );

    const presetButtons = h('div', { style: 'display:flex;gap:6px;flex-wrap:wrap' },
      h('button', {
        type: 'button',
        class: 'fr-btn fr-btn-quiet',
        onclick: () => {
          Object.keys(atlas.sim).forEach((k) => { atlas.sim[k] = false; });
          refreshAllPanels(true);
        }
      }, 'PR #151 baseline'),
      h('button', {
        type: 'button',
        class: 'fr-btn fr-btn-quiet',
        onclick: () => {
          atlas.sim.reinit = true; atlas.sim.jit = true;
          atlas.sim.frontend = false; atlas.sim.core_rebuild = false; atlas.sim.native_swap = false;
          refreshAllPanels(true);
        }
      }, 'Default (#156 + JIT)'),
      h('button', {
        type: 'button',
        class: 'fr-btn fr-btn-quiet',
        onclick: () => {
          Object.keys(atlas.sim).forEach((k) => { atlas.sim[k] = true; });
          refreshAllPanels(true);
        }
      }, 'All 5 capabilities')
    );

    const simBoxes = h('div', { class: 'atlas-checks', role: 'group', 'aria-label': 'What-if capabilities' },
      SIM_TOGGLES.map((t) => {
        const checked = atlas.sim[t.key];
        const inp = h('input', {
          type: 'checkbox',
          checked: checked ? 'checked' : null,
          onchange: (ev) => {
            atlas.sim[t.key] = ev.target.checked;
            refreshAllPanels(true);
          }
        });
        return h('label', { class: 'atlas-chk' + (checked ? ' is-on' : '') }, inp, h('span', null, t.label));
      })
    );

    const focusedMeta = F.survey.find((e) => e.engine === atlas.focusEngine) || F.survey[0];

    const panel1Card = h('div', { class: 'fr-card au-card' },
      h('div', { class: 'au-head' },
        h('h3', { class: 'au-h3' }, 'Panel 1 \u00b7 Six-engine coverage Marimekko & capability simulator'),
        badge('Survey ruleset 1.1', 'good'),
        badge('Static source-tree share, not workload trace', 'warn')
      ),
      p('Column width scales with engine size; vertical height splits every function or code line into its install path. Toggle capabilities below to recompute live shares across all six engines without approximation. Diagonal hatching marks classifications resting on declared paths, names, or defaults rather than syntax.'),
      h('div', { class: 'atlas-bar' },
        h('div', { class: 'atlas-row' }, unitButtons, scaleButtons, presetButtons),
        simBoxes
      ),
      h('ul', { class: 'fr-keys' },
        MEKKO_SEGS.filter((sg) => atlas.unit === 'lines_all' || (sg.key !== 'native_live' && sg.key !== 'native_restart')).map((sg) =>
          h('li', null, h('span', { class: 'fr-k ' + sg.cls }), sg.label)
        ),
        h('li', null, h('span', { class: 'sv-solid-key' }), 'AST-pinned (decorator/init)'),
        h('li', null, h('span', { class: 'sv-hatch-key' }), 'Rule-scoped (path reach)')
      ),
      h('div', { class: 'mekko-box', id: prefixId + '-mekko', 'data-filter': engineFilterFn ? 'cov' : 'all' }),
      h('p', { class: 'fr-note' }, 'Click any engine column to inspect its two-level subsystem hierarchy in Panel 2 and sync its runtime process topology in Panel 3. Engines on main: vLLM, Megatron-LM, FastVideo; adapter branches in progress: SGLang, verl, miles.')
    );

    const engChips = h('div', { class: 'fr-picker', style: 'margin:0' },
      enginesShown.map((e) => h('button', {
        type: 'button',
        class: 'fr-chip',
        'aria-pressed': String(atlas.focusEngine === e.engine),
        onclick: () => {
          atlas.focusEngine = e.engine;
          atlas.focusSubsystem = null;
          atlas.selectedCell.engine = e.engine;
          refreshAllPanels(false);
        }
      }, e.engine + ' \u00b7 ' + e.subsystems_total + ' subsystems'))
    );

    const panel2Card = h('div', { class: 'fr-card au-card' },
      h('div', { class: 'au-head', style: 'justify-content:space-between;align-items:center' },
        h('div', { class: 'au-head' },
          h('h3', { class: 'au-h3' }, 'Panel 2 \u00b7 Two-level subsystem icicle drill-down: ' + focusedMeta.engine),
          badge(focusedMeta.on_main ? 'On main' : 'Adapter branch', focusedMeta.on_main ? 'good' : 'warn')
        ),
        h('a', {
          href: location.hash === '#atlas' ? '#' + prefixId + '-topo-sec' : '#boundaries',
          class: 'fr-btn fr-btn-quiet',
          onclick: (ev) => {
            atlas.selectedCell.engine = focusedMeta.engine;
            if (location.hash === '#atlas') {
              ev.preventDefault();
              const el = document.getElementById(prefixId + '-topo-sec');
              if (el) el.scrollIntoView({ behavior: 'smooth' });
            }
          }
        }, 'Inspect ' + focusedMeta.engine + ' process tree & risks \u2192')
      ),
      engChips,
      p('Left-to-right hierarchy: whole engine \u2192 top-level package \u2192 two-level subsystem directory (' + focusedMeta.repo + ', ' + focusedMeta.version + '). Bar fill reflects each subsystem\u2019s live share under the active capability toggles above; click any bar to pin its blocker diagnosis and unlock recipe.'),
      h('div', { class: 'icicle-grid' },
        h('div', { class: 'icicle-box', id: prefixId + '-icicle' }),
        h('div', { class: 'sub-insp', id: prefixId + '-sub-insp' })
      )
    );

    return h('section', { class: 'au-sec', id: prefixId + '-sec' }, panel1Card, panel2Card);
  }

  function drawMarimekko(boxId, animate) {
    const box = document.getElementById(boxId);
    if (!box || !window.d3 || !box.clientWidth) return;
    const isCovFiltered = box.dataset.filter === 'cov' && covState.engine !== 'All';
    const list = F.survey.filter((e) => !isCovFiltered || e.engine === covState.engine);
    if (!list.length) return;

    const W = box.clientWidth;
    const H = W < 620 ? 330 : 370;
    const pad = { t: 48, r: 14, b: 36, l: 42 };
    const innerW = W - pad.l - pad.r;
    const innerH = H - pad.t - pad.b;
    const gap = Math.min(10, Math.max(4, Math.floor(innerW * 0.012)));

    const computed = list.map((e) => {
      const cur = computeSlice(e, e, atlas.unit, atlas.sim);
      const base = computeSlice(e, e, atlas.unit, BASELINE_SIM);
      const rawWeight = atlas.widthMode === 'equal' ? 1 : atlas.widthMode === 'sqrt' ? Math.sqrt(Math.max(1, cur.total)) : Math.max(1, cur.total);
      return { meta: e, cur, base, weight: rawWeight };
    });

    const sumW = computed.reduce((acc, d) => acc + d.weight, 0);
    const availW = Math.max(120, innerW - gap * (computed.length - 1));
    let cursorX = pad.l;
    computed.forEach((d) => {
      d.w = Math.max(38, (d.weight / sumW) * availW);
      d.x = cursorX;
      cursorX += d.w + gap;
    });
    // Normalize exact fit to innerW
    const scaleFix = innerW / Math.max(1, cursorX - gap - pad.l);
    let normX = pad.l;
    computed.forEach((d) => {
      d.w = d.w * scaleFix;
      d.x = normX;
      normX += d.w + gap * scaleFix;
    });

    let svg = d3.select(box).select('svg.mekko-svg');
    const patId = boxId + '-hatch';
    if (svg.empty() || +svg.attr('data-cols') !== computed.length || Math.abs(+svg.attr('width') - W) > 4) {
      box.replaceChildren();
      svg = d3.select(box).append('svg')
        .attr('class', 'fr-svg mekko-svg')
        .attr('data-cols', computed.length)
        .attr('viewBox', '0 0 ' + W + ' ' + H)
        .attr('width', W)
        .attr('height', H)
        .attr('role', 'img')
        .attr('aria-label', 'Six-engine Marimekko coverage chart');

      const defs = svg.append('defs');
      const pat = defs.append('pattern')
        .attr('id', patId)
        .attr('patternUnits', 'userSpaceOnUse')
        .attr('width', 7)
        .attr('height', 7)
        .attr('patternTransform', 'rotate(45)');
      pat.append('line')
        .attr('x1', 0).attr('y1', 0).attr('x2', 0).attr('y2', 7)
        .attr('stroke', 'var(--color-bg)')
        .attr('stroke-width', 2.2)
        .attr('stroke-opacity', 0.34);

      const yScale = d3.scaleLinear().domain([0, 100]).range([pad.t + innerH, pad.t]);
      [0, 25, 50, 75, 100].forEach((yv) => {
        svg.append('line').attr('class', 'gl').attr('x1', pad.l).attr('x2', W - pad.r).attr('y1', yScale(yv)).attr('y2', yScale(yv));
        svg.append('text').attr('x', pad.l - 6).attr('y', yScale(yv) + 4).attr('text-anchor', 'end').text(yv + '%');
      });
      svg.append('g').attr('class', 'cols-layer');
    }

    let tip = box.querySelector('.au-tipbox');
    const hideTip = () => { if (tip) { tip.remove(); tip = null; } };
    const showTip = (d, segSpec, segVal, ev) => {
      hideTip();
      const [ptrX, ptrY] = d3.pointer(ev, box);
      const uncCount = Math.max(0, segVal.v - segVal.firm);
      tip = h('div', { class: 'fr-tip au-tipbox' },
        h('div', null, h('b', null, d.meta.engine), ' \u00b7 ', segSpec.label),
        h('div', null, h('span', { class: 'tv' }, num(segVal.v)), ' of ', num(d.cur.total), ' (' + pct(segVal.v, d.cur.total) + ')'),
        h('div', { class: 'tm' }, 'AST-pinned: ' + num(segVal.firm) + ' \u00b7 Path-scoped: ' + num(uncCount)),
        h('div', { class: 'tm' }, 'Live total: ' + pct(d.cur.live, d.cur.total) + ' (PR #151 base: ' + pct(d.base.live, d.base.total) + ')')
      );
      box.append(tip);
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      tip.style.left = Math.max(4, Math.min(W - tw - 4, ptrX + 12)) + 'px';
      tip.style.top = Math.max(4, (ptrY - th - 10 < 0 ? ptrY + 14 : ptrY - th - 10)) + 'px';
    };

    const dur = animate ? 360 : 0;
    const colsLayer = svg.select('g.cols-layer');
    const colJoin = colsLayer.selectAll('g.mekko-col').data(computed, (d) => d.meta.key);
    colJoin.exit().remove();

    const colEnter = colJoin.enter().append('g').attr('class', 'mekko-col');
    colEnter.append('text').attr('class', 'mekko-head').attr('text-anchor', 'middle');
    colEnter.append('text').attr('class', 'mekko-pct').attr('text-anchor', 'middle');
    colEnter.append('text').attr('class', 'mekko-gain').attr('text-anchor', 'middle');
    colEnter.append('g').attr('class', 'segs');
    colEnter.append('rect').attr('class', 'col-frame');
    colEnter.append('text').attr('class', 'mekko-sub').attr('text-anchor', 'middle');

    const colAll = colEnter.merge(colJoin);
    colAll.classed('is-active', (d) => d.meta.engine === atlas.focusEngine)
      .on('click', (ev, d) => {
        ev.stopPropagation();
        atlas.focusEngine = d.meta.engine;
        atlas.focusSubsystem = null;
        atlas.selectedCell.engine = d.meta.engine;
        refreshAllPanels(false);
      });

    colAll.each(function (d) {
      const colG = d3.select(this);
      const cx = d.x + d.w / 2;
      const curPct = d.cur.total ? (100 * d.cur.live / d.cur.total) : 0;
      const basePct = d.base.total ? (100 * d.base.live / d.base.total) : 0;
      const deltaPct = curPct - basePct;

      colG.select('text.mekko-head')
        .attr('x', cx).attr('y', pad.t - 28)
        .text(d.w < 68 && d.meta.engine === 'Megatron-LM' ? 'Megatron' : d.meta.engine);

      colG.select('text.mekko-pct')
        .attr('x', cx).attr('y', pad.t - 14)
        .text(curPct.toFixed(1) + '% live');

      colG.select('text.mekko-gain')
        .attr('x', cx).attr('y', pad.t - 3)
        .text(Math.abs(deltaPct) >= 0.05 ? (deltaPct > 0 ? '+' : '') + deltaPct.toFixed(1) + ' pts' : '');

      colG.select('rect.col-frame')
        .attr('x', d.x - 1).attr('y', pad.t - 1)
        .attr('width', d.w + 2).attr('height', innerH + 2);

      const shortCount = d.cur.total >= 1000000 ? (d.cur.total / 1000000).toFixed(2) + 'M' : d.cur.total >= 1000 ? (d.cur.total / 1000).toFixed(1) + 'k' : String(d.cur.total);
      colG.select('text.mekko-sub')
        .attr('x', cx).attr('y', pad.t + innerH + 16)
        .text(shortCount + (d.w >= 86 ? ' (' + pct(d.cur.uncertain, d.cur.total) + ' path)' : ''));

      // Stack segments from top (live classes first, then respawn, restart, uncovered at bottom)
      let yCursor = pad.t;
      const stackData = MEKKO_SEGS.map((spec) => {
        const sv = d.cur.segs[spec.key];
        const hPx = d.cur.total > 0 ? (sv.v / d.cur.total) * innerH : 0;
        const uncRatio = sv.v > 0 ? Math.max(0, sv.v - sv.firm) / sv.v : 0;
        const item = { spec, sv, x: d.x, w: d.w, y: yCursor, h: hPx, uncH: hPx * uncRatio, parent: d };
        yCursor += hPx;
        return item;
      });

      const segJoin = colG.select('g.segs').selectAll('g.m-seg').data(stackData, (item) => item.spec.key);
      const segEnter = segJoin.enter().append('g').attr('class', 'm-seg');
      segEnter.append('rect').attr('class', 'base-fill').attr('x', d.x).attr('width', d.w).attr('y', pad.t + innerH).attr('height', 0);
      segEnter.append('rect').attr('class', 'hatch-fill').attr('fill', 'url(#' + patId + ')').attr('pointer-events', 'none').attr('x', d.x).attr('width', d.w).attr('y', pad.t + innerH).attr('height', 0);
      segEnter.append('text').attr('class', (item) => 'mekko-seg-lbl' + (item.spec.inkLabel ? ' ink' : '')).attr('text-anchor', 'middle');

      const segAll = segEnter.merge(segJoin);
      segAll.on('pointermove', (ev, item) => { if (item.sv.v > 0) showTip(item.parent, item.spec, item.sv, ev); })
        .on('pointerleave', hideTip);

      segAll.select('rect.base-fill')
        .attr('fill', (item) => item.spec.cssVar)
        .transition().duration(dur)
        .attr('x', (item) => item.x)
        .attr('width', (item) => item.w)
        .attr('y', (item) => item.y)
        .attr('height', (item) => Math.max(0, item.h - (item.h > 1.5 ? 0.6 : 0)));

      segAll.select('rect.hatch-fill')
        .transition().duration(dur)
        .attr('x', (item) => item.x)
        .attr('width', (item) => item.w)
        .attr('y', (item) => item.y + (item.h - item.uncH))
        .attr('height', (item) => (item.uncH > 1 ? item.uncH : 0));

      segAll.select('text.mekko-seg-lbl')
        .transition().duration(dur)
        .attr('x', (item) => item.x + item.w / 2)
        .attr('y', (item) => item.y + item.h / 2 + 3.5)
        .text((item) => (item.h >= 18 && item.w >= 54 ? pct(item.sv.v, item.parent.cur.total) : ''));
    });
  }

  function drawIcicle(icicleId, inspId) {
    const box = document.getElementById(icicleId);
    const insp = document.getElementById(inspId);
    if (!box || !insp || !window.d3 || !box.clientWidth) return;

    const engMeta = F.survey.find((e) => e.engine === atlas.focusEngine) || F.survey[0];
    const subs = engMeta.subsystems.map((s) => {
      const cur = computeSlice(s, engMeta, atlas.unit, atlas.sim);
      return { raw: s, cur, name: s.name, parent: s.parent };
    }).filter((s) => s.cur.total > 0);

    // Group into Level 1 parents -> Level 2 subsystems
    const groupsMap = new Map();
    subs.forEach((s) => {
      if (!groupsMap.has(s.parent)) groupsMap.set(s.parent, { name: s.parent, total: 0, live: 0, kids: [] });
      const grp = groupsMap.get(s.parent);
      grp.total += s.cur.total;
      grp.live += s.cur.live;
      grp.kids.push(s);
    });
    const groupsArr = Array.from(groupsMap.values()).sort((a, b) => b.total - a.total);
    groupsArr.forEach((g) => g.kids.sort((a, b) => b.cur.total - a.cur.total));

    const wholeCur = computeSlice(engMeta, engMeta, atlas.unit, atlas.sim);
    const W = box.clientWidth;
    const H = 350;
    const colX = [0, Math.round(W * 0.16), Math.round(W * 0.50), W];
    box.replaceChildren();

    const svg = d3.select(box).append('svg')
      .attr('class', 'fr-svg')
      .attr('viewBox', '0 0 ' + W + ' ' + H)
      .attr('width', W)
      .attr('height', H)
      .attr('role', 'img')
      .attr('aria-label', engMeta.engine + ' two-level subsystem icicle chart');

    const liveColor = (liveAmt, totAmt) => {
      const ratio = totAmt > 0 ? liveAmt / totAmt : 0;
      if (ratio >= 0.72) return 'color-mix(in oklab, var(--good) 72%, var(--color-panel))';
      if (ratio >= 0.42) return 'color-mix(in oklab, var(--fr-fast) 68%, var(--color-panel))';
      if (ratio >= 0.15) return 'color-mix(in oklab, var(--warn) 65%, var(--color-panel))';
      return 'color-mix(in oklab, var(--bad) 58%, var(--color-panel))';
    };

    // Column 0: Engine root
    const rootG = svg.append('g').attr('class', 'icicle-cell' + (!atlas.focusSubsystem ? ' is-active' : ''))
      .on('click', () => { atlas.focusSubsystem = null; drawIcicle(icicleId, inspId); });
    rootG.append('rect').attr('x', colX[0]).attr('y', 0).attr('width', colX[1] - colX[0] - 2).attr('height', H)
      .attr('rx', 4).attr('fill', liveColor(wholeCur.live, wholeCur.total));
    rootG.append('text').attr('class', 'icicle-lbl halo').attr('x', colX[0] + 6).attr('y', 18).text(engMeta.engine);
    rootG.append('text').attr('class', 'icicle-lbl halo').attr('x', colX[0] + 6).attr('y', 33).text(pct(wholeCur.live, wholeCur.total) + ' live');

    // Column 1 & Column 2
    const sumTotal = groupsArr.reduce((acc, g) => acc + g.total, 0) || 1;
    let y1 = 0;
    groupsArr.forEach((grp) => {
      const h1 = (grp.total / sumTotal) * H;
      const gNode = svg.append('g').attr('class', 'icicle-cell')
        .on('click', () => {
          if (grp.kids.length) {
            atlas.focusSubsystem = grp.kids[0].name;
            drawIcicle(icicleId, inspId);
          }
        });
      gNode.append('rect')
        .attr('x', colX[1]).attr('y', y1)
        .attr('width', colX[2] - colX[1] - 2).attr('height', Math.max(1, h1))
        .attr('fill', liveColor(grp.live, grp.total));
      if (h1 >= 15) {
        const shortGrp = grp.name.replace(/^python\//, '');
        gNode.append('text').attr('class', 'icicle-lbl halo')
          .attr('x', colX[1] + 5).attr('y', y1 + Math.min(14, h1 / 2 + 4))
          .text(shortGrp + ' (' + pct(grp.live, grp.total) + ')');
      }

      let y2 = y1;
      grp.kids.forEach((subNode) => {
        const h2 = (subNode.cur.total / sumTotal) * H;
        const active = atlas.focusSubsystem === subNode.name;
        const sNode = svg.append('g').attr('class', 'icicle-cell' + (active ? ' is-active' : ''))
          .on('click', () => {
            atlas.focusSubsystem = subNode.name;
            drawIcicle(icicleId, inspId);
          });
        sNode.append('rect')
          .attr('x', colX[2]).attr('y', y2)
          .attr('width', colX[3] - colX[2]).attr('height', Math.max(1, h2))
          .attr('fill', liveColor(subNode.cur.live, subNode.cur.total));
        if (h2 >= 14) {
          const leafLbl = subNode.name.startsWith(grp.name + '/') ? subNode.name.slice(grp.name.length + 1) : subNode.name;
          sNode.append('text').attr('class', 'icicle-lbl halo')
            .attr('x', colX[2] + 6).attr('y', y2 + Math.min(13, h2 / 2 + 3.5))
            .text(leafLbl + ' \u00b7 ' + pct(subNode.cur.live, subNode.cur.total));
        }
        y2 += h2;
      });
      y1 += h1;
    });

    // Render Inspector Card on the right
    const uncovScore = (s) => s.cur.uncovered + (atlas.unit === 'lines_all' && !atlas.sim.native_swap ? s.cur.nativeLines : 0);
    const picked = subs.find((s) => s.name === atlas.focusSubsystem) ||
      subs.slice().sort((a, b) => uncovScore(b) - uncovScore(a))[0] ||
      subs[0];

    if (!picked) {
      insp.replaceChildren(p('No subsystem records available.'));
      return;
    }
    const diag = diagnoseSubsystem(picked.raw, engMeta);
    const miniSegs = MEKKO_SEGS.filter((sg) => picked.cur.segs[sg.key].v > 0);
    const miniBar = h('div', { class: 'sv-bar big' },
      miniSegs.map((sg) => {
        const v = picked.cur.segs[sg.key].v;
        return h('span', {
          class: 'sv-s ' + sg.cls + (sg.inkLabel ? ' ink' : ''),
          style: 'flex:' + v + ' 0 0',
          title: sg.label + ': ' + num(v) + ' (' + pct(v, picked.cur.total) + ')'
        }, v / picked.cur.total >= 0.14 ? pct(v, picked.cur.total) : null);
      })
    );

    insp.replaceChildren(
      h('div', { class: 'au-head', style: 'justify-content:space-between' },
        h('span', { class: 'au-src' }, atlas.focusSubsystem ? 'Selected subsystem' : 'Largest uncovered subsystem'),
        badge(pct(picked.cur.live, picked.cur.total) + ' live', picked.cur.live / Math.max(1, picked.cur.total) >= 0.65 ? 'good' : 'warn')
      ),
      h('h4', null, picked.name),
      miniBar,
      kv([
        ['Python functions', picked.raw.functions.total],
        ['Python non-blank lines', picked.raw.lines.total],
        ['Native C++/CUDA files / lines', num(picked.raw.native.files) + ' / ' + num(picked.raw.native.lines)],
        ['Live covered (' + UNIT_LABELS[atlas.unit] + ')', num(picked.cur.live) + ' (' + pct(picked.cur.live, picked.cur.total) + ')'],
        ['Warm respawn / Cold restart', num(picked.cur.respawn) + ' / ' + num(picked.cur.restart)],
        ['Uncovered Python', num(picked.cur.uncovered)]
      ]),
      h('div', { class: 'sub-recipe' },
        h('div', null, h('b', null, 'Why blocked: '), diag.why),
        h('div', null, h('b', null, 'Unlock recipe: '), diag.unlock)
      )
    );
  }

  // ================================================================ Panel 3 & Panel 4: Process-Tree Map + 8x6 Interface Risk Heatmap & Drawer
  const BLAST_LABELS = {
    all: 'All blast radii',
    silent_corruption: 'Silent corruption',
    rank_hang: 'Rank hang / deadlock',
    orphan_leak: 'Orphan process / leak',
    cold_fallback: 'Cold fallback',
    none: 'Not applicable'
  };

  const BLAST_BADGE = {
    silent_corruption: ['Silent corruption', 'bad'],
    rank_hang: ['Rank hang', 'warn'],
    orphan_leak: ['Orphan / leak', 'warn'],
    cold_fallback: ['Cold fallback', 'mute'],
    none: ['N/A', 'mute']
  };

  function cellMapKey(boundary, engine) { return boundary + '::' + engine; }
  const riskCellIndex = {};
  F.risks.cells.forEach((c) => { riskCellIndex[cellMapKey(c.boundary, c.engine)] = c; });

  function cellColorClass(c) {
    if (c.risk_level === 'high') return 'rk-high';
    if (c.risk_level === 'medium') return 'rk-medium';
    if (c.status === 'gpu_verified') return 'rk-good';
    return 'rk-low';
  }

  function cellSummaryLabel(c) {
    if (c.status === 'not_applicable') return 'N/A';
    if (c.status === 'unbuilt') return 'Unbuilt';
    if (c.risk_level === 'high') return 'High risk';
    if (c.risk_level === 'medium') return 'Med risk';
    if (c.status === 'gpu_verified') return 'Fail-closed';
    return 'Verified';
  }

  function buildPanel3And4(prefixId) {
    const R = F.risks;
    const curEng = R.engines.find((e) => e.engine === atlas.selectedCell.engine) || R.engines[0];

    const topoChips = h('div', { class: 'fr-picker', style: 'margin:0' },
      R.engines.map((e) => h('button', {
        type: 'button',
        class: 'fr-chip',
        'aria-pressed': String(curEng.engine === e.engine),
        onclick: () => {
          atlas.selectedCell.engine = e.engine;
          atlas.focusEngine = e.engine;
          refreshAllPanels(false);
        }
      }, e.engine + ' \u00b7 ' + e.kind))
    );

    const clearFilterBtn = atlas.focusBoundary ? h('button', {
      type: 'button',
      class: 'fr-btn fr-btn-quiet',
      onclick: () => { atlas.focusBoundary = null; refreshAllPanels(false); }
    }, 'Clear boundary highlight (' + atlas.focusBoundary + ') \u00d7') : null;

    const panel3Card = h('div', { class: 'fr-card au-card', id: prefixId + '-topo-sec' },
      h('div', { class: 'au-head', style: 'justify-content:space-between;align-items:center' },
        h('div', { class: 'au-head' },
          h('h3', { class: 'au-h3' }, 'Panel 3 \u00b7 Interactive process-tree & blast-radius topology: ' + curEng.engine),
          badge(curEng.status_label, curEng.key === 'vllm' || curEng.key === 'megatron' || curEng.key === 'fastvideo' ? 'good' : 'warn')
        ),
        clearFilterBtn
      ),
      topoChips,
      p(curEng.topology.subtitle + '. Solid boxes have a kernel participant; dashed boxes are unreached processes. Click any labeled boundary edge to highlight its downstream restart propagation chain and inspect that boundary cell in Panel 4.'),
      h('div', { class: 'topo-box', id: prefixId + '-topo' }),
      h('ul', { class: 'fr-keys' },
        h('li', null, h('span', { class: 'fr-k', style: 'background:var(--bad)' }), 'High residual risk edge / invalidated chain'),
        h('li', null, h('span', { class: 'fr-k', style: 'background:var(--warn)' }), 'Medium residual risk edge'),
        h('li', null, h('span', { class: 'fr-k', style: 'background:var(--good)' }), 'Low / fail-closed edge'),
        h('li', null, h('span', { class: 'fr-k-dash' }), 'Unreached process (no participant)')
      )
    );

    const blastChips = h('div', { class: 'fr-picker', style: 'margin:0' },
      ['all', 'silent_corruption', 'rank_hang', 'orphan_leak', 'cold_fallback'].map((bk) => h('button', {
        type: 'button',
        class: 'fr-chip',
        'aria-pressed': String(atlas.blastFilter === bk),
        onclick: () => { atlas.blastFilter = bk; refreshAllPanels(false); }
      }, BLAST_LABELS[bk]))
    );

    const hmHeader = h('tr', null,
      h('th', null, 'Boundary (8)'),
      R.engines.map((e) => h('th', { class: 'eng-col' }, e.engine))
    );

    const hmRows = R.boundaries.map((b) => {
      const rowDim = atlas.focusBoundary && atlas.focusBoundary !== b.id;
      return h('tr', { class: 'hm-row' + (rowDim ? ' is-filtered' : '') },
        h('td', { class: 'hm-rowhead' },
          h('code', null, b.code + ' ' + b.name),
          h('span', { class: 'sub' }, b.title)
        ),
        R.engines.map((e) => {
          const c = riskCellIndex[cellMapKey(b.id, e.engine)];
          const isSelected = atlas.selectedCell.boundary === b.id && atlas.selectedCell.engine === e.engine;
          const matchesBlast = atlas.blastFilter === 'all' || c.primary_blast_radius === atlas.blastFilter ||
            c.residual_risks.some((rr) => rr.blast_radius === atlas.blastFilter) ||
            c.closed_bugs.some((cbItem) => cbItem.blast_radius === atlas.blastFilter);
          return h('td', {
            class: 'hm-cell ' + cellColorClass(c) + (isSelected ? ' is-selected' : '') + (!matchesBlast ? ' is-dim' : ''),
            tabindex: '0',
            role: 'button',
            'aria-label': b.code + ' ' + b.name + ' on ' + e.engine + ': ' + cellSummaryLabel(c) + ', tier ' + c.verification_tier,
            onclick: () => {
              atlas.selectedCell = { boundary: b.id, engine: e.engine };
              atlas.focusBoundary = b.id;
              atlas.focusEngine = e.engine;
              refreshAllPanels(false);
            },
            onkeydown: (ev) => {
              if (ev.key === 'Enter' || ev.key === ' ') {
                ev.preventDefault();
                atlas.selectedCell = { boundary: b.id, engine: e.engine };
                atlas.focusBoundary = b.id;
                atlas.focusEngine = e.engine;
                refreshAllPanels(false);
              }
            }
          },
            h('div', { class: 'hm-top' },
              h('span', { class: 'hm-sev' }, cellSummaryLabel(c)),
              h('span', { class: 'hm-tier', title: 'Verification tier' }, c.verification_tier)
            ),
            h('div', { class: 'hm-meta' },
              c.closed_bugs.length + ' closed \u00b7 ' + c.residual_risks.length + ' open'
            )
          );
        })
      );
    });

    const selected = riskCellIndex[cellMapKey(atlas.selectedCell.boundary, atlas.selectedCell.engine)] || R.cells[0];
    const selectedBound = R.boundaries.find((b) => b.id === selected.boundary) || R.boundaries[0];
    const blastInfo = BLAST_BADGE[selected.primary_blast_radius] || [selected.primary_blast_radius, 'mute'];

    const drawer = h('div', { class: 'drawer', id: prefixId + '-drawer' },
      h('div', { class: 'drawer-head' },
        h('div', { class: 'au-head' },
          h('h4', { class: 'drawer-title' }, selectedBound.code + ' ' + selectedBound.name + ' \u00d7 ' + selected.engine),
          badge(cellSummaryLabel(selected), selected.risk_level === 'high' ? 'bad' : selected.risk_level === 'medium' ? 'warn' : 'good'),
          badge('Blast: ' + blastInfo[0], blastInfo[1]),
          badge('Tier: ' + selected.verification_tier, selected.verification_tier === 'GPU' ? 'good' : selected.verification_tier === 'CPU' ? 'warn' : 'mute')
        ),
        h('span', { class: 'au-src' }, selectedBound.files)
      ),
      h('div', null,
        h('h5', { class: 'au-h3', style: 'margin-bottom:6px' }, 'Contract invariants'),
        h('ul', { class: 'au-list' }, selected.invariants.map((inv) => h('li', null, rich(inv))))
      ),
      h('div', { class: 'drawer-cols' },
        h('div', { style: 'display:grid;gap:8px;align-content:start' },
          h('h5', { class: 'au-h3' }, 'Audit & conformance bugs closed (' + selected.closed_bugs.length + ')'),
          selected.closed_bugs.length ? selected.closed_bugs.map((bug) => {
            const bb = BLAST_BADGE[bug.blast_radius] || [bug.blast_radius, 'mute'];
            return h('div', { class: 'bug-item is-closed' },
              h('div', { class: 'au-head', style: 'justify-content:space-between' },
                h('b', null, bug.id + ' \u00b7 ' + bug.fixed_in_pr),
                h('div', { style: 'display:flex;gap:4px' }, badge(bb[0], bb[1]), badge(bug.verification, 'good'))
              ),
              h('div', null, rich(bug.summary))
            );
          }) : p('No historical bug entries recorded for this cell.')
        ),
        h('div', { style: 'display:grid;gap:8px;align-content:start' },
          h('h5', { class: 'au-h3' }, 'Residual / post-merge risks (' + selected.residual_risks.length + ')'),
          selected.residual_risks.length ? selected.residual_risks.map((rr) => {
            const rb = BLAST_BADGE[rr.blast_radius] || [rr.blast_radius, 'mute'];
            return h('div', { class: 'bug-item is-open sev-' + rr.severity },
              h('div', { class: 'au-head', style: 'justify-content:space-between' },
                h('b', null, rr.id + ' (' + rr.severity + ')'),
                badge(rb[0], rb[1])
              ),
              h('div', null, rich(rr.summary)),
              h('div', { style: 'color:var(--fr-muted)' }, h('b', { style: 'color:var(--fr-ink)' }, 'Mitigation: '), rich(rr.mitigation)),
              h('div', { class: 'bug-cmd' }, rr.reproducer)
            );
          }) : p('Not applicable: Ray spawns isolated worker processes without parent-child CUDA fork.')
        )
      )
    );

    const panel4Card = h('div', { class: 'fr-card au-card' },
      h('div', { class: 'au-head' },
        h('h3', { class: 'au-h3' }, 'Panel 4 \u00b7 8 \u00d7 6 interface risk heatmap & failure-mode inspector'),
        badge('48 boundary cells', 'mute')
      ),
      p('Cell color shows residual risk severity after open pull requests #144\u2013#156; the corner badge shows the verification tier (GPU, CPU, Toy, or -). Click any cell to inspect its contract invariants, closed audit findings, and open failure modes below.'),
      blastChips,
      h('div', { class: 'fr-tablewrap' }, h('table', { class: 'hm-grid' }, h('thead', null, hmHeader), h('tbody', null, hmRows))),
      drawer,
      src(R.source)
    );

    return h('section', { class: 'au-sec' }, panel3Card, panel4Card);
  }

  function drawTopology(topoBoxId) {
    const box = document.getElementById(topoBoxId);
    if (!box || !window.d3 || !box.clientWidth) return;
    const R = F.risks;
    const eng = R.engines.find((e) => e.engine === atlas.selectedCell.engine) || R.engines[0];
    const topo = eng.topology;

    const W = Math.max(880, box.clientWidth);
    const H = 268;
    const nodeW = 190, nodeH = 62;
    const colGap = (W - 24 - nodeW * 4) / 3;
    const rowY = [48, 178];

    const pos = {};
    topo.nodes.forEach((nd) => {
      pos[nd.id] = {
        x: 12 + nd.col * (nodeW + colGap),
        y: rowY[nd.row] || 48,
        cx: 12 + nd.col * (nodeW + colGap) + nodeW / 2,
        cy: (rowY[nd.row] || 48) + nodeH / 2,
        nd
      };
    });

    const activeBnd = atlas.focusBoundary;
    const activeEdges = topo.edges.filter((ed) => ed.boundary === activeBnd);
    const chainNodes = new Set();
    activeEdges.forEach((ed) => (ed.chain || []).forEach((nid) => chainNodes.add(nid)));

    box.replaceChildren();
    const svg = d3.select(box).append('svg')
      .attr('class', 'fr-svg')
      .attr('viewBox', '0 0 ' + W + ' ' + H)
      .attr('width', W)
      .attr('height', H)
      .attr('role', 'img')
      .attr('aria-label', eng.engine + ' process topology and restart propagation map');

    const defs = svg.append('defs');
    defs.append('clipPath').attr('id', topoBoxId + '-nclip')
      .append('rect').attr('x', 0).attr('y', 0).attr('width', nodeW - 8).attr('height', nodeH);
    [['arr-muted', 'var(--fr-muted)'], ['arr-high', 'var(--bad)'], ['arr-medium', 'var(--warn)'], ['arr-low', 'var(--good)']].forEach(([id, col]) => {
      defs.append('marker')
        .attr('id', topoBoxId + '-' + id)
        .attr('viewBox', '0 -4 8 8').attr('refX', 7).attr('refY', 0)
        .attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
        .append('path').attr('d', 'M0,-3.5L8,0L0,3.5').attr('fill', col);
    });

    // Draw edges first
    topo.edges.forEach((ed, idx) => {
      const a = pos[ed.from], b = pos[ed.to];
      if (!a || !b) return;
      const cellInfo = riskCellIndex[cellMapKey(ed.boundary, eng.engine)];
      const sev = cellInfo ? cellInfo.risk_level : 'low';
      const bndObj = R.boundaries.find((x) => x.id === ed.boundary);
      const bCode = bndObj ? bndObj.code : ed.boundary;

      const longTopArc = a.nd.row === 0 && b.nd.row === 0 && Math.abs(a.nd.col - b.nd.col) > 1;
      let sx = a.cx, sy = a.cy, tx = b.cx, ty = b.cy;
      if (longTopArc) {
        sy = a.y; ty = b.y;
      } else {
        if (a.nd.col < b.nd.col) { sx = a.x + nodeW; tx = b.x; }
        else if (a.nd.col > b.nd.col) { sx = a.x; tx = b.x + nodeW; }
        if (a.nd.col === b.nd.col) {
          sy = a.y < b.y ? a.y + nodeH : a.y;
          ty = a.y < b.y ? b.y : b.y + nodeH;
        }
      }

      const midX = (sx + tx) / 2;
      const curveOffset = (idx % 2 === 0 ? -9 : 9);
      const midY = longTopArc ? 6 : (sy + ty) / 2 + curveOffset;
      const pathD = 'M' + sx + ',' + sy + ' Q' + midX + ',' + midY + ' ' + tx + ',' + ty;

      const isAct = activeBnd === ed.boundary;
      const isDim = activeBnd && !isAct;
      const markerKey = sev === 'high' ? 'arr-high' : sev === 'medium' ? 'arr-medium' : 'arr-low';

      const eg = svg.append('g')
        .attr('class', 'topo-edge sev-' + sev + (isAct ? ' is-active' : '') + (isDim ? ' dim' : ''))
        .on('click', (ev) => {
          ev.stopPropagation();
          atlas.focusBoundary = atlas.focusBoundary === ed.boundary ? null : ed.boundary;
          atlas.selectedCell = { boundary: ed.boundary, engine: eng.engine };
          refreshAllPanels(false);
        });

      eg.append('path')
        .attr('d', pathD)
        .attr('marker-end', 'url(#' + topoBoxId + '-' + markerKey + ')');

      const pillText = bCode + ' ' + ed.boundary;
      const pw = pillText.length * 6.1 + 12;
      const lx = (sx + 2 * midX + tx) / 4;
      const ly = (sy + 2 * midY + ty) / 4;
      const pg = eg.append('g').attr('class', 'topo-pill').attr('transform', 'translate(' + (lx - pw / 2) + ',' + (ly - 9) + ')');
      pg.append('rect').attr('width', pw).attr('height', 18);
      pg.append('text').attr('x', pw / 2).attr('y', 12.5).attr('text-anchor', 'middle').text(pillText);
    });

    // Draw process/state nodes
    topo.nodes.forEach((nd) => {
      const pNode = pos[nd.id];
      const inChain = chainNodes.has(nd.id);
      const isDim = activeBnd && !inChain;
      const ng = svg.append('g')
        .attr('class', 'topo-node' + (!nd.reached ? ' unreached' : '') + (inChain ? ' in-chain' : '') + (isDim ? ' dim' : ''))
        .attr('transform', 'translate(' + pNode.x + ',' + pNode.y + ')');
      ng.append('rect').attr('width', nodeW).attr('height', nodeH);
      const tg = ng.append('g').attr('clip-path', 'url(#' + topoBoxId + '-nclip)');
      tg.append('text').attr('class', 'topo-title').attr('x', 10).attr('y', 20).text(nd.label);
      tg.append('text').attr('class', 'topo-sub').attr('x', 10).attr('y', 36).text(nd.sub);
      tg.append('text').attr('class', 'topo-cost').attr('x', 10).attr('y', 52).text((nd.reached ? 'live \u00b7 ' : 'unreached \u00b7 ') + nd.cost);
    });
  }

  // ================================================================ Coverage and conversion (#coverage)
  const covState = { engine: 'All' };
  const STATUS = {
    merged_gpu_verified: ['GPU-verified', 'good'],
    merged_cpu_only: ['CPU-tested only', 'warn'],
    measured_not_in_repo: ['Measured earlier, not in the repo', 'mute'],
    planned: ['Planned', 'mute']
  };
  const CLS = [['swap', 'Swap in place'], ['swap_recapture', 'Swap plus recapture'], ['rebuild', 'Rebuild in place'], ['respawn', 'Respawn from zygote'], ['restart', 'Restart'], ['none', 'No path yet']];
  const GAIN = { swap: 'swap in place', swap_recapture: 'graph recapture', rebuild: 'in-place rebuild', respawn: 'zygote respawn', jit_rebuild: 'kernel rebuild', constructor_reinit: 'constructor re-init' };
  const gained = (stepObj) => stepObj.gained.map((g) => GAIN[g] || g).join(' + ');

  function stack(u, big) {
    return h('div', { class: 'sv-bar' + (big ? ' big' : ''), role: 'img', 'aria-label': CLS.map(([, l], i) => l + ' ' + pct(u.segs[i], u.total)).join(', ') },
      CLS.map(([k, l], i) => u.segs[i] > 0 && h('span', { class: 'sv-s sv-' + k, style: 'flex:' + u.segs[i] + ' 0 0', title: l + ': ' + num(u.segs[i]) + ' (' + pct(u.segs[i], u.total) + ')' }, big && u.segs[i] / u.total >= 0.075 ? pct(u.segs[i], u.total) : null)));
  }

  function measuredCard(m) {
    const body = [];
    if (m.id === 'megatron-edit-kinds') {
      body.push(table(m.columns.map((c) => c), m.rows, ''));
      const s = m.commit_survey;
      body.push(h('h4', { class: 'au-h3' }, 'Commit survey'), kv([['Agent-written commits classified', s.total], ['No code change', s.no_code_change], ['A function swap works', s.function_swap_works], ['Needed a restart', s.needed_a_restart]]),
        p(s.restart_share_note + '. Why a function swap was not enough:'), kv(Object.entries(s.restart_reasons)),
        kv([['Sampled restart commits installed', m.sample.n], ['Installed in place', m.sample.live_editable], ['Needed a partial reset', m.sample.partial_reset], ['Fell back to a restart', m.sample.restart]]));
    } else if (m.id === 'vllm-pr-survey') {
      body.push(table(['Mechanism the PR needs', { h: 'Share of 80', n: 1 }, 'Groups', 'Class'], m.rows.map((r) => [r.mechanism, r.share_of_80_pct + '%', groups(r.groups), r.class])));
    } else if (m.id === 'megatron-upstream-replay') {
      const a = F.prs.engines.find((e) => e.engine === 'Megatron-LM').aggregate;
      body.push(kv([['Pull requests tried', a.attempted], ['Installed live', a.live_committed], ['Refused, then a code restart', a.refused_then_code_restart], ['Need a manual rebase (not run)', a.need_manual_rebase], ['Do not apply to the lab\u2019s tree', a.do_not_apply]]));
      m = Object.assign({}, m, { source: a.source });
    } else if (m.id === 'conformance') {
      body.push(table(['Lab', { h: 'Cases passed', n: 1 }], Object.entries(m.results).map(([k, v]) => [k, v == null ? nr() : v])));
    } else if (m.counts) {
      body.push(kv(Object.entries(m.counts).filter(([k]) => k !== 'counted_by').map(([k, v]) => [pretty(k), v])));
      if (m.after_fixes) body.push(p('After the fixes: ' + m.after_fixes.live_in_place + ' live in place, ' + m.after_fixes.forked_restart_from_warm_process + ' by forked restart. ' + m.after_fixes.note));
      if (m.verified_live_on_gpu) body.push(p('Changed live on a GPU: ', m.verified_live_on_gpu.map((f) => h('code', null, f)), '. Implemented but not run on a GPU: ', m.implemented_live_not_gpu_run.map((f) => h('code', null, f)), '. Flag counts were counted from the matrix rows for this page.'));
    }
    return h('div', { class: 'fr-card au-card' },
      h('div', { class: 'au-head' }, h('h3', { class: 'au-h3' }, m.what), badge(m.engine === 'all' ? 'All engines' : m.engine, 'mute')),
      p(h('b', null, 'Counts '), m.unit, m.when ? ['. ', h('b', null, 'When: '), m.when] : null, '.'),
      m.method && p(rich(m.method)), body,
      (m.caveat || m.note) && h('p', { class: 'fr-note' }, rich(m.caveat || m.note)), src(m.source));
  }

  function renderCoverage() {
    const root = document.querySelector('[data-view="coverage"]');
    if (!root) return;
    const on = (e) => covState.engine === 'All' || e === covState.engine || e === 'all';
    const C = F.coverage, V = F.conversion, engs = V.engines.filter((e) => on(e.engine));
    const folders = [...new Set(V.engines.flatMap((e) => Object.keys(e.derived_from_git.current_tree_by_folder)))];
    const g = (e) => e.derived_from_git;
    const gitRows = [
      ['Commits that touch the engine\u2019s folder', (e) => num(g(e).commits_touching_folder)],
      ['First to last commit', (e) => g(e).first_commit + ' to ' + g(e).last_commit],
      ['Lines added, all history', (e) => num(g(e).lines_added_all_history)],
      ['Lines deleted, all history', (e) => num(g(e).lines_deleted_all_history)],
      ['Merged PRs titled for the engine', (e) => num(g(e).merged_prs_titled_for_engine.count)],
      ['Lab Python, not tests (files / lines)', (e) => num(g(e).livelab_python.non_test_files) + ' / ' + num(g(e).livelab_python.non_test_lines)],
      ['Lab Python, tests (files / lines)', (e) => num(g(e).livelab_python.test_files) + ' / ' + num(g(e).livelab_python.test_lines)]
    ].concat(folders.map((f) => ['Lines now in ' + f + '/', (e) => (g(e).current_tree_by_folder[f] ? num(g(e).current_tree_by_folder[f].lines) : '0')]))
      .concat([['Upstream patch files kept in the repo', (e) => num(e.upstream_patches.count)]]);

    const W = V.meta.whole_repo, K = V.meta.shared_core;
    const mech = F.engines.engines.filter((e) => on(e.engine));
    const allMech = F.engines.engines.flatMap((e) => e.mechanisms), cnt = (s) => allMech.filter((m) => m.status === s).length;
    const histEngines = F.survey.filter((e) => e.history && on(e.engine));
    const baseUnit = atlas.unit === 'functions' ? 'functions' : 'lines';

    root.replaceChildren(
      h('p', { class: 'au-intro' }, 'What is measured about how much of each engine can be edited live, the six-engine static coverage Marimekko and capability simulator, two-level subsystem drill-downs, what the conversion of each engine left in the repo, and what the toolsmith built.'),
      picker(covState, ALL_ENGINES, null, () => { renderCoverage(); drawCoverageCharts(false); }),
      buildPanel1And2('cov', (engName) => on(engName)),
      histEngines.length ? sec('Live share as each lab on main grew',
        p('The share of ' + (baseUnit === 'functions' ? 'functions' : 'lines of Python') + ' with a live install path at each commit that touched the lab\u2019s folder.'),
        h('div', { class: 'fr-multi sv-multi' }, histEngines.map((e) => h('div', { class: 'fr-mini' }, h('h4', null, e.engine),
          h('div', { class: 'fr-mini-read' }, e.history.n + ' lab commits, ' + e.history.first + ' to ' + e.history.last),
          h('div', { class: 'sv-step', 'data-e': e.engine }),
          h('ul', { class: 'au-list sv-steps' }, e.history.steps.filter((stepObj) => stepObj.gained.length).map((stepObj) => h('li', null, stepObj.date + ': ' + gained(stepObj) + (stepObj.pr ? ' (FastRSI PR #' + stepObj.pr + ')' : '') + ', live share ' + pct(stepObj[baseUnit], e[baseUnit].total)))))))) : null,
      sec('Where the code sits: the 10 largest top-level modules per engine',
        F.survey.filter((e) => on(e.engine)).map((e) => [
          h('p', { class: 'au-p' }, h('b', null, e.engine), ': 10 of ' + e.modules_total + ' top-level modules.'),
          table(['Module', { h: baseUnit === 'functions' ? 'Functions' : 'Lines of Python', n: 1 }, 'Split by install class', { h: 'Live path', n: 1 }, { h: 'No path yet', n: 1 }, { h: 'Native files', n: 1 }],
            e.modules.map((m) => [h('code', null, m.name), num(m[baseUnit].total), h('td', { class: 'sv-cell' }, stack(m[baseUnit])), pct(m[baseUnit].live, m[baseUnit].total), pct(m[baseUnit].segs[5], m[baseUnit].total), num(m.native_files)]), 'au-w2')
        ]),
        h('p', { class: 'au-src' }, 'Source: static survey tool in core/livelab/survey (ruleset ' + F.survey[0].ruleset_version + '), run on ' + F.survey[0].scan_date + '.')
      ),
      C.measured.some((m) => on(m.engine)) ? sec('What is measured: coverage by edit kind and by sampled change',
        p('These are the tested results on the three engines on main. They count kinds of edit, sampled commits, pull requests and command-line flags.'),
        h('div', { class: 'au-cards' }, C.measured.filter((m) => on(m.engine)).map(measuredCard)),
        LEGACY_ENGINES.some(on) ? [
          h('h3', { class: 'au-h3' }, 'Not covered, as the repo states'),
          table(['Engine', 'Edits or setups the lab does not cover'], LEGACY_ENGINES.filter(on).map((e) => [h('b', null, e), h('ul', { class: 'au-list' }, C.not_covered_stated_by_repo[e].map((x) => h('li', null, rich(x))))]), ''),
          src(C.not_covered_stated_by_repo.source)
        ] : null
      ) : null,
      engs.length ? sec('What converting each engine left in the repo',
        h('div', { class: 'au-head' }, badge('Derived from git history', 'mute')),
        p('These rows are computed from the git history of the main branch on 2026-11-10. The whole repo has ' + W.commits_on_main + ' commits and ' + W.merged_prs + ' merged pull requests between ' + W.first_commit + ' and ' + W.last_commit + '. The kernel all labs share is ' + num(K.kernel.lines) + ' lines in ' + K.kernel.files + ' files, with ' + num(K.conformance.lines) + ' lines of conformance cases.'),
        table([''].concat(engs.map((e) => ({ h: e.engine, n: 1 }))), gitRows.map(([k, fn]) => [k].concat(engs.map(fn))), ''),
        src(engs[0].derived_from_git.source),
        h('ul', { class: 'au-list' }, V.meta.caution.concat(engs.flatMap((e) => e.notes.map((n) => e.engine + ': ' + n))).map((c) => h('li', null, rich(c)))),
        h('h3', { class: 'au-h3' }, 'Upstream patches'),
        engs.map((e) => {
          const u = e.upstream_patches;
          return h('div', { class: 'au-sec' },
            p(h('b', null, e.engine + ': ' + u.count + (u.count === 1 ? ' patch file. ' : ' patch files. ')), u.kind ? pretty(u.kind) + '. ' : '', u.total_added != null ? 'Together +' + num(u.total_added) + ' \u2212' + num(u.total_removed) + ' lines. ' : '', u.result ? u.result + '. ' : '', rich(u.note)),
            u.patches && table(['Patch file', { h: 'Files', n: 1 }, { h: 'Diff', n: 1 }, 'Recorded gain', 'Upstream'], u.patches.map((x) => [h('code', null, x.file), x.files, '+' + x.added + ' \u2212' + x.removed, x.gain, x.upstream]), 'au-w3'),
            src(u.source));
        }),
        h('h3', { class: 'au-h3' }, 'Recorded in the repo'),
        table(['Engine', 'Fact', 'Source'], engs.flatMap((e) => e.recorded_facts.map((f) => [h('b', null, e.engine), rich(f.fact), h('td', { class: 'au-src' }, srcText(f.source))])), 'au-w3')
      ) : null,
      engs.length ? h('section', { class: 'au-nm' },
        h('div', { class: 'au-head' }, h('h2', { class: 'fr-h2' }, 'Conversion wall time'), badge('LOC & Conformance Recorded', 'ok')),
        p('No record exists of how long the offline conversion of any engine took, how many agents worked on it, or how many rounds it needed.'),
        table(['Engine', 'Wall time', 'Agents', 'Rounds'], engs.map((e) => [h('b', null, e.engine), nr(), nr(), nr()]), '')
      ) : null,
      mech.length ? sec('What the toolsmith built',
        p('Every mechanism listed for each engine on main, with its status and the time before and after where one was measured. Across the three engines: ' + cnt('merged_gpu_verified') + ' GPU-verified, ' + cnt('merged_cpu_only') + ' CPU-tested only, ' + cnt('measured_not_in_repo') + ' measured earlier but not in the repo, ' + cnt('planned') + ' planned.'),
        mech.map((e) => [h('h3', { class: 'au-h3' }, e.engine + ' \u00b7 ' + e.mechanisms.length + ' mechanisms'),
          table(['Mechanism', 'Start-up cost it removes', 'Status', 'Before \u2192 after', 'Source'], e.mechanisms.map((m) => {
            const s = m.measured_saving, statusPair = STATUS[m.status] || [m.status, 'mute'];
            return [[h('b', null, rich(m.name)), sub(rich(m.what)), m.note ? sub(rich(m.note)) : null], rich(m.startup_cost_removed), badge(statusPair[0], statusPair[1]),
              s ? [h('span', { class: 'fr-num' }, fmt(s.before, s.unit), ' \u2192 ', fmt(s.after, s.unit)), typeof s.before === 'string' || typeof s.after === 'string' ? sub('Values in ' + (s.unit === 's' ? 'seconds' : s.unit)) : null, s.capture_ms != null ? sub('capture ' + s.capture_ms + ' ms, restore ' + s.restore_ms + ' ms') : null, sub(rich(s.setup))] : nr(),
              h('td', { class: 'au-src' }, srcText(m.source))];
          }), 'au-w1'), hint()]),
        h('ul', { class: 'au-list' }, F.engines.caveats.map((c) => h('li', null, srcText(c))))
      ) : null
    );
  }

  function drawSteps() {
    if (!window.d3) return;
    const baseUnit = atlas.unit === 'functions' ? 'functions' : 'lines';
    document.querySelectorAll('.sv-step').forEach((box) => {
      const e = F.survey.find((x) => x.engine === box.dataset.e), W = box.clientWidth;
      if (!e || !e.history || !W) return;
      const H = 190, m = { t: 16, r: 12, b: 24, l: 40 }, n = e.history.n, tot = e[baseUnit].total;
      const x = d3.scaleLinear().domain([0, Math.max(1, n - 1)]).range([m.l, W - m.r]), y = d3.scaleLinear().domain([0, 100]).range([H - m.b, m.t]);
      box.replaceChildren();
      const svg = d3.select(box).append('svg').attr('class', 'fr-svg').attr('viewBox', '0 0 ' + W + ' ' + H).attr('width', W).attr('height', H).attr('role', 'img').attr('aria-label', e.engine + ': live share by lab commit.');
      [0, 50, 100].forEach((v) => { svg.append('line').attr('class', 'gl').attr('x1', m.l).attr('x2', W - m.r).attr('y1', y(v)).attr('y2', y(v)); svg.append('text').attr('x', m.l - 6).attr('y', y(v) + 4).attr('text-anchor', 'end').text(v + '%'); });
      svg.append('text').attr('x', m.l).attr('y', H - 6).text('first lab commit');
      svg.append('text').attr('x', W - m.r).attr('y', H - 6).attr('text-anchor', 'end').text('commit ' + n);
      const pts = e.history.steps.map((stepObj) => [stepObj.idx, 100 * stepObj[baseUnit] / tot]).concat([[n - 1, 100 * e.history.steps[e.history.steps.length - 1][baseUnit] / tot]]);
      svg.append('path').attr('class', 'ln fastrsi').attr('d', d3.line().x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveStepAfter)(pts));
      let prev = -1;
      e.history.steps.filter((stepObj) => stepObj.gained.length).forEach((stepObj) => {
        const v = 100 * stepObj[baseUnit] / tot, px = x(stepObj.idx), py = y(v), flat = v === prev; prev = v;
        svg.append('circle').attr('cx', px).attr('cy', py).attr('r', 4).attr('fill', flat ? 'var(--color-panel)' : 'var(--fr-fast)').attr('stroke', 'var(--fr-fast)').attr('stroke-width', 2);
        const labelTxt = gained(stepObj);
        const right = px + 10 + labelTxt.length * 6.2 < W - m.r;
        const lw = labelTxt.length * 6.2, lx = right ? px + 8 : Math.max(m.l + 4, Math.min(px - 8 - lw, W - m.r - lw));
        svg.append('text').attr('class', 'sv-lb halo').attr('x', lx).attr('y', flat ? py + 16 : py - 8).text(labelTxt);
      });
    });
  }

  // ================================================================ Architecture & Risk (#boundaries) + Unified Atlas (#atlas)
  function renderBoundaries() {
    const root = document.querySelector('[data-view="boundaries"]');
    if (!root) return;
    root.replaceChildren(
      h('p', { class: 'au-intro' }, 'How FastRSI hooks into each engine\u2019s runtime process tree, how restarts propagate across state and graph boundaries, and where silent-corruption, hang, or cold-fallback risks remain across all 8 boundaries \u00d7 6 engines.'),
      buildPanel3And4('bnd')
    );
  }

  function renderAtlas() {
    const root = document.querySelector('[data-view="atlas"]');
    if (!root) return;
    root.replaceChildren(
      h('p', { class: 'au-intro' }, 'Coordinated four-panel Coverage & Interface Risk Atlas across vLLM, Megatron-LM, FastVideo, SGLang, verl, and miles. Selecting an engine, capability toggle, or boundary updates all four views together.'),
      buildPanel1And2('atl', null),
      buildPanel3And4('atl')
    );
  }

  function drawCoverageCharts(animate) {
    drawMarimekko('cov-mekko', animate);
    drawIcicle('cov-icicle', 'cov-sub-insp');
    drawSteps();
  }

  function refreshAllPanels(animateMekko) {
    const scrollPos = window.scrollY;
    const activeView = currentView();
    if (activeView === 'coverage') {
      if (!animateMekko) renderCoverage();
      else {
        // Re-sync checkbox classes and panel 2 header without destroying mekko SVG transitions
        const root = document.querySelector('[data-view="coverage"]');
        const oldMekko = document.getElementById('cov-mekko');
        const svgKeep = oldMekko && oldMekko.querySelector('svg.mekko-svg');
        renderCoverage();
        const newMekko = document.getElementById('cov-mekko');
        if (svgKeep && newMekko) newMekko.appendChild(svgKeep);
        void root;
      }
      drawCoverageCharts(animateMekko);
    } else if (activeView === 'boundaries') {
      renderBoundaries();
      drawTopology('bnd-topo');
    } else if (activeView === 'atlas') {
      const oldMekko = document.getElementById('atl-mekko');
      const svgKeep = animateMekko && oldMekko && oldMekko.querySelector('svg.mekko-svg');
      renderAtlas();
      const newMekko = document.getElementById('atl-mekko');
      if (svgKeep && newMekko) newMekko.appendChild(svgKeep);
      drawMarimekko('atl-mekko', animateMekko);
      drawIcicle('atl-icicle', 'atl-sub-insp');
      drawTopology('atl-topo');
    }
    window.scrollTo(0, scrollPos);
  }

  // ================================================================ Faults (#faults) & Conventions (#conventions)
  function renderFaults() {
    const root = document.querySelector('[data-view="faults"]');
    if (!root) return;
    const X = F.faults;
    const cols = ['Fault', 'Expected', 'Observed', 'Source'];
    const row = (r) => [[rich(r.fault), r.case ? sub(h('code', null, r.case)) : null], rich(r.expected), [r.known_bug ? [badge('Known gap', 'bad'), ' '] : null, rich(r.observed), r.detail ? sub(rich(r.detail)) : null, r.note ? sub(rich(r.note)) : null], h('td', { class: 'au-src' }, srcText(r.source))];
    const block = (title, tag, setup, rows, note) => h('section', { class: 'au-sec' }, h('div', { class: 'au-head' }, h('h3', { class: 'au-h3' }, title), tag), setup && p(rich(setup)), table(cols, rows.map(row), 'au-w2'), note && h('p', { class: 'fr-note' }, rich(note)));
    const earlier = () => badge('Earlier build of the lab', 'warn'), repo = () => badge('Public repo', 'mute');
    const a = X.earlier_megatron_fault_injection, b = X.earlier_kernel_fault_timing;
    const nRepo = X.repo_conformance_cases.reduce((s, g) => s + g.cases.length, 0) + X.repo_pr_fault_receipts.reduce((s, g) => s + g.cases.length, 0);
    root.replaceChildren(
      h('p', { class: 'au-intro' }, 'Every recorded fault-injection result: the fault, what should happen, what was observed, and where the record is. It does not cover faults nobody injected; a hung CUDA context, for one, was not tested.'),
      h('p', { class: 'au-callout is-gap' }, h('b', null, 'Known gap. '), b.known_bug + ' The restart path reports the failure; the live kernel worker returns nothing. This is a bug, listed in the kernel timing table below.'),
      h('p', { class: 'au-callout' }, h('b', null, 'Two builds. '), X.meta.caveat),
      sec('From an earlier build of the lab: ' + (a.rows.length + b.rows.length) + ' rows',
        block(a.title, earlier(), a.setup + ' ' + X.meta.count_note + ' The detail lines are read from the table\u2019s source ledger, one row per injected commit.', a.rows),
        block(b.title, earlier(), b.setup + '.', b.rows)),
      sec('From the public repo: ' + nRepo + ' rows',
        X.repo_conformance_cases.map((g) => block(g.engine + ': conformance fault cases', repo(), g.setup, g.cases)),
        X.repo_pr_fault_receipts.map((g) => block(g.engine + ': fault receipts', repo(), g.setup + '. Source: ' + srcText(g.source) + '.', g.cases, g.note))),
      sec('Related: drift guard', p(X.related_drift_guard.fact), src(X.related_drift_guard.source)));
  }

  function renderConformance() {
    const box = document.getElementById('conf-results');
    if (!box) return;
    const m = F.coverage.measured.find((x) => x.id === 'conformance');
    box.replaceChildren(table(['Lab', { h: 'Cases passed', n: 1 }], Object.entries(m.results).map(([k, v]) => [k, v == null ? nr() : v]), ''), h('p', { class: 'fr-note' }, m.note), src(m.source));
  }

  // ================================================================ Routing & Resize
  function currentView() {
    const raw = location.hash.slice(1);
    if (VIEWS.includes(raw)) return raw;
    return window.ATLAS_DEFAULT_TAB || 'prs';
  }

  function route() {
    const v = currentView();
    document.querySelectorAll('.au-view').forEach((s) => { s.hidden = s.dataset.view !== v; });
    document.querySelectorAll('#au-tabs a, .pg-tabs a[data-tab]').forEach((a) => {
      if (a.dataset.tab === v) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    });
    if (v === 'prs') drawChart();
    else if (v === 'coverage') { renderCoverage(); drawCoverageCharts(false); }
    else if (v === 'boundaries') { renderBoundaries(); drawTopology('bnd-topo'); }
    else if (v === 'atlas') { renderAtlas(); drawMarimekko('atl-mekko', false); drawIcicle('atl-icicle', 'atl-sub-insp'); drawTopology('atl-topo'); }
  }

  document.addEventListener('click', () => { const t = document.querySelector('.au-tipbox'); if (t) t.remove(); });
  let rz;
  window.addEventListener('resize', () => {
    clearTimeout(rz);
    rz = setTimeout(() => {
      const v = currentView();
      if (v === 'prs') drawChart();
      else if (v === 'coverage') drawCoverageCharts(false);
      else if (v === 'boundaries') drawTopology('bnd-topo');
      else if (v === 'atlas') { drawMarimekko('atl-mekko', false); drawIcicle('atl-icicle', 'atl-sub-insp'); drawTopology('atl-topo'); }
    }, 120);
  });

  renderPrs();
  renderFaults();
  renderConformance();
  window.addEventListener('hashchange', () => { route(); window.scrollTo(0, 0); });
  route();
})();

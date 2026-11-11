// FastRSI Audit: renders three views from the inlined facts (window.FACTS) and routes on location.hash.
(function () {
  const F = window.FACTS, VIEWS = ['prs', 'coverage', 'conventions', 'faults'], ENGINES = ['vLLM', 'Megatron-LM', 'FastVideo'];
  const NR = 'not recorded';
  function h(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const k in attrs || {}) { const v = attrs[k]; if (v == null || v === false) continue; if (k === 'class') n.className = v; else if (k.slice(0, 2) === 'on') n.addEventListener(k.slice(2), v); else n.setAttribute(k, v === true ? '' : v); }
    for (const k of kids.flat(Infinity)) { if (k == null || k === false) continue; n.append(k.nodeType ? k : document.createTextNode(String(k))); }
    return n;
  }
  // text with `code` spans
  const rich = (s) => String(s).split('`').map((p, i) => (i % 2 ? h('code', null, p) : p));
  const nr = () => h('span', { class: 'au-nr' }, NR);
  const val = (v) => (v == null || v === '' ? nr() : rich(v));
  const fmt = (v, unit) => (v == null ? nr() : typeof v === 'number' ? v + ' ' + unit : rich(v));
  const num = (v) => (v == null ? nr() : typeof v === 'number' ? v.toLocaleString('en-US') : v);
  const srcText = (s) => String(s).replace(/\bPR #/g, 'FastRSI PR #');
  const src = (s, tag) => h(tag || 'p', { class: 'au-src' }, s ? srcText(s) : NR);
  const badge = (t, kind) => h('span', { class: 'au-b ' + (kind || '') }, t);
  const groups = (g) => (g && g.length ? g.map((x) => h('span', { class: 'au-g' }, x)) : nr());
  const sub = (...k) => h('span', { class: 'sub' }, ...k);
  const pretty = (k) => { const s = k.replace(/_/g, ' '); return s[0].toUpperCase() + s.slice(1); };
  function table(cols, rows, cls) {
    const t = h('table', { class: 'fr-table au-t ' + (cls || '') },
      h('thead', null, h('tr', null, cols.map((c) => h('th', { class: c.n ? 'n' : null, scope: 'col' }, c.h || c)))),
      h('tbody', null, rows.map((r) => h('tr', null, r.map((cell, i) => (cell && cell.nodeName === 'TD' ? cell : h('td', { class: cols[i].n ? 'n' : cols[i].k ? 'k' : null }, cell)))))));
    return h('div', { class: 'fr-tablewrap', tabindex: '0', role: 'region', 'aria-label': 'Table, scrolls sideways on a narrow screen' }, t);
  }
  const kv = (pairs) => h('dl', { class: 'au-kv' }, pairs.map(([k, v]) => [h('dt', null, k), h('dd', null, num(v))]));
  const sec = (title, ...kids) => h('section', { class: 'au-sec' }, title && h('h2', { class: 'fr-h2' }, title), ...kids);
  const p = (...k) => h('p', { class: 'au-p' }, ...k);
  const hint = () => h('p', { class: 'fr-note au-scrollhint' }, 'Wide table: scroll it sideways.');
  function picker(state, counts, redraw) {
    return h('div', { class: 'fr-picker', role: 'group', 'aria-label': 'Engine' }, ['All'].concat(ENGINES).map((e) =>
      h('button', { type: 'button', class: 'fr-chip', 'aria-pressed': String(state.engine === e), onclick: () => { state.engine = e; redraw(); } }, counts ? e + ' · ' + counts[e] : e)));
  }

  // ---------------------------------------------------------------- Supported PRs
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
    if (x.added != null) return [h('span', { class: 'fr-num' }, '+' + x.added + ' −' + x.removed), sub(x.files + (x.files === 1 ? ' file' : ' files'))];
    return nr();
  }
  function renderPrs() {
    const root = document.querySelector('[data-view="prs"]'), list = allPrs.filter((r) => prState.engine === 'All' || r.engine === prState.engine);
    const counts = { All: allPrs.length }; ENGINES.forEach((e) => { counts[e] = allPrs.filter((r) => r.engine === e).length; });
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
      h('td', { class: 'au-src' }, srcText(r.source)),
    ]);
    root.replaceChildren(
      h('p', { class: 'au-intro' }, 'The upstream pull requests that were replayed in a live lab: whether each edit installed into the running engine, how long the install and check took, and the restart it replaces. It does not show speedups, and it leaves out the pull requests the labs could not take.'),
      h('p', { class: 'au-callout' }, 'None of these upstream pull requests showed a measured speedup on the workload. They test whether an edit installs and runs live.'),
      sec(allPrs.length + ' replayed pull requests, ' + live + ' installed live',
        p(nlTotal + ' other pull requests were tried and are not listed here (' + Object.keys(nl).map((e) => nl[e] + ' ' + e).join(', ') + '): they did not apply to the lab’s tree or needed a manual rebase. The vLLM replay covered ' + v.first_pass.live_in_place + ' in place and ' + v.first_pass.forked_restart + ' by forked restart, 19 pull requests in all; the repo names only 6 of them, so the other 13 are not listed either.'),
        picker(prState, counts, renderPrs),
        table([{ h: 'PR', k: 1 }, 'Title', 'Kind', 'Function groups', 'Outcome', 'Time to install and check', 'Restart time it replaces', { h: 'Diff', k: 1 }, 'Source'], rows, 'au-w1'),
        hint(),
        h('p', { class: 'fr-note' }, 'Titles and diff sizes come from GitHub (the pull request’s additions, deletions and changed files, read on 2026-11-11). Everything else comes from the FastRSI repo at the cited lines. The three vLLM rows citing mechanisms.md are examples from an earlier survey, not from the 19-PR replay.')),
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
    const box = document.getElementById('pr-chart'); if (!box || !window.d3 || !box.clientWidth) return;
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
        h('div', null, h('span', { class: 'tv' }, '+' + d.added + ' −' + d.removed), ' in ' + d.files + (d.files === 1 ? ' file' : ' files')),
        h('div', null, 'FastRSI: ', d.t == null ? 'no time recorded for the route taken' : [h('span', { class: 'tv' }, d.t + ' s'), ' · ' + d.t_what]),
        h('div', null, 'Restart: ', d.restart == null ? NR : [h('span', { class: 'tv' }, (Array.isArray(d.restart) ? d.restart.join('–') : d.restart) + ' s'), ' · ' + d.restart_what]));
      box.append(tip);
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      tip.style.left = Math.max(0, Math.min(W - tw, px + 12)) + 'px'; tip.style.top = (py - th - 10 < 0 ? py + 14 : py - th - 10) + 'px';
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
  document.addEventListener('click', () => { const t = document.querySelector('.au-tipbox'); if (t) t.remove(); });
  let rz; window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { drawChart(); drawSteps(); }, 120); });

  // ---------------------------------------------------------------- Coverage and conversion
  const covState = { engine: 'All', unit: 'functions' };
  const STATUS = { merged_gpu_verified: ['GPU-verified', 'good'], merged_cpu_only: ['CPU-tested only', 'warn'], measured_not_in_repo: ['Measured earlier, not in the repo', 'mute'], planned: ['Planned', 'mute'] };
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
      body.push(kv([['Pull requests tried', a.attempted], ['Installed live', a.live_committed], ['Refused, then a code restart', a.refused_then_code_restart], ['Need a manual rebase (not run)', a.need_manual_rebase], ['Do not apply to the lab’s tree', a.do_not_apply]]));
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
    const root = document.querySelector('[data-view="coverage"]'), on = (e) => covState.engine === 'All' || e === covState.engine || e === 'all';
    const C = F.coverage, V = F.conversion, engs = V.engines.filter((e) => on(e.engine));
    const folders = [...new Set(V.engines.flatMap((e) => Object.keys(e.derived_from_git.current_tree_by_folder)))];
    const g = (e) => e.derived_from_git, tot = (e) => Object.values(g(e).current_tree_by_folder).reduce((a, b) => a + b.lines, 0);
    const gitRows = [
      ['Commits that touch the engine’s folder', (e) => num(g(e).commits_touching_folder)],
      ['First to last commit', (e) => g(e).first_commit + ' to ' + g(e).last_commit],
      ['Lines added, all history', (e) => num(g(e).lines_added_all_history)],
      ['Lines deleted, all history', (e) => num(g(e).lines_deleted_all_history)],
      ['Merged PRs titled for the engine', (e) => num(g(e).merged_prs_titled_for_engine.count)],
      ['Lab Python, not tests (files / lines)', (e) => num(g(e).livelab_python.non_test_files) + ' / ' + num(g(e).livelab_python.non_test_lines)],
      ['Lab Python, tests (files / lines)', (e) => num(g(e).livelab_python.test_files) + ' / ' + num(g(e).livelab_python.test_lines)],
    ].concat(folders.map((f) => ['Lines now in ' + f + '/', (e) => (g(e).current_tree_by_folder[f] ? num(g(e).current_tree_by_folder[f].lines) : '0')]))
      .concat([['Upstream patch files kept in the repo', (e) => num(e.upstream_patches.count)]]);
    const W = V.meta.whole_repo, K = V.meta.shared_core;
    const mech = F.engines.engines.filter((e) => on(e.engine));
    const allMech = F.engines.engines.flatMap((e) => e.mechanisms), cnt = (s) => allMech.filter((m) => m.status === s).length;
    root.replaceChildren(
      h('p', { class: 'au-intro' }, 'What is measured about how much of each engine can be edited live, a static estimate of how much of each source tree has a live install path, what the conversion of each engine left in the repo, and what the toolsmith built. It does not show coverage of what a workload runs, and it does not show how long a conversion took: that was not recorded.'),
      picker(covState, null, renderCoverage),
      sec('What is measured: coverage by edit kind and by sampled change',
        p('These are the tested results. They count kinds of edit, sampled commits, pull requests and command-line flags, not functions or lines. The static survey further down counts functions and lines, as an estimate.'),
        h('div', { class: 'au-cards' }, C.measured.filter((m) => on(m.engine)).map(measuredCard)),
        h('h3', { class: 'au-h3' }, 'Not covered, as the repo states'),
        table(['Engine', 'Edits or setups the lab does not cover'], ENGINES.filter(on).map((e) => [h('b', null, e), h('ul', { class: 'au-list' }, C.not_covered_stated_by_repo[e].map((x) => h('li', null, rich(x))))]), ''),
        src(C.not_covered_stated_by_repo.source)),
      surveySection(on),
      sec('What converting each engine left in the repo',
        h('div', { class: 'au-head' }, badge('Derived from git history', 'mute')),
        p('These rows are computed from the git history of the main branch on 2026-11-10, not logged by the conversion itself. Main is squash-merged, so one commit is about one pull request. The whole repo has ' + W.commits_on_main + ' commits and ' + W.merged_prs + ' merged pull requests between ' + W.first_commit + ' and ' + W.last_commit + '. The kernel all three labs share is ' + num(K.kernel.lines) + ' lines in ' + K.kernel.files + ' files, with ' + num(K.conformance.lines) + ' lines of conformance cases.'),
        table([''].concat(engs.map((e) => ({ h: e.engine, n: 1 }))), gitRows.map(([k, f]) => [k].concat(engs.map(f))), ''),
        src(engs[0].derived_from_git.source),
        h('ul', { class: 'au-list' }, V.meta.caution.concat(engs.flatMap((e) => e.notes.map((n) => e.engine + ': ' + n))).map((c) => h('li', null, rich(c)))),
        h('h3', { class: 'au-h3' }, 'Upstream patches'),
        engs.map((e) => { const u = e.upstream_patches; return h('div', { class: 'au-sec' },
          p(h('b', null, e.engine + ': ' + u.count + (u.count === 1 ? ' patch file. ' : ' patch files. ')), u.kind ? pretty(u.kind) + '. ' : '', u.total_added != null ? 'Together +' + num(u.total_added) + ' −' + num(u.total_removed) + ' lines. ' : '', u.result ? u.result + '. ' : '', rich(u.note)),
          u.patches && table(['Patch file', { h: 'Files', n: 1 }, { h: 'Diff', n: 1 }, 'Recorded gain', 'Upstream'], u.patches.map((x) => [h('code', null, x.file), x.files, '+' + x.added + ' −' + x.removed, x.gain, x.upstream]), 'au-w3'),
          src(u.source)); }),
        h('h3', { class: 'au-h3' }, 'Recorded in the repo'),
        table(['Engine', 'Fact', 'Source'], engs.flatMap((e) => e.recorded_facts.map((f) => [h('b', null, e.engine), rich(f.fact), h('td', { class: 'au-src' }, srcText(f.source))])), 'au-w3')),
      h('section', { class: 'au-nm' },
        h('div', { class: 'au-head' }, h('h2', { class: 'fr-h2' }, 'Conversion wall time'), badge('Not measured yet', 'warn')),
        p('No record exists of how long the offline conversion of any engine took, how many agents worked on it, or how many rounds it needed. The project’s own description says only that the toolsmith runs offline, once per repository. Commit dates bound the calendar window but not the effort. A measurement would log, per engine, the wall time from the first toolsmith action to a passing conformance run, the number of agents, and the number of rounds.'),
        table(['Engine', 'Wall time', 'Agents', 'Rounds'], engs.map((e) => [h('b', null, e.engine), nr(), nr(), nr()]), '')),
      sec('What the toolsmith built',
        p('Every mechanism listed for each engine, with its status and the time before and after where one was measured. Across the three engines: ' + cnt('merged_gpu_verified') + ' GPU-verified, ' + cnt('merged_cpu_only') + ' CPU-tested only, ' + cnt('measured_not_in_repo') + ' measured earlier but not in the repo, ' + cnt('planned') + ' planned.'),
        mech.map((e) => [h('h3', { class: 'au-h3' }, e.engine + ' · ' + e.mechanisms.length + ' mechanisms'),
          table(['Mechanism', 'Start-up cost it removes', 'Status', 'Before → after', 'Source'], e.mechanisms.map((m) => { const s = m.measured_saving, st = STATUS[m.status] || [m.status, 'mute'];
            return [[h('b', null, rich(m.name)), sub(rich(m.what)), m.note ? sub(rich(m.note)) : null], rich(m.startup_cost_removed), badge(st[0], st[1]),
              s ? [h('span', { class: 'fr-num' }, fmt(s.before, s.unit), ' → ', fmt(s.after, s.unit)), typeof s.before === 'string' || typeof s.after === 'string' ? sub('Values in ' + (s.unit === 's' ? 'seconds' : s.unit)) : null, s.capture_ms != null ? sub('capture ' + s.capture_ms + ' ms, restore ' + s.restore_ms + ' ms') : null, sub(rich(s.setup))] : nr(),
              h('td', { class: 'au-src' }, srcText(m.source))]; }), 'au-w1'), hint()]),
        h('ul', { class: 'au-list' }, F.engines.caveats.map((c) => h('li', null, srcText(c))))));
    drawSteps();
  }

  // ---- static survey: bars, step charts, top modules
  const CLS = [['swap', 'Swap in place'], ['swap_recapture', 'Swap plus recapture'], ['rebuild', 'Rebuild in place'], ['respawn', 'Respawn from zygote'], ['restart', 'Restart'], ['none', 'No path yet']];
  const GAIN = { swap: 'swap in place', swap_recapture: 'graph recapture', rebuild: 'in-place rebuild', respawn: 'zygote respawn', jit_rebuild: 'kernel rebuild' };
  const UNIT = { functions: 'functions', lines: 'lines of Python' };
  const pct = (a, b) => (b ? (100 * a / b).toFixed(1) : '0.0') + '%';
  const gained = (st) => st.gained.map((g) => GAIN[g] || g).join(' + ');
  function stack(u, big) {
    return h('div', { class: 'sv-bar' + (big ? ' big' : ''), role: 'img', 'aria-label': CLS.map(([, l], i) => l + ' ' + pct(u.segs[i], u.total)).join(', ') },
      CLS.map(([k, l], i) => u.segs[i] > 0 && h('span', { class: 'sv-s sv-' + k, style: 'flex:' + u.segs[i] + ' 0 0', title: l + ': ' + num(u.segs[i]) + ' (' + pct(u.segs[i], u.total) + ')' }, big && u.segs[i] / u.total >= 0.075 ? pct(u.segs[i], u.total) : null)));
  }
  function surveySection(on) {
    const U = covState.unit, list = F.survey.filter((e) => on(e.engine));
    const unitPick = h('div', { class: 'fr-picker', role: 'group', 'aria-label': 'Count by' }, Object.keys(UNIT).map((k) => h('button', { type: 'button', class: 'fr-chip', 'aria-pressed': String(U === k), onclick: () => { covState.unit = k; const y = scrollY; renderCoverage(); scrollTo(0, y); } }, k === 'functions' ? 'Functions' : 'Lines of Python')));
    return h('section', { class: 'au-sec', id: 'survey' },
      h('div', { class: 'au-head' }, h('h2', { class: 'fr-h2' }, 'Static survey: how much of each engine has a live install path'), badge('Static estimate, not measured coverage', 'warn')),
      h('ul', { class: 'au-list sv-limits' },
        h('li', null, h('b', null, 'Shares are of the whole source tree, not of what a workload runs. '), 'Training, evaluation and tooling code in the tree counts too.'),
        h('li', null, h('b', null, 'Most classifications rest on heuristics. '), 'Only the syntax rules are firm; the rest come from declared paths, names or a default. By function, the heuristic share is ', F.survey.map((e) => e.engine + ' ' + pct(e.functions.uncertain, e.functions.total)).join(', '), '. It is printed next to each bar.'),
        h('li', null, h('b', null, '"Has a live path" is not a tested result. '), 'It means the lab contains that mechanism and reaches the file. No edit was installed for this survey. The replayed pull requests under ', h('a', { href: '#prs' }, 'Supported PRs'), ' are the tested ones.'),
        h('li', null, h('b', null, 'Native files are counted apart, as restart: '), F.survey.map((e) => e.engine + ' ' + num(e.native.files)).join(', '), '. They are not in the bars.')),
      unitPick,
      h('div', { class: 'fr-card au-card' },
        h('ul', { class: 'fr-keys' }, CLS.map(([k, l]) => h('li', null, h('span', { class: 'fr-k sv-' + k }), l))),
        list.map((e) => { const u = e[U]; return h('div', { class: 'sv-row' },
          h('div', { class: 'sv-rowhead' }, h('b', null, e.engine), h('span', null, num(u.total) + ' ' + UNIT[U]), h('span', null, 'Live path (first three classes): ', h('b', { class: 'fr-num' }, pct(u.live, u.total))), h('span', { class: 'sv-unc' }, 'Rests on heuristics: ', h('b', { class: 'fr-num' }, pct(u.uncertain, u.total)))),
          stack(u, true),
          h('p', { class: 'fr-note' }, e.repo + ', ' + e.version + '. ' + num(e.native.files) + ' native files (' + num(e.native.lines) + ' lines) counted apart as restart.')); }),
        h('p', { class: 'fr-note' }, 'A function is one a module or class body defines; nested functions count with the function around them. Code outside every function counts in lines only, as respawn, which is why the line shares differ from the function shares. A share is printed on a segment when the segment is wide enough; hover or read the table below for the rest.')),
      h('h3', { class: 'au-h3' }, 'Live share as each lab grew'),
      p('The share of ' + UNIT[U] + ' with a live install path at each commit that touched the lab’s folder. The upstream tree and the rules stay fixed, so a step is the commit whose files first provide a mechanism.'),
      h('div', { class: 'fr-multi sv-multi' }, list.map((e) => h('div', { class: 'fr-mini' }, h('h4', null, e.engine),
        h('div', { class: 'fr-mini-read' }, e.history.n + ' lab commits, ' + e.history.first + ' to ' + e.history.last),
        h('div', { class: 'sv-step', 'data-e': e.engine }),
        h('ul', { class: 'au-list sv-steps' }, e.history.steps.filter((st) => st.gained.length).map((st) => h('li', null, st.date + ': ' + gained(st) + (st.pr ? ' (FastRSI PR #' + st.pr + ')' : '') + ', live share ' + pct(st[U], e[U].total))))))),
      h('h3', { class: 'au-h3' }, 'Where the code sits: the 10 largest modules per engine'),
      list.map((e) => [h('p', { class: 'au-p' }, h('b', null, e.engine), ': 10 of ' + e.modules_total + ' top-level modules, by lines of Python.'),
        table(['Module', { h: UNIT[U][0].toUpperCase() + UNIT[U].slice(1), n: 1 }, 'Split by install class', { h: 'Live path', n: 1 }, { h: 'No path yet', n: 1 }, { h: 'Native files', n: 1 }],
          e.modules.map((m) => [h('code', null, m.name), num(m[U].total), h('td', { class: 'sv-cell' }, stack(m[U])), pct(m[U].live, m[U].total), pct(m[U].segs[5], m[U].total), num(m.native_files)]), 'au-w2')]),
      h('p', { class: 'au-src' }, 'Source: static survey tool in core/livelab/survey (FastRSI PR #151, draft), run on ' + F.survey[0].scan_date + '.'));
  }
  function drawSteps() {
    if (!window.d3) return;
    document.querySelectorAll('.sv-step').forEach((box) => {
      const e = F.survey.find((x) => x.engine === box.dataset.e), U = covState.unit, W = box.clientWidth; if (!W) return;
      const H = 190, m = { t: 16, r: 12, b: 24, l: 40 }, n = e.history.n, tot = e[U].total;
      const x = d3.scaleLinear().domain([0, n - 1]).range([m.l, W - m.r]), y = d3.scaleLinear().domain([0, 100]).range([H - m.b, m.t]);
      box.replaceChildren();
      const svg = d3.select(box).append('svg').attr('class', 'fr-svg').attr('viewBox', '0 0 ' + W + ' ' + H).attr('width', W).attr('height', H).attr('role', 'img').attr('aria-label', e.engine + ': live share by lab commit. The list below gives each step.');
      [0, 50, 100].forEach((v) => { svg.append('line').attr('class', 'gl').attr('x1', m.l).attr('x2', W - m.r).attr('y1', y(v)).attr('y2', y(v)); svg.append('text').attr('x', m.l - 6).attr('y', y(v) + 4).attr('text-anchor', 'end').text(v + '%'); });
      svg.append('text').attr('x', m.l).attr('y', H - 6).text('first lab commit'); svg.append('text').attr('x', W - m.r).attr('y', H - 6).attr('text-anchor', 'end').text('commit ' + n);
      const pts = e.history.steps.map((st) => [st.idx, 100 * st[U] / tot]).concat([[n - 1, 100 * e.history.steps[e.history.steps.length - 1][U] / tot]]);
      svg.append('path').attr('class', 'ln fastrsi').attr('d', d3.line().x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveStepAfter)(pts));
      let prev = -1;
      e.history.steps.filter((st) => st.gained.length).forEach((st) => {
        const v = 100 * st[U] / tot, px = x(st.idx), py = y(v), flat = v === prev; prev = v;
        svg.append('circle').attr('cx', px).attr('cy', py).attr('r', 4).attr('fill', flat ? 'var(--color-panel)' : 'var(--fr-fast)').attr('stroke', 'var(--fr-fast)').attr('stroke-width', 2);
        const right = px + 10 + gained(st).length * 6.2 < W - m.r;
        const lw = gained(st).length * 6.2, lx = right ? px + 8 : Math.max(m.l + 4, Math.min(px - 8 - lw, W - m.r - lw));
        svg.append('text').attr('class', 'sv-lb halo').attr('x', lx).attr('y', flat ? py + 16 : py - 8).text(gained(st));
      });
    });
  }

  // ---------------------------------------------------------------- Faults
  function renderFaults() {
    const root = document.querySelector('[data-view="faults"]'), X = F.faults;
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
        block(a.title, earlier(), a.setup + ' ' + X.meta.count_note + ' The detail lines are read from the table’s source ledger, one row per injected commit.', a.rows),
        block(b.title, earlier(), b.setup + '.', b.rows)),
      sec('From the public repo: ' + nRepo + ' rows',
        X.repo_conformance_cases.map((g) => block(g.engine + ': conformance fault cases', repo(), g.setup, g.cases)),
        X.repo_pr_fault_receipts.map((g) => block(g.engine + ': fault receipts', repo(), g.setup + '. Source: ' + srcText(g.source) + '.', g.cases, g.note))),
      sec('Related: drift guard', p(X.related_drift_guard.fact), src(X.related_drift_guard.source)));
  }

  // ---------------------------------------------------------------- Conventions: recorded conformance results
  function renderConformance() {
    const m = F.coverage.measured.find((x) => x.id === 'conformance');
    document.getElementById('conf-results').replaceChildren(table(['Lab', { h: 'Cases passed', n: 1 }], Object.entries(m.results).map(([k, v]) => [k, v == null ? nr() : v]), ''), h('p', { class: 'fr-note' }, m.note), src(m.source));
  }

  // ---------------------------------------------------------------- routing
  function route() {
    let v = location.hash.slice(1); if (!VIEWS.includes(v)) v = 'prs';
    document.querySelectorAll('.au-view').forEach((s) => { s.hidden = s.dataset.view !== v; });
    document.querySelectorAll('#au-tabs a').forEach((a) => { if (a.dataset.tab === v) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    if (v === 'prs') drawChart(); if (v === 'coverage') drawSteps();
  }
  renderPrs(); renderCoverage(); renderFaults(); renderConformance();
  window.addEventListener('hashchange', () => { route(); window.scrollTo(0, 0); });
  route();
})();

(() => {
  'use strict';

  const TABS = ['overview', 'charts', 'tracks', 'cluster', 'matrix', 'activity'];
  const POLL_INTERVAL_MS = 5000;

  const state = {
    facts: null,
    factsError: null,
    live: null,
    mode: 'static', // 'live' | 'static'
    paused: false,
    activeTab: 'overview',
    lastUpdated: null,
    matrixFilter: 'all', // 'all' | 'done' | 'active'
    matrixSearch: '',
    pollTimer: null,
    isFetchingState: false,
  };

  function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function resolveAssetPath(relPath) {
    if (!relPath) return '';
    if (/^(https?:)?\/\//.test(relPath) || relPath.startsWith('/')) {
      return relPath;
    }
    const clean = relPath.replace(/^\.?\//, '');
    const p = window.location.pathname;
    const idx = p.indexOf('/site/dashboard');
    if (idx !== -1) {
      const prefix = p.slice(0, idx);
      return `${prefix}/${clean}`;
    }
    return `/${clean}`;
  }

  async function fetchDeliveryMatrix() {
    const p = window.location.pathname;
    const candidates = [];
    const idx = p.indexOf('/site/dashboard');
    if (idx !== -1) {
      const prefix = p.slice(0, idx);
      candidates.push(`${prefix}/site/facts/delivery-matrix.json`, '../facts/delivery-matrix.json');
    } else {
      candidates.push('/site/facts/delivery-matrix.json', '../facts/delivery-matrix.json', '../../site/facts/delivery-matrix.json');
    }
    const seen = new Set();
    let lastStatus = null;
    for (const url of candidates) {
      if (seen.has(url)) continue;
      seen.add(url);
      try {
        const res = await fetch(url, { cache: 'no-store' });
        if (res.ok) {
          state.facts = await res.json();
          state.factsError = null;
          return true;
        }
        lastStatus = res.status;
      } catch (_) {
        // try next candidate
      }
    }
    state.facts = null;
    state.factsError = `未能加载 delivery-matrix.json（HTTP ${lastStatus || 'network error'}）。请确认 site/facts/delivery-matrix.json 存在且可访问。`;
    return false;
  }

  async function fetchLiveState() {
    if (state.isFetchingState) return;
    state.isFetchingState = true;
    try {
      const res = await fetch('/api/state', { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        if (data && typeof data === 'object' && data.host) {
          state.live = data;
          state.mode = 'live';
          state.lastUpdated = new Date();
          return;
        }
      }
      state.live = null;
      state.mode = 'static';
    } catch (_) {
      // Silent fallback to static mode without console spam
      state.live = null;
      state.mode = 'static';
    } finally {
      state.isFetchingState = false;
      if (!state.lastUpdated) {
        state.lastUpdated = new Date();
      }
    }
  }

  function isDoneStatus(status) {
    const s = String(status || '').toLowerCase();
    return (
      s.includes('done') ||
      s.includes('verified') ||
      s.includes('complete') ||
      s.includes('pass') ||
      s.includes('closed') ||
      s.includes('ready') ||
      s === '已闭环' ||
      s === '已完成' ||
      s === '已验收'
    );
  }

  function isActiveOrPlannedStatus(status) {
    const s = String(status || '').toLowerCase();
    return (
      s.includes('progress') ||
      s.includes('plan') ||
      s.includes('pending') ||
      s.includes('active') ||
      s.includes('wip') ||
      s.includes('推进') ||
      s.includes('规划')
    );
  }

  function statusBadge(status, labelOverride) {
    const raw = String(status || 'unknown');
    const text = labelOverride || raw;
    if (isDoneStatus(raw)) {
      return `<span class="badge badge-emerald">${escapeHtml(text)}</span>`;
    }
    if (isActiveOrPlannedStatus(raw)) {
      return `<span class="badge badge-cobalt">${escapeHtml(text)}</span>`;
    }
    const low = raw.toLowerCase();
    if (low.includes('warn') || low.includes('block') || low.includes('partial')) {
      return `<span class="badge badge-amber">${escapeHtml(text)}</span>`;
    }
    if (low.includes('fail') || low.includes('error')) {
      return `<span class="badge badge-red">${escapeHtml(text)}</span>`;
    }
    return `<span class="badge badge-slate">${escapeHtml(text)}</span>`;
  }

  function renderTickBar(ratio, toneClass) {
    const total = 20;
    const clamped = Math.max(0, Math.min(1, Number.isFinite(ratio) ? ratio : 0));
    const filled = Math.round(clamped * total);
    let html = '<div class="tick-bar" aria-hidden="true">';
    for (let i = 0; i < total; i += 1) {
      const cls = i < filled ? `tick ${toneClass}` : 'tick';
      html += `<span class="${cls}"></span>`;
    }
    html += '</div>';
    return html;
  }

  function renderEvidenceChip(evidence) {
    if (!evidence || (Array.isArray(evidence) && evidence.length === 0)) {
      return '<span class="card-subtitle">—</span>';
    }
    const list = Array.isArray(evidence) ? evidence : [evidence];
    return list
      .map((item) => {
        const text = String(item);
        return `<button type="button" class="evidence-btn" data-copy="${escapeHtml(text)}" title="${escapeHtml(text)}">
          <span class="evidence-path">${escapeHtml(text)}</span>
          <span class="evidence-copy-tag">复制</span>
        </button>`;
      })
      .join(' ');
  }

  function formatTime(d) {
    if (!d) return '—';
    const dt = d instanceof Date ? d : new Date(d);
    if (Number.isNaN(dt.getTime())) return String(d);
    return dt.toLocaleTimeString('zh-CN', { hour12: false });
  }

  function renderTopHeader() {
    const facts = state.facts;
    const live = state.live;

    const repoSlug = document.getElementById('repo-slug');
    if (repoSlug) {
      repoSlug.textContent = (facts && facts.repo) || 'hao-ai-lab/rsibox';
    }

    const modePill = document.getElementById('pill-mode');
    const modeText = document.getElementById('pill-mode-text');
    if (modePill && modeText) {
      if (state.mode === 'live' && live) {
        modePill.className = 'pill is-live';
        modeText.textContent = `Live · ${live.host}`;
      } else {
        modePill.className = 'pill';
        modeText.textContent = 'Static Mode';
      }
    }

    const headText = document.getElementById('pill-head-text');
    if (headText) {
      if (state.mode === 'live' && live && live.git && live.git.head) {
        headText.textContent = `${live.git.head} (${live.git.branch || 'HEAD'})`;
      } else if (facts && facts.generated_at) {
        headText.textContent = String(facts.generated_at).slice(0, 10);
      } else {
        headText.textContent = 'main';
      }
    }

    const actPill = document.getElementById('pill-activity');
    const actText = document.getElementById('pill-activity-text');
    if (actPill && actText) {
      if (state.mode === 'live' && live) {
        const prs = Array.isArray(live.open_prs) ? live.open_prs.length : 0;
        const dirty = live.git && Number(live.git.dirty_count) ? Number(live.git.dirty_count) : 0;
        if (dirty > 0) {
          actPill.className = 'pill is-warn';
          actText.textContent = `${prs} Open PRs · ${dirty} dirty`;
        } else {
          actPill.className = 'pill';
          actText.textContent = `${prs} Open PRs · Clean`;
        }
      } else {
        actPill.className = 'pill';
        actText.textContent = '静态基线视图';
      }
    }

    const dot = document.getElementById('status-dot');
    const updated = document.getElementById('last-updated');
    const pauseBtn = document.getElementById('btn-pause');

    if (dot) {
      if (state.paused) {
        dot.className = 'status-dot is-paused';
      } else if (state.mode === 'live') {
        dot.className = 'status-dot is-live';
      } else {
        dot.className = 'status-dot is-static';
      }
    }
    if (updated) {
      const prefix = state.paused ? '已暂停' : state.mode === 'live' ? 'Live' : 'Static';
      updated.textContent = `${prefix} · ${formatTime(state.lastUpdated)}`;
    }
    if (pauseBtn) {
      pauseBtn.setAttribute('aria-pressed', state.paused ? 'true' : 'false');
      pauseBtn.textContent = state.paused ? '继续' : '暂停';
    }

    const genAt = document.getElementById('facts-generated-at');
    if (genAt) {
      genAt.textContent = facts && facts.generated_at ? `facts: ${facts.generated_at}` : 'facts: unavailable';
    }
  }

  function renderAlertStrip() {
    const strip = document.getElementById('alert-strip');
    const body = document.getElementById('alert-body');
    const action = document.getElementById('alert-action');
    if (!strip || !body || !action) return;

    const facts = state.facts;
    const live = state.live;
    const targets = (facts && Array.isArray(facts.targets)) ? facts.targets : [];

    if (state.mode === 'live' && live) {
      const warnings = [];
      if (live.disk && typeof live.disk.free_gb === 'number' && live.disk.free_gb < 300) {
        warnings.push(`磁盘余量偏低：${live.disk.free_gb} GB (< 300 GB)`);
      }
      if (live.git && live.git.dirty_count > 0) {
        warnings.push(`工作区存在 ${live.git.dirty_count} 个未提交变更文件`);
      }
      if (warnings.length > 0) {
        strip.className = 'alert-strip';
        body.innerHTML = `<strong>主机告警 (${escapeHtml(live.host)})：</strong>${escapeHtml(warnings.join(' · '))}`;
        action.href = live.git && live.git.dirty_count > 0 ? '#activity' : '#cluster';
        action.textContent = live.git && live.git.dirty_count > 0 ? '查看工作区 →' : '查看主机状态 →';
        return;
      }
      strip.className = 'alert-strip is-healthy';
      const targetSummary = targets.length
        ? `当前仍有 ${targets.length} 项未完 Roadmap Target（${targets.map((t) => t.id).join(' / ')}）持续推进中`
        : '主机运行健康，暂无告警';
      body.innerHTML = `<strong>Live 集群健康 (${escapeHtml(live.host)})：</strong>磁盘与工作区状态正常 · ${escapeHtml(targetSummary)}`;
      action.href = '#tracks';
      action.textContent = '查看 Targets →';
      return;
    }

    strip.className = 'alert-strip';
    if (targets.length > 0) {
      const avgPct = Math.round(
        targets.reduce((acc, t) => acc + (Number(t.progress_pct) || 0), 0) / targets.length
      );
      const ids = targets.map((t) => `${t.id} (${t.progress_pct || 0}%)`).join(' · ');
      body.innerHTML = `<strong>Static 基线模式 · ${targets.length} 项未完 Target (平均进度 ${avgPct}%)：</strong>${escapeHtml(ids)}`;
    } else {
      body.innerHTML = '<strong>Static 基线模式：</strong>当前以只读静态模式运行，启动 <code class="chip">site/dashboard/server.py</code> 可解锁 B200 实时 GPU 与 Git 遥测。';
    }
    action.href = '#tracks';
    action.textContent = '查看 Targets →';
  }

  function renderKpis() {
    const grid = document.getElementById('kpi-grid');
    if (!grid) return;

    const facts = state.facts;
    const k = (facts && facts.kpis) || {};
    const targets = (facts && Array.isArray(facts.targets)) ? facts.targets : [];

    const enginesDone = Number(k.engines_done ?? 0);
    const enginesTarget = Number(k.engines_target ?? 10);
    const invDone = Number(k.invariants_done ?? 0);
    const invTotal = Number(k.invariants_total ?? 12);
    const expDone = Number(k.experiments_done ?? 0);
    const expTotal = Number(k.experiments_total ?? 1);
    const tracesVerified = typeof k.traces_verified === 'number'
      ? k.traces_verified.toLocaleString()
      : String(k.traces_verified ?? '0');
    const leasesVerified = Number(k.leases_verified ?? 0);
    const targetsDone = Number(k.targets_done ?? 0);
    const targetsTotal = Number(k.targets_total ?? targets.length);

    const avgTargetPct = targets.length
      ? Math.round(targets.reduce((acc, t) => acc + (Number(t.progress_pct) || 0), 0) / targets.length)
      : 0;

    let clusterValue = `${enginesDone} Engines Verified`;
    let clusterSub = '3× NVIDIA B200 基线已验证';
    let clusterRatio = 0.9;
    let clusterTone = 'is-slate';

    if (state.mode === 'live' && state.live) {
      const gpus = Array.isArray(state.live.gpus) ? state.live.gpus : [];
      if (gpus.length > 0) {
        const active = gpus.filter(
          (g) => Number(g.utilization_gpu) > 5 || Number(g.memory_used_mb) > 2000
        ).length;
        const usedMem = gpus.reduce((s, g) => s + (Number(g.memory_used_mb) || 0), 0);
        const totalMem = gpus.reduce((s, g) => s + (Number(g.memory_total_mb) || 0), 0);
        clusterValue = `${active} of ${gpus.length} 活跃`;
        clusterSub = `${usedMem.toLocaleString()} / ${totalMem.toLocaleString()} MiB VRAM`;
        clusterRatio = gpus.length ? active / gpus.length : 0;
        clusterTone = active > 0 ? 'is-emerald' : 'is-cobalt';
      } else {
        const freeGb = state.live.disk ? state.live.disk.free_gb : '—';
        clusterValue = 'Host Online';
        clusterSub = `无本地 GPU · 磁盘可用 ${freeGb} GB`;
        clusterRatio = 1;
        clusterTone = 'is-cobalt';
      }
    }

    const cards = [
      {
        href: '#matrix',
        title: '引擎交付',
        tag: 'Engines',
        value: `${enginesDone} of ${enginesTarget}`,
        sub: `${Math.round((enginesDone / Math.max(1, enginesTarget)) * 100)}% 引擎适配器达标`,
        ratio: enginesDone / Math.max(1, enginesTarget),
        tone: 'is-cobalt',
      },
      {
        href: '#matrix',
        title: '核心不变量',
        tag: 'Invariants',
        value: `${invDone} of ${invTotal}`,
        sub: '100% 运行时与撤销不变量闭环',
        ratio: invDone / Math.max(1, invTotal),
        tone: 'is-emerald',
      },
      {
        href: '#matrix',
        title: 'B200 实验矩阵',
        tag: 'EXP-A ~ H2',
        value: `${expDone} of ${expTotal}`,
        sub: `${tracesVerified} traces · ${leasesVerified.toLocaleString()} leases`,
        ratio: expDone / Math.max(1, expTotal),
        tone: 'is-emerald',
      },
      {
        href: '#tracks',
        title: '未完 Target',
        tag: 'T1–T5',
        value: `${targetsDone} of ${targetsTotal}`,
        sub: `T1–T5 规划/推进中 · 平均 ${avgTargetPct}%`,
        ratio: avgTargetPct / 100,
        tone: 'is-cobalt',
      },
      {
        href: '#cluster',
        title: 'B200 GPU 集群',
        tag: 'Cluster',
        value: clusterValue,
        sub: clusterSub,
        ratio: clusterRatio,
        tone: clusterTone,
      },
    ];

    grid.innerHTML = cards
      .map(
        (c) => `<a class="kpi-card" href="${c.href}">
          <div>
            <div class="kpi-head">
              <span class="kpi-title">${escapeHtml(c.title)}</span>
              <span class="kpi-tag">${escapeHtml(c.tag)}</span>
            </div>
            <div class="kpi-value">${escapeHtml(c.value)}</div>
            <div class="kpi-sub" title="${escapeHtml(c.sub)}">${escapeHtml(c.sub)}</div>
          </div>
          ${renderTickBar(c.ratio, c.tone)}
        </a>`
      )
      .join('');
  }

  function renderClusterSummaryContent(compact) {
    const live = state.live;
    if (state.mode === 'live' && live) {
      const disk = live.disk || {};
      const sys = live.system || {};
      const gpus = Array.isArray(live.gpus) ? live.gpus : [];
      const diskWarn = typeof disk.free_gb === 'number' && disk.free_gb < 300;
      const diskFillClass = diskWarn ? 'progress-fill is-amber' : 'progress-fill is-emerald';

      let html = `
        <div class="grid-3" style="margin-bottom: 12px;">
          <div class="gpu-card">
            <div class="metric-row"><span class="metric-label">主机</span><span class="metric-val">${escapeHtml(live.host)}</span></div>
            <div class="metric-row"><span class="metric-label">采样时间</span><span class="metric-val">${escapeHtml(formatTime(live.timestamp))}</span></div>
          </div>
          <div class="gpu-card">
            <div class="metric-row"><span class="metric-label">磁盘可用</span><span class="metric-val">${disk.free_gb ?? '—'} / ${disk.total_gb ?? '—'} GB</span></div>
            <div class="progress-track"><div class="${diskFillClass}" style="width: ${Math.min(100, Number(disk.used_pct) || 0)}%"></div></div>
          </div>
          <div class="gpu-card">
            <div class="metric-row"><span class="metric-label">CPU 负载 (1m/5m)</span><span class="metric-val">${escapeHtml(sys.load_1m ?? '—')} / ${escapeHtml(sys.load_5m ?? '—')}</span></div>
            <div class="metric-row"><span class="metric-label">内存可用</span><span class="metric-val">${sys.mem_available_gb ?? '—'} / ${sys.mem_total_gb ?? '—'} GB</span></div>
          </div>
        </div>
      `;

      if (gpus.length > 0) {
        html += `<div class="gpu-grid">${gpus
          .map((g) => {
            const pct = g.memory_total_mb
              ? Math.round((Number(g.memory_used_mb) / Number(g.memory_total_mb)) * 100)
              : 0;
            const util = Number(g.utilization_gpu) || 0;
            return `<div class="gpu-card">
              <div class="metric-row">
                <strong>GPU ${escapeHtml(g.index)} · ${escapeHtml(g.name)}</strong>
                <span class="badge ${util > 5 ? 'badge-emerald' : 'badge-slate'}">${util}% Util</span>
              </div>
              <div class="metric-row">
                <span class="metric-label">VRAM</span>
                <span class="metric-val">${Number(g.memory_used_mb).toLocaleString()} / ${Number(g.memory_total_mb).toLocaleString()} MiB (${pct}%)</span>
              </div>
              <div class="progress-track"><div class="progress-fill ${pct > 85 ? 'is-amber' : 'is-cobalt'}" style="width: ${pct}%"></div></div>
            </div>`;
          })
          .join('')}</div>`;
      } else {
        html += `<div class="empty-state">当前主机未检测到 NVIDIA GPU（<code>nvidia-smi</code> 不可用或返回空）。在 B200 节点运行 <code>server.py</code> 后将自动显示每卡 VRAM 与利用率。</div>`;
      }
      return html;
    }

    return `
      <div class="grid-2">
        <div class="gpu-card">
          <div class="metric-row"><strong>已验证 B200 硬件基线</strong><span class="badge badge-emerald">Static Verified</span></div>
          <div class="metric-row"><span class="metric-label">集群拓扑</span><span class="metric-val">3× NVIDIA B200 (183,359 MiB/GPU)</span></div>
          <div class="metric-row"><span class="metric-label">热态显存占用</span><span class="metric-val">~93.3 GiB (95,494–95,626 MiB)</span></div>
          <div class="metric-row"><span class="metric-label">基线归还水位</span><span class="metric-val">18,894 MiB · 0% Util</span></div>
        </div>
        ${
          compact
            ? `<div class="gpu-card">
                <div class="metric-row"><strong>开启 Live GPU 遥测</strong><span class="badge badge-slate">Static Mode</span></div>
                <p class="card-subtitle" style="margin:0;">在 B200 节点执行以下命令启动只读看板服务，即可每 5 秒轮询 <code class="chip">/api/state</code> 获取实时 <code class="chip">nvidia-smi</code>、磁盘与 Git 状态：</p>
                <code class="chip">python3 site/dashboard/server.py --port 8765</code>
              </div>`
            : `<div class="gpu-card">
                <div class="metric-row"><strong>开启 Live GPU 遥测</strong><span class="badge badge-slate">Static Mode</span></div>
                <p class="card-subtitle" style="margin:0;">当前处于静态文件模式（无法访问 <code class="chip">/api/state</code>）。请在 B200 主机仓库根目录运行：</p>
                <code class="chip">python3 site/dashboard/server.py --host 127.0.0.1 --port 8765</code>
                <p class="card-subtitle" style="margin:0;">服务启动后本页将自动切入 Live 模式并展示每张 B200 的实时 VRAM 进度条、CPU/RAM/磁盘空间和工作树状态。</p>
              </div>`
        }
      </div>
    `;
  }

  function renderOverview() {
    const el = document.getElementById('panel-overview');
    if (!el) return;
    const facts = state.facts;
    if (!facts) {
      el.innerHTML = `<div class="empty-state">等待交付事实数据 <code class="chip">delivery-matrix.json</code>…</div>`;
      return;
    }

    const targets = Array.isArray(facts.targets) ? facts.targets : [];
    const engines = Array.isArray(facts.engines) ? facts.engines : [];
    const invariants = Array.isArray(facts.invariants) ? facts.invariants : [];

    const targetsHtml = targets.length
      ? targets
          .map(
            (t) => `<a href="#tracks" class="mini-tile">
              <div class="mini-tile-top">
                <span class="mini-tile-name"><span class="chip">${escapeHtml(t.id)}</span> ${escapeHtml(t.title)}</span>
                ${statusBadge(t.status)}
              </div>
              <div class="track-progress-row">
                <div class="progress-track"><div class="progress-fill" style="width:${Math.min(100, Number(t.progress_pct) || 0)}%"></div></div>
                <span class="track-pct">${Number(t.progress_pct) || 0}%</span>
              </div>
              <div class="mini-tile-meta">
                <span>负责人/轨道: ${escapeHtml(t.owner_track || '—')}</span>
              </div>
            </a>`
          )
          .join('')
      : '<div class="empty-state">暂无未完 Target 记录</div>';

    const enginesHtml = engines
      .map(
        (e) => `<a href="#matrix" class="mini-tile">
          <div class="mini-tile-top">
            <span class="mini-tile-name">${escapeHtml(e.name)}</span>
            ${statusBadge(e.status)}
          </div>
          <div class="mini-tile-meta">
            <span>${escapeHtml(e.category || 'Engine')}</span>
            <span class="mono">${escapeHtml(e.speedup || '—')}</span>
          </div>
          <div class="mini-tile-meta">
            <span class="mono">${escapeHtml(e.conformance || '—')}</span>
          </div>
        </a>`
      )
      .join('');

    const invariantsHtml = invariants
      .map(
        (inv) => `<a href="#matrix" class="inv-pill" title="${escapeHtml(inv.detail || inv.title)}">
          <span class="inv-pill-label"><span class="mono">${escapeHtml(inv.id)}</span> · ${escapeHtml(inv.title)}</span>
          ${statusBadge(inv.status, isDoneStatus(inv.status) ? '已闭环' : null)}
        </a>`
      )
      .join('');

    el.innerHTML = `
      <div class="section-stack">
        <div class="grid-2">
          <div class="card">
            <div class="card-header">
              <h2 class="card-title">B200 集群与主机状态</h2>
              <a class="card-subtitle" href="#cluster">详情 →</a>
            </div>
            ${renderClusterSummaryContent(true)}
          </div>

          <div class="card">
            <div class="card-header">
              <h2 class="card-title">T1–T5 未完目标概览</h2>
              <a class="card-subtitle" href="#tracks">展开全部 →</a>
            </div>
            <div class="section-stack" style="gap:8px;">
              ${targetsHtml}
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h2 class="card-title">9 引擎适配器总览</h2>
            <a class="card-subtitle" href="#matrix">进入验收矩阵 →</a>
          </div>
          <div class="engine-mini-grid">${enginesHtml}</div>
        </div>

        <div class="card">
          <div class="card-header">
            <h2 class="card-title">12 项核心不变量闭环状态</h2>
            <a class="card-subtitle" href="#matrix">查看详细校验 →</a>
          </div>
          <div class="invariant-mini-grid">${invariantsHtml}</div>
        </div>

        ${renderActivitySummaryCard()}
      </div>
    `;
  }

  function renderActivitySummaryCard() {
    const live = state.live;
    if (state.mode === 'live' && live && live.git) {
      const commits = Array.isArray(live.git.recent_commits) ? live.git.recent_commits.slice(0, 5) : [];
      const prs = Array.isArray(live.open_prs) ? live.open_prs.slice(0, 5) : [];
      return `
        <div class="grid-2">
          <div class="card">
            <div class="card-header">
              <h2 class="card-title">最近提交 (main / HEAD)</h2>
              <a class="card-subtitle" href="#activity">全部 →</a>
            </div>
            ${
              commits.length
                ? `<div class="list-stack">${commits
                    .map(
                      (c) => `<div class="list-row">
                        <div class="list-main">
                          <span class="chip">${escapeHtml(c.sha)}</span>
                          <span class="list-title">${escapeHtml(c.subject)}</span>
                        </div>
                        <span class="list-meta">${escapeHtml(c.date)}</span>
                      </div>`
                    )
                    .join('')}</div>`
                : '<div class="empty-state">暂无提交记录</div>'
            }
          </div>

          <div class="card">
            <div class="card-header">
              <h2 class="card-title">活跃 Open PRs</h2>
              <a class="card-subtitle" href="#activity">全部 →</a>
            </div>
            ${
              prs.length
                ? `<div class="list-stack">${prs
                    .map(
                      (pr) => `<div class="list-row">
                        <div class="list-main">
                          <span class="chip">#${escapeHtml(pr.number)}</span>
                          <span class="list-title">${escapeHtml(pr.title)}</span>
                        </div>
                        <span class="list-meta mono">${escapeHtml(pr.headRefName || '')}</span>
                      </div>`
                    )
                    .join('')}</div>`
                : '<div class="empty-state">当前无 Open PR（或 gh CLI 离线）</div>'
            }
          </div>
        </div>
      `;
    }

    const exps = (state.facts && Array.isArray(state.facts.experiments)) ? state.facts.experiments : [];
    const recentExps = exps.slice(-4);
    return `
      <div class="card">
        <div class="card-header">
          <h2 class="card-title">近期里程碑与最新动态</h2>
          <a class="card-subtitle" href="#activity">Activity 详情 →</a>
        </div>
        <div class="list-stack">
          ${
            recentExps.length
              ? recentExps
                  .map(
                    (e) => `<div class="list-row">
                      <div class="list-main">
                        <span class="chip">${escapeHtml(e.id)}</span>
                        <span class="list-title">${escapeHtml(e.title)}</span>
                      </div>
                      ${statusBadge(e.status)}
                    </div>`
                  )
                  .join('')
              : '<div class="empty-state">Static 模式：启动 server.py 后可实时拉取 Git 主干提交与 Open PRs。</div>'
          }
        </div>
      </div>
    `;
  }

  function renderCharts() {
    const el = document.getElementById('panel-charts');
    if (!el) return;
    const charts = (state.facts && Array.isArray(state.facts.charts)) ? state.facts.charts : [];
    if (!charts.length) {
      el.innerHTML = '<div class="empty-state">delivery-matrix.json 中暂无图表定义。</div>';
      return;
    }

    el.innerHTML = `
      <div class="charts-grid">
        ${charts
          .map((c) => {
            const src = resolveAssetPath(c.path);
            return `<article class="card chart-card">
              <div class="card-header" style="margin-bottom:0;">
                <h2 class="card-title">${escapeHtml(c.title)}</h2>
                <span class="chip">${escapeHtml(c.id)}</span>
              </div>
              <div class="chart-frame">
                <a href="${escapeHtml(src)}" target="_blank" rel="noopener" title="在新标签页打开原图">
                  <img src="${escapeHtml(src)}" alt="${escapeHtml(c.title)}" loading="lazy">
                </a>
              </div>
              <p class="chart-caption">${escapeHtml(c.caption || '')}</p>
              <div>${renderEvidenceChip(c.path)}</div>
            </article>`;
          })
          .join('')}
      </div>
    `;
  }

  function renderChecklistItem(item) {
    if (typeof item === 'string') {
      return `<li><span class="check-dot" aria-hidden="true">•</span><span>${escapeHtml(item)}</span></li>`;
    }
    if (item && typeof item === 'object') {
      const done = Boolean(item.done) || isDoneStatus(item.status);
      const label = item.title || item.label || item.text || JSON.stringify(item);
      return `<li><span class="check-dot ${done ? 'is-done' : ''}" aria-hidden="true">${done ? '✓' : ''}</span><span>${escapeHtml(label)}</span></li>`;
    }
    return '';
  }

  function renderTracks() {
    const el = document.getElementById('panel-tracks');
    if (!el) return;
    const targets = (state.facts && Array.isArray(state.facts.targets)) ? state.facts.targets : [];

    const tracksHtml = targets.length
      ? targets
          .map((t) => {
            const pct = Math.min(100, Math.max(0, Number(t.progress_pct) || 0));
            const items = Array.isArray(t.items) ? t.items : [];
            return `<article class="card track-card">
              <div class="card-header" style="margin-bottom:0;">
                <div>
                  <span class="chip">${escapeHtml(t.id)}</span>
                  <strong style="margin-left:6px;font-size:0.95rem;">${escapeHtml(t.title)}</strong>
                </div>
                <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                  ${t.owner_track ? `<span class="badge badge-slate">${escapeHtml(t.owner_track)}</span>` : ''}
                  ${statusBadge(t.status)}
                </div>
              </div>
              <div class="track-progress-row">
                <div class="progress-track"><div class="progress-fill ${pct >= 100 ? 'is-emerald' : ''}" style="width:${pct}%"></div></div>
                <span class="track-pct">${pct}%</span>
              </div>
              <p class="chart-caption">${escapeHtml(t.summary || '')}</p>
              ${items.length ? `<ul class="checklist">${items.map(renderChecklistItem).join('')}</ul>` : ''}
              <div>${renderEvidenceChip(t.evidence)}</div>
            </article>`;
          })
          .join('')
      : '<div class="empty-state">暂无 Target 追踪项。</div>';

    let liveExtra = '';
    if (state.mode === 'live' && state.live) {
      const worktrees = (state.live.git && Array.isArray(state.live.git.worktrees)) ? state.live.git.worktrees : [];
      const prs = Array.isArray(state.live.open_prs) ? state.live.open_prs : [];
      liveExtra = `
        <div class="grid-2">
          <div class="card">
            <div class="card-header">
              <h2 class="card-title">本地 Git Worktrees</h2>
              <span class="badge badge-slate">${worktrees.length}</span>
            </div>
            ${
              worktrees.length
                ? `<div class="list-stack">${worktrees
                    .map(
                      (w) => `<div class="list-row">
                        <div class="list-main">
                          <span class="chip">${escapeHtml(w.branch || 'detached')}</span>
                          <span class="list-title mono">${escapeHtml(w.name || w.path)}</span>
                        </div>
                        <span class="list-meta mono">${escapeHtml(w.head || '')}</span>
                      </div>`
                    )
                    .join('')}</div>`
                : '<div class="empty-state">无 worktree 记录</div>'
            }
          </div>

          <div class="card">
            <div class="card-header">
              <h2 class="card-title">Open PRs (Live)</h2>
              <span class="badge badge-cobalt">${prs.length}</span>
            </div>
            ${
              prs.length
                ? `<div class="list-stack">${prs
                    .map(
                      (pr) => `<div class="list-row">
                        <div class="list-main">
                          <span class="chip">#${escapeHtml(pr.number)}</span>
                          <span class="list-title">${escapeHtml(pr.title)}</span>
                        </div>
                        <span class="list-meta mono">${escapeHtml(pr.headRefName || '')}</span>
                      </div>`
                    )
                    .join('')}</div>`
                : '<div class="empty-state">暂无 Open PR</div>'
            }
          </div>
        </div>
      `;
    }

    el.innerHTML = `<div class="section-stack">${tracksHtml}${liveExtra}</div>`;
  }

  function renderCluster() {
    const el = document.getElementById('panel-cluster');
    if (!el) return;

    const live = state.live;
    let extra = '';
    if (state.mode === 'live' && live && live.git) {
      const worktrees = Array.isArray(live.git.worktrees) ? live.git.worktrees : [];
      extra = `
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">节点 Git 工作树状态</h2>
            <span class="chip">${escapeHtml(live.git.branch || 'HEAD')} · ${escapeHtml(live.git.head || '')}</span>
          </div>
          ${
            worktrees.length
              ? `<div class="list-stack">${worktrees
                  .map(
                    (w) => `<div class="list-row">
                      <div class="list-main">
                        <span class="chip">${escapeHtml(w.branch || 'detached')}</span>
                        <span class="mono">${escapeHtml(w.name || w.path)}</span>
                      </div>
                      <span class="list-meta mono">${escapeHtml(w.head || '')}</span>
                    </div>`
                  )
                  .join('')}</div>`
              : '<div class="empty-state">无额外 worktree</div>'
          }
        </div>
      `;
    }

    el.innerHTML = `
      <div class="section-stack">
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">B200 集群与主机遥测</h2>
            <span class="badge ${state.mode === 'live' ? 'badge-emerald' : 'badge-slate'}">${state.mode === 'live' ? 'Live /api/state' : 'Static Baseline'}</span>
          </div>
          ${renderClusterSummaryContent(false)}
        </div>
        ${extra}
      </div>
    `;
  }

  function matchesFilter(item, fields) {
    if (state.matrixFilter === 'done' && !isDoneStatus(item.status)) return false;
    if (state.matrixFilter === 'active' && isDoneStatus(item.status)) return false;
    const q = state.matrixSearch.trim().toLowerCase();
    if (!q) return true;
    return fields.some((f) => String(f || '').toLowerCase().includes(q));
  }

  function renderMatrix() {
    const el = document.getElementById('panel-matrix');
    if (!el) return;
    const facts = state.facts;
    if (!facts) {
      el.innerHTML = '<div class="empty-state">等待 delivery-matrix.json 数据…</div>';
      return;
    }

    const engines = (Array.isArray(facts.engines) ? facts.engines : []).filter((e) =>
      matchesFilter(e, [e.id, e.name, e.category, e.status, e.conformance, e.tiers, e.speedup, e.summary, e.evidence])
    );
    const invariants = (Array.isArray(facts.invariants) ? facts.invariants : []).filter((inv) =>
      matchesFilter(inv, [inv.id, inv.title, inv.status, inv.detail])
    );
    const experiments = (Array.isArray(facts.experiments) ? facts.experiments : []).filter((exp) =>
      matchesFilter(exp, [exp.id, exp.title, exp.status, exp.result, exp.traces, exp.evidence])
    );

    const engineRows = engines
      .map((e) => {
        const tierList = Array.isArray(e.tiers)
          ? e.tiers
          : String(e.tiers || '—').split(' + ').map((s) => s.trim()).filter(Boolean);
        const tiers = tierList.map((t) => `<span class="chip">${escapeHtml(t)}</span>`).join('');
        return `<tr>
          <th><div>${escapeHtml(e.name)}</div><div class="card-subtitle mono">${escapeHtml(e.id)}</div></th>
          <td data-label="Category"><span class="badge badge-slate">${escapeHtml(e.category || '—')}</span></td>
          <td data-label="状态">${statusBadge(e.status)}</td>
          <td data-label="Conformance"><span class="mono">${escapeHtml(e.conformance || '—')}</span></td>
          <td data-label="Capabilities / Tiers">
            <div class="tier-list">${tiers}</div>
            ${e.summary ? `<div class="cell-summary" style="margin-top:4px;font-size:0.75rem;">${escapeHtml(e.summary)}</div>` : ''}
          </td>
          <td data-label="加速比"><strong class="mono">${escapeHtml(e.speedup || '—')}</strong></td>
          <td data-label="证据路径">${renderEvidenceChip(e.evidence)}</td>
        </tr>`;
      })
      .join('');

    const invCards = invariants
      .map(
        (inv) => `<div class="invariant-card">
          <div class="invariant-top">
            <span class="invariant-title"><span class="chip">${escapeHtml(inv.id)}</span> ${escapeHtml(inv.title)}</span>
            ${statusBadge(inv.status, isDoneStatus(inv.status) ? '已闭环' : null)}
          </div>
          <p class="invariant-detail">${escapeHtml(inv.detail || '')}</p>
        </div>`
      )
      .join('');

    const expRows = experiments
      .map(
        (exp) => `<tr>
          <th><span class="chip">${escapeHtml(exp.id)}</span></th>
          <td data-label="实验项"><strong>${escapeHtml(exp.title)}</strong></td>
          <td data-label="状态">${statusBadge(exp.status)}</td>
          <td data-label="实测结论 & Traces">
            <div class="cell-summary">${escapeHtml(exp.result || '—')}</div>
            ${exp.traces ? `<div style="margin-top:4px;"><span class="chip">${escapeHtml(exp.traces)}</span></div>` : ''}
          </td>
          <td data-label="证据路径">${renderEvidenceChip(exp.evidence)}</td>
        </tr>`
      )
      .join('');

    el.innerHTML = `
      <div class="section-stack">
        <div class="card">
          <div class="matrix-toolbar">
            <div class="filter-pills" role="group" aria-label="状态筛选">
              <button type="button" class="filter-pill" data-filter="all" aria-pressed="${state.matrixFilter === 'all'}">全部</button>
              <button type="button" class="filter-pill" data-filter="done" aria-pressed="${state.matrixFilter === 'done'}">已验收 / Completed</button>
              <button type="button" class="filter-pill" data-filter="active" aria-pressed="${state.matrixFilter === 'active'}">推进中 / Planned</button>
            </div>
            <input type="search" class="search-input" id="matrix-search" placeholder="筛选引擎、不变量、实验 ID 或路径…" value="${escapeHtml(state.matrixSearch)}" aria-label="筛选验收矩阵">
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h2 class="card-title" id="tbl-engines-cap">9 引擎能力验收表</h2>
            <span class="badge badge-cobalt">${engines.length} 项</span>
          </div>
          ${
            engines.length
              ? `<div tabindex="0" role="region" aria-labelledby="tbl-engines-cap" class="table-wrapper responsive-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Engine</th>
                        <th>Category</th>
                        <th>状态</th>
                        <th>Conformance</th>
                        <th>Capabilities / Tiers</th>
                        <th>加速比</th>
                        <th>证据路径</th>
                      </tr>
                    </thead>
                    <tbody>${engineRows}</tbody>
                  </table>
                </div>`
              : '<div class="empty-state">无匹配引擎记录</div>'
          }
        </div>

        <div class="card">
          <div class="card-header">
            <h2 class="card-title">12 项核心不变量校验</h2>
            <span class="badge badge-emerald">${invariants.length} 项</span>
          </div>
          ${invariants.length ? `<div class="invariants-grid">${invCards}</div>` : '<div class="empty-state">无匹配不变量记录</div>'}
        </div>

        <div class="card">
          <div class="card-header">
            <h2 class="card-title" id="tbl-exp-cap">多 Agent 与系统实验验收 (EXP-A ~ EXP-H2)</h2>
            <span class="badge badge-emerald">${experiments.length} 项</span>
          </div>
          ${
            experiments.length
              ? `<div tabindex="0" role="region" aria-labelledby="tbl-exp-cap" class="table-wrapper responsive-table">
                  <table>
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>实验项</th>
                        <th>状态</th>
                        <th>实测结论 &amp; Traces</th>
                        <th>证据路径</th>
                      </tr>
                    </thead>
                    <tbody>${expRows}</tbody>
                  </table>
                </div>`
              : '<div class="empty-state">无匹配实验记录</div>'
          }
        </div>
      </div>
    `;

    const searchInput = document.getElementById('matrix-search');
    if (searchInput) {
      searchInput.addEventListener('input', (ev) => {
        const pos = ev.target.selectionStart;
        state.matrixSearch = ev.target.value;
        renderMatrix();
        const nextInput = document.getElementById('matrix-search');
        if (nextInput) {
          nextInput.focus();
          if (typeof pos === 'number') {
            nextInput.setSelectionRange(pos, pos);
          }
        }
      });
    }

    el.querySelectorAll('.filter-pill').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.matrixFilter = btn.getAttribute('data-filter') || 'all';
        renderMatrix();
      });
    });
  }

  function renderActivity() {
    const el = document.getElementById('panel-activity');
    if (!el) return;

    const live = state.live;
    if (state.mode !== 'live' || !live) {
      el.innerHTML = `
        <div class="section-stack">
          <div class="card">
            <div class="card-header">
              <h2 class="card-title">提交与动态 (Static Mode)</h2>
              <span class="badge badge-slate">Live Polling Inactive</span>
            </div>
            <p class="chart-caption">当前运行在纯静态模式，未能连接 <code class="chip">/api/state</code>。在仓库根目录运行 <code class="chip">python3 site/dashboard/server.py --port 8765</code> 后，本页每 5 秒自动同步：</p>
            <ul class="checklist" style="margin-top:10px;">
              <li><span class="check-dot" aria-hidden="true">•</span><span>主干分支近期 Git 提交记录（SHA、作者、相对时间、提交信息）</span></li>
              <li><span class="check-dot" aria-hidden="true">•</span><span>GitHub Open PR 列表（通过服务端 30 秒缓存的 <code class="chip">gh pr list</code>）</span></li>
              <li><span class="check-dot" aria-hidden="true">•</span><span>本地工作区未提交文件状态与分支隔离情况</span></li>
            </ul>
          </div>
          ${renderActivitySummaryCard()}
        </div>
      `;
      return;
    }

    const git = live.git || {};
    const commits = Array.isArray(git.recent_commits) ? git.recent_commits : [];
    const prs = Array.isArray(live.open_prs) ? live.open_prs : [];
    const dirtyFiles = Array.isArray(git.dirty_files) ? git.dirty_files : [];

    el.innerHTML = `
      <div class="section-stack">
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">当前工作区状态 (${escapeHtml(git.branch || 'HEAD')})</h2>
            <span class="badge ${git.dirty_count > 0 ? 'badge-amber' : 'badge-emerald'}">${git.dirty_count > 0 ? `${git.dirty_count} 个未提交文件` : 'Working Tree Clean'}</span>
          </div>
          <div class="metric-row" style="margin-bottom:8px;">
            <span class="metric-label">HEAD Commit</span>
            <span class="chip">${escapeHtml(git.head_full || git.head || '—')}</span>
          </div>
          ${
            dirtyFiles.length
              ? `<div class="list-stack">${dirtyFiles
                  .map(
                    (f) => `<div class="list-row">
                      <div class="list-main">
                        <span class="chip">${escapeHtml(f.status)}</span>
                        <span class="mono">${escapeHtml(f.path)}</span>
                      </div>
                    </div>`
                  )
                  .join('')}</div>`
              : '<div class="empty-state">工作区干净，无未提交改动。</div>'
          }
        </div>

        <div class="grid-2">
          <div class="card">
            <div class="card-header">
              <h2 class="card-title">最近提交记录</h2>
              <span class="badge badge-slate">${commits.length}</span>
            </div>
            ${
              commits.length
                ? `<div class="list-stack">${commits
                    .map(
                      (c) => `<div class="list-row">
                        <div class="list-main">
                          <span class="chip">${escapeHtml(c.sha)}</span>
                          <span class="list-title">${escapeHtml(c.subject)}</span>
                        </div>
                        <span class="list-meta">${escapeHtml(c.author)} · ${escapeHtml(c.date)}</span>
                      </div>`
                    )
                    .join('')}</div>`
                : '<div class="empty-state">暂无提交记录</div>'
            }
          </div>

          <div class="card">
            <div class="card-header">
              <h2 class="card-title">Open Pull Requests</h2>
              <span class="badge badge-cobalt">${prs.length}</span>
            </div>
            ${
              prs.length
                ? `<div class="list-stack">${prs
                    .map(
                      (pr) => `<div class="list-row">
                        <div class="list-main">
                          <span class="chip">#${escapeHtml(pr.number)}</span>
                          ${
                            pr.url
                              ? `<a class="list-title" href="${escapeHtml(pr.url)}" target="_blank" rel="noopener" style="text-decoration:underline;">${escapeHtml(pr.title)}</a>`
                              : `<span class="list-title">${escapeHtml(pr.title)}</span>`
                          }
                        </div>
                        <span class="list-meta mono">${escapeHtml(pr.headRefName || '')}</span>
                      </div>`
                    )
                    .join('')}</div>`
                : '<div class="empty-state">当前无 Open PR（或未登录 gh CLI）</div>'
            }
          </div>
        </div>
      </div>
    `;
  }

  function renderActivePanel() {
    const tab = state.activeTab;
    if (tab === 'overview') renderOverview();
    else if (tab === 'charts') renderCharts();
    else if (tab === 'tracks') renderTracks();
    else if (tab === 'cluster') renderCluster();
    else if (tab === 'matrix') renderMatrix();
    else if (tab === 'activity') renderActivity();
  }

  function renderAll() {
    const errEl = document.getElementById('facts-error-banner');
    if (errEl) {
      if (state.factsError) {
        errEl.hidden = false;
        errEl.textContent = state.factsError;
      } else {
        errEl.hidden = true;
      }
    }
    renderTopHeader();
    renderAlertStrip();
    renderKpis();
    renderActivePanel();
  }

  function setActiveTab(tabName) {
    const target = TABS.includes(tabName) ? tabName : 'overview';
    state.activeTab = target;

    document.querySelectorAll('#tab-nav .tab-link').forEach((link) => {
      const isCurrent = link.getAttribute('data-tab') === target;
      if (isCurrent) {
        link.setAttribute('aria-current', 'page');
        link.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      } else {
        link.removeAttribute('aria-current');
      }
    });

    TABS.forEach((name) => {
      const panel = document.getElementById(`panel-${name}`);
      if (panel) {
        panel.hidden = name !== target;
      }
    });

    renderActivePanel();
  }

  function syncFromHash() {
    const raw = (window.location.hash || '').replace(/^#/, '').trim();
    setActiveTab(TABS.includes(raw) ? raw : 'overview');
  }

  async function copyText(text, btn) {
    let ok = false;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        ok = true;
      }
    } catch (_) {
      ok = false;
    }
    if (!ok) {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try {
        ok = document.execCommand('copy');
      } catch (_) {
        ok = false;
      }
      document.body.removeChild(ta);
    }

    if (ok && btn) {
      const tag = btn.querySelector('.evidence-copy-tag');
      btn.classList.add('is-copied');
      if (tag) tag.textContent = '已复制';
      window.setTimeout(() => {
        btn.classList.remove('is-copied');
        if (tag) tag.textContent = '复制';
      }, 1400);
    }
  }

  function startPolling() {
    if (state.pollTimer) {
      window.clearInterval(state.pollTimer);
    }
    state.pollTimer = window.setInterval(async () => {
      if (state.paused) return;
      await fetchLiveState();
      renderAll();
    }, POLL_INTERVAL_MS);
  }

  async function init() {
    document.addEventListener('click', (ev) => {
      const copyBtn = ev.target.closest('.evidence-btn');
      if (copyBtn) {
        const text = copyBtn.getAttribute('data-copy');
        if (text) copyText(text, copyBtn);
      }
    });

    const pauseBtn = document.getElementById('btn-pause');
    if (pauseBtn) {
      pauseBtn.addEventListener('click', () => {
        state.paused = !state.paused;
        renderTopHeader();
      });
    }

    const refreshBtn = document.getElementById('btn-refresh');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', async () => {
        await Promise.all([fetchDeliveryMatrix(), fetchLiveState()]);
        renderAll();
      });
    }

    window.addEventListener('hashchange', syncFromHash);

    await Promise.all([fetchDeliveryMatrix(), fetchLiveState()]);
    syncFromHash();
    renderAll();
    startPolling();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

// Shared helpers for the FastRSI sections. Each section script works alone if this is missing a piece.
window.FR = window.FR || {};
(function (FR) {
  FR.reduced = () => !window.matchMedia || window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  FR.num = (v) => (v === '' || v == null ? null : Number(v));
  FR.ok = (id) => { const r = dash.data(id); return r.status === 'ok' ? r : null; };
  FR.arms = { codex: 'Baseline (restart per check)', fastrsi: 'FastRSI (live checks)' };
  FR.short = { codex: 'Baseline', fastrsi: 'FastRSI' };
  // Build a data-where value: every name and value percent-encoded, joined by &.
  FR.where = (o) => Object.entries(o).map(([k, v]) => encodeURIComponent(k) + '=' + encodeURIComponent(String(v))).join('&');
  FR.mark = (node, source, field, where) => {
    node.setAttribute('data-source', source);
    node.setAttribute('data-field', field);
    if (where) node.setAttribute('data-where', FR.where(where));
    return node;
  };
  FR.svg = (tag, attrs, parent) => {
    const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    if (attrs) for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };
  FR.el = (tag, cls, text, parent) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  };
  // Run fn once when el first scrolls into view.
  FR.onceVisible = (el, fn, threshold) => {
    if (!('IntersectionObserver' in window)) { fn(); return () => {}; }
    const io = new IntersectionObserver((es) => {
      if (es.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
    }, { root: document, threshold: threshold || 0.3 });
    io.observe(el);
    return () => io.disconnect();
  };
  // Tooltip kept inside a relative box.
  FR.tip = (box) => {
    const t = FR.el('div', 'fr-tip', null, box);
    t.hidden = true;
    t.setAttribute('role', 'status');
    return {
      node: t,
      show(x, y, build) {
        t.replaceChildren();
        build(t);
        t.hidden = false;
        const bw = box.clientWidth, tw = t.offsetWidth, th = t.offsetHeight;
        let left = x + 14; if (left + tw > bw) left = Math.max(0, x - tw - 14); if (left + tw > bw) left = Math.max(0, bw - tw);
        let top = y - th - 10; if (top < 0) top = y + 14;
        t.style.left = left + 'px'; t.style.top = top + 'px';
      },
      hide() { t.hidden = true; },
    };
  };
  // Width of text as drawn (0 until in the page).
  FR.textW = (svg, str, cls) => {
    const n = FR.svg('text', { class: cls || '', x: -9999, y: -9999 }, svg);
    n.textContent = str;
    const w = n.getComputedTextLength ? n.getComputedTextLength() : 0;
    n.remove();
    return w || str.length * 7;
  };
})(window.FR);

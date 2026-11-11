// A small stand-in for the dashboard's `dash` API, so its section scripts run unchanged on a plain page.
(function () {
  const calcs = {}, draws = {}, ons = [], made = new Map(), parsed = {};
  window.dash = {
    data(id) {
      if (RAW[id] != null) { const d = parsed[id] || (parsed[id] = d3.csvParse(RAW[id])); return { status: 'ok', data: d, columns: d.columns }; }
      if (calcs[id]) { const c = calcs[id]; if (c.v === undefined) c.v = c.fn(...c.inputs.map((i) => dash.data(i).data)); return { status: 'ok', data: c.v }; }
      return { status: 'missing', data: [] };
    },
    calc(id, o) { calcs[id] = { ...o }; }, loader() {}, onData(fn) { ons.push(fn); },
    draw(name, f) { if (f) { draws[name] = typeof f === 'function' ? { update: (el) => f(el) } : f; return; } run(name); },
    params() { return {}; }, setParams() {}, setLink() {},
  };
  function run(name) {
    const d = draws[name]; if (!d) return;
    document.querySelectorAll('[data-draw="' + name + '"]').forEach((el) => {
      if (!made.has(el)) { el.replaceChildren(); made.set(el, d.mount ? d.mount(el) : undefined); }
      d.update(el, made.get(el));
    });
  }
  const all = () => Object.keys(draws).forEach(run);
  window.addEventListener('DOMContentLoaded', () => { ons.forEach((f) => f()); all(); });
  let w = window.innerWidth, t;
  window.addEventListener('resize', () => { if (window.innerWidth === w) return; w = window.innerWidth; clearTimeout(t); t = setTimeout(all, 120); });
})();

/* Interactive versions of the RSIBox blog figures (haoailab.com/blogs/rsibox).
   Each <div data-fig="ladder|bottleneck|engines|races"> becomes one plate. Numbers come from
   data/*.csv (a link to site/landing/dash/data), except the engines figure, which carries the
   values of blog Figure 4 below. */
(()=>{
'use strict';
const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const esc=t=>String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const PLAY='<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>';

function parseCSV(text){
  const rows=[];let row=[],f='',q=false;
  for(let i=0;i<text.length;i++){const c=text[i];
    if(q){if(c==='"'){if(text[i+1]==='"'){f+='"';i++;}else q=false;}else f+=c;}
    else if(c==='"')q=true;else if(c===','){row.push(f);f='';}
    else if(c==='\n'||c==='\r'){if(c==='\r'&&text[i+1]==='\n')i++;row.push(f);rows.push(row);row=[];f='';}
    else f+=c;}
  if(f||row.length){row.push(f);rows.push(row);}
  const h=rows.shift();
  return rows.filter(r=>r.length>1||r[0]).map(r=>Object.fromEntries(h.map((k,i)=>[k,r[i]??''])));
}
const cache={};
const csv=n=>cache[n]||(cache[n]=fetch(`data/${n}.csv`).then(r=>{if(!r.ok)throw new Error(`data/${n}.csv: HTTP ${r.status}`);return r.text();}).then(parseCSV));

const tip=document.createElement('div');tip.className='pl-tip';tip.setAttribute('role','tooltip');document.body.appendChild(tip);
const showTip=(html,x,y)=>{tip.innerHTML=html;tip.style.left=x+'px';tip.style.top=y+'px';tip.classList.add('on');};
const hideTip=()=>tip.classList.remove('on');
addEventListener('scroll',hideTip,{passive:true});
/* start once the plate's top reaches the lower quarter of the screen; tall plates on phones never reach a ratio threshold */
function onView(el,fn){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){io.disconnect();fn();}},{threshold:0,rootMargin:'0px 0px -25% 0px'});io.observe(el);}
const fmtS=s=>s>=50?`${Math.round(s)} s`:s>=1?`${s.toFixed(1)} s`:`${s.toFixed(2)} s`;
const fmtX=v=>v>=10?`${Math.round(v)}×`:`${v.toFixed(1)}×`;
function niceStep(max,target=5){const raw=max/target,p=10**Math.floor(Math.log10(raw)),m=raw/p;return (m<=1?1:m<=2?2:m<=2.5?2.5:m<=5?5:10)*p;}

/* ---------- Figure 2: restart overhead ladder ---------- */
const FIXED={kernel_autotune:'var(--pl-r1)',weight_load:'var(--pl-r2)',cuda_graph_capture:'var(--pl-r3)',engine_init:'var(--pl-r4)',
  run_workload:'var(--pl-blue)',logprob_check:'var(--pl-blue-2)',quality_check:'#6E93E6',live_install:'var(--pl-blue-3)'};
const RUST=['var(--pl-r1)','var(--pl-r2)','var(--pl-r3)','var(--pl-r4)','var(--pl-r5)'];
async function ladder(el){
  const [L,P,R,W]=await Promise.all(['eval_ladder','eval_phases','ladder_rungs','race_workloads'].map(csv));
  const PH=Object.fromEntries(P.map(p=>[p.phase,p]));
  const engines=W.filter(w=>L.some(l=>l.engine===w.workload)).sort((a,b)=>+a.order-+b.order);
  el.innerHTML=`<div class="pl-head"><div><h4>Cutting the overhead, one phase at a time</h4><p class="pl-sub" data-sub></p></div>
    <div class="pl-tabs" role="group" aria-label="Engine">${engines.map((w,i)=>`<button type="button" data-e="${esc(w.workload)}" aria-pressed="${i===0}">${esc(w.engine_label)}</button>`).join('')}</div></div>
    <div class="lf-body"><div><div class="lf-rows" data-rows></div><div class="lf-axis" data-axis></div></div><div class="pl-legend-col" data-legend></div></div>
    <div class="lf-detail" data-detail aria-live="polite"></div>
    <div class="pl-foot"><button type="button" class="pl-btn blue" data-play>${PLAY}<span>Step through</span></button></div>`;
  const VAL='var(--lf-val)';let rows=[],timers=[];
  const kindOf=p=>PH[p]?.kind==='check'?1:0;
  const label=p=>PH[p]?.label||p;
  function build(engine){
    timers.forEach(clearTimeout);timers=[];
    const w=engines.find(x=>x.workload===engine);
    $$('.pl-tabs button',el).forEach(b=>b.setAttribute('aria-pressed',b.dataset.e===engine));
    $('[data-sub]',el).textContent='The same check run again and again, each row keeping one more piece of start-up warm.';
    const steps=R.filter(r=>r.engine===engine).sort((a,b)=>+a.step-+b.step);
    rows=steps.map(s=>{const segs=L.filter(l=>l.engine===engine&&l.step===s.step).map(l=>({ph:l.phase,sec:+l.seconds}))
      .sort((a,b)=>kindOf(a.ph)-kindOf(b.ph)||(+PH[a.ph]?.order||0)-(+PH[b.ph]?.order||0));
      return {...s,segs,total:segs.reduce((t,x)=>t+x.sec,0)};}).filter(r=>r.segs.length);
    const color={};const present=[...new Set(rows.flatMap(r=>r.segs.map(s=>s.ph)))];
    const used=new Set(present.map(p=>FIXED[p]).filter(Boolean));const pool=RUST.filter(c=>!used.has(c));let k=0;
    present.forEach(p=>{color[p]=FIXED[p]||(kindOf(p)?'#6E93E6':(pool.length?pool[k++%pool.length]:RUST[k++%5]));});
    const max=Math.max(...rows.map(r=>r.total)),base=rows[0].total,step=niceStep(max,4),ticks=[];for(let v=0;v<=max*1.001;v+=step)ticks.push(v);
    const W_=f=>`calc((100% - ${VAL}) * ${f.toFixed(5)})`;
    let h=`<div class="lf-grid">${ticks.map(v=>`<i style="left:${W_(v/max)}"></i>`).join('')}</div>`,grp='';
    rows.forEach((r,i)=>{
      if(r.group!==grp){grp=r.group;h+=`<div class="lf-group">${esc(grp)}</div>`;}
      const derived=r.measured==='no';
      h+=`<div class="lf-row${derived?' derived':''}" data-i="${i}" tabindex="0"><div class="lf-lab">${esc(r.label)}${derived?'<span class="dv">derived</span>':''}</div>
        <div class="lf-bar"><div class="lf-segs" style="width:${W_(r.total/max)}">${r.segs.map(s=>`<i style="flex:${s.sec} 1 0;background:${color[s.ph]}" data-ph="${esc(s.ph)}" data-sec="${s.sec}"></i>`).join('')}</div>
        <span class="lf-val"><b>${fmtS(r.total)}</b>${i?fmtX(base/r.total):''}</span></div></div>`;});
    $('[data-rows]',el).innerHTML=h;
    $('[data-axis]',el).innerHTML=ticks.map(v=>`<span style="left:${W_(v/max)}">${v} s</span>`).join('');
    const leg=k=>present.filter(p=>kindOf(p)===k).map(p=>`<span><i style="background:${color[p]}"></i>${esc(label(p))}</span>`).join('');
    $('[data-legend]',el).innerHTML=`<b>Start-up</b>${leg(0)}<b>Test</b>${leg(1)}`;
    detail(-1);
    /* tell the page which engine's edit is on screen (the edit card beside the figure) */
    el.dispatchEvent(new CustomEvent('fig-engine',{bubbles:true,detail:{label:w.engine_label,setup:w.ladder_setup,first:rows[0].total,last:rows[rows.length-1].total}}));
  }
  function detail(i){
    $$('.lf-row',el).forEach(r=>r.classList.toggle('sel',+r.dataset.i===i));
    const d=$('[data-detail]',el),w=engines.find(x=>x.workload===current);
    if(i<0){d.innerHTML=`<b>Floor.</b> ${esc(w.ladder_floor||'')}`;return;}
    const r=rows[i];
    if(i===0){d.innerHTML=`<b>${fmtS(r.total)} per edit.</b> Every check restarts the engine: start-up phases (rust) run again before the test itself (blue) can start.`;return;}
    const prev=Object.fromEntries(rows[i-1].segs.map(s=>[s.ph,s.sec])),now=Object.fromEntries(r.segs.map(s=>[s.ph,s.sec]));
    const cut=Object.keys(prev).map(p=>[p,prev[p]-(now[p]||0)]).filter(([,v])=>v>.5).sort((a,b)=>b[1]-a[1]);
    const what=cut.length?cut.map(([p,v])=>`${esc(label(p))} (${fmtS(v)})`).join(', '):'only small differences between runs';
    d.innerHTML=`<b>${fmtS(rows[i-1].total)} → ${fmtS(r.total)}.</b> Removes ${what}. Now ${fmtX(rows[0].total/r.total)} faster than restarting for every check.`;
  }
  let current=engines[0].workload;
  async function play(){
    const rs=$$('.lf-row',el);rs.forEach(r=>r.classList.add('hide'));
    for(let i=0;i<rs.length;i++){
      await new Promise(res=>timers.push(setTimeout(res,i?(reduce?0:1100):150)));
      rs[i].classList.remove('hide');detail(i);
    }
    timers.push(setTimeout(()=>detail(-1),reduce?0:2600));
  }
  el.addEventListener('click',e=>{
    const t=e.target.closest('[data-e]');if(t){current=t.dataset.e;build(current);play();return;}
    if(e.target.closest('[data-play]')){build(current);play();return;}
    const r=e.target.closest('.lf-row');if(r){timers.forEach(clearTimeout);detail(+r.dataset.i);}
  });
  el.addEventListener('keydown',e=>{const r=e.target.closest('.lf-row');if(r&&(e.key==='Enter'||e.key===' ')){e.preventDefault();detail(+r.dataset.i);}});
  el.addEventListener('pointermove',e=>{const s=e.target.closest('.lf-segs i');if(!s){hideTip();return;}
    const tot=[...s.parentNode.children].reduce((t,x)=>t+ +x.dataset.sec,0),sec=+s.dataset.sec;
    showTip(`<b>${esc(label(s.dataset.ph))}</b> ${fmtS(sec)} <span>${Math.round(sec/tot*100)}% of this row</span>`,e.clientX,e.clientY);});
  el.addEventListener('pointerleave',hideTip);
  build(current);
  onView(el,play);
}

/* ---------- Figure 1: evaluation dominates the loop ---------- */
async function bottleneck(el){
  const [B,C]=await Promise.all([csv('fig2_breakdown'),csv('fig2_claim')]);
  const rows=B.map(b=>({...b,x:Math.max(1,Math.round(+b.tok_s/50)),h:+b.total_h,ev:+b.evaluation_pct}));
  const maxH=Math.ceil(Math.max(...rows.map(r=>r.h)));
  el.innerHTML=`<div class="pl-head"><div><h4>Evaluation dominates the RSI loop</h4><p class="pl-sub">Replay of a 6-hour vLLM optimization trace while changing the token generation speed.</p></div>
    <div class="pl-legend"><span><i style="background:var(--pl-eval)"></i>Waiting on evaluation</span><span><i style="background:var(--pl-gray)"></i>Agent generating tokens</span></div></div>
    <div class="bf-rows" data-rows>${rows.map((r,i)=>`<div class="bf-row" data-i="${i}" tabindex="0"><div class="bf-who"><b>${r.x===1?'Baseline speed':`${r.x}× faster model`}</b><small>${esc(r.tok_s_label)}</small></div>
      <div class="bf-bar"><div class="bf-segs" style="width:0"><i class="ev" style="width:${r.ev}%">${r.ev}%</i><i class="gen" style="flex:1">${100-r.ev}%</i></div>
      <span class="bf-val"><b>${r.h} h</b>${r.change_pct?`${String(r.change_pct).replace('-','−')}%`:''}</span></div></div>`).join('')}</div>
    <div class="bf-axis"><div></div><div>${Array.from({length:maxH+1},(_,h)=>`<span style="left:calc((100% - 110px) * ${h/maxH})">${h} h</span>`).join('')}</div></div>
    <div class="pl-foot"><div class="pl-tabs" role="group" aria-label="Model speed">${rows.map((r,i)=>`<button type="button" data-k="${i}" aria-pressed="${i===rows.length-1}">${r.x}×</button>`).join('')}</div><span class="bf-call" data-call aria-live="polite"></span></div>`;
  function pick(i){
    $$('.bf-row',el).forEach(r=>r.classList.toggle('sel',+r.dataset.i===i));$$('[data-k]',el).forEach(b=>b.setAttribute('aria-pressed',+b.dataset.k===i));
    const r=rows[i];
    $('[data-call]',el).innerHTML=i===0?`<b>${rows[0].ev}%</b> of the run is spent waiting on evaluation at today’s speed.`
      :`<b>${String(r.change_pct).replace('-','−')}%</b> shorter run with a ${r.x}× faster model, because evaluation grows to ${r.ev}% of the clock.`;
  }
  const grow=()=>$$('.bf-segs',el).forEach((s,i)=>setTimeout(()=>{s.style.width=`calc((100% - 110px) * ${(rows[i].h/maxH).toFixed(4)})`;},reduce?0:i*160));
  el.addEventListener('click',e=>{const b=e.target.closest('[data-k]');if(b)pick(+b.dataset.k);const r=e.target.closest('.bf-row');if(r)pick(+r.dataset.i);});
  el.addEventListener('pointermove',e=>{const r=e.target.closest('.bf-row');if(!r){hideTip();return;}const d=rows[+r.dataset.i];
    showTip(`<b>${d.h} h</b> total <span>${d.ev}% waiting on evaluation, ${100-d.ev}% generating tokens</span>`,e.clientX,e.clientY);});
  el.addEventListener('pointerleave',hideTip);
  pick(rows.length-1);onView(el,grow);
}

/* ---------- System overhead per engine: the restart row of every ladder, split start-up vs test ---------- */
async function overhead(el){
  const [L,P,W]=await Promise.all(['eval_ladder','eval_phases','race_workloads'].map(csv));
  const PH=Object.fromEntries(P.map(p=>[p.phase,p]));
  const rows=W.filter(w=>L.some(l=>l.engine===w.workload&&l.step==='1')).sort((a,b)=>+a.order-+b.order).map(w=>{
    const ph=L.filter(l=>l.engine===w.workload&&l.step==='1').map(l=>({ph:l.phase,sec:+l.seconds,test:PH[l.phase]?.kind==='check'}));
    const su=ph.filter(p=>!p.test),te=ph.filter(p=>p.test),sum=a=>a.reduce((t,p)=>t+p.sec,0);
    return {w,su,te,S:sum(su),T:sum(te),tot:sum(ph)};});
  const max=Math.max(...rows.map(r=>r.tot)),step=niceStep(max,4),ticks=[];for(let v=0;v<=max*1.001;v+=step)ticks.push(v);
  const pct=x=>Math.round(x*100);
  const lab=p=>PH[p]?.label||p;
  el.innerHTML=`<div class="pl-head"><div><h4>System overhead dominates the evaluation</h4><p class="pl-sub">One check of one edit on each engine, restarting the engine as agents do today. Measured wall time.</p></div>
    <div class="pl-legend"><span><i style="background:var(--pl-r2)"></i>Engine start-up</span><span><i style="background:var(--pl-blue)"></i>The test itself</span></div></div>
    <div class="bf-rows of-rows" data-rows>${rows.map((r,i)=>`<div class="bf-row of-row" data-i="${i}" tabindex="0" aria-label="${esc(r.w.engine_label)}: drag or use the arrow keys to move"><div class="bf-who"><span class="of-grip" aria-hidden="true"></span><b>${esc(r.w.engine_label)}</b><small>${pct(r.S/r.tot)}% start-up</small></div>
      <div class="bf-bar"><div class="bf-segs" data-segs style="width:0"><i class="su" style="flex:${r.S} 1 0">${fmtS(r.S)}</i><i class="te" style="flex:${r.T} 1 0">${fmtS(r.T)}</i></div>
      <span class="bf-val"><b>${fmtS(r.tot)}</b></span></div></div>`).join('')}</div>
    <div class="bf-axis"><div></div><div data-axis></div></div>
    <div class="pl-foot"><div class="pl-tabs" role="group" aria-label="Scale"><button type="button" data-m="abs" aria-pressed="true">Wall time</button><button type="button" data-m="pct" aria-pressed="false">Share of the check</button></div></div>`;
  let mode='abs',shown=false;
  const VAL='110px',box=$('[data-rows]',el);
  const rowOf=s=>rows[+s.closest('.of-row').dataset.i];
  const width=r=>mode==='pct'?'100%':`calc((100% - ${VAL}) * ${(r.tot/max).toFixed(4)})`;
  function draw(){
    $$('.pl-tabs button',el).forEach(b=>b.setAttribute('aria-pressed',b.dataset.m===mode));
    $$('[data-segs]',el).forEach(s=>{const r=rowOf(s);
      s.style.width=shown?width(r):'0';
      const [a,b]=s.children;
      a.textContent=mode==='pct'?`${pct(r.S/r.tot)}%`:fmtS(r.S);b.textContent=mode==='pct'?`${pct(r.T/r.tot)}%`:fmtS(r.T);});
    $$('.of-row .bf-val',el).forEach(v=>{v.hidden=mode==='pct';});
    setTimeout(fit,reduce?0:850);
    $('[data-axis]',el).innerHTML=(mode==='pct'?[0,25,50,75,100].map(v=>[v/100,`${v}%`]):ticks.map(v=>[v/max,`${v} s`]))
      .map(([f,t])=>`<span style="left:${mode==='pct'?`${f*100}%`:`calc((100% - ${VAL}) * ${f})`}">${t}</span>`).join('');
  }
  /* hide a segment's number when it does not fit inside the segment */
  const fit=()=>$$('[data-segs] i',el).forEach(i=>{i.classList.remove('mute');if(i.scrollWidth>i.clientWidth+1)i.classList.add('mute');});
  addEventListener('resize',fit,{passive:true});
  /* reorder rows: move the DOM node, then slide the others from where they were (FLIP) */
  function flip(change,skip){
    const others=$$('.of-row',box).filter(x=>x!==skip),before=new Map(others.map(x=>[x,x.offsetTop]));
    change();
    if(reduce)return;
    others.forEach(x=>{const d=before.get(x)-x.offsetTop;if(d)x.animate([{transform:`translateY(${d}px)`},{transform:'none'}],{duration:220,easing:'cubic-bezier(.2,.8,.2,1)'});});
  }
  let drag=null;
  box.addEventListener('pointerdown',e=>{
    const r=e.target.closest('.of-row');if(!r||e.button!==0)return;
    if(e.pointerType!=='mouse'&&!e.target.closest('.of-grip'))return;
    e.preventDefault();hideTip();r.focus({preventScroll:true});
    drag={r,grab:e.clientY-r.getBoundingClientRect().top,id:e.pointerId};r.setPointerCapture(e.pointerId);r.classList.add('drag');
  });
  box.addEventListener('pointermove',e=>{
    if(!drag||e.pointerId!==drag.id)return;const {r}=drag,top=box.getBoundingClientRect().top,y=e.clientY-top;
    let before=null;for(const s of $$('.of-row',box)){if(s===r)continue;if(y<s.offsetTop+s.offsetHeight/2){before=s;break;}}
    if(r.nextElementSibling!==before)flip(()=>box.insertBefore(r,before),r);
    r.style.transform=`translateY(${e.clientY-top-drag.grab-r.offsetTop}px)`;
  });
  const drop=e=>{if(!drag||e.pointerId!==drag.id)return;const {r}=drag;drag=null;r.classList.remove('drag');
    const t=r.style.transform;r.style.transform='';if(t&&!reduce)r.animate([{transform:t},{transform:'none'}],{duration:180,easing:'ease-out'});};
  box.addEventListener('pointerup',drop);box.addEventListener('pointercancel',drop);
  el.addEventListener('click',e=>{const m=e.target.closest('[data-m]');if(m){mode=m.dataset.m;draw();}});
  el.addEventListener('keydown',e=>{const r=e.target.closest('.of-row');if(!r||(e.key!=='ArrowUp'&&e.key!=='ArrowDown'))return;e.preventDefault();
    const to=e.key==='ArrowUp'?r.previousElementSibling:r.nextElementSibling?.nextElementSibling??null;
    if(e.key==='ArrowUp'&&!to)return;if(e.key==='ArrowDown'&&!r.nextElementSibling)return;
    flip(()=>box.insertBefore(r,to),null);r.focus({preventScroll:true});});
  el.addEventListener('pointermove',e=>{if(drag)return;const s=e.target.closest('[data-segs] i');if(!s){hideTip();return;}
    const r=rowOf(s),list=s.classList.contains('su')?r.su:r.te;
    showTip(`<b>${s.classList.contains('su')?'Start-up':'Test'}: ${fmtS(list.reduce((t,p)=>t+p.sec,0))}</b> <span>${[...list].sort((a,b)=>b.sec-a.sec).map(p=>`${esc(lab(p.ph))} ${fmtS(p.sec)}`).join(' · ')}</span>`,e.clientX,e.clientY);});
  el.addEventListener('pointerleave',hideTip);
  draw();
  onView(el,()=>{shown=true;$$('[data-segs]',el).forEach((s,i)=>setTimeout(()=>{s.style.width=width(rowOf(s));},reduce?0:i*140));setTimeout(fit,reduce?0:rows.length*140+900);});
}

/* ---------- Figure 4: across systems, minutes to seconds (values from blog Figure 4) ---------- */
const SYS=[
  {n:'vLLM',d:'Scheduler edit (Nemotron 3.5 Super)',probe:2.4,full:23.6,base:228,sp:'95×',spFull:'9.6×',subP:'9.6× full'},
  {n:'vLLM',d:'Triton top-p kernel fix',full:2.7,base:105,approx:true,sp:'~39×'},
  {n:'SGLang',d:'Decode CUDA graph recapture',full:.84,base:25,approx:true,sp:'~30×'},
  {n:'Megatron-LM',d:'Backward kernel + state restore',probe:.5,full:5.4,base:58,approx:true,sp:'>100×',spFull:'~11×',subP:'~11× full'},
  {n:'FastVideo',d:'Attention edit, full video test',full:[6,7],base:[68,72],cold:510,coldL:'8.5 min cold',sp:'~11×',subP:'~75× vs. cold'},
  {n:'veRL',d:'Distributed RL advantage edit',full:9,base:224,approx:true,cold:1020,coldL:'17 min NFS',sp:'~24×',subP:'>100× vs. NFS'},
  {n:'Miles',d:'Multi-actor RL swap + 1-step GRPO',probe:2.5,full:8,base:[60,95],approx:true,sp:'~24–38×',spFull:'~8–12×',subP:'~8–12× full'},
];
function engines(el){
  const LO=Math.log10(.3),HI=Math.log10(1500),X=v=>(Math.log10(v)-LO)/(HI-LO)*100;
  const lo=v=>Array.isArray(v)?v[0]:v,hi=v=>Array.isArray(v)?v[1]:v;
  const lab=(v,a)=>Array.isArray(v)?`${v[0]}–${v[1]} s`:`${a?'~':''}${v} s`;
  const ticks=[[.5,'0.5 s'],[1,'1 s'],[10,'10 s'],[100,'100 s'],[1000,'1,000 s']];
  el.dataset.mode='probe';
  el.innerHTML=`<div class="pl-head"><div><h4>Across six AI systems, testing one edit drops from minutes to seconds</h4><p class="pl-sub">Measured on NVIDIA GB200, 1 to 4 GPUs per workload.</p></div></div>
    <div class="pl-legend" style="margin-bottom:12px"><span><i class="dot" style="background:var(--pl-gray-2)"></i>Baseline: restart, then test</span><span><i class="dot" style="background:var(--pl-blue)"></i>RSIBox: live full check</span><span><i class="ring" style="--c:var(--pl-blue)"></i>RSIBox: targeted probe</span><span><i class="ring" style="--c:var(--pl-gray-2)"></i>Cold cache / NFS baseline</span></div>
    <div class="ef-head"><span></span><span></span><span>Speedup</span></div>
    ${SYS.map((s,i)=>{const fastest=s.probe??lo(s.full);return `<div class="ef-row" data-i="${i}"><div class="ef-who"><b>${s.n}</b><small>${esc(s.d)}</small></div>
      <div class="ef-track">${ticks.map(([v])=>`<i class="gl" style="left:${X(v)}%"></i>`).join('')}
        ${s.cold?`<i class="ef-line dotted" style="left:${X(hi(s.base))}%;width:${X(s.cold)-X(hi(s.base))}%"></i><i class="ef-dot cold" style="left:${X(s.cold)}%"></i><span class="ef-lab top gray" style="left:${X(s.cold)}%">${s.coldL}</span>`:''}
        <i class="ef-line" data-from="${X(lo(s.full))}" style="left:${X(lo(s.base))}%;width:0%"></i>
        ${s.probe?`<i class="ef-line blue" data-from="${X(s.probe)}" data-to="${X(lo(s.full))}" style="left:${X(lo(s.base))}%;width:0%"></i>`:''}
        ${Array.isArray(s.base)?`<i class="ef-dot base range" style="left:${X(s.base[0])}%;width:calc(${X(s.base[1])-X(s.base[0])}% + 13px);margin-left:-6.5px"></i>`:`<i class="ef-dot base" style="left:${X(s.base)}%"></i>`}
        <span class="ef-lab r" style="left:${X(hi(s.base))}%">${lab(s.base,s.approx)}</span>
        ${Array.isArray(s.full)?`<i class="ef-dot full range" data-x="${X(s.full[0])}" data-w="${X(s.full[1])-X(s.full[0])}" style="left:${X(lo(s.base))}%;width:calc(${X(s.full[1])-X(s.full[0])}% + 13px);margin-left:-6.5px"></i>`:`<i class="ef-dot full" data-x="${X(s.full)}" style="left:${X(lo(s.base))}%"></i>`}
        ${s.probe?`<i class="ef-dot probe" data-x="${X(s.probe)}" style="left:${X(lo(s.base))}%"></i><span class="ef-lab top fullonly" data-x="${X(s.full)}" style="left:${X(s.full)}%;opacity:0">${s.full} s</span>`:''}
        <span class="ef-lab l${s.probe?' probe':''}" data-x="${X(fastest)}" style="left:${X(lo(s.base))}%;opacity:0">${s.probe?`${s.probe} s`:lab(s.full,s.approx)}</span>
        ${s.probe?`<span class="ef-lab l fullonly2" data-x="${X(s.full)}" style="left:${X(s.full)}%;opacity:0">${s.full} s</span>`:''}
      </div><div class="ef-sp" data-sp><b>${s.sp}</b>${s.subP?`<small>${s.subP}</small>`:''}</div></div>`;}).join('')}
    <div class="ef-axis"><div></div><div>${ticks.map(([v,l])=>`<span style="left:${X(v)}%">${l}</span>`).join('')}<span class="t">Time to test one edit (log scale)</span></div><div></div></div>`;
  function mode(m){
    el.dataset.mode=m;
    $$('.ef-row',el).forEach((row,i)=>{const s=SYS[i];if(!s.probe)return;
      const l=$('.ef-lab.l.probe',row),l2=$('.fullonly2',row),top=$('.fullonly',row);
      l.style.opacity=m==='probe'?1:0;l2.style.opacity=m==='full'?1:0;top.style.opacity=m==='probe'?1:0;
      $('[data-sp]',row).innerHTML=m==='probe'?`<b>${s.sp}</b><small>${s.subP}</small>`:`<b>${s.spFull}</b><small>full check</small>`;});
  }
  function settle(){
    $$('.ef-row',el).forEach((row,i)=>setTimeout(()=>{
      $$('[data-x]',row).forEach(d=>{if(!d.classList.contains('fullonly')&&!d.classList.contains('fullonly2'))d.style.left=d.dataset.x+'%';});
      const gl=$('.ef-line:not(.blue):not(.dotted)',row),s=SYS[i];gl.style.left=gl.dataset.from+'%';gl.style.width=(X(lo(s.base))-+gl.dataset.from)+'%';
      const bl=$('.ef-line.blue',row);if(bl){bl.style.left=bl.dataset.from+'%';bl.style.width=(+bl.dataset.to-+bl.dataset.from)+'%';}
      $$('.ef-lab.l',row).forEach(x=>{if(!x.classList.contains('fullonly2'))x.style.opacity=1;});
      if(s.probe&&el.dataset.mode==='probe')$('.fullonly',row).style.opacity=1;
      mode(el.dataset.mode);
    },reduce?0:i*130));
  }
  el.addEventListener('pointermove',e=>{const r=e.target.closest('.ef-row');if(!r){hideTip();return;}const s=SYS[+r.dataset.i];
    showTip(`<b>${s.n}</b> <span>${esc(s.d)}:</span> ${s.probe?`probe ${s.probe} s, `:''}full check ${lab(s.full,s.approx)}, restart ${lab(s.base,s.approx)}`,e.clientX,e.clientY);});
  el.addEventListener('pointerleave',hideTip);
  onView(el,settle);
  return Promise.resolve();
}

/* ---------- Figure 5: end-to-end races, all at once ----------
   The same six races, runs and rule as Figure 5 of the RSIBox blog post. Each tile draws every
   run of one race (race_runs). Solid lines: the Baseline run with the best final result, and the
   first RSIBox run to reach that result. Dotted lines: the other runs. The headline is the wall
   time RSIBox saved reaching it: 1 - t_rsibox / t_baseline. data-extra/ holds the races that are
   not in the shared data (T2VA, and SGLang and Miles from the blog's race picker). */
const extraCache={};
const csvExtra=n=>extraCache[n]||(extraCache[n]=fetch(`data-extra/${n}.csv`).then(r=>r.ok?r.text().then(parseCSV):[]).catch(()=>[]));
const armOf=a=>a==='fastrsi'?'rsibox':a;            /* restored rows still name the live arm "fastrsi" */
/* Tiles in the blog's order. edits: edits checked by the run that checked the most on each side
   [RSIBox, Baseline], from the blog's Figure 5. base: the blog's own pick for the solid Baseline
   line where it is not the run with the best final result (Miles). */
const RACE_TILES=[
  {w:'vllm-d8x32k',n:'vLLM',about:'DeepSeek-V4.1 Flash decode, 4× GB200',edits:[173,101]},
  {w:'sglang',n:'SGLang',about:'Qwen3.8 Flash Next decode, 2× GB200',edits:[125,101]},
  {w:'megatron',n:'Megatron-LM',about:'Qwen3-30B-A3B training, 4× GB200',edits:[98,52]},
  {w:'t2va',n:'FastVideo T2VA',about:'MiniMax H3 text to audio-video, DGX Spark',edits:[82,53]},
  {w:'ref2va',n:'FastVideo Ref2VA',about:'MiniMax H3 reference to audio-video, DGX Spark',edits:[34,18]},
  {w:'miles',n:'Miles',about:'Qwen3-0.6B GRPO, 2× GB200',edits:[70,53],base:'Baseline r3'}];
async function raceGrid(el){
  const [a,b]=await Promise.all([csv('race_runs'),csvExtra('race_runs')]);
  const have=new Set(a.map(r=>r.workload));
  const rows=[...a,...b.filter(r=>!have.has(r.workload))].map(r=>({...r,arm:armOf(r.arm)}));
  const fh=t=>`${(+t).toFixed(1)} h`,fx=v=>{const s=(+v).toFixed(3);return (s.endsWith('0')?s.slice(0,-1):s)+'×';};
  const races=RACE_TILES.map(T=>{
    const runs={codex:{},rsibox:{}};
    rows.filter(r=>r.workload===T.w&&runs[r.arm]).forEach(r=>(runs[r.arm][r.run]||(runs[r.arm][r.run]=[])).push({t:+r.t_h,v:+r.value}));
    const ser=a=>Object.entries(runs[a]).map(([name,p])=>{p.sort((x,y)=>x.t-y.t);if(!p.length||p[0].t>0)p.unshift({t:0,v:1});
      return {name,p,fin:Math.max(...p.map(q=>q.v))};});
    const C=ser('codex'),R=ser('rsibox');if(!C.length||!R.length)return null;
    const B=T.w==='t2va'||T.w==='ref2va'?8:6;
    const reach=(s,v)=>{const q=s.p.find(q=>q.v>=v-1e-6);return q?q.t:null;};
    const base=C.find(s=>s.name===T.base)||C.reduce((x,y)=>y.fin>x.fin?y:x),top=base.fin,tc=reach(base,top);
    const hit=R.map(s=>({s,t:reach(s,top)})).filter(h=>h.t!=null).sort((x,y)=>x.t-y.t)[0];
    const lead=hit?hit.s:R.reduce((x,y)=>y.fin>x.fin?y:x),tr=hit?hit.t:null;
    return {...T,B,C,R,base,lead,top,tc,tr,save:tr==null?null:1-tr/tc,fc:Math.max(...C.map(s=>s.fin)),fr:Math.max(...R.map(s=>s.fin))};
  }).filter(Boolean);
  if(!races.length)throw new Error('no race data');
  const saves=races.map(r=>r.save).filter(v=>v!=null),pc=v=>`${Math.round(v*100)}%`;
  el.innerHTML=`<div class="pl-head"><div><h4>End to end, the agent reaches the baseline’s best result with ${pc(Math.min(...saves))}–${pc(Math.max(...saves))} less wall time</h4>
      <p class="pl-sub">Each race runs the same coding agent with the same prompt and GPUs: restarting the engine for every check (Baseline) or on RSIBox. Lines are the best speedup found so far.</p></div>
    <div class="pl-legend"><span><i class="ln" style="background:var(--pl-gray)"></i>Baseline, best run</span><span><i class="ln" style="background:var(--pl-blue)"></i>RSIBox, first run to reach it</span><span><i class="dotl"></i>Other runs</span><span><i class="dash"></i>Baseline’s best final</span><span><i style="background:var(--pl-blue-bg);box-shadow:inset 0 0 0 1px var(--pl-blue-2)"></i>Wall time saved</span></div></div>
    <div class="rg">${races.map((r,i)=>`<div class="rg-t" data-i="${i}">
        <div class="rg-h"><b>${esc(r.n)}</b><small title="${esc(r.about)}, ${r.B} h">${esc(r.about)}, ${r.B} h</small></div>
        <div class="rg-big">${r.save==null?'not reached':`${pc(r.save)} <span>less time</span>`}</div>
        <div class="rg-sub">${r.tr==null?'':`<b>${fh(r.tr)}</b> vs <b>${fh(r.tc)}</b> to reach ${fx(r.top)}, ${(r.tc/r.tr).toFixed(1)}× sooner`}</div>
        <div class="rg-c" data-c></div>
        <div class="rg-f">Best final ${fx(r.fr)} on RSIBox, ${fx(r.fc)} on Baseline<br>Edits checked: <b>${r.edits[0]}</b> on RSIBox, <b>${r.edits[1]}</b> on Baseline</div></div>`).join('')}</div>`;

  function draw(tile,r){
    const box=$('[data-c]',tile),Wd=Math.max(220,Math.round(box.clientWidth)),Ht=136,ml=38,mr=8,mt=8,mb=20;
    const all=[...r.C,...r.R],lo=1,hi=Math.max(...all.map(s=>s.fin));
    const pad=(hi-lo)*.08,y0=lo-pad,y1=hi+pad;
    const X=t=>ml+t/r.B*(Wd-ml-mr),Y=v=>mt+(1-(v-y0)/(y1-y0))*(Ht-mt-mb);
    const path=s=>{let d='';s.p.forEach((q,k)=>{const x=X(Math.min(q.t,r.B)).toFixed(1),y=Y(q.v).toFixed(1);d+=k?`H${x}V${y}`:`M${x},${y}`;});return d+`H${X(r.B).toFixed(1)}`;};
    const step=niceStep(y1-y0,5),ticks=[];for(let v=Math.ceil(y0/step)*step;v<=y1+1e-9;v+=step)ticks.push(v);
    const hs=[];for(let h=0;h<=r.B;h+=r.B>6?2:1)hs.push(h);
    const dp=step<.05?2:step<.5?2:1;
    let s=`<svg class="rg-svg" viewBox="0 0 ${Wd} ${Ht}" width="${Wd}" height="${Ht}" role="img" aria-label="${esc(r.n)}: best speedup over ${r.B} hours, every Baseline and RSIBox run">`;
    ticks.forEach(v=>{s+=`<line class="gl" x1="${ml}" x2="${Wd-mr}" y1="${Y(v)}" y2="${Y(v)}"/><text class="ax" x="${ml-5}" y="${Y(v)+3.5}" text-anchor="end">${v.toFixed(dp)}×</text>`;});
    hs.forEach(h=>{s+=`<text class="ax" x="${X(h)}" y="${Ht-5}" text-anchor="${h===0?'start':h===r.B?'end':'middle'}">${h} h</text>`;});
    if(r.tr!=null)s+=`<rect class="saved" x="${X(r.tr)}" y="${mt}" width="${X(r.tc)-X(r.tr)}" height="${Ht-mt-mb}"/>`;
    s+=`<line class="tgt" x1="${ml}" x2="${Wd-mr}" y1="${Y(r.top)}" y2="${Y(r.top)}"/>`;
    r.C.filter(x=>x!==r.base).forEach(x=>{s+=`<path class="ln codex alt" d="${path(x)}"/>`;});
    r.R.filter(x=>x!==r.lead).forEach(x=>{s+=`<path class="ln rsibox alt" d="${path(x)}"/>`;});
    s+=`<path class="ln codex" d="${path(r.base)}"/><path class="ln rsibox" d="${path(r.lead)}"/>`;
    if(r.tr!=null)s+=`<circle class="hit rsibox" cx="${X(r.tr)}" cy="${Y(r.top)}" r="4"/><circle class="hit codex" cx="${X(r.tc)}" cy="${Y(r.top)}" r="4"/>`;
    s+=`<line class="rule" x1="0" x2="0" y1="${mt}" y2="${Ht-mb}" hidden/></svg>`;
    box.innerHTML=s;
    const svg=$('svg',box),rule=$('.rule',svg);
    const at=(x,t)=>{let v=null;x.p.forEach(q=>{if(q.t<=t+1e-9)v=v==null?q.v:Math.max(v,q.v);});return v;};
    svg.addEventListener('pointermove',e=>{const b=svg.getBoundingClientRect(),x=(e.clientX-b.left)*Wd/b.width;
      const t=Math.max(0,Math.min(r.B,(x-ml)/(Wd-ml-mr)*r.B));rule.removeAttribute('hidden');rule.setAttribute('x1',X(t));rule.setAttribute('x2',X(t));
      showTip(`<b>${fh(t)}</b> into the race: Baseline ${fx(at(r.base,t))}, RSIBox ${fx(at(r.lead,t))}`,e.clientX,e.clientY);});
    svg.addEventListener('pointerleave',()=>{rule.setAttribute('hidden','');hideTip();});
    return svg;
  }
  const tiles=$$('.rg-t[data-i]',el);let svgs=[],builtW=0;
  function build(){builtW=Math.round(el.clientWidth);svgs=tiles.map(t=>draw(t,races[+t.dataset.i]));}
  build();
  new ResizeObserver(()=>{const w=Math.round(el.clientWidth);if(w&&Math.abs(w-builtW)>2)build();}).observe(el);
  /* one orchestrated reveal: the solid lines draw in, then the saved band appears */
  if(!reduce){
    el.classList.add('rg-wait');
    onView(el,()=>{svgs.forEach((svg,i)=>{$$('.ln:not(.alt)',svg).forEach(p=>{const L=p.getTotalLength();p.style.strokeDasharray=L;p.style.strokeDashoffset=L;
      p.animate([{strokeDashoffset:L},{strokeDashoffset:0}],{duration:1400,delay:i*120,easing:'cubic-bezier(.3,0,.2,1)',fill:'forwards'}).onfinish=()=>{p.style.strokeDasharray='';p.style.strokeDashoffset='';};});});
      el.classList.remove('rg-wait');});
  }
}

const FIGS={ladder,bottleneck,overhead,engines,raceGrid};
$$('[data-fig]').forEach(el=>{el.classList.add('fplate');const f=FIGS[el.dataset.fig];if(!f)return;
  Promise.resolve().then(()=>f(el)).catch(e=>{console.error(e);el.innerHTML=`<p class="pl-sub">This figure could not load its data: ${esc(e.message)}</p>`;});});
})();

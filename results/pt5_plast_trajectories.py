"""Regenerate the pt5 plasticity accuracy-trajectory dashboard from the existing logs.

READ-ONLY over results/*.log — creates NO training runs. Parses every learned-plasticity CL run
(+ its per-task "After task k/5 | seen tasks: [...]" trajectory) out of four pt5 studies, tags each
run by its five ablation axes, and emits a self-contained HTML artifact (inline CSS/JS/SVG, no
external assets -> Artifact-CSP-safe).

Sources (all seed 42, lr 1e-3, ep 5, buffer 1000, learned projection unless noted, 1 seed, ORACLE):
  pt5_iter3            : standalone (NO buffer) + naive/er baselines + +ER(curr) x {classil,taskil} x {sgd,adam}
  pt5_iter3_metareplay : standalone + buffer (meta-replay)  x {classil,taskil} x {sgd,adam}
  pt5_plast_init       : +ER(own task-id, init 0.5)         x classil x {sgd,adam}   (class-IL only)
  pt5_er_task_id       : disjoint-projection within-study own-vs-curr (classil, sgd, per-neuron)

The five axes and where they render:
  standalone vs +ER      -> blue vs orange hue
  buffer vs no buffer    -> blue solid vs dashed        (standalone only)
  own vs curr task-id    -> orange solid vs dashed      (+ER only; learned own is cross-study, class-IL only)
  SGD vs Adam            -> panel columns
  class-IL vs task-IL    -> panel rows

Trajectory shown = mean accuracy over tasks-seen-so-far after each task (avg incremental accuracy);
the final point equals the reported final accuracy (verified == logged avg_final, 0 mismatches).
Standalone plasticity has NO own/curr split (its meta-loss applies ONE gate to the whole summed
gradient), so its buffer line carries no task-id variant.

Run:  uv run python results/pt5_plast_trajectories.py   ->  results/pt5_plast_trajectories.html
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
LOGS = ["pt5_iter3", "pt5_iter3_metareplay", "pt5_plast_init", "pt5_er_task_id"]

AFTER = re.compile(r"After task \d+/\d+ \| seen tasks: \[([^\]]*)\]")
DEBUG = re.compile(r"\[pt5 debug\] (.*)")
SUMM = re.compile(r"^>>> (.*?)\s+acc=([\d.]+)\s+forget=([\d.]+)")


def _tags(s):
    return {k: v for k, v in (kv.split("=", 1) for kv in s.split() if "=" in kv)}


def parse():
    recs = []
    for name in LOGS:
        traj, dbg = [], None
        for line in (HERE / f"{name}.log").read_text().splitlines():
            m = AFTER.search(line)
            if m:
                traj.append([float(x) for x in m.group(1).split(",") if x.strip()]); continue
            m = DEBUG.search(line)
            if m:
                dbg = _tags(m.group(1)); continue
            m = SUMM.match(line)
            if not m:
                continue
            label, acc, forget = m.group(1), float(m.group(2)), float(m.group(3))
            br = re.match(r"\[(\w+)\s+(\w+)\]", label) or re.match(r"\[(\w+)\]", label)
            metric, opt = (br.group(1), br.group(2)) if (br and br.lastindex == 2) else \
                          ("classil", br.group(1)) if br else ("classil", "?")
            base = dict(acc=acc, forget=forget, opt=opt, metric=metric, src=name,
                        avg_seen=[round(sum(t) / len(t), 4) for t in traj])
            if "plast" in label and traj and dbg:
                meta = dbg.get("meta_replay") == "True"
                arm = "er" if dbg.get("method") == "er" else "standalone"
                task_id = ("own" if dbg.get("er_task_id") == "True" else "curr") if arm == "er" else "na"
                recs.append(dict(kind="plast",
                                 mech=("neuron" if dbg.get("granularity") == "neuron" else "synapse"),
                                 proj=dbg.get("projection"), arm=arm, buffer=(arm == "er") or meta,
                                 task_id=task_id, init=dbg.get("plast_init", "0.5"), **base))
            elif traj and ("naive (baseline)" in label or "er (baseline)" in label):
                recs.append(dict(kind="baseline", mech="-", proj="-",
                                 arm=("naive" if "naive" in label else "er"),
                                 buffer=("er" in label), task_id="-", init="-", **base))
            traj, dbg = [], None
    return recs


RECS = parse()


def find(**q):
    hits = [r for r in RECS if all(r.get(k) == v for k, v in q.items())]
    return hits[0] if hits else None


ROLE_LABEL = {
    "naive": "naive baseline", "er_base": "ER baseline",
    "sa_nobuf": "standalone, no buffer", "sa_buf": "standalone, + buffer (meta-replay)",
    "er_curr": "+ER, curr task-id P[t]", "er_own": "+ER, own task-id P[j]",
}


def series_for(mech, metric, opt):
    out = []

    def add(role, rec):
        if rec:
            out.append(dict(role=role, traj=rec["avg_seen"], acc=rec["acc"], forget=rec["forget"]))
    add("naive", find(kind="baseline", arm="naive", opt=opt, metric=metric))
    add("er_base", find(kind="baseline", arm="er", opt=opt, metric=metric))
    add("sa_nobuf", find(kind="plast", src="pt5_iter3", proj="learned", mech=mech, arm="standalone",
                         buffer=False, opt=opt, metric=metric))
    add("sa_buf", find(kind="plast", src="pt5_iter3_metareplay", proj="learned", mech=mech,
                       arm="standalone", buffer=True, opt=opt, metric=metric))
    add("er_curr", find(kind="plast", src="pt5_iter3", proj="learned", mech=mech, arm="er",
                        task_id="curr", opt=opt, metric=metric))
    add("er_own", find(kind="plast", src="pt5_plast_init", proj="learned", mech=mech, arm="er",
                       task_id="own", init="0.5", opt=opt, metric=metric))
    return out


GRID = {f"{mech}|{metric}|{opt}": series_for(mech, metric, opt)
        for mech in ("neuron", "synapse") for metric in ("classil", "taskil") for opt in ("sgd", "adam")}

DISJOINT = []
for role, tid in (("er_curr", "curr"), ("er_own", "own")):
    r = find(kind="plast", src="pt5_er_task_id", proj="disjoint", mech="neuron", arm="er",
             task_id=tid, opt="sgd", metric="classil")
    DISJOINT.append(dict(role=role, traj=r["avg_seen"], acc=r["acc"], forget=r["forget"]))

table = [dict(mech=k.split("|")[0], metric=k.split("|")[1], opt=k.split("|")[2],
              role=s["role"], label=ROLE_LABEL[s["role"]], acc=s["acc"], forget=s["forget"])
         for k, ser in GRID.items() for s in ser]

# integrity check: parsed final trajectory point must equal the logged final accuracy
_bad = [r for r in RECS if abs(r["avg_seen"][-1] - r["acc"]) > 1e-3]
assert not _bad, f"{len(_bad)} trajectory/final-acc mismatches — parse is wrong"

payload = dict(grid=GRID, disjoint=DISJOINT, table=table, role_label=ROLE_LABEL)
DATA = json.dumps(payload, separators=(",", ":"))

HTML = r"""<title>Plasticity trajectories — pt5 continual-learning ablations</title>
<style>
:root{
  --bg:#f6f7f9; --surface:#ffffff; --surface-2:#fbfcfd;
  --ink:#161a20; --ink-2:#535b67; --ink-3:#8b93a0; --border:#e3e6eb; --grid:#eef0f4;
  --c-sa:#2a78d6; --c-er:#eb6834; --c-naive:#9aa3af; --c-erbase:#5b6472;
  --accent:#2a78d6;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
  --bg:#101216; --surface:#181b21; --surface-2:#1c2027;
  --ink:#f1f3f6; --ink-2:#b4bbc6; --ink-3:#7c8593; --border:#2a2f39; --grid:#232833;
  --c-sa:#3987e5; --c-er:#e0733f; --c-naive:#6b7482; --c-erbase:#9aa3b1; --accent:#3987e5;
}}
:root[data-theme="dark"]{
  --bg:#101216; --surface:#181b21; --surface-2:#1c2027;
  --ink:#f1f3f6; --ink-2:#b4bbc6; --ink-3:#7c8593; --border:#2a2f39; --grid:#232833;
  --c-sa:#3987e5; --c-er:#e0733f; --c-naive:#6b7482; --c-erbase:#9aa3b1; --accent:#3987e5;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:44px 24px 80px}
h1{font-size:27px;line-height:1.15;margin:0 0 8px;letter-spacing:-.02em;text-wrap:balance;font-weight:640}
.sub{color:var(--ink-2);font-size:15px;max-width:64ch;margin:0 0 4px}
.prov{color:var(--ink-3);font-size:12.5px;margin:14px 0 0;font-family:var(--mono)}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-2);
  margin:46px 0 4px;font-weight:600}
h2 .n{color:var(--accent);margin-right:8px;font-family:var(--mono);font-size:13px}
.rule{height:1px;background:var(--border);margin:10px 0 22px}
.lead{color:var(--ink-2);font-size:14px;max-width:70ch;margin:0 0 20px}
.lead b{color:var(--ink);font-weight:600}

/* reading guide */
.guide{display:grid;grid-template-columns:1fr 1fr;gap:14px 30px;background:var(--surface);
  border:1px solid var(--border);border-radius:12px;padding:18px 20px;margin:6px 0 8px}
@media(max-width:680px){.guide{grid-template-columns:1fr}}
.guide h3{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-3);margin:0 0 9px}
.legend{display:flex;flex-direction:column;gap:7px}
.lg{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--ink)}
.lg svg{flex:none}
.lg .t2{color:var(--ink-2);font-size:12px}
.axmap{display:flex;flex-direction:column;gap:6px;font-size:13px}
.axmap div{display:flex;gap:8px;align-items:baseline}
.axmap .k{color:var(--ink-2);min-width:132px;font-size:12.5px}
.axmap .v{color:var(--ink)}
.pill{font-family:var(--mono);font-size:11px;padding:1px 6px;border-radius:5px;
  background:var(--surface-2);border:1px solid var(--border);color:var(--ink-2)}

/* findings */
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:8px 0 6px}
@media(max-width:760px){.cards{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 17px}
.card .big{font-family:var(--mono);font-size:22px;font-weight:600;letter-spacing:-.02em;color:var(--ink)}
.card .big .u{color:var(--ink-3);font-size:14px;font-weight:500}
.card p{margin:7px 0 0;font-size:12.7px;color:var(--ink-2);line-height:1.5}
.card p b{color:var(--ink);font-weight:600}

/* panel grids */
.mech-h{display:flex;align-items:baseline;gap:12px;margin:34px 0 2px}
.mech-h .lbl{font-size:16px;font-weight:640;color:var(--ink);letter-spacing:-.01em}
.mech-h .meta{font-size:12.5px;color:var(--ink-3);font-family:var(--mono)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px}
@media(max-width:680px){.grid{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:12px 12px 8px;position:relative}
.panel .ph{display:flex;justify-content:space-between;align-items:baseline;margin:0 2px 4px}
.panel .ph .tt{font-size:12.5px;font-weight:600;color:var(--ink)}
.panel .ph .metric{font-family:var(--mono);font-size:11px;color:var(--ink-3)}
.panel svg.chart{display:block;width:100%;height:auto;overflow:visible}
.tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .08s;
  background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:8px 9px;font-size:11px;box-shadow:0 6px 20px rgba(0,0,0,.14);z-index:5;min-width:150px}
.tip .th{font-family:var(--mono);font-size:10.5px;color:var(--ink-3);margin-bottom:5px;
  text-transform:uppercase;letter-spacing:.05em}
.tip .row{display:flex;align-items:center;gap:7px;justify-content:space-between;padding:1.5px 0}
.tip .row .nm{display:flex;align-items:center;gap:6px;color:var(--ink-2)}
.tip .row .vv{font-family:var(--mono);color:var(--ink);font-variant-numeric:tabular-nums}
.sw{width:14px;height:0;border-top-width:2.5px;border-top-style:solid;flex:none}

/* disjoint exhibit */
.exhibit{display:grid;grid-template-columns:320px 1fr;gap:24px;align-items:center;
  background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 20px}
@media(max-width:680px){.exhibit{grid-template-columns:1fr}}
.exhibit .txt{font-size:13px;color:var(--ink-2)}
.exhibit .txt b{color:var(--ink)}

/* table */
.tablewrap{overflow-x:auto;border:1px solid var(--border);border-radius:12px;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:12.5px;min-width:640px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
th{background:var(--surface-2);color:var(--ink-2);font-weight:600;font-size:11px;
  text-transform:uppercase;letter-spacing:.05em;position:sticky;top:0}
td{color:var(--ink)}
td.num{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:var(--surface-2)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:-1px}
.tag{font-family:var(--mono);font-size:11px;color:var(--ink-2)}
.best{color:var(--c-er);font-weight:600}

.foot{margin-top:40px;color:var(--ink-3);font-size:12px;line-height:1.6;max-width:78ch}
.foot b{color:var(--ink-2)}
</style>

<div class="wrap">
  <h1>Plasticity mechanisms: accuracy trajectories</h1>
  <p class="sub">Every learned-plasticity continual-learning run I have on disk, read straight from the
  pt5 logs — no new training. Five ablation axes, laid over average-incremental-accuracy curves on
  Split-MNIST (5 tasks &times; 2 classes).</p>
  <p class="prov">source: results/pt5_iter3 · pt5_iter3_metareplay · pt5_plast_init · pt5_er_task_id ·
  learned projection · seed 42 · lr 1e-3 · ep 5 · buffer 1000 · 1 seed · ORACLE task id</p>

  <h2><span class="n">A</span>How to read this</h2>
  <div class="rule"></div>
  <div class="guide">
    <div>
      <h3>Line identity</h3>
      <div class="legend" id="legend"></div>
    </div>
    <div>
      <h3>The five axes → where they live</h3>
      <div class="axmap">
        <div><span class="k">standalone vs +ER</span><span class="v"><span class="pill">blue</span> vs <span class="pill">orange</span> hue</span></div>
        <div><span class="k">buffer vs no buffer</span><span class="v">blue <span class="pill">solid</span> vs <span class="pill">dashed</span></span></div>
        <div><span class="k">own vs curr task-id</span><span class="v">orange <span class="pill">solid</span> vs <span class="pill">dashed</span></span></div>
        <div><span class="k">SGD vs Adam</span><span class="v">panel <span class="pill">columns</span></span></div>
        <div><span class="k">class-IL vs task-IL</span><span class="v">panel <span class="pill">rows</span></span></div>
      </div>
    </div>
  </div>
  <p class="lead" style="margin-top:16px">Y axis is <b>mean accuracy over tasks seen so far</b> after
  finishing each task (T1…T5); the last point is the reported final accuracy. A flat-high line retains;
  a line that sags is forgetting. Shared 0–1 scale across all panels — so the near-flat task-IL rows
  are the finding, not a rendering quirk. Hover any panel for exact values; all final-accuracy and
  forgetting numbers are in the table at the end.</p>

  <h2><span class="n">B</span>What the curves say</h2>
  <div class="rule"></div>
  <div class="cards" id="cards"></div>

  <div id="sections"></div>

  <h2><span class="n">D</span>Own vs curr, cleanly (disjoint projection)</h2>
  <div class="rule"></div>
  <p class="lead">On the learned projection, own-vs-curr is a cross-study pairing (curr from pt5_iter3,
  own from pt5_plast_init). The one <b>within-study</b> own-vs-curr for plasticity lives on the
  <b>disjoint</b> projection (the pt5_er_task_id study): same run, flag flipped. Class-IL, SGD, per-neuron.</p>
  <div class="exhibit">
    <div class="panel" style="border:none;padding:0"><div id="disjoint-chart"></div></div>
    <div class="txt">Gating each replayed sample by its <b>own</b> task mask P[j] instead of the current
    task's P[t] lifts final accuracy <b>0.4483 → 0.4833</b> (+0.035) and cuts forgetting
    <b>0.458 → 0.348</b> (−0.110). The correction helps most exactly where the wrong-task mask was
    scrambling replayed samples — but the cell is still far below replay's reach: a rescue of the
    ablation, not a win.</div>
  </div>

  <h2><span class="n">E</span>Every cell — final accuracy &amp; forgetting</h2>
  <div class="rule"></div>
  <div class="tablewrap"><table id="tbl"><thead><tr>
    <th>mechanism</th><th>metric</th><th>opt</th><th>arm (line)</th>
    <th class="num">final acc</th><th class="num">forgetting</th></tr></thead><tbody></tbody></table></div>

  <p class="foot"><b>Caveats, kept honest.</b> 1 seed; ORACLE task id at eval (task-IL-style even on the
  class-IL metric); plasticity+Adam is a first-order surrogate (grads gated before <span class="tag">.step()</span>),
  so SGD is the clean read. <b>Coverage gaps:</b> +ER own-task-id on the learned projection was only ever
  run at class-IL (pt5_plast_init), so the task-IL panels have no orange-solid line; standalone "+buffer"
  is the meta-replay arm (pt5_iter3_metareplay), which reproduces pt5_plast_init exactly where they
  overlap. Standalone plasticity has no own/curr split — its meta-loss applies one gate to the whole
  summed gradient — so its buffer line carries no task-id variant. Numbers are single runs; treat
  sub-0.02 gaps as noise.</p>
</div>

<script>
const P=""" + f"{DATA}" + r""";
const RS={
  naive:{c:"var(--c-naive)",d:"3 3",w:1.5,lab:"naive baseline",t2:""},
  er_base:{c:"var(--c-erbase)",d:"1 3",w:1.5,lab:"ER baseline",t2:""},
  sa_nobuf:{c:"var(--c-sa)",d:"5 4",w:2,lab:"standalone",t2:"no buffer"},
  sa_buf:{c:"var(--c-sa)",d:"none",w:2.6,lab:"standalone",t2:"+ buffer"},
  er_curr:{c:"var(--c-er)",d:"5 4",w:2,lab:"+ER",t2:"curr task-id"},
  er_own:{c:"var(--c-er)",d:"none",w:2.6,lab:"+ER",t2:"own task-id"},
};
const ORDER=["naive","er_base","sa_nobuf","sa_buf","er_curr","er_own"];
const METL={classil:"class-IL",taskil:"task-IL"};
const MECHL={neuron:"per-neuron",synapse:"per-synapse"};
const OPTL={sgd:"SGD",adam:"Adam"};
const SVGNS="http://www.w3.org/2000/svg";
function el(t,a){const e=document.createElementNS(SVGNS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}

// legend
(function(){const L=document.getElementById("legend");
 ORDER.forEach(r=>{const s=RS[r];const row=document.createElement("div");row.className="lg";
   const sv=`<svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="${s.c}" stroke-width="${s.w}" stroke-dasharray="${s.d==='none'?'':s.d}" stroke-linecap="round"/></svg>`;
   row.innerHTML=sv+`<span>${s.lab}</span>`+(s.t2?`<span class="t2">· ${s.t2}</span>`:"");
   L.appendChild(row);});})();

// cards
(function(){const data=[
  {big:"≤ 0.09",u:"forget",p:"<b>task-IL eval hides forgetting.</b> Every arm keeps forgetting ≤0.09 and accuracy ≥0.90 under task-IL, while the same runs forget up to <b>0.59</b> under class-IL. The 2-way eval masks the damage; it does not repair it."},
  {big:"+0.04",u:"acc",p:"<b>The buffer helps standalone only where forgetting is fast.</b> Meta-replay lifts standalone Adam class-IL (neuron 0.387→0.424) and task-IL (0.909→0.950), but moves SGD ~0 — SGD drifts slowly, so there is little to protect."},
  {big:"+0.092",u:"acc",p:"<b>Own vs curr task-id is optimizer-dependent.</b> Correct task-ids lift learned SGD +ER (neuron 0.568→0.660) yet are ~neutral under Adam. The gate matters where the wrong mask scrambled replay."},
];const C=document.getElementById("cards");
 data.forEach(d=>{const e=document.createElement("div");e.className="card";
   e.innerHTML=`<div class="big">${d.big} <span class="u">${d.u}</span></div><p>${d.p}</p>`;C.appendChild(e);});})();

function chart(series,opts){
  opts=opts||{};
  const W=328,H=196,mL=30,mR=12,mT=10,mB=24;
  const iw=W-mL-mR, ih=H-mT-mB, n=5;
  const X=i=>mL+(n<=1?0:iw*i/(n-1));
  const Y=v=>mT+ih*(1-v);
  const svg=el("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,preserveAspectRatio:"xMidYMid meet",role:"img"});
  // gridlines
  [0,.25,.5,.75,1].forEach(g=>{
    svg.appendChild(el("line",{x1:mL,y1:Y(g),x2:W-mR,y2:Y(g),stroke:"var(--grid)","stroke-width":1}));
    const tx=el("text",{x:mL-6,y:Y(g)+3,"text-anchor":"end","font-size":9,fill:"var(--ink-3)","font-family":"var(--mono)"});
    tx.textContent=g.toFixed(2).replace(/0$/,"");svg.appendChild(tx);});
  // x labels
  for(let i=0;i<n;i++){const tx=el("text",{x:X(i),y:H-8,"text-anchor":"middle","font-size":9,fill:"var(--ink-3)","font-family":"var(--mono)"});tx.textContent="T"+(i+1);svg.appendChild(tx);}
  // series (baselines first, neuromod on top)
  const sorted=series.slice().sort((a,b)=>ORDER.indexOf(a.role)-ORDER.indexOf(b.role));
  sorted.forEach(s=>{const st=RS[s.role];
    const pts=s.traj.map((v,i)=>[X(i),Y(v)]);
    const d=pts.map((p,i)=>(i?"L":"M")+p[0].toFixed(1)+" "+p[1].toFixed(1)).join(" ");
    const pl=el("path",{d,fill:"none",stroke:st.c,"stroke-width":st.w,"stroke-linejoin":"round","stroke-linecap":"round"});
    if(st.d!=="none")pl.setAttribute("stroke-dasharray",st.d);
    svg.appendChild(pl);
    if(s.role==="sa_buf"||s.role==="er_own"||s.role==="er_curr"||s.role==="sa_nobuf"){
      svg.appendChild(el("circle",{cx:pts[n-1][0],cy:pts[n-1][1],r:2.6,fill:st.c}));}
  });
  // hover layer
  const guide=el("line",{x1:0,y1:mT,x2:0,y2:mT+ih,stroke:"var(--ink-3)","stroke-width":1,"stroke-dasharray":"2 2",opacity:0});
  svg.appendChild(guide);
  const hit=el("rect",{x:mL,y:mT,width:iw,height:ih,fill:"transparent"});svg.appendChild(hit);
  const tip=opts.tip;
  function move(ev){const r=svg.getBoundingClientRect();const sx=(ev.clientX-r.left)/r.width*W;
    let i=Math.round((sx-mL)/(iw/(n-1)));i=Math.max(0,Math.min(n-1,i));
    guide.setAttribute("x1",X(i));guide.setAttribute("x2",X(i));guide.setAttribute("opacity",1);
    if(tip){let rows=sorted.slice().sort((a,b)=>b.traj[i]-a.traj[i]).map(s=>{const st=RS[s.role];
      return `<div class="row"><span class="nm"><span class="sw" style="border-top-color:${st.c};border-top-style:${st.d==='none'?'solid':'dashed'}"></span>${st.lab}${st.t2?" · "+st.t2:""}</span><span class="vv">${s.traj[i].toFixed(3)}</span></div>`;}).join("");
      tip.innerHTML=`<div class="th">after T${i+1}</div>`+rows;tip.style.opacity=1;
      const pr=opts.panel.getBoundingClientRect();
      let lx=(ev.clientX-pr.left)+14; if(lx+165>pr.width)lx=(ev.clientX-pr.left)-165;
      tip.style.left=lx+"px";tip.style.top=Math.max(4,(ev.clientY-pr.top)-10)+"px";}
  }
  hit.addEventListener("mousemove",move);
  hit.addEventListener("mouseleave",()=>{guide.setAttribute("opacity",0);if(tip)tip.style.opacity=0;});
  return svg;
}

// build mechanism sections
(function(){const root=document.getElementById("sections");
 const secs=[["neuron","C"],["synapse","C"]];
 ["neuron","synapse"].forEach((mech,mi)=>{
   const h2=document.createElement("h2");h2.innerHTML=`<span class="n">${mi===0?'C1':'C2'}</span>${MECHL[mech]} plasticity`;
   root.appendChild(h2);
   const rule=document.createElement("div");rule.className="rule";root.appendChild(rule);
   const mh=document.createElement("div");mh.className="mech-h";
   mh.innerHTML=`<span class="meta">rows = class-IL / task-IL · columns = SGD / Adam · baselines in grey</span>`;
   root.appendChild(mh);
   const grid=document.createElement("div");grid.className="grid";root.appendChild(grid);
   ["classil","taskil"].forEach(metric=>{["sgd","adam"].forEach(opt=>{
     const key=`${mech}|${metric}|${opt}`;const series=P.grid[key];
     const panel=document.createElement("div");panel.className="panel";
     panel.innerHTML=`<div class="ph"><span class="tt">${METL[metric]} · ${OPTL[opt]}</span><span class="metric">${series.length} arms</span></div>`;
     const tip=document.createElement("div");tip.className="tip";panel.appendChild(tip);
     panel.appendChild(chart(series,{tip,panel}));
     grid.appendChild(panel);});});
 });})();

// disjoint chart
(function(){const host=document.getElementById("disjoint-chart");
 const panel=host.closest(".panel");
 const tip=document.createElement("div");tip.className="tip";panel.appendChild(tip);
 host.appendChild(chart(P.disjoint,{tip,panel}));})();

// table
(function(){const tb=document.querySelector("#tbl tbody");
 const rows=P.table.slice().sort((a,b)=>{
   const o=["neuron","synapse"],m=["classil","taskil"],op=["sgd","adam"];
   return o.indexOf(a.mech)-o.indexOf(b.mech)||m.indexOf(a.metric)-m.indexOf(b.metric)
     ||op.indexOf(a.opt)-op.indexOf(b.opt)||ORDER.indexOf(a.role)-ORDER.indexOf(b.role);});
 rows.forEach(r=>{const tr=document.createElement("tr");
   const st=RS[r.role];const col=st.c;
   tr.innerHTML=`<td>${MECHL[r.mech]}</td><td class="tag">${METL[r.metric]}</td><td class="tag">${OPTL[r.opt]}</td>`+
     `<td><span class="dot" style="background:${col}"></span>${st.lab}${st.t2?' · '+st.t2:''}</td>`+
     `<td class="num">${r.acc.toFixed(4)}</td><td class="num">${r.forget.toFixed(4)}</td>`;
   tb.appendChild(tr);});})();
</script>
"""

(HERE / "pt5_plast_trajectories.html").write_text(HTML)
print("wrote pt5_plast_trajectories.html", len(HTML), "bytes")

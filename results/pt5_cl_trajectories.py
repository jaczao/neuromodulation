"""Regenerate the pt5 continual-learning accuracy-trajectory dashboard from the existing logs.

READ-ONLY over results/*.log — creates NO training runs. Parses every learned-projection GAIN and
PLASTICITY CL run (+ its per-task "After task k/5 | seen tasks: [...]" trajectory) out of the pt5
studies, tags each by its five ablation axes, and emits a self-contained HTML artifact (inline
CSS/JS/SVG, no external assets -> Artifact-CSP-safe).

Selection is by (source log + label substring) rather than debug-tag inference, because the four
studies were logged at different times with different [pt5 debug] field sets. Runs pulled (all
learned projection, gain uses gain_form=unbounded, seed 42, lr 1e-3, ep 5, buffer 1000, 1 seed):
  role      plasticity source                 gain source
  sa_nobuf  pt5_iter3 (standalone, no buffer) pt5_iter3
  sa_buf    pt5_iter3_metareplay (meta-replay)pt5_iter3_gain_metareplay (meta-replay, per-task=own)
  er_curr   pt5_iter3 (+ER, P[t])             pt5_iter3 (+ER, P[t])
  er_own    pt5_plast_init (+ER, own, init.5) pt5_gain_forms_buffer (+ER, own)
  naive/er baselines: pt5_iter3 (shared by both targets, class-IL & task-IL, SGD & Adam).

Five axes -> encoding: standalone vs +ER = blue vs orange hue; buffer vs no-buffer = blue solid vs
dashed; own vs curr task-id = orange solid vs dashed; SGD vs Adam = panel columns; class-IL vs
task-IL = panel rows. Four mechanism grids (plast per-neuron/synapse, gain per-neuron/synapse).

CAVEAT on sa_buf task-id: gain's standalone-buffer line is OWN (its meta-loss forwards each task
under its own gate P[j], per-task by default); plasticity's is CURR-forced (its meta-loss gates the
whole summed gradient once, no per-task split). So the blue-solid line means own for gain, curr for
plasticity — the legend calls it "+buffer"; this note is the difference. +ER own on the learned
projection was only ever run at class-IL, so the task-IL panels have no orange-solid (er_own) line.

Trajectory shown = mean accuracy over tasks-seen-so-far after each task (avg incremental accuracy);
the final point equals the reported final accuracy (asserted == logged avg_final, 0 mismatches).

Run:  uv run python results/pt5_cl_trajectories.py   ->  results/pt5_cl_trajectories.html
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
LOGS = ["pt5_iter3", "pt5_iter3_metareplay", "pt5_iter3_gain_metareplay",
        "pt5_plast_init", "pt5_gain_forms_buffer", "pt5_er_task_id"]

AFTER = re.compile(r"After task \d+/\d+ \| seen tasks: \[([^\]]*)\]")
SUMM = re.compile(r"^>>> (.*?)\s+acc=([\d.]+)\s+forget=([\d.]+)")


def parse():
    recs = []
    for name in LOGS:
        traj = []
        for line in (HERE / f"{name}.log").read_text().splitlines():
            m = AFTER.search(line)
            if m:
                traj.append([float(x) for x in m.group(1).split(",") if x.strip()]); continue
            m = SUMM.match(line)
            if not m:
                continue
            label, acc, forget = m.group(1), float(m.group(2)), float(m.group(3))
            if traj:
                br = re.match(r"\[(\w+)\s+(\w+)\]", label) or re.match(r"\[(\w+)\]", label)
                metric, opt = (br.group(1), br.group(2)) if (br and br.lastindex == 2) else \
                              ("classil", br.group(1)) if br else ("classil", "?")
                recs.append(dict(src=name, label=label, opt=opt, metric=metric, acc=acc,
                                 forget=forget, avg_seen=[round(sum(t) / len(t), 4) for t in traj]))
            traj = []
    return recs


RECS = parse()
_bad = [r for r in RECS if abs(r["avg_seen"][-1] - r["acc"]) > 1e-3]
assert not _bad, f"{len(_bad)} trajectory/final-acc mismatches — parse is wrong"


def pick(src, contains, not_contains, opt, metric):
    for r in RECS:
        if r["src"] != src or r["opt"] != opt or r["metric"] != metric:
            continue
        if all(c in r["label"] for c in contains) and not any(n in r["label"] for n in not_contains):
            return r
    return None


# (target, role) -> (source log, [label substrings], [excluded substrings]); {m} = neuron|synapse
SPEC = {
    ("plast", "sa_nobuf"):   ("pt5_iter3", ["plast-{m} neurom"], ["+er"]),
    ("plast", "sa_buf_cur"): ("pt5_iter3_metareplay", ["plast-{m} meta_replay=ON"], []),
    ("plast", "er_curr"):    ("pt5_iter3", ["plast-{m} neurom+er"], []),
    ("plast", "er_own"):     ("pt5_plast_init", ["plast-{m} er init=0.5"], []),
    ("gain", "sa_nobuf"):    ("pt5_iter3", ["gain-{m} neurom"], ["+er"]),
    ("gain", "sa_buf_cur"):  ("pt5_gain_forms_buffer", ["gain-{m} unbounded buf-meta-cur"], []),
    ("gain", "sa_buf_own"):  ("pt5_iter3_gain_metareplay", ["gain-{m} meta_replay=ON"], []),
    ("gain", "er_curr"):     ("pt5_iter3", ["gain-{m} neurom+er"], []),
    ("gain", "er_own"):      ("pt5_gain_forms_buffer", ["gain-{m} unbounded er-own"], []),
}
# plasticity has NO standalone-buffer OWN variant (its meta-loss gates the whole summed gradient
# ONCE -> curr only, no per-task split), so sa_buf_own is gain-only. gain's sa_buf_cur/er_own live
# in pt5_gain_forms_buffer, which is class-IL only.
TARGET_ROLES = {
    "plast": ["sa_nobuf", "sa_buf_cur", "er_curr", "er_own"],
    "gain":  ["sa_nobuf", "sa_buf_cur", "sa_buf_own", "er_curr", "er_own"],
}
ROLE_LABEL = {
    "naive": "naive baseline", "er_base": "ER baseline",
    "sa_nobuf": "standalone, no buffer",
    "sa_buf_cur": "standalone, +buffer, curr task-id",
    "sa_buf_own": "standalone, +buffer, own task-id",
    "er_curr": "+ER, curr task-id P[t]", "er_own": "+ER, own task-id P[j]",
}
ROLE_ORDER = ["naive", "er_base", "sa_nobuf", "sa_buf_cur", "sa_buf_own", "er_curr", "er_own"]


def series_for(target, mech, metric, opt):
    out = []

    def add(role, rec):
        if rec:
            out.append(dict(role=role, traj=rec["avg_seen"], acc=rec["acc"], forget=rec["forget"]))
    add("naive", pick("pt5_iter3", ["naive (baseline)"], [], opt, metric))
    add("er_base", pick("pt5_iter3", ["er (baseline)"], [], opt, metric))
    for role in TARGET_ROLES[target]:
        src, contains, excl = SPEC[(target, role)]
        add(role, pick(src, [c.format(m=mech) for c in contains], excl, opt, metric))
    return out


GRID = {f"{tgt}-{mech}|{metric}|{opt}": series_for(tgt, mech, metric, opt)
        for tgt in ("plast", "gain") for mech in ("neuron", "synapse")
        for metric in ("classil", "taskil") for opt in ("sgd", "adam")}


# within-study own-vs-curr exhibits (er_task_id study, disjoint projection, class-IL SGD)
def disjoint(tgt):
    lab = "gain-synapse" if tgt == "gain" else "plast-neuron"  # gain-synapse = the striking case
    off = pick("pt5_er_task_id", [lab, "er_task_id=OFF"], [], "sgd", "classil")
    on = pick("pt5_er_task_id", [lab, "er_task_id=ON"], [], "sgd", "classil")
    if not (off and on):
        return None
    return [dict(role="er_curr", traj=off["avg_seen"], acc=off["acc"], forget=off["forget"]),
            dict(role="er_own", traj=on["avg_seen"], acc=on["acc"], forget=on["forget"])]


DISJOINT = {"gain": disjoint("gain"), "plast": disjoint("plast")}

table = [dict(target=k.split("-")[0], mech=k.split("-")[1].split("|")[0],
              metric=k.split("|")[1], opt=k.split("|")[2], role=s["role"],
              acc=s["acc"], forget=s["forget"])
         for k, ser in GRID.items() for s in ser]

payload = dict(grid=GRID, disjoint=DISJOINT, table=table, role_label=ROLE_LABEL,
               role_order=ROLE_ORDER)
DATA = json.dumps(payload, separators=(",", ":"))

HTML = r"""<title>Continual-learning trajectories — gain &amp; plasticity (pt5)</title>
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
.sub{color:var(--ink-2);font-size:15px;max-width:66ch;margin:0 0 4px}
.prov{color:var(--ink-3);font-size:12.5px;margin:14px 0 0;font-family:var(--mono);line-height:1.7}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-2);
  margin:46px 0 4px;font-weight:600}
h2 .n{color:var(--accent);margin-right:8px;font-family:var(--mono);font-size:13px}
.rule{height:1px;background:var(--border);margin:10px 0 22px}
.lead{color:var(--ink-2);font-size:14px;max-width:72ch;margin:0 0 20px}
.lead b{color:var(--ink);font-weight:600}

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

.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:8px 0 6px}
@media(max-width:760px){.cards{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 17px}
.card .big{font-family:var(--mono);font-size:20px;font-weight:600;letter-spacing:-.02em;color:var(--ink)}
.card .big .u{color:var(--ink-3);font-size:13px;font-weight:500}
.card p{margin:7px 0 0;font-size:12.7px;color:var(--ink-2);line-height:1.5}
.card p b{color:var(--ink);font-weight:600}

.mech-h{display:flex;align-items:baseline;gap:12px;margin:8px 0 2px}
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

.exhibit{display:grid;grid-template-columns:300px 1fr;gap:22px;align-items:center;
  background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 20px;margin-bottom:14px}
@media(max-width:680px){.exhibit{grid-template-columns:1fr}}
.exhibit .txt{font-size:12.7px;color:var(--ink-2)}
.exhibit .txt b{color:var(--ink)}
.exhibit .et{font-size:12px;font-weight:640;color:var(--ink);margin-bottom:6px}

.tablewrap{overflow-x:auto;border:1px solid var(--border);border-radius:12px;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:12.5px;min-width:680px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
th{background:var(--surface-2);color:var(--ink-2);font-weight:600;font-size:11px;
  text-transform:uppercase;letter-spacing:.05em;position:sticky;top:0}
td{color:var(--ink)}
td.num{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:var(--surface-2)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:-1px}
.tag{font-family:var(--mono);font-size:11px;color:var(--ink-2)}

.foot{margin-top:40px;color:var(--ink-3);font-size:12px;line-height:1.6;max-width:80ch}
.foot b{color:var(--ink-2)}
</style>

<div class="wrap">
  <h1>Continual-learning trajectories: gain &amp; plasticity</h1>
  <p class="sub">Every learned-projection gain and plasticity run I have on disk, read straight from
  the pt5 logs — no new training. Five ablation axes over average-incremental-accuracy curves on
  Split-MNIST (5 tasks &times; 2 classes).</p>
  <p class="prov">gain: pt5_iter3 · pt5_iter3_gain_metareplay · pt5_gain_forms_buffer<br>
  plasticity: pt5_iter3 · pt5_iter3_metareplay · pt5_plast_init<br>
  own-vs-curr aside: pt5_er_task_id (disjoint) · learned proj · gain_form unbounded · seed 42 · 1 seed · ORACLE task id</p>

  <h2><span class="n">A</span>How to read this</h2>
  <div class="rule"></div>
  <div class="guide">
    <div>
      <h3>Line identity (same in every panel)</h3>
      <div class="legend" id="legend"></div>
    </div>
    <div>
      <h3>The five axes → where they live</h3>
      <div class="axmap">
        <div><span class="k">standalone vs +ER</span><span class="v"><span class="pill">blue</span> vs <span class="pill">orange</span> hue</span></div>
        <div><span class="k">buffer vs no buffer</span><span class="v">blue <span class="pill">dashed/solid</span> vs <span class="pill">dotted</span></span></div>
        <div><span class="k">own vs curr task-id</span><span class="v"><span class="pill">solid</span> vs <span class="pill">dashed</span> · blue &amp; orange</span></div>
        <div><span class="k">SGD vs Adam</span><span class="v">panel <span class="pill">columns</span></span></div>
        <div><span class="k">class-IL vs task-IL</span><span class="v">panel <span class="pill">rows</span></span></div>
      </div>
    </div>
  </div>
  <p class="lead" style="margin-top:16px">Y axis is <b>mean accuracy over tasks seen so far</b> after
  finishing each task (T1…T5); the last point is the reported final accuracy. Flat-high = retains; a
  sagging line = forgetting. Shared 0–1 scale across all panels, so the near-flat task-IL rows are the
  finding (task-IL eval hides forgetting), not a rendering quirk. Hover any panel for exact values;
  the full table is at the end. <b>Blue dash = standalone task-id quality:</b> dotted = no buffer,
  dashed = +buffer curr (P[t] on every replayed sample), solid = +buffer own (each task its own P[j]).
  Plasticity has no standalone-buffer <i>own</i> variant — its meta-loss gates the whole summed
  gradient once — so its buffer line is always curr (blue dashed), never solid.</p>

  <h2><span class="n">B</span>What the curves say</h2>
  <div class="rule"></div>
  <div class="cards" id="cards"></div>

  <div id="sections"></div>

  <h2><span class="n">D</span>Own vs curr, cleanly (disjoint projection)</h2>
  <div class="rule"></div>
  <p class="lead">On the learned projection, own-vs-curr is a cross-study pairing (curr from pt5_iter3,
  own from pt5_plast_init / pt5_gain_forms_buffer). The one <b>within-study</b> own-vs-curr lives on
  the <b>disjoint</b> projection (pt5_er_task_id): same run, flag flipped. Class-IL, SGD, per-neuron.</p>
  <div id="disjoint-wrap"></div>

  <h2><span class="n">E</span>Every cell — final accuracy &amp; forgetting</h2>
  <div class="rule"></div>
  <div class="tablewrap"><table id="tbl"><thead><tr>
    <th>target</th><th>mechanism</th><th>metric</th><th>opt</th><th>arm (line)</th>
    <th class="num">final acc</th><th class="num">forgetting</th></tr></thead><tbody></tbody></table></div>

  <p class="foot"><b>Caveats, kept honest.</b> 1 seed; ORACLE task id at eval (task-IL-style even on the
  class-IL metric); plasticity+Adam is a first-order surrogate (grads gated before <span class="tag">.step()</span>),
  so SGD is the clean read for plasticity; gain is a forward target (Adam legitimate). <b>Coverage
  gaps:</b> +ER own-task-id on the learned projection was only ever run at class-IL, so the task-IL
  panels have no orange-solid line; gain's +buffer <i>curr</i> line (pt5_gain_forms_buffer) is also
  class-IL only, so gain task-IL shows only no-buffer + own. Plasticity has no standalone-buffer
  <i>own</i> variant (its meta-loss gates the summed gradient once), so it never shows blue-solid.
  Gain uses gain_form=unbounded throughout. Numbers are single runs; treat sub-0.02 gaps as noise.</p>
</div>

<script>
const P=""" + DATA + r""";
const RS={
  naive:{c:"var(--c-naive)",d:"3 3",w:1.5,lab:"naive baseline",t2:""},
  er_base:{c:"var(--c-erbase)",d:"1 3",w:1.5,lab:"ER baseline",t2:""},
  sa_nobuf:{c:"var(--c-sa)",d:"2 3",w:1.8,lab:"standalone",t2:"no buffer"},
  sa_buf_cur:{c:"var(--c-sa)",d:"6 4",w:2,lab:"standalone",t2:"+buf · curr"},
  sa_buf_own:{c:"var(--c-sa)",d:"none",w:2.6,lab:"standalone",t2:"+buf · own"},
  er_curr:{c:"var(--c-er)",d:"6 4",w:2,lab:"+ER",t2:"curr task-id"},
  er_own:{c:"var(--c-er)",d:"none",w:2.6,lab:"+ER",t2:"own task-id"},
};
const ORDER=P.role_order;
const METL={classil:"class-IL",taskil:"task-IL"};
const TGTL={gain:"gain",plast:"plasticity"};
const MECHL={neuron:"per-neuron",synapse:"per-synapse"};
const OPTL={sgd:"SGD",adam:"Adam"};
const SVGNS="http://www.w3.org/2000/svg";
function el(t,a){const e=document.createElementNS(SVGNS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}

(function(){const L=document.getElementById("legend");
 ORDER.forEach(r=>{const s=RS[r];const row=document.createElement("div");row.className="lg";
   const sv=`<svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="${s.c}" stroke-width="${s.w}" stroke-dasharray="${s.d==='none'?'':s.d}" stroke-linecap="round"/></svg>`;
   row.innerHTML=sv+`<span>${s.lab}</span>`+(s.t2?`<span class="t2">· ${s.t2}</span>`:"");
   L.appendChild(row);});})();

(function(){const data=[
  {big:"≤ 0.09",u:"forget",p:"<b>task-IL eval hides forgetting.</b> Every arm keeps forgetting ≤0.09 and accuracy ≥0.90 under task-IL, while the same runs forget up to <b>0.59</b> under class-IL. The 2-way eval masks the damage; it does not repair it."},
  {big:"0.74 → 0.99",u:"own gate",p:"<b>Gain's buffer win is mostly the per-task OWN gate.</b> gain-synapse class-IL SGD: +buffer <i>curr</i> 0.739, +buffer <i>own</i> <b>0.987</b> (+0.248); the buffer alone barely clears no-buffer (0.629). Plasticity's buffer (curr-only) stays inert at 0.642."},
  {big:"0.99",u:"gain +ER Adam",p:"<b>Gain +ER with own task-ids ≈ ceiling under Adam.</b> gain-neuron/synapse reach 0.989/0.990 vs ER 0.905 — the learned-projection win. Plasticity+ER stays ≈ER (≤0.906) at best."},
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
  [0,.25,.5,.75,1].forEach(g=>{
    svg.appendChild(el("line",{x1:mL,y1:Y(g),x2:W-mR,y2:Y(g),stroke:"var(--grid)","stroke-width":1}));
    const tx=el("text",{x:mL-6,y:Y(g)+3,"text-anchor":"end","font-size":9,fill:"var(--ink-3)","font-family":"var(--mono)"});
    tx.textContent=g.toFixed(2).replace(/0$/,"");svg.appendChild(tx);});
  for(let i=0;i<n;i++){const tx=el("text",{x:X(i),y:H-8,"text-anchor":"middle","font-size":9,fill:"var(--ink-3)","font-family":"var(--mono)"});tx.textContent="T"+(i+1);svg.appendChild(tx);}
  const sorted=series.slice().sort((a,b)=>ORDER.indexOf(a.role)-ORDER.indexOf(b.role));
  sorted.forEach(s=>{const st=RS[s.role];
    const pts=s.traj.map((v,i)=>[X(i),Y(v)]);
    const d=pts.map((p,i)=>(i?"L":"M")+p[0].toFixed(1)+" "+p[1].toFixed(1)).join(" ");
    const pl=el("path",{d,fill:"none",stroke:st.c,"stroke-width":st.w,"stroke-linejoin":"round","stroke-linecap":"round"});
    if(st.d!=="none")pl.setAttribute("stroke-dasharray",st.d);
    svg.appendChild(pl);
    if(s.role!=="naive"&&s.role!=="er_base"){
      svg.appendChild(el("circle",{cx:pts[n-1][0],cy:pts[n-1][1],r:2.6,fill:st.c}));}
  });
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

function buildPanel(series,title,metricTxt){
  const panel=document.createElement("div");panel.className="panel";
  panel.innerHTML=`<div class="ph"><span class="tt">${title}</span><span class="metric">${metricTxt}</span></div>`;
  const tip=document.createElement("div");tip.className="tip";panel.appendChild(tip);
  panel.appendChild(chart(series,{tip,panel}));
  return panel;
}

(function(){const root=document.getElementById("sections");
 const MECHS=[["plast","neuron","C1"],["plast","synapse","C2"],["gain","neuron","C3"],["gain","synapse","C4"]];
 MECHS.forEach(([tgt,mech,tag])=>{
   const h2=document.createElement("h2");h2.innerHTML=`<span class="n">${tag}</span>${MECHL[mech]} ${TGTL[tgt]}`;
   root.appendChild(h2);
   const rule=document.createElement("div");rule.className="rule";root.appendChild(rule);
   const mh=document.createElement("div");mh.className="mech-h";
   const note=tgt==="gain"?"blue: dotted no-buf · dashed +buf curr · solid +buf own":"blue: dotted no-buf · dashed +buf curr (no own variant)";
   mh.innerHTML=`<span class="meta">rows = class-IL / task-IL · columns = SGD / Adam · ${note}</span>`;
   root.appendChild(mh);
   const grid=document.createElement("div");grid.className="grid";root.appendChild(grid);
   ["classil","taskil"].forEach(metric=>{["sgd","adam"].forEach(opt=>{
     const series=P.grid[`${tgt}-${mech}|${metric}|${opt}`];
     grid.appendChild(buildPanel(series,`${METL[metric]} · ${OPTL[opt]}`,`${series.length} arms`));});});
 });})();

(function(){const wrap=document.getElementById("disjoint-wrap");
 [["gain","per-synapse gain","own lifts <b>0.2576→0.6133</b> (+0.356): the wrong-task mask P[t] was scrambling replayed samples through the wrong synapse gate; routing each through its own P[j] fixes it. (Per-<i>neuron</i> gain is ~neutral, 0.826→0.816 — its disjoint subnet already isolates each task, so mis-routing barely matters.)"],
  ["plast","per-neuron plasticity","own lifts <b>0.4483→0.4833</b> (+0.035) and cuts forgetting 0.458→0.348 (−0.110): a rescue of the ablation, still far below replay's reach."]].forEach(([tgt,mlab,txt])=>{
   const d=P.disjoint[tgt];if(!d)return;
   const ex=document.createElement("div");ex.className="exhibit";
   const left=document.createElement("div");left.className="panel";left.style.border="none";left.style.padding="0";
   const tip=document.createElement("div");tip.className="tip";left.appendChild(tip);
   left.appendChild(chart(d,{tip,panel:left}));
   const right=document.createElement("div");right.className="txt";
   right.innerHTML=`<div class="et">${mlab} · disjoint · class-IL · SGD</div>${txt}`;
   ex.appendChild(left);ex.appendChild(right);wrap.appendChild(ex);});})();

(function(){const tb=document.querySelector("#tbl tbody");
 const T=["gain","plast"],M=["neuron","synapse"],me=["classil","taskil"],op=["sgd","adam"];
 const rows=P.table.slice().sort((a,b)=>
   T.indexOf(a.target)-T.indexOf(b.target)||M.indexOf(a.mech)-M.indexOf(b.mech)||
   me.indexOf(a.metric)-me.indexOf(b.metric)||op.indexOf(a.opt)-op.indexOf(b.opt)||
   ORDER.indexOf(a.role)-ORDER.indexOf(b.role));
 rows.forEach(r=>{const tr=document.createElement("tr");const st=RS[r.role];
   tr.innerHTML=`<td>${TGTL[r.target]}</td><td>${MECHL[r.mech]}</td><td class="tag">${METL[r.metric]}</td>`+
     `<td class="tag">${OPTL[r.opt]}</td>`+
     `<td><span class="dot" style="background:${st.c}"></span>${st.lab}${st.t2?' · '+st.t2:''}</td>`+
     `<td class="num">${r.acc.toFixed(4)}</td><td class="num">${r.forget.toFixed(4)}</td>`;
   tb.appendChild(tr);});})();
</script>
"""

(HERE / "pt5_cl_trajectories.html").write_text(HTML)
print("panels:", len(GRID), "| table rows:", len(table),
      "| disjoint gain:", [round(x["acc"], 4) for x in DISJOINT["gain"]] if DISJOINT["gain"] else None,
      "| plast:", [round(x["acc"], 4) for x in DISJOINT["plast"]] if DISJOINT["plast"] else None)
for k in ("gain-neuron|classil|sgd", "gain-synapse|classil|sgd", "gain-neuron|taskil|adam"):
    print(k, [(s["role"], s["acc"]) for s in GRID[k]])
print("wrote pt5_cl_trajectories.html", len(HTML), "bytes")

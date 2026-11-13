"""Campus Silence Heatmap - single-file web app (FastAPI + ML + real campus map).
Run:  python app.py   ->  http://localhost:8000
Phone on same Wi-Fi: http://<your-computer-IP>:8000
Admin key: env ADMIN_API_KEY (default campus123)
"""
import json, os, uuid, threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor

BASE = Path(__file__).resolve().parent.parent
DATA_FILE = BASE / "campus_data.json"
LOCK = threading.Lock()
ADMIN_KEY = os.getenv("ADMIN_API_KEY", "campus123")   # change in production!

CFG_FILE = BASE / "campus_config.json"
CAMPUS_DEFAULT = {"name": "Yashoda Technical Campus, Satara", "lat": 17.68589, "lng": 73.99333}  # Satara centre; fine-tune in Admin

def campus() -> dict:
    try:
        return {**CAMPUS_DEFAULT, **json.loads(CFG_FILE.read_text())}
    except Exception:
        return dict(CAMPUS_DEFAULT)

SLOTS = ["morning", "afternoon", "evening", "night"]
CATS = ["Library", "Classroom", "Cafeteria", "Hostel", "Sports", "Lab", "Outdoor"]
# event context: dB offset applied on top of the model output
EVENTS = {"normal": 0, "exam": -6, "event": 8, "holiday": -4}

# ---------- schemas ----------
class ZoneIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    cat: str
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)
    db: Dict[str, float]          # {"morning":.., "afternoon":.., ...} each 20-110

class CampusIn(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)

class PosIn(BaseModel):
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)

class Zone(ZoneIn):
    id: str

# ---------- storage (creator-defined fixed data) ----------
SAMPLE = [
 ("Central Library","Library",20,22,[34,38,36,30]),("Reading Hall","Library",32,30,[36,40,38,32]),
 ("Main Canteen","Cafeteria",70,40,[62,78,70,48]),("Block A Classrooms","Classroom",45,55,[58,64,50,36]),
 ("Computer Lab","Lab",25,62,[44,48,46,40]),("Boys Hostel","Hostel",78,78,[52,50,60,42]),
 ("Girls Hostel","Hostel",55,82,[50,48,58,40]),("Sports Ground","Sports",85,18,[68,72,76,38]),
 ("Garden Bench","Outdoor",12,80,[42,46,44,36]),("Admin Block","Classroom",62,20,[55,60,48,34])]

def sample_zones() -> List[dict]:
    return [dict(id=str(uuid.uuid4()), name=n, cat=c, x=x, y=y, db=dict(zip(SLOTS, d)))
            for n, c, x, y, d in SAMPLE]

def load() -> List[dict]:
    if not DATA_FILE.exists():
        save(sample_zones())
    return json.loads(DATA_FILE.read_text())

def save(zones: List[dict]):
    DATA_FILE.parent.mkdir(exist_ok=True)
    tmp = DATA_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(zones, indent=2))
    tmp.replace(DATA_FILE)
    train()                       # retrain whenever creator data changes

# ---------- ML ----------
model: Optional[RandomForestRegressor] = None

def features(z: dict, slot: str) -> List[float]:
    cat = [1.0 if z["cat"] == c else 0.0 for c in CATS]
    vals = [z["db"][s] for s in SLOTS]
    return cat + [SLOTS.index(slot), z["x"], z["y"], float(np.mean(vals)), float(np.std(vals))]

def train():
    """RandomForest learns: (category, time slot, position, zone profile) -> dB."""
    global model
    zones = json.loads(DATA_FILE.read_text())
    if len(zones) < 3:
        model = None
        return
    X = [features(z, s) for z in zones for s in SLOTS]
    y = [z["db"][s] for z in zones for s in SLOTS]
    model = RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=42).fit(X, y)

def predict_db(z: dict, slot: str, event: str = "normal") -> float:
    own = z["db"][slot]
    base = own if model is None else 0.6 * own + 0.4 * float(model.predict([features(z, slot)])[0])
    return round(float(np.clip(base + EVENTS.get(event, 0), 20, 110)), 1)

def category_of(db: float) -> str:
    return "quiet" if db < 45 else "moderate" if db < 60 else "noisy" if db < 70 else "very noisy"

def current_slot() -> str:
    h = datetime.now().hour
    return "morning" if 5 <= h < 12 else "afternoon" if h < 17 else "evening" if h < 21 else "night"

def check_zone(z: ZoneIn):
    if z.cat not in CATS:
        raise HTTPException(422, f"cat must be one of {CATS}")
    if set(z.db) != set(SLOTS) or not all(20 <= v <= 110 for v in z.db.values()):
        raise HTTPException(422, "db needs all 4 slots, each 20-110")

def admin(x_api_key: str = Header(default="")):
    if x_api_key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin API key")

# ---------- app ----------
app = FastAPI(title="Campus Silence Heatmap")

@app.on_event("startup")
def startup():
    load(); train()

# public (read-only for users)
@app.get("/api/meta")
def meta():
    return {"slots": SLOTS, "cats": CATS, "events": list(EVENTS), "current_slot": current_slot(), "campus": campus()}

@app.get("/api/zones", response_model=List[Zone])
def zones():
    return load()

@app.get("/api/predict")
def predict(slot: str = "afternoon", event: str = "normal"):
    """ML-predicted dB for every zone in a slot (sorted quietest first)."""
    if slot not in SLOTS or event not in EVENTS:
        raise HTTPException(422, "bad slot or event")
    out = [{"id": z["id"], "name": z["name"], "cat": z["cat"], "entered": z["db"][slot],
            "predicted": (p := predict_db(z, slot, event)), "level": category_of(p)} for z in load()]
    return sorted(out, key=lambda r: r["predicted"])

@app.get("/api/forecast/{zone_id}")
def forecast(zone_id: str, event: str = "normal"):
    """Future prediction: next 4 time slots starting from the current slot."""
    z = next((z for z in load() if z["id"] == zone_id), None)
    if not z:
        raise HTTPException(404, "zone not found")
    i = SLOTS.index(current_slot())
    seq = [SLOTS[(i + k) % 4] for k in range(1, 5)]
    return {"zone": z["name"], "forecast": [{"slot": s, "predicted": predict_db(z, s, event),
            "level": category_of(predict_db(z, s, event))} for s in seq]}

@app.get("/api/quietest")
def quietest(slot: str = "afternoon", cat: str = ""):
    rows = [r for r in predict(slot) if not cat or r["cat"] == cat]
    if not rows:
        raise HTTPException(404, "no zones")
    return rows[0]

# admin (API key required)
@app.post("/api/admin/zones", response_model=Zone, dependencies=[Depends(admin)])
def add_zone(z: ZoneIn):
    check_zone(z)
    with LOCK:
        data = load()
        name = z.name.strip()
        if any(d["name"].lower() == name.lower() for d in data):
            raise HTTPException(409, f"Zone '{name}' already exists")
        new = {"id": str(uuid.uuid4()), **z.dict(), "name": name}
        save(data + [new])
    return new

@app.delete("/api/admin/zones/{zone_id}", dependencies=[Depends(admin)])
def delete_zone(zone_id: str):
    save([z for z in load() if z["id"] != zone_id])
    return {"ok": True}

@app.put("/api/admin/zones/{zone_id}/pos", dependencies=[Depends(admin)])
def move_zone(zone_id: str, p: PosIn):
    with LOCK:
        data = load()
        z = next((z for z in data if z["id"] == zone_id), None)
        if not z:
            raise HTTPException(404, "zone not found")
        z["x"], z["y"] = p.x, p.y
        save(data)
    return z

@app.post("/api/admin/campus", dependencies=[Depends(admin)])
def set_campus(c: CampusIn):
    CFG_FILE.write_text(json.dumps({"lat": c.lat, "lng": c.lng}))
    return campus()

@app.post("/api/admin/sample", dependencies=[Depends(admin)])
def load_sample():
    save(sample_zones()); return {"ok": True}

@app.delete("/api/admin/zones", dependencies=[Depends(admin)])
def clear_all():
    save([]); return {"ok": True}

@app.post("/api/admin/check", dependencies=[Depends(admin)])
def check_key():
    return {"ok": True}

# frontend (embedded, so the whole app is ONE file)
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Campus Silence Heatmap</title>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<style>
:root{--bg:#0a1320;--panel:#0f1a2b;--line:#1f3048;--text:#f2f6fa;--mut:#8fa0b5;--acc:#4fe0b0;--sel:#1e3050;--q:#4fe0b0;--m:#f2cc54;--n:#f19a56;--v:#ee5d6c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:Manrope,system-ui,sans-serif;min-height:100vh}
.mono{font-family:'JetBrains Mono',ui-monospace,monospace;letter-spacing:.12em;text-transform:uppercase;font-size:12px}
header{display:flex;align-items:center;gap:16px;padding:20px 20px 0;max-width:900px;margin:auto}
.logo{width:46px;height:46px;border:1px solid var(--acc);background:#10352f;display:grid;place-items:center;color:var(--acc);font-size:20px}
h1{font-size:18px;margin:0;letter-spacing:.06em}h1 span{color:var(--acc);font-size:13px;margin-left:6px}
.wrap{max-width:900px;margin:auto;padding:0 20px 60px}
.btn{background:var(--acc);color:#04201a;border:0;padding:14px 18px;font:700 15px Manrope;cursor:pointer;margin:22px 0}
.btn.ghost{background:transparent;color:var(--text);border:1px solid var(--line);margin:0;padding:8px 12px;font-size:13px}
nav{display:flex;border-bottom:1px solid var(--line);overflow-x:auto;margin:0 -20px;padding:0 20px}
nav button{background:none;border:0;color:var(--mut);padding:14px 16px;font:400 14px 'JetBrains Mono',monospace;letter-spacing:.08em;text-transform:uppercase;cursor:pointer;white-space:nowrap}
nav button.on{background:var(--sel);color:var(--text)}
.eyebrow{color:var(--acc);margin:28px 0 8px}
h2{font-size:44px;font-weight:300;margin:0 0 10px;line-height:1.05;letter-spacing:-.02em}
.sub{color:var(--mut);font-size:18px;line-height:1.5;margin:0 0 22px;max-width:560px}
.box{border:1px solid var(--line);padding:18px;margin-bottom:20px;background:var(--panel)}
.pill{display:flex;gap:12px;align-items:center;padding:14px 18px;border:1px solid var(--line);margin-bottom:22px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--acc);display:inline-block}
label{display:block;color:var(--mut);margin:14px 0 8px}label:first-child{margin-top:0}
select,input{width:100%;background:#0a1220;border:1px solid var(--line);color:var(--text);padding:16px;font:400 17px Manrope;border-radius:0}
select:focus,input:focus,button:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.map{position:relative;aspect-ratio:1/1.05;border:1px solid var(--line);background:linear-gradient(var(--line) 1px,transparent 1px) 0 0/12.5% 12.5%,linear-gradient(90deg,var(--line) 1px,transparent 1px) 0 0/12.5% 12.5%;overflow:hidden}
.road{position:absolute;background:#1b2b45;width:9%;height:140%;left:55%;top:-20%;transform:rotate(25deg)}
.road.b{width:9%;height:130%;left:45%;top:-15%;transform:rotate(-76deg)}
.sec{position:absolute;color:#34496a}.sec.n{top:6%;left:6%}.sec.s{bottom:4%;right:6%}
.empty{position:absolute;inset:0;display:grid;place-content:center;text-align:center;padding:20px}
.empty b{font-size:22px;font-weight:600}.empty p{color:var(--mut);margin:8px 0 0}
.zone{position:absolute;transform:translate(-50%,-50%);border-radius:50%;border:2px solid rgba(255,255,255,.5);cursor:pointer;display:grid;place-items:center;font:700 11px 'JetBrains Mono';color:#06121c;transition:transform .15s}
.zone:hover,.zone.hl{transform:translate(-50%,-50%) scale(1.25);z-index:5}
.zone.hl{box-shadow:0 0 0 6px rgba(79,224,176,.35)}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 18px;color:var(--mut);font-size:14px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px}
.tip{margin-top:14px;padding:14px;border:1px solid var(--line);font-size:15px;min-height:56px;color:var(--mut)}
.tip b{color:var(--text)}
.slots{display:flex;border:1px solid var(--line);margin-top:16px}
.slots button{flex:1;background:none;border:0;color:var(--mut);padding:14px 6px;font:400 13px 'JetBrains Mono';text-transform:uppercase;cursor:pointer}
.slots button.on{background:var(--sel);color:var(--text)}
.row{display:grid;grid-template-columns:1fr 70px;gap:6px 12px;align-items:center;padding:14px 0;border-bottom:1px solid var(--line);cursor:pointer}
.row small{color:var(--mut);grid-column:1/-1}.bar{height:6px;background:#16253b;grid-column:1/-1}.bar i{display:block;height:100%}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px}
.stat{border:1px solid var(--line);padding:18px;background:var(--panel)}
.stat b{display:block;font:700 30px 'JetBrains Mono';margin-top:18px;overflow-wrap:anywhere}
.chart{display:flex;align-items:flex-end;gap:18px;height:240px;border-bottom:1px solid var(--line);padding:0 10px;margin-top:20px}
.chart div{flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;height:100%;font-size:13px;color:var(--mut)}
.chart i{width:100%;background:var(--acc);min-height:2px;margin-top:4px}
.axis{display:flex;gap:18px;padding:8px 10px;color:var(--mut)}.axis span{flex:1;text-align:center}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.zl{display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-bottom:1px solid var(--line);gap:10px}
.note{color:var(--mut);font-size:13px;margin-top:10px}
@media(max-width:520px){h2{font-size:38px}.grid{grid-template-columns:1fr}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}

.mapwrap{position:relative}.lmap,#map{height:430px;border:1px solid var(--line);background:#0a1220}
.empty{z-index:500;pointer-events:none;background:rgba(10,19,32,.7)}
.pin{width:34px;height:34px;border-radius:50%;border:2px solid rgba(255,255,255,.8);display:grid;place-items:center;font:700 11px 'JetBrains Mono';color:#06121c;box-shadow:0 2px 8px rgba(0,0,0,.5)}
.pin.hl{box-shadow:0 0 0 6px rgba(79,224,176,.45);transform:scale(1.2)}
.leaflet-container{font-family:Manrope,sans-serif}
</style>
</head>
<body>
<header><div class="logo"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 5 6 9H2v6h4l5 4z"/><path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a10 10 0 0 1 0 14"/></svg></div><h1>Campus Silence <span class="mono">Heatmap</span></h1></header>
<div class="wrap">
<button class="btn" id="quietBtn"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:8px"><path d="m12 3 1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 15v4M17 17h4"/></svg>Find quietest</button>
<nav id="tabs"><button data-t="heat" class="on"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:8px"><circle cx="12" cy="12" r="10"/><path d="m16 8-2 6-6 2 2-6z"/></svg>Heatmap</button><button data-t="pred"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:8px"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>Predictions</button><button data-t="ins"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:8px"><path d="M3 3v18h18"/><path d="M18 17V9M13 17V5M8 17v-3"/></svg>Insights</button><button data-t="admin"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:8px"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/></svg>Admin</button></nav>

<section id="heat">
<div class="mono eyebrow">Campus acoustic monitor / 01</div>
<h2>Find your quiet corner.</h2>
<p class="sub">Explore creator-defined noise readings and choose the right place to focus.</p>
<div class="pill mono"><span class="dot"></span>Fixed dataset <b id="cnt">0 zones</b></div>
<div class="box">
<label class="mono">Time slot</label><select id="slot"></select>
<label class="mono">Noise category</label><select id="cat"></select>
<label class="mono">Search locations</label><input id="q" placeholder="Library, room code...">
</div>
<div class="box">
<div class="mono" style="color:var(--acc)">Yashoda Technical Campus, Satara</div>
<h3 id="mapTitle" style="font-size:26px;font-weight:300;margin:12px 0">Noise field</h3>
<div class="legend"><span><i style="background:var(--q)"></i>quiet</span><span><i style="background:var(--m)"></i>moderate</span><span><i style="background:var(--n)"></i>noisy</span><span><i style="background:var(--v)"></i>very noisy</span></div>
<div class="mapwrap"><div id="map"></div><div class="empty" id="empty"><div><b>No fixed zones yet</b><p>Add a location from Admin to populate the map.</p></div></div></div>
<div class="tip" id="tip">Tap a zone to see its reading and AI forecast.</div>
<p class="note mono">Readings are creator-defined, not live microphone data &middot; <b id="vis">0 visible</b></p>
</div>
</section>

<section id="pred" hidden>
<div class="box" style="margin-top:24px">
<div class="mono" style="color:var(--mut)">Random Forest forecast model</div>
<h3 style="font-size:26px;font-weight:300;margin:8px 0">Likely conditions by time slot</h3>
<p class="sub" style="font-size:17px;margin:0 0 6px">Trained on the creator's fixed readings. Tap a zone for its next-4-slots forecast.</p>
<label class="mono">Campus context</label><select id="event"></select>
<div class="slots" id="pslots"></div></div>
<div id="plist"></div>
<div class="box" id="fc" hidden></div>
</section>

<section id="ins" hidden>
<div class="mono eyebrow">Campus acoustic monitor / 01</div>
<h2>A clearer read on campus.</h2>
<p class="sub">Aggregate view of the creator-entered readings.</p>
<div class="stats">
<div class="stat"><span class="mono" style="color:var(--mut)">Mapped zones</span><b id="s1">0</b></div>
<div class="stat"><span class="mono" style="color:var(--mut)">Quiet zones</span><b id="s2">0</b></div>
<div class="stat"><span class="mono" style="color:var(--mut)">Campus average</span><b id="s3">0 dB</b></div>
<div class="stat"><span class="mono" style="color:var(--mut)">Best study spot</span><b id="s4" style="font-size:20px">&mdash;</b></div></div>
<div class="box"><div class="mono" style="color:var(--acc)">Campus rhythm</div>
<h3 style="font-size:26px;font-weight:300;margin:12px 0 4px">Average noise by time slot</h3>
<div class="chart" id="chart"></div><div class="axis mono" id="axis"></div></div>
</section>

<section id="admin" hidden>
<div class="mono eyebrow">Admin</div>
<h2>Add a zone.</h2>
<p class="sub">Only the creator can change data. Enter the admin API key, then add zones (map x: west&rarr;east, y: north&rarr;south, 0&ndash;100).</p>
<div class="box"><label class="mono">Admin API key</label><input id="key" type="password" placeholder="X-API-Key"></div>
<div class="box">
<div class="lmap" id="amap"></div><p class="note">Tap the map to place a new zone &middot; drag saved markers to reposition them.</p><button class="btn ghost" id="setC" style="margin:0 0 6px">Set campus centre to current map view</button>
<label class="mono">Location name</label><input id="aName" placeholder="Central Library, Room 204">
<label class="mono">Category</label><select id="aCat"></select>
<div class="grid"><div><label class="mono">Map x (0-100)</label><input id="aX" type="number" min="0" max="100" value="50"></div>
<div><label class="mono">Map y (0-100)</label><input id="aY" type="number" min="0" max="100" value="50"></div></div>
<div class="grid" id="aDb"></div>
<button class="btn" id="aAdd" style="margin-bottom:0">Save zone</button>
<p class="note" id="aMsg"></p></div>
<div class="box"><div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap">
<b>Saved zones</b><span><button class="btn ghost" id="sample">Reset to sample data</button> <button class="btn ghost" id="clear">Clear all</button></span></div>
<div id="zl"></div></div>
</section>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script>
let HM,AM,HL,AL,TMP,CAMPUS={lat:17.68589,lng:73.99333},BOX;
const clamp=n=>Math.max(0,Math.min(100,n));
function setBox(){const a=.0016,b=.0017;BOX={n:CAMPUS.lat+a,s:CAMPUS.lat-a,w:CAMPUS.lng-b,e:CAMPUS.lng+b}}
const toLL=(x,y)=>[BOX.n-y/100*(BOX.n-BOX.s),BOX.w+x/100*(BOX.e-BOX.w)];
const toXY=ll=>[clamp((ll.lng-BOX.w)/(BOX.e-BOX.w)*100),clamp((BOX.n-ll.lat)/(BOX.n-BOX.s)*100)];
function mkMap(id){const m=L.map(id).setView([CAMPUS.lat,CAMPUS.lng],18);
 const sat=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,attribution:'Imagery &copy; Esri'}).addTo(m);
 const st=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; OpenStreetMap'});
 L.control.layers({Satellite:sat,Street:st}).addTo(m);
 L.rectangle([[BOX.s,BOX.w],[BOX.n,BOX.e]],{color:'#4fe0b0',weight:1,fill:false,dashArray:'4'}).addTo(m);return m}
const icon=(d,h)=>L.divIcon({className:'',html:`<div class="pin ${h?'hl':''}" style="background:${col(d)}">${Math.round(d)}</div>`,iconSize:[34,34],iconAnchor:[17,17]});
function drawPins(){if(!HL)return;HL.clearLayers();
 visible().forEach(z=>L.marker(toLL(z.x,z.y),{icon:icon(z.db[slot],z.id===hl)}).bindTooltip(z.name).on('click',()=>showTip(z.id)).addTo(HL))}
function drawAdminPins(){if(!AL)return;AL.clearLayers();
 zones.forEach(z=>L.marker(toLL(z.x,z.y),{icon:icon(z.db[slot],false),draggable:true}).bindTooltip(z.name).addTo(AL).on('dragend',async e=>{
  const[x,y]=toXY(e.target.getLatLng());try{await api('/admin/zones/'+z.id+'/pos',{method:'PUT',body:JSON.stringify({x,y})},true);z.x=x;z.y=y;msg('Moved '+z.name+'.')}catch(err){msg('Error: '+err.message);loadZones()}}))}
let SLOTS=[],CATS=[],EVENTS=[],zones=[],slot='afternoon',pslot='afternoon',hl=null;
const $=id=>document.getElementById(id);
const avg=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:0;
const level=db=>db<45?['quiet','--q']:db<60?['moderate','--m']:db<70?['noisy','--n']:['very noisy','--v'];
const col=db=>`var(${level(db)[1]})`;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function api(path,opts={},admin=false){
 const h={'Content-Type':'application/json'};if(admin)h['X-API-Key']=$('key').value;
 const r=await fetch('/api'+path,{...opts,headers:h});
 if(!r.ok){let m=r.statusText;try{const j=await r.json();m=typeof j.detail==='string'?j.detail:JSON.stringify(j.detail)}catch(e){}throw new Error(m)}
 return r.json()}
const loadZones=async()=>{zones=await api('/zones');render()};

async function init(){
 const m=await api('/meta');SLOTS=m.slots;CATS=m.cats;EVENTS=m.events;slot=pslot=m.current_slot;CAMPUS=m.campus;setBox();
 $('slot').innerHTML=SLOTS.map(s=>`<option ${s===slot?'selected':''}>${s}</option>`).join('');
 $('cat').innerHTML='<option value="">All categories</option>'+CATS.map(c=>`<option>${c}</option>`).join('');
 $('aCat').innerHTML=CATS.map(c=>`<option>${c}</option>`).join('');
 $('event').innerHTML=EVENTS.map(e=>`<option>${e}</option>`).join('');
 $('aDb').innerHTML=SLOTS.map(s=>`<div><label class="mono">${s} dB</label><input type="number" min="20" max="110" value="45" data-s="${s}"></div>`).join('');
 $('pslots').innerHTML=SLOTS.map(s=>`<button data-s="${s}">${s}</button>`).join('');
 HM=mkMap('map');HL=L.layerGroup().addTo(HM);AM=mkMap('amap');AL=L.layerGroup().addTo(AM);
 AM.on('click',e=>{const[x,y]=toXY(e.latlng);$('aX').value=x.toFixed(1);$('aY').value=y.toFixed(1);
  if(TMP)AM.removeLayer(TMP);TMP=L.circleMarker(e.latlng,{radius:9,color:'#fff',fillColor:'#4fe0b0',fillOpacity:1}).addTo(AM);msg('Position set. Fill the details and Save zone.')});
 $('setC').onclick=async()=>{const c=AM.getCenter();try{await api('/admin/campus',{method:'POST',body:JSON.stringify({lat:c.lat,lng:c.lng})},true);CAMPUS={...CAMPUS,lat:c.lat,lng:c.lng};setBox();msg('Campus centre saved. Zone positions now map around it.');loadZones()}catch(e){msg('Error: '+e.message)}};
 await loadZones()}

$('tabs').onclick=e=>{const t=e.target.closest('button');if(t)go(t.dataset.t)};
function go(t){document.querySelectorAll('#tabs button').forEach(b=>b.classList.toggle('on',b.dataset.t===t));
 ['heat','pred','ins','admin'].forEach(id=>$(id).hidden=id!==t);render();setTimeout(()=>{HM&&HM.invalidateSize();AM&&AM.invalidateSize()},50)}

// heatmap
const visible=()=>{const q=$('q').value.trim().toLowerCase(),c=$('cat').value;
 return zones.filter(z=>(!c||z.cat===c)&&(!q||(z.name+' '+z.cat).toLowerCase().includes(q)))};
function drawHeat(){
 const v=visible();$('cnt').textContent=zones.length+' zones';$('vis').textContent=v.length+' visible';
 $('mapTitle').textContent='Noise field / '+slot;
 $('empty').style.display=v.length?'none':'grid';
 $('empty').innerHTML=zones.length?'<div><b>No matches</b><p>Try another category or search term.</p></div>':'<div><b>No fixed zones yet</b><p>Add a location from Admin to populate the map.</p></div>';
 drawPins()}
async function showTip(id){const z=zones.find(z=>z.id===id);if(!z)return;hl=id;const d=z.db[slot];drawHeat();if(HM)HM.panTo(toLL(z.x,z.y));
 $('tip').innerHTML=`<b>${esc(z.name)}</b> (${z.cat}) &mdash; ${d} dB, ${level(d)[0]} in the ${slot}.<br>All slots: ${SLOTS.map(s=>s+' '+z.db[s]).join(' / ')}<br><span id="nx">Loading forecast…</span>`;
 try{const f=await api('/forecast/'+id);$('nx').innerHTML='<b>AI forecast:</b> '+f.forecast.map(x=>`${x.slot} ${x.predicted} dB`).join(' &rarr; ')}catch(e){$('nx').textContent=''}}
$('slot').onchange=e=>{slot=e.target.value;drawHeat()};$('cat').onchange=drawHeat;$('q').oninput=drawHeat;

$('quietBtn').onclick=async()=>{go('heat');
 try{const b=await api(`/quietest?slot=${slot}&cat=${encodeURIComponent($('cat').value)}`);
  await showTip(b.id);$('tip').innerHTML=`<b>Quietest (AI-predicted): ${esc(b.name)}</b> &mdash; ${b.predicted} dB in the ${slot}.`;
  $('map').scrollIntoView({behavior:'smooth',block:'center'})}
 catch(e){$('tip').innerHTML='<b>Nothing to rank yet.</b> Add zones in Admin first.'}};

// predictions (from backend ML model)
async function drawPred(){
 document.querySelectorAll('#pslots button').forEach(b=>b.classList.toggle('on',b.dataset.s===pslot));
 if(!zones.length){$('plist').innerHTML='<div style="text-align:center;padding:60px 20px"><b style="font-size:20px">Predictions will appear here</b><p style="color:var(--mut)">Add fixed campus zones to generate forecasts.</p></div>';return}
 const list=await api(`/predict?slot=${pslot}&event=${$('event').value||'normal'}`);
 $('plist').innerHTML=list.map((r,i)=>`<div class="row" data-id="${r.id}"><div><b>${i+1}. ${esc(r.name)}</b></div><div class="mono" style="text-align:right;color:${col(r.predicted)}">${r.predicted} dB</div><small>${r.cat} &middot; expected ${r.level} &middot; entered ${r.entered} dB</small><div class="bar"><i style="width:${Math.min(100,r.predicted)}%;background:${col(r.predicted)}"></i></div></div>`).join('')}
$('pslots').onclick=e=>{const b=e.target.closest('button');if(b){pslot=b.dataset.s;drawPred()}};
$('event').onchange=drawPred;
$('plist').onclick=async e=>{const r=e.target.closest('.row');if(!r)return;
 const f=await api(`/forecast/${r.dataset.id}?event=${$('event').value}`);
 $('fc').hidden=false;$('fc').innerHTML=`<div class="mono" style="color:var(--acc)">Future prediction</div><h3 style="font-weight:300;margin:10px 0">${esc(f.zone)}: next 4 slots</h3>`+
 f.forecast.map(x=>`<div class="zl"><span class="mono">${x.slot}</span><b style="color:${col(x.predicted)}">${x.predicted} dB &middot; ${x.level}</b></div>`).join('');
 $('fc').scrollIntoView({behavior:'smooth'})};

// insights
function drawIns(){
 const all=zones.flatMap(z=>SLOTS.map(s=>z.db[s]));
 $('s1').textContent=zones.length;$('s2').textContent=zones.filter(z=>z.db[slot]<45).length;
 $('s3').textContent=Math.round(avg(all))+' dB';
 const best=[...zones].sort((a,b)=>avg(SLOTS.map(s=>a.db[s]))-avg(SLOTS.map(s=>b.db[s])))[0];
 $('s4').textContent=best?best.name:'\u2014';
 const av=SLOTS.map(s=>zones.length?avg(zones.map(z=>z.db[s])):0),max=Math.max(100,...av);
 $('chart').innerHTML=av.map(a=>`<div>${a?Math.round(a):''}<i style="height:${a/max*85}%;${a?'background:'+col(a):''}"></i></div>`).join('');
 $('axis').innerHTML=SLOTS.map(s=>`<span>${s}</span>`).join('')}

// admin (API-key protected)
const msg=t=>$('aMsg').textContent=t;
$('aAdd').onclick=async()=>{
 const name=$('aName').value.trim();if(!name)return msg('Enter a location name first.');
 const db={};document.querySelectorAll('#aDb input').forEach(i=>db[i.dataset.s]=+i.value);
 try{await api('/admin/zones',{method:'POST',body:JSON.stringify({name,cat:$('aCat').value,x:+$('aX').value,y:+$('aY').value,db})},true);
  $('aName').value='';msg('Saved '+name+'. Model retrained.');HM=mkMap('map');HL=L.layerGroup().addTo(HM);AM=mkMap('amap');AL=L.layerGroup().addTo(AM);
 AM.on('click',e=>{const[x,y]=toXY(e.latlng);$('aX').value=x.toFixed(1);$('aY').value=y.toFixed(1);
  if(TMP)AM.removeLayer(TMP);TMP=L.circleMarker(e.latlng,{radius:9,color:'#fff',fillColor:'#4fe0b0',fillOpacity:1}).addTo(AM);msg('Position set. Fill the details and Save zone.')});
 $('setC').onclick=async()=>{const c=AM.getCenter();try{await api('/admin/campus',{method:'POST',body:JSON.stringify({lat:c.lat,lng:c.lng})},true);CAMPUS={...CAMPUS,lat:c.lat,lng:c.lng};setBox();msg('Campus centre saved. Zone positions now map around it.');loadZones()}catch(e){msg('Error: '+e.message)}};
 await loadZones()}catch(e){msg('Error: '+e.message)}};
$('sample').onclick=async()=>{try{await api('/admin/sample',{method:'POST'},true);msg('Reset to sample data.');HM=mkMap('map');HL=L.layerGroup().addTo(HM);AM=mkMap('amap');AL=L.layerGroup().addTo(AM);
 AM.on('click',e=>{const[x,y]=toXY(e.latlng);$('aX').value=x.toFixed(1);$('aY').value=y.toFixed(1);
  if(TMP)AM.removeLayer(TMP);TMP=L.circleMarker(e.latlng,{radius:9,color:'#fff',fillColor:'#4fe0b0',fillOpacity:1}).addTo(AM);msg('Position set. Fill the details and Save zone.')});
 $('setC').onclick=async()=>{const c=AM.getCenter();try{await api('/admin/campus',{method:'POST',body:JSON.stringify({lat:c.lat,lng:c.lng})},true);CAMPUS={...CAMPUS,lat:c.lat,lng:c.lng};setBox();msg('Campus centre saved. Zone positions now map around it.');loadZones()}catch(e){msg('Error: '+e.message)}};
 await loadZones()}catch(e){msg('Error: '+e.message)}};
$('clear').onclick=async()=>{if(zones.length&&confirm('Delete all zones?')){try{await api('/admin/zones',{method:'DELETE'},true);hl=null;HM=mkMap('map');HL=L.layerGroup().addTo(HM);AM=mkMap('amap');AL=L.layerGroup().addTo(AM);
 AM.on('click',e=>{const[x,y]=toXY(e.latlng);$('aX').value=x.toFixed(1);$('aY').value=y.toFixed(1);
  if(TMP)AM.removeLayer(TMP);TMP=L.circleMarker(e.latlng,{radius:9,color:'#fff',fillColor:'#4fe0b0',fillOpacity:1}).addTo(AM);msg('Position set. Fill the details and Save zone.')});
 $('setC').onclick=async()=>{const c=AM.getCenter();try{await api('/admin/campus',{method:'POST',body:JSON.stringify({lat:c.lat,lng:c.lng})},true);CAMPUS={...CAMPUS,lat:c.lat,lng:c.lng};setBox();msg('Campus centre saved. Zone positions now map around it.');loadZones()}catch(e){msg('Error: '+e.message)}};
 await loadZones()}catch(e){msg('Error: '+e.message)}}};
$('zl').onclick=async e=>{const b=e.target.closest('button[data-d]');if(!b)return;
 try{await api('/admin/zones/'+b.dataset.d,{method:'DELETE'},true);HM=mkMap('map');HL=L.layerGroup().addTo(HM);AM=mkMap('amap');AL=L.layerGroup().addTo(AM);
 AM.on('click',e=>{const[x,y]=toXY(e.latlng);$('aX').value=x.toFixed(1);$('aY').value=y.toFixed(1);
  if(TMP)AM.removeLayer(TMP);TMP=L.circleMarker(e.latlng,{radius:9,color:'#fff',fillColor:'#4fe0b0',fillOpacity:1}).addTo(AM);msg('Position set. Fill the details and Save zone.')});
 $('setC').onclick=async()=>{const c=AM.getCenter();try{await api('/admin/campus',{method:'POST',body:JSON.stringify({lat:c.lat,lng:c.lng})},true);CAMPUS={...CAMPUS,lat:c.lat,lng:c.lng};setBox();msg('Campus centre saved. Zone positions now map around it.');loadZones()}catch(e){msg('Error: '+e.message)}};
 await loadZones()}catch(err){msg('Error: '+err.message)}};
function drawAdmin(){drawAdminPins();$('zl').innerHTML=zones.length?zones.map(z=>`<div class="zl"><span><b>${esc(z.name)}</b><br><small style="color:var(--mut)">${z.cat} &middot; ${SLOTS.map(s=>z.db[s]).join(' / ')} dB</small></span><button class="btn ghost" data-d="${z.id}">Delete</button></div>`).join(''):'<p class="note">No zones saved yet.</p>'}

function render(){drawHeat();drawPred();drawIns();drawAdmin()}
init();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

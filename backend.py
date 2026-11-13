"""Campus Silence Heatmap - BACKEND (FastAPI + scikit-learn).
REST API under /api/*, and it also serves the website files (index.html, style.css, app.js...) sitting next to it.
Run:  python backend.py   ->  http://localhost:8000   (API docs: /docs)
Phone on same Wi-Fi: http://<your-computer-IP>:8000
Admin key: env ADMIN_API_KEY (default campus123)
"""
import json, os, uuid, threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "campus_data.json"   # creator's fixed dataset (auto-created)
LOCK = threading.Lock()
ADMIN_KEY = os.getenv("ADMIN_API_KEY", "campus123")   # change in production!

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
    tmp = DATA_FILE.with_suffix(".tmp")          # atomic write: no half-written file
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
# lets the frontend be hosted on a different domain if you want
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup():
    load(); train()

# public (read-only for users)
@app.get("/api/meta")
def meta():
    return {"slots": SLOTS, "cats": CATS, "events": list(EVENTS), "current_slot": current_slot()}

@app.get("/api/campus")
def campus():
    """Real-world location of the campus (override with CAMPUS_LAT / CAMPUS_LNG env vars)."""
    return {"name": os.getenv("CAMPUS_NAME", "Yashoda Technical Campus, Wadhe, Satara"),
            "lat": float(os.getenv("CAMPUS_LAT", "17.72158")),
            "lng": float(os.getenv("CAMPUS_LNG", "74.02065")), "zoom": 17}

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

@app.post("/api/admin/sample", dependencies=[Depends(admin)])
def load_sample():
    save(sample_zones()); return {"ok": True}

@app.delete("/api/admin/zones", dependencies=[Depends(admin)])
def clear_all():
    save([]); return {"ok": True}

@app.post("/api/admin/check", dependencies=[Depends(admin)])
def check_key():
    return {"ok": True}

# ---------- serve the website files (kept LAST so /api routes win) ----------
PUBLIC_FILES = ["index.html", "style.css", "app.js", "manifest.json", "sw.js", "icon-192.png", "icon-512.png"]

@app.get("/")
def home():
    return FileResponse(BASE / "index.html")

@app.get("/{name}")
def public_file(name: str):
    if name not in PUBLIC_FILES:       # backend.py and the data file are never exposed
        raise HTTPException(404, "Not found")
    return FileResponse(BASE / name)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

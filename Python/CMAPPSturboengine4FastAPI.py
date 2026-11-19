"""
FD002 RUL inference service (FastAPI).

Loads artifacts saved by CMAPPSturboengine3saveModels.py — no retraining.
Run from project root or PdM/:

  py -3 -m uvicorn PdM.CMAPPSturboengine4FastAPI:app --reload
  # or, from PdM/:
  py -3 -m uvicorn CMAPPSturboengine4FastAPI:app --reload

  just do :  py -3 PdM\CMAPPSturboengine4FastAPI.py behind \cursor>
  or just run python file normally

  if you want to upload a new file with measurements -> upload measurements
  Else, just pick engine_id at start since they are all from the test file (not the train)

Then open http://127.0.0.1:8000/
"""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from tensorflow import keras  # type: ignore

ARTIFACT_DIR = Path(__file__).resolve().parent
COLS = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Populated at startup from disk.
STATE: dict[str, Any] = {}


def load_artifacts() -> None:
    """Load regime pipeline + models once for the process lifetime."""
    required = [
        "fd002_regime_clusterer.joblib",
        "fd002_regime_scalers.joblib",
        "fd002_gpr.joblib",
        "fd002_gpr_2.joblib",
        "fd002_rf.joblib",
        "fd002_cnn.keras",
        "fd002_meta.joblib",
    ]
    missing = [name for name in required if not (ARTIFACT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing FD002 artifacts in "
            f"{ARTIFACT_DIR}: {missing}. "
            "Run CMAPPSturboengine3saveModels.py first."
        )

    print(f"Loading FD002 artifacts from {ARTIFACT_DIR} ...")
    meta = joblib.load(ARTIFACT_DIR / "fd002_meta.joblib")
    STATE["meta"] = meta
    STATE["op_cols"] = list(meta["op_cols"])
    STATE["sensor_cols"] = list(meta["sensor_cols"])
    STATE["window"] = int(meta["window"])
    STATE["regime_clusterer"] = joblib.load(
        ARTIFACT_DIR / "fd002_regime_clusterer.joblib"
    )
    STATE["regime_scalers"] = joblib.load(
        ARTIFACT_DIR / "fd002_regime_scalers.joblib"
    )
    STATE["gpr"] = joblib.load(ARTIFACT_DIR / "fd002_gpr.joblib")
    STATE["gpr_2"] = joblib.load(ARTIFACT_DIR / "fd002_gpr_2.joblib")
    STATE["rf"] = joblib.load(ARTIFACT_DIR / "fd002_rf.joblib")
    STATE["cnn"] = keras.models.load_model(ARTIFACT_DIR / "fd002_cnn.keras")
    print("Artifacts loaded.")


def regime_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign regimes and z-score sensors within each regime (train-fitted)."""
    op_cols = STATE["op_cols"]
    sensor_cols = STATE["sensor_cols"]
    out = frame.copy()
    out["regime"] = STATE["regime_clusterer"].predict(out[op_cols].values)
    out[sensor_cols] = out[sensor_cols].astype(float)
    for r, scaler in STATE["regime_scalers"].items():
        mask = out["regime"] == r
        if mask.any():
            out.loc[mask, sensor_cols] = scaler.transform(
                out.loc[mask, sensor_cols].values
            )
    return out


def make_per_cycle_windows(sensors: np.ndarray, window: int) -> np.ndarray:
    """Build a WINDOW x n_sensors tensor ending at each cycle (front-pad if needed)."""
    X = []
    for i in range(len(sensors)):
        w = sensors[max(0, i - window + 1) : i + 1]
        if len(w) < window:
            pad = np.repeat(w[:1], window - len(w), axis=0)
            w = np.vstack([pad, w])
        X.append(w)
    return np.array(X)


def parse_engine_table(raw: bytes | str, filename: str | None = None) -> pd.DataFrame:
    """
    Accept whitespace-separated CMAPSS-style rows (no header) or CSV with headers.
    Requires cycle, op_setting_1..3, and the 10 trending sensors (engine_id optional).
    """
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    # Try headered CSV first; fall back to CMAPSS whitespace, no header.
    try:
        df = pd.read_csv(io.StringIO(text))
        if "cycle" not in df.columns:
            raise ValueError("no cycle column")
    except Exception:
        df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None, engine="python")
        if df.shape[1] == len(COLS):
            df.columns = COLS
        elif df.shape[1] == len(COLS) - 1:
            # engine_id omitted
            df.columns = COLS[1:]
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Expected {len(COLS)} or {len(COLS) - 1} columns "
                    f"(got {df.shape[1]}). Use CMAPSS FD002 layout."
                ),
            )

    required = ["cycle"] + STATE["op_cols"] + STATE["sensor_cols"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {missing}",
        )

    if "engine_id" not in df.columns:
        df["engine_id"] = 1

    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    if filename:
        df.attrs["source"] = filename
    return df


def recommend(rul: float) -> str:
    """Simple maintenance-facing text from predicted RUL (cycles)."""
    if rul < 20:
        return "Critical: schedule inspection / maintenance immediately."
    if rul < 50:
        return "High priority: plan maintenance within the next few dozen cycles."
    if rul < 100:
        return "Elevated risk: increase monitoring and prepare spare parts."
    return "Nominal: continue routine monitoring."


def predict_engine(df_raw: pd.DataFrame, engine_id: int | None = None) -> dict[str, Any]:
    """Regime-normalize one engine and return RUL trajectories + last-cycle summary."""
    ids = df_raw["engine_id"].unique()
    if engine_id is None:
        if len(ids) != 1:
            raise HTTPException(
                status_code=400,
                detail=f"Multiple engine_ids {list(ids)}; pass engine_id to select one.",
            )
        engine_id = int(ids[0])
    elif engine_id not in ids:
        raise HTTPException(
            status_code=404,
            detail=f"engine_id {engine_id} not in upload (found {list(ids)}).",
        )

    eng_raw = df_raw[df_raw["engine_id"] == engine_id].copy()
    eng = regime_normalize(eng_raw).sort_values("cycle")
    sensor_cols = STATE["sensor_cols"]
    window = STATE["window"]
    X = eng[sensor_cols].values
    cycles = eng["cycle"].astype(int).tolist()

    p1, s1 = STATE["gpr"].predict(X, return_std=True)
    p2, s2 = STATE["gpr_2"].predict(X, return_std=True)
    pred_rf = STATE["rf"].predict(X)
    pred_cnn = STATE["cnn"].predict(make_per_cycle_windows(X, window), verbose=0).ravel()

    # Primary reported RUL: GPR-2 (trajectory-trained) with 95% band at last cycle.
    last = -1
    rul_primary = float(p2[last])
    std_primary = float(s2[last])
    lo = float(p2[last] - 1.96 * s2[last])
    hi = float(p2[last] + 1.96 * s2[last])

    return {
        "engine_id": int(engine_id),
        "n_cycles": len(cycles),
        "last_cycle": int(cycles[last]),
        "rul_primary_model": "gpr_2",
        "rul_cycles": rul_primary,
        "rul_std": std_primary,
        "rul_95ci": [lo, hi],
        "recommendation": recommend(rul_primary),
        "last_cycle_by_model": {
            "gpr": float(p1[last]),
            "gpr_2": float(p2[last]),
            "rf": float(pred_rf[last]),
            "cnn": float(pred_cnn[last]),
        },
        "trajectory": {
            "cycle": cycles,
            "gpr": p1.tolist(),
            "gpr_std": s1.tolist(),
            "gpr_2": p2.tolist(),
            "gpr_2_std": s2.tolist(),
            "rf": pred_rf.tolist(),
            "cnn": pred_cnn.tolist(),
        },
    }


def trajectory_plot_base64(result: dict[str, Any]) -> str:
    """PNG (base64) of predicted RUL vs cycle with GPR uncertainty bands."""
    t = result["trajectory"]
    cycles = np.asarray(t["cycle"])
    p1 = np.asarray(t["gpr"])
    s1 = np.asarray(t["gpr_std"])
    p2 = np.asarray(t["gpr_2"])
    s2 = np.asarray(t["gpr_2_std"])
    pred_rf = np.asarray(t["rf"])
    pred_cnn = np.asarray(t["cnn"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cycles, p1, "b-", label="GPR 1 (7000 random rows)")
    ax.fill_between(cycles, p1 - 1.96 * s1, p1 + 1.96 * s1, color="blue", alpha=0.15)
    ax.plot(cycles, p2, "g-", label="GPR 2 (first 20 engines)")
    ax.fill_between(cycles, p2 - 1.96 * s2, p2 + 1.96 * s2, color="green", alpha=0.15)
    ax.plot(cycles, pred_rf, color="tab:orange", label="Random Forest")
    ax.plot(cycles, pred_cnn, color="tab:red", label="1D CNN")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Predicted RUL (cycles remaining)")
    ax.set_title(f"RUL trajectory — engine {result['engine_id']}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_sample_test_engine(engine_id: int = 1) -> pd.DataFrame:
    """Pull one engine from official test_FD002.txt for demos."""
    path = ARTIFACT_DIR / "test_FD002.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS, engine="python")
    if engine_id not in df["engine_id"].values:
        raise HTTPException(status_code=404, detail=f"No test engine_id={engine_id}")
    return df[df["engine_id"] == engine_id].copy()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_artifacts()
    yield
    STATE.clear()


app = FastAPI(
    title="CMAPSS FD002 RUL API",
    description="Inference-only service using saved FD002 models and regime pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictJSON(BaseModel):
    """Optional JSON body: list of cycle rows (same fields as CMAPSS / CSV)."""

    rows: list[dict[str, float | int]] = Field(
        ...,
        description="One dict per cycle with cycle, op_setting_*, sensor_* keys.",
    )
    engine_id: int | None = None
    include_plot: bool = True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifacts_dir": str(ARTIFACT_DIR),
        "models_loaded": sorted(
            k for k in STATE if k in {"gpr", "gpr_2", "rf", "cnn", "regime_clusterer"}
        ),
        "meta": STATE.get("meta"),
    }


@app.post("/predict")
async def predict_upload(
    file: UploadFile = File(...),
    engine_id: int | None = None,
    include_plot: bool = True,
):
    """Upload a CMAPSS-style table (whitespace or CSV) and get RUL + optional plot."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    df = parse_engine_table(raw, filename=file.filename)
    result = predict_engine(df, engine_id=engine_id)
    if include_plot:
        result["plot_png_base64"] = trajectory_plot_base64(result)
    return JSONResponse(result)


@app.post("/predict/json")
def predict_json(body: PredictJSON):
    df = pd.DataFrame(body.rows)
    result = predict_engine(df, engine_id=body.engine_id)
    if body.include_plot:
        result["plot_png_base64"] = trajectory_plot_base64(result)
    return JSONResponse(result)


@app.get("/predict/sample/{engine_id}")
def predict_sample(engine_id: int = 1, include_plot: bool = True):
    """Score an engine from packaged test_FD002.txt (handy for demos)."""
    df = load_sample_test_engine(engine_id)
    result = predict_engine(df, engine_id=engine_id)
    if include_plot:
        result["plot_png_base64"] = trajectory_plot_base64(result)
    return JSONResponse(result)


@app.get("/", response_class=HTMLResponse)
def ui():
    """Minimal UI: sample engine or file upload → RUL + plot."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>CMAPSS FD002 RUL</title>
  <style>
    body { font-family: Georgia, serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem;
           background: #f7f4ef; color: #1a1a1a; }
    h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
    .sub { color: #555; margin-bottom: 1.5rem; }
    section { margin-bottom: 1.5rem; padding: 1rem; background: #fff; border: 1px solid #ddd; }
    button, input[type=file] { margin-top: 0.5rem; }
    button { padding: 0.4rem 0.9rem; cursor: pointer; }
    #out { white-space: pre-wrap; font-family: Consolas, monospace; font-size: 0.85rem; }
    img { max-width: 100%; margin-top: 1rem; border: 1px solid #ccc; }
    label { display: inline-block; margin-right: 0.75rem; }
  </style>
</head>
<body>
  <h1>CMAPSS FD002 — RUL service</h1>
  <p class="sub">Loads saved models (no retraining). Primary RUL from GPR-2 with 95% CI.</p>

  <section>
    <h2>Sample test engine</h2>
    <label>engine_id <input id="sampleId" type="number" value="1" min="1"/></label>
    <button onclick="runSample()">Predict sample</button>
  </section>

  <section>
    <h2>Upload measurements</h2>
    <p>Whitespace CMAPSS rows or CSV with cycle / op_setting_* / sensor_* columns.</p>
    <input id="file" type="file"/>
    <button onclick="runUpload()">Predict upload</button>
  </section>

  <section>
    <h2>Result</h2>
    <div id="out">Waiting…</div>
    <img id="plot" alt="" hidden/>
  </section>

<script>
async function showResult(data) {
  const { plot_png_base64, trajectory, ...summary } = data;
  document.getElementById('out').textContent = JSON.stringify(summary, null, 2);
  const img = document.getElementById('plot');
  if (plot_png_base64) {
    img.src = 'data:image/png;base64,' + plot_png_base64;
    img.hidden = false;
  } else {
    img.hidden = true;
  }
}
async function runSample() {
  const id = document.getElementById('sampleId').value || 1;
  document.getElementById('out').textContent = 'Loading…';
  const res = await fetch('/predict/sample/' + id);
  if (!res.ok) { document.getElementById('out').textContent = await res.text(); return; }
  showResult(await res.json());
}
async function runUpload() {
  const f = document.getElementById('file').files[0];
  if (!f) { alert('Choose a file first'); return; }
  document.getElementById('out').textContent = 'Loading…';
  const fd = new FormData();
  fd.append('file', f);
  const res = await fetch('/predict?include_plot=true', { method: 'POST', body: fd });
  if (!res.ok) { document.getElementById('out').textContent = await res.text(); return; }
  showResult(await res.json());
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    # Loading large joblibs (esp. GPR / RF) can take a minute on first start.
    uvicorn.run(
        "CMAPPSturboengine4FastAPI:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )

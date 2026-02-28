from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import pmdarima as pm
from datetime import datetime, timedelta, timezone
import logging
from dataclasses import dataclass

# -----------------------------
# Configuration & logging
# -----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# -----------------------------
# Incremental ARIMA
# -----------------------------

@dataclass(frozen=True)
class IncrementalState:
    css: float
    y_lags: np.ndarray
    e_lags: np.ndarray
    diff_lags: np.ndarray

    @staticmethod
    def initial(p: int, q: int, d: int):
        return IncrementalState(
            css=0.0,
            y_lags=np.zeros(p),
            e_lags=np.zeros(q),
            diff_lags=np.zeros(d),
        )

def transition(state: IncrementalState, y_raw: float,
               phi: np.ndarray, theta: np.ndarray, c: float) -> IncrementalState:
    # Differencing (any d)
    x = y_raw
    diff_lags = state.diff_lags.copy()

    for i in range(len(diff_lags)):
        delta = x - diff_lags[i]
        diff_lags[i] = x
        x = delta

    y = x  # fully differenced observation

    # Prediction
    y_hat = c
    if len(phi):
        y_hat += phi @ state.y_lags
    if len(theta):
        y_hat += theta @ state.e_lags

    eps = y - y_hat

    # Lag updates
    y_lags = state.y_lags.copy()
    e_lags = state.e_lags.copy()
    if len(y_lags):
        y_lags[1:], y_lags[0] = y_lags[:-1], y
    if len(e_lags):
        e_lags[1:], e_lags[0] = e_lags[:-1], eps

    return IncrementalState(
        css=state.css + eps * eps,
        y_lags=y_lags,
        e_lags=e_lags,
        diff_lags=diff_lags,
    )


def full_table_state(ys, phi, theta, c, d):
    state = IncrementalState.initial(len(phi), len(theta), d)
    for y in ys:
        state = transition(state, y, phi, theta, c)
    return state


class GeometricArima:
    def __init__(self, p, d, q, phi, theta, c, history, tol=0.05):
        self.phi = np.array(phi)
        self.theta = np.array(theta)
        self.p = p
        self.q = q
        self.c = c
        self.d = d
        self.tol = tol

        self.centre_state = full_table_state(history, self.phi, self.theta, c, d)

        params = np.concatenate([self.phi, self.theta])
        p_len = len(self.phi)
        self.vertices = []

        for i in range(len(params)):
            v = params.copy()
            v[i] += tol if v[i] >= 0 else -tol
            self.vertices.append({
                "phi": v[:p_len],
                "theta": v[p_len:],
                "state": full_table_state(history, v[:p_len], v[p_len:], c, d),
            })

        dist = np.linalg.norm(params)
        scale = 1.0 - tol / dist if dist > tol else 0.5
        v = params * scale
        self.vertices.append({
            "phi": v[:p_len],
            "theta": v[p_len:],
            "state": full_table_state(history, v[:p_len], v[p_len:], c, d),
        })

    def update(self, new_ys: List[float]) -> bool:
        centre = self.centre_state
        verts = [v["state"] for v in self.vertices]

        for y in new_ys:
            centre = transition(centre, y, self.phi, self.theta, self.c)
            verts = [
                transition(s, y, v["phi"], v["theta"], self.c)
                for s, v in zip(verts, self.vertices)
            ]
        
        self.centre_state = centre

        # No vertices, retrain
        if self.p + self.q == 0:
            return False

        if min(v.css for v in verts) < centre.css * (1 - 1e-12):
            return False  # breach & retrain

        for v, s in zip(self.vertices, verts):
            v["state"] = s
        return True

# -----------------------------
# In-memory application state
# -----------------------------

class State:
    def __init__(self):
        self.data = pd.DataFrame(columns=["date", "value"])
        self.model = None
        self.geo_arima: Optional[GeometricArima] = None
        self.mode = "naive"

state = State()

# -----------------------------
# API models
# -----------------------------

class Record(BaseModel):
    start_timestamp: datetime
    time_index: float
    global_index: Optional[float] = None
    value: float

class BatchRecords(BaseModel):
    records: List[Record]

class ForecastRequest(BaseModel):
    horizon: int

class Config(BaseModel):
    mode: str  # "naive" | "geometric"

# -----------------------------
# API endpoints
# -----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/config")
def set_config(config: Config):
    state.mode = config.mode
    return {"status": "ok", "mode": state.mode}

@app.post("/setup_single")
def setup_single(records: List[Record]):
    _reset_state()
    _add_records(records)
    return {"status": "ok"}

@app.post("/setup_batch")
def setup_batch(records: List[Record]):
    return setup_single(records)

@app.post("/add_single")
def add_single(record: Record):
    _add_records([record])
    if state.mode == "geometric" and state.geo_arima:
        if not state.geo_arima.update([record.value]):
            logger.info("Breach detected - retraining")
            _train()
    return {"status": "ok"}

@app.post("/add_batch")
def add_batch(batch: BatchRecords):
    _add_records(batch.records)
    if state.mode == "geometric" and state.geo_arima:
        ys = [r.value for r in batch.records]
        if not state.geo_arima.update(ys):
            logger.info("Breach detected - retraining")
            _train()
    return {"status": "ok"}

@app.post("/forecast")
def forecast(req: ForecastRequest):
    if len(state.data) < 10:
        raise HTTPException(status_code=400, detail="Not enough data")

    if state.mode == "naive" or state.model is None:
        _train()

    forecast_vals = state.model.predict(n_periods=req.horizon)

    last_date = state.data["date"].iloc[-1]
    step = state.data["date"].iloc[-1] - state.data["date"].iloc[-2]

    return [
        {
            "forecast_date": last_date + step * (i + 1),
            "forecast_value": float(v),
        }
        for i, v in enumerate(forecast_vals)
    ]

@app.get("/loss")
def get_loss() -> float:
    # Geometric ARIMA - use incremental CSS
    if state.mode == "geometric" and state.geo_arima is not None:
        return float(state.geo_arima.centre_state.css)

    # Naive ARIMA
    if state.model is not None:
        resid = state.model.resid()
        return float(np.sum(resid * resid))

    raise HTTPException(status_code=400, detail="Model not trained")

@app.post("/teardown")
def teardown():
    _reset_state()
    return {"status": "ok"}

# -----------------------------
# Helpers
# -----------------------------

def _reset_state():
    state.data = pd.DataFrame(columns=["date", "value"])
    state.model = None
    state.geo_arima = None
    logger.info("State reset")

def _add_records(records: List[Record]):
    rows = []
    for r in records:
        idx = r.global_index if r.global_index is not None else r.time_index
        ts = r.start_timestamp + timedelta(seconds=idx)
        rows.append({"date": ts, "value": r.value})
    if rows:
        df = pd.DataFrame(rows)
        state.data = pd.concat([state.data, df]).sort_values("date").reset_index(drop=True)
    if (state.mode == "geometric" and not state.geo_arima) or not state.model:
        _train()

def _train():
    logger.info(f"Training AutoARIMA on {len(state.data)} points")
    np.random.seed(42)
    model = pm.auto_arima(
        state.data["value"],
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=5, max_q=5, max_d=2,
    )
    state.model = model
    p, d, q = model.order

    if state.mode == "geometric":
        phi, theta, c = [], [], 0.0
        for name, val in zip(model.arima_res_.param_names, model.arima_res_.params):
            if name in ("const", "intercept"):
                c = val
            elif name.startswith("ar.L"):
                phi.append(val)
            elif name.startswith("ma.L"):
                theta.append(val)

        state.geo_arima = GeometricArima(
            p, d, q, phi, theta, c, state.data["value"].values
        )

# -----------------------------
# Entrypoint
# -----------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

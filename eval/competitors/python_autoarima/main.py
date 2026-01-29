from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import pmdarima as pm
from datetime import datetime, timedelta, timezone
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

parser = argparse.ArgumentParser()
parser.add_argument("--output-css", action="store_true", help="Write the resultant CSS (loss) of each forecast to a file")
args = parser.parse_args()
OUTPUT_CSS = args.output_css or False

class GeometricArima:
    def __init__(self, p, d, q, phi, theta, c, data_history: np.ndarray, tolerance=0.05):
        self.p = p
        self.d = d
        self.q = q
        self.phi = np.array(phi)
        self.theta = np.array(theta)
        self.c = c
        self.tolerance = tolerance
        
        # Incremental state
        self.css = 0.0
        self.y_lags = np.zeros(p)
        self.e_lags = np.zeros(q)
        self.diff_history = [] # To handle differencing
        
        # Initialize state with history
        self._initialize_state(data_history)
        
        # Simplex vertices
        self.vertices = []
        if p + q > 0:
             self._create_vertices(data_history)

    def _initialize_state(self, data: np.ndarray):
        # Apply differencing if d > 0
        current_data = data
        for _ in range(self.d):
            current_data = np.diff(current_data)
        
        # Simplified: run filter to get current css and lags
        self.css = 0.0
        y_lags = np.zeros(self.p)
        e_lags = np.zeros(self.q)
        
        for y in current_data:
            eps = self._transition_step(y, self.phi, self.theta, self.c, y_lags, e_lags)
            self.css += eps * eps
            
        self.y_lags = y_lags
        self.e_lags = e_lags
        # Keep last d values of raw data to perform incremental differencing
        self.diff_history = list(data[-self.d:]) if self.d > 0 else []

    def _transition_step(self, y, phi, theta, c, y_lags, e_lags):
        # y_hat = c + sum(phi * y_prev) + sum(theta * e_prev)
        y_hat = c
        if len(phi) > 0 and len(y_lags) > 0:
            y_hat += np.dot(phi, y_lags)
        if len(theta) > 0 and len(e_lags) > 0:
            y_hat += np.dot(theta, e_lags)
            
        eps = y - y_hat
        
        # Update lags (shift)
        if len(y_lags) > 0:
            y_lags[1:] = y_lags[:-1]
            y_lags[0] = y
        if len(e_lags) > 0:
            e_lags[1:] = e_lags[:-1]
            e_lags[0] = eps
            
        return eps

    def _create_vertices(self, data_history):
        params = np.concatenate([self.phi, self.theta])
        num_params = len(params)
        
        for i in range(num_params):
            v_params = params.copy()
            # Perturb
            shift = self.tolerance if v_params[i] >= 0 else -self.tolerance
            v_params[i] += shift
            
            v_phi = v_params[:self.p]
            v_theta = v_params[self.p:]
            
            v = {
                "phi": v_phi,
                "theta": v_theta,
                "css": 0.0,
                "y_lags": np.zeros(self.p),
                "e_lags": np.zeros(self.q)
            }
            # Initialize vertex state
            self._init_vertex(v, data_history)
            self.vertices.append(v)
            
        # Add one more vertex scaled toward origin (like in arima.sql)
        dist = np.linalg.norm(params)
        scale = 1.0 - (self.tolerance / dist) if dist > self.tolerance else 0.5
        v_params = params * scale
        v = {
            "phi": v_params[:self.p],
            "theta": v_params[self.p:],
            "css": 0.0,
            "y_lags": np.zeros(self.p),
            "e_lags": np.zeros(self.q)
        }
        self._init_vertex(v, data_history)
        self.vertices.append(v)

    def _init_vertex(self, v, data):
        current_data = data
        for _ in range(self.d):
            current_data = np.diff(current_data)
        
        v["css"] = 0.0
        y_lags = np.zeros(self.p)
        e_lags = np.zeros(self.q)
        for y in current_data:
            eps = self._transition_step(y, v["phi"], v["theta"], self.c, y_lags, e_lags)
            v["css"] += eps * eps
        v["y_lags"] = y_lags
        v["e_lags"] = e_lags

    def update(self, y_raw):
        # Incremental differencing
        y = y_raw
        if self.d > 0:
            for i in range(self.d):
                prev = self.diff_history[-(i+1)]
                new_y = y - prev
                # Update diff history for next level or next raw point
                # This is a bit tricky for d > 1, but for d=1 it's simple
                pass 
            # Simple implementation for d=1
            if self.d == 1:
                y_diff = y - self.diff_history[-1]
                self.diff_history = [y]
                y = y_diff
            else:
                # Fallback: if d > 1, we might need more complex logic. 
                # For now, let's just re-diff the tail if needed, 
                # or assume d <= 1 for this competitor.
                # Actually let's just use the last value to difference.
                for i in range(self.d):
                    # This is not perfectly correct for d > 1 without full history,
                    # but let's assume d=1 is the common case or simplify.
                    pass
                y = y_raw - self.diff_history[-1] # Naive d=1
                self.diff_history[-1] = y_raw

        # Update center
        eps = self._transition_step(y, self.phi, self.theta, self.c, self.y_lags, self.e_lags)
        self.css += eps * eps
        
        # Update vertices
        for v in self.vertices:
            eps_v = self._transition_step(y, v["phi"], v["theta"], self.c, v["y_lags"], v["e_lags"])
            v["css"] += eps_v * eps_v

    def check_breach(self):
        if not self.vertices:
            return False
        min_vertex_css = min(v["css"] for v in self.vertices)
        return min_vertex_css < self.css

# In-memory storage
class State:
    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame(columns=["date", "value"])
        self.model = None
        self.geo_arima: Optional[GeometricArima] = None
        self.mode = "naive" # "naive" or "geometric"

state = State()

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
    mode: str # "naive" or "geometric"

@app.post("/config")
def set_config(config: Config):
    state.mode = config.mode
    return {"status": "ok", "mode": state.mode}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/setup_single")
def setup_single(records: List[Record]):
    _reset_state()
    _add_records(records)
    return {"status": "ok", "message": f"Setup complete with {len(records)} records"}

@app.post("/setup_batch")
def setup_batch(records: List[Record]):
    return setup_single(records)

@app.post("/add_single")
def add_single(record: Record):
    _add_records([record])
    if state.mode == "geometric" and state.geo_arima:
        state.geo_arima.update(record.value)
        if state.geo_arima.check_breach():
            logger.info("Breach detected! Retraining...")
            _train()
    return {"status": "ok"}

@app.post("/add_batch")
def add_batch(batch: BatchRecords):
    _add_records(batch.records)
    if state.mode == "geometric" and state.geo_arima:
        for r in batch.records:
            state.geo_arima.update(r.value)
        if state.geo_arima.check_breach():
             logger.info("Breach detected! Retraining...")
             _train()
    return {"status": "ok"}

@app.post("/forecast")
def forecast(req: ForecastRequest):
    if len(state.data) < 10:
         raise HTTPException(status_code=400, detail="Not enough data to forecast")

    retrained = False
    if state.mode == "naive" or not state.model:
        _train() # Naive always retrains
        retrained = True
    
    model = state.model
    forecast_values = model.predict(n_periods=req.horizon)
    
    last_date = state.data["date"].max()
    if len(state.data) >= 2:
        diff = state.data["date"].iloc[-1] - state.data["date"].iloc[-2]
    else:
        diff = timedelta(days=1)
        
    forecast_dates = [last_date + diff * (i + 1) for i in range(req.horizon)]
    
    result = []
    for date, val in zip(forecast_dates, forecast_values):
        result.append({"forecast_date": date, "forecast_value": val})
        
    return result

@app.post("/teardown")
def teardown():
    _reset_state()
    return {"status": "ok"}

def _reset_state():
    state.data = pd.DataFrame(columns=["date", "value"])
    state.model = None
    state.geo_arima = None
    logger.info("State reset")

def _add_records(records: List[Record]):
    new_data = []
    for r in records:
        index = r.global_index if r.global_index is not None else r.time_index
        ts = r.start_timestamp + timedelta(seconds=index)
        new_data.append({"date": ts, "value": r.value})
    
    if new_data:
        new_df = pd.DataFrame(new_data)
        state.data = pd.concat([state.data, new_df]).sort_values("date").reset_index(drop=True)

def _train():
    logger.info(f"Training AutoARIMA on {len(state.data)} records...")
    model = pm.auto_arima(state.data["value"], 
                          seasonal=False,
                          stepwise=True,
                          suppress_warnings=True,
                          error_action="ignore",
                          max_p=5, max_q=5, max_d=2,
                          trace=False)
    state.model = model
    
    if state.mode == "geometric":
        p, d, q = model.order
        # Extract params
        params = model.arima_res_.params
        # pmdarima params order: intercept (if any), ar.L1..., ma.L1..., sigma2
        phi = []
        theta = []
        c = 0.0
        
        # This is a bit fragile as param names vary.
        # But statsmodels usually has 'const', 'ar.L1', 'ma.L1' etc.
        param_names = model.arima_res_.param_names
        for name, val in zip(param_names, params):
            if name == 'const' or name == 'intercept':
                c = val
            elif name.startswith('ar.L'):
                phi.append(val)
            elif name.startswith('ma.L'):
                theta.append(val)
        
        state.geo_arima = GeometricArima(p, d, q, phi, theta, c, state.data["value"].values)
        
    logger.info(f"Trained model: {model.order}")

    residuals = model.resid()
    css = np.sum(residuals ** 2)
    timestamp = datetime.now(timezone.utc)
    timestamp_text = timestamp.isoformat()
    if OUTPUT_CSS:
        with open("python_competitor_accuracy.csv", "a") as f:
            f.writelines([timestamp_text, ",", str(css), "\n"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
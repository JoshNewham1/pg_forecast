from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from typing import Dict, List, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import logging
import math
import gc

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# -----------------------------
# Configuration & logging
# -----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# -----------------------------
# In-memory application state
# -----------------------------

class State:
    def __init__(self):
        self.data = pd.DataFrame()
        self.model = None
        self.mode = "naive"
        self.n_lags = 5
        self.n_features = 50
        self.single_target = False

state = State()

# -----------------------------
# Forecasting model
# -----------------------------
class XGBTimeSeriesModel:
    def __init__(self, lags: int, max_features: int, single_target: bool = False):
        self.lags = lags
        self.max_features = max_features
        self.single_target = single_target
        self.model = XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            max_leaves=8,
            max_bin=1000
        )
        self.train_X = None
        self.train_y = None
        self.last_window = None
        self.n_raw_features_ = None
        self.n_lagged_features_ = None

        if single_target:
            self.max_features = lags

    def _cap_features(self, x: np.ndarray) -> np.ndarray:
        """
        Caps the last dimension of x to max_features.
        """
        if self.max_features is None:
            return x

        if x.ndim == 1:
            return x[: self.max_features]

        return x[:, : self.max_features]

    def _make_supervised(self, values: np.ndarray):
        """
        values: (T, n_raw_features)
        X: (T-lags, <= max_features)
        y: (T-lags, n_target_features)
        """
        values = np.nan_to_num(values, nan=0.0)
        T, n_raw = values.shape
        self.n_raw_features_ = n_raw

        X, y = [], []

        for t in range(self.lags, T):
            features = []

            # build lagged features sequentially per raw feature
            for f in range(n_raw):
                for l in range(self.lags, 0, -1):
                    features.append(values[t - l, f])
                    if len(features) >= self.max_features:
                        break
                if len(features) >= self.max_features:
                    break

            X.append(features)

            # y must match the *current-time* features we included
            n_target_features = math.ceil(len(features) / self.lags)
            y.append(values[t, :n_target_features])

        self.n_lagged_features_ = len(X[0])

        return np.array(X), np.array(y)


    def fit(self, values: np.ndarray):
        X, y = self._make_supervised(values)

        if len(X) == 0:
            raise ValueError("Not enough data to train XGBoost model")

        self.train_X = X
        self.train_y = y

        self.model.fit(X, y)

        with open("xgboost.log", "a") as f:
            f.write(
                f"{datetime.now()}: trained on {X.shape[0]} samples, "
                f"X shape={X.shape}, "
                f"y shape={y.shape}, "
                f"lags={self.lags}, max_features={self.max_features}, "
                f"n_zeroes_X={int((X == 0).all(axis=1).sum())}, "
                f"n_zeroes_y={int((y == 0).all(axis=1).sum())}\n"
            )

        # Store only the needed raw features for rolling
        n_target = y.shape[1]
        self.last_window = values[-self.lags :, :n_target]


    def predict(self, n_periods: int):
        """
        Returns: (n_periods, n_target_features)
        """
        preds = []
        window = self.last_window.copy()  # (lags, n_target_features)
        window = np.nan_to_num(window, nan=0.0)

        for _ in range(n_periods):
            features = []

            for f in range(window.shape[1]):
                for l in range(self.lags):
                    features.append(window[-l, f])
                    if len(features) >= self.max_features:
                        break
                if len(features) >= self.max_features:
                    break

            x = np.array(features).reshape(1, -1)
            y_hat = self.model.predict(x)[0]
            if y_hat.ndim == 1:
                y_hat = y_hat.reshape(1, -1)

            preds.append(y_hat)
            window = np.vstack([window[1:], y_hat])

        return np.array(preds)



    def resid(self):
        preds = self.model.predict(self.train_X)
        if preds.ndim == 1 and self.train_y.ndim == 2:
            preds = preds.reshape(-1, self.train_y.shape[1])
        
        return self.train_y - preds

# -----------------------------
# API models
# -----------------------------

class Record(BaseModel):
    index: int
    timestamp: datetime
    values: Dict[str, float]
    
class BatchRecords(BaseModel):
    records: List[Record]

class ForecastRequest(BaseModel):
    horizon: int

class Config(BaseModel):
    mode: str  # "naive" | "geometric"
    n_lags: int
    n_features: int
    single_target: bool = False

# -----------------------------
# API endpoints
# -----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/config")
def set_config(config: Config):
    state.mode = config.mode
    state.n_lags = config.n_lags
    state.n_features = config.n_features
    state.single_target = config.single_target
    return {"status": "ok", "mode": state.mode, "n_lags": state.n_lags, "n_features": state.n_features, "single_target": state.single_target}

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
    return {"status": "ok"}

@app.post("/add_batch")
def add_batch(data: dict):
    _add_records(data.get("records", []))
    return {"status": "ok"}

@app.post("/forecast")
def forecast(req: ForecastRequest):
    if len(state.data) < 10:
        raise HTTPException(status_code=400, detail="Not enough data")

    _train()  # train on forecast to take data transformation / insert out of the question

    forecast_vals = state.model.predict(req.horizon)
    feature_cols = [c for c in state.data.columns if c != "date" and c != "values"]

    last_date = pd.to_datetime(state.data["date"].iloc[-1])
    if len(state.data) >= 2:
        step = last_date - pd.to_datetime(state.data["date"].iloc[-2])
    else:
        step = pd.Timedelta(days=1)

    result = []
    for i, row in enumerate(forecast_vals):
        forecast_dt = last_date + step * (i + 1)

        forecast_val = None
        if isinstance(row, np.ndarray):
            forecast_val = dict(zip(feature_cols, row.tolist()))
        else: # single value
            forecast_val = float(row)

        result.append({
            "forecast_date": forecast_dt.isoformat(),
            "forecast_value": forecast_val
        })

    return result

@app.get("/loss")
def get_loss() -> float:
    if state.model is None:
        raise HTTPException(status_code=400, detail="Model not trained")

    resid = state.model.resid()
    loss = float(np.sum(resid ** 2))
    return loss

@app.post("/teardown")
def teardown():
    _reset_state()
    return {"status": "ok"}

# -----------------------------
# Helpers
# -----------------------------

def _reset_state():
    state.data = pd.DataFrame()
    state.model = None
    logger.info("State reset")

def _add_records(records: Union[list[Record], list[dict]]):
    if not records:
        return
    
    data_list = []
    for r in records:
        if isinstance(r, Record):
            row = r.values.copy()
            row['date'] = r.timestamp
        else:
            row = r['values'].copy()
            row['date'] = r['timestamp']
        
        # Lowercase keys (except date)
        new_row = {}
        for k, v in row.items():
            if k == 'date':
                new_row[k] = v
            else:
                new_row[k.lower()] = v
        data_list.append(new_row)

    df = pd.DataFrame(data_list)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.replace("NULL", np.nan, inplace=True)

    # Forward fill (Last Observation Carried Forward)
    df.fillna(method='ffill', inplace=True)

    # Fill any remaining NaNs (at the start) with 0
    df.fillna(0.0, inplace=True)

    state.data = pd.concat([state.data, df], ignore_index=True)

def _train():
    logger.info(f"Training XGBoost on {len(state.data)} rows")

    feature_cols = [c for c in state.data.columns if c != "date" and c != "values"]
    values = state.data[feature_cols].astype(np.float32).to_numpy(copy=True)

    # Reset model before retraining
    state.model = XGBTimeSeriesModel(
        state.n_lags,
        state.n_features,
        state.single_target
    )
    
    state.model.fit(values)


# -----------------------------
# Entrypoint
# -----------------------------
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        size = int(request.headers.get("content-length", 0))
        if size > self.max_upload_size:
            return JSONResponse(
                {"detail": f"Request too large: {size} bytes"}, status_code=413
            )
        return await call_next(request)

app.add_middleware(LimitUploadSizeMiddleware, max_upload_size=2_000_000_000) # 2GB


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

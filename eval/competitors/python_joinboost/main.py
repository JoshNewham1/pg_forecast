import sys
import os

# Add JoinBoost to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../JoinBoost/src")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Union, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import duckdb
import math
import shutil

from joinboost.joingraph import JoinGraph
from joinboost.app import DecisionTree, GradientBoosting
from joinboost.aggregator import agg_to_sql

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

LEARNING_RATE = 0.1
NUM_LEAVES = 8
DEPTH = 3
ITERATIONS = 50

class State:
    def __init__(self):
        self.data = pd.DataFrame()
        self.model = None
        self.mode = "naive"
        self.n_lags = 5
        self.n_features = 50
        self.single_target = False
        # DuckDB connection
        self.con = duckdb.connect(database=':memory:')

state = State()

# -----------------------------
# JoinBoost Model Wrapper
# -----------------------------

class JoinBoostTimeSeriesModel:
    def __init__(self, lags: int, max_features: int, con, single_target: bool = False):
        self.lags = lags
        self.max_features = max_features
        self.con = con
        self.single_target = single_target
        self.models = {}
        self.last_window = None
        self.feature_cols_used = []
        self.target_cols = []

    def fit(self, df: pd.DataFrame):
        # Prepare data with lags
        
        # Identify feature columns (everything except date)
        # In the input df, 'values' contains the target(s) if it exists.
        # df columns are: date, <target_1>, <target_2>, ...
        
        # We assume all non-date columns are targets to be predicted autoregressively.
        all_targets = [c for c in df.columns if c != 'date']
        
        logger.debug(f"JoinBoost fit: single_target={self.single_target}, n_lags={self.lags}, max_features={self.max_features}, all_targets_len={len(all_targets)}")

        if self.single_target:
            self.target_cols = all_targets[:1]
        else:
            # Match the logic: math.ceil(len(features) / self.lags)
            # len(features) is min(len(all_lag_cols), self.max_features)
            n_raw = len(all_targets)
            all_lag_cols_len = n_raw * self.lags
            features_len = min(all_lag_cols_len, self.max_features)
            n_targets = math.ceil(features_len / self.lags)
            self.target_cols = all_targets[:n_targets]
        
        logger.debug(f"JoinBoost target_cols: {self.target_cols}")
        
        # Filter df to only keep target cols + date
        keep_cols = ['date'] + self.target_cols
        # Use intersection to be safe against missing cols
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols].copy()

        # Create lagged features for all targets
        lagged_data = df.copy()
        lag_cols = []
        
        for col in self.target_cols:
            for i in range(1, self.lags + 1):
                col_name = f"{col}_lag_{i}"
                lagged_data[col_name] = df[col].shift(i)
                lag_cols.append(col_name)
        
        # Drop rows with NaN (initial lags)
        lagged_data.dropna(inplace=True)
        
        if len(lagged_data) == 0:
            raise ValueError("Not enough data to train JoinBoost model")
        
        # Filter features if max_features is set
        self.feature_cols_used = lag_cols
        if self.max_features and len(lag_cols) > self.max_features:
             self.feature_cols_used = lag_cols[:self.max_features]

        # Register data in DuckDB and materialize it into a table
        # This is critical for performance because JoinBoost will query this many times
        # and we don't want DuckDB to re-compute the lags/shifts every time.
        self.con.register("lagged_data_view", lagged_data)
        table_name = "train_data"
        self.con.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM lagged_data_view")
        
        # Train a model for each target column
        for target in self.target_cols:
            logger.info(f"Training JoinBoost model for target {target} with features {self.feature_cols_used}")
            
            # Create JoinGraph
            dataset = JoinGraph(self.con)
            
            dataset.add_relation(table_name, self.feature_cols_used, y=target)
            
            # Train
            # Using GradientBoosting
            reg = GradientBoosting(learning_rate=LEARNING_RATE, max_depth=DEPTH, n_estimators=ITERATIONS, num_leaves=NUM_LEAVES)
            reg.fit(dataset)
            
            self.models[target] = reg
            
        # Store last window for prediction
        # We need the last 'lags' rows of the original dataframe
        self.last_window = df.iloc[-self.lags:].copy()

    def predict(self, n_periods: int):
        preds = []
        
        # We need a working copy of history that we append predictions to
        # history needs to contain the target columns
        current_history = self.last_window.copy()
        
        for _ in range(n_periods):
            # Construct feature row for t+1
            row_dict = {}
            for col in self.target_cols:
                for i in range(1, self.lags + 1):
                    # history[-1] is t, history[-2] is t-1
                    # lag_1 is t => -1
                    # lag_i is -i
                    if len(current_history) < i:
                         # Should not happen if self.lags is correct
                         val = 0.0
                    else:
                        val = current_history.iloc[-i][col]
                    col_name = f"{col}_lag_{i}"
                    row_dict[col_name] = val
            
            # Filter features
            # We only keep features that are in self.feature_cols_used
            # But we must ensure they are present in row_dict
            
            # Add dummy target columns
            for target in self.target_cols:
                row_dict[target] = 0.0
            
            pred_df = pd.DataFrame([row_dict])
            
            # Register for prediction
            pred_table = "pred_input"
            self.con.register(pred_table, pred_df)
            
            step_preds = {}
            for target, model in self.models.items():
                jg = JoinGraph(self.con)
                jg.add_relation(pred_table, self.feature_cols_used, y=target)
                
                # Predict
                y_pred = model.predict(jg)
                step_preds[target] = float(y_pred[0])
            
            preds.append(step_preds)
            
            # Update history
            # Create a new row with predicted values
            # We need to match the columns of current_history (date + targets)
            # We can ignore date for history purposes as we only index by position
            # But pandas concat requires matching columns
            
            new_row = step_preds.copy()
            # Add dummy date
            new_row['date'] = datetime.now() # Not used
            
            new_row_df = pd.DataFrame([new_row])
            # Align columns
            new_row_df = new_row_df[current_history.columns]
            
            current_history = pd.concat([current_history, new_row_df], ignore_index=True)
            
            # Keep only last 'lags' rows
            if len(current_history) > self.lags:
                current_history = current_history.iloc[-self.lags:]
                
        return preds

    def get_loss(self) -> float:
        total_sse = 0.0
        table_name = "train_data"
        
        for target, model in self.models.items():
            pred_agg = model.get_prediction_aggregate()
            case_sql = agg_to_sql(pred_agg, qualified=False)
            
            # DuckDB query to calculate SSE
            sql = f"SELECT SUM(POW({target} - ({case_sql}), 2)) FROM {table_name}"
            
            try:
                res = self.con.execute(sql).fetchone()
                if res and res[0] is not None:
                    total_sse += float(res[0])
            except Exception as e:
                logger.error(f"Error calculating loss for {target}: {e}")
        
        return total_sse

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
    n_lags: int = 5
    n_features: int = 50
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

    _train()  # train on forecast

    # Predict
    forecast_results = state.model.predict(req.horizon)
    
    last_date = pd.to_datetime(state.data["date"].iloc[-1])
    if len(state.data) >= 2:
        step = last_date - pd.to_datetime(state.data["date"].iloc[-2])
    else:
        step = pd.Timedelta(days=1)

    result = []
    for i, row in enumerate(forecast_results):
        forecast_dt = last_date + step * (i + 1)
        result.append({
            "forecast_date": forecast_dt.isoformat(),
            "forecast_value": row
        })

    return result

@app.get("/loss")
def get_loss() -> float:
    if state.model:
        return state.model.get_loss()
    return 0.0

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
        
        # Lowercase keys
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
    logger.info(f"Training JoinBoost on {len(state.data)} rows")
    
    model = JoinBoostTimeSeriesModel(state.n_lags, state.n_features, state.con, state.single_target)
    model.fit(state.data)

    state.model = model

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

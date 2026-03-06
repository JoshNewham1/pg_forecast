import sys
import os

# Add JoinBoost to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../JoinBoost/src")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from typing import Dict, List, Union, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import duckdb
import math
import shutil
import time
import re

from joinboost.joingraph import JoinGraph
from joinboost.app import DecisionTree, GradientBoosting
from joinboost.aggregator import agg_to_sql
from joinboost.executor import DuckdbExecutor

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# -----------------------------
# Configuration & logging
# -----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class OptimizedDuckdbExecutor(DuckdbExecutor):
    def _execute_query(self, q):
        # CTAS Rewrite for DuckDB (performs CASE in memory instead of in UPDATE)
        # DuckDB UPDATE is slow, CTAS is fast.
        if "UPDATE" in q.upper() and "SET S =" in q.upper():
            # Extract the math inside the UPDATE statement
            start_case = q.upper().find("(CASE")
            if start_case != -1:
                case_statement = q[start_case:].rstrip("; \n")
                # Find table name: UPDATE <table_name> SET
                parts = q.split()
                try:
                    update_idx = next(i for i, p in enumerate(parts) if p.upper() == "UPDATE")
                    table_name = parts[update_idx + 1]
                    
                    # Get column names (excluding s)
                    cols = [col[1] for col in self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
                    non_s_cols = [c for c in cols if c.lower() != "s"]
                    select_cols = ", ".join(non_s_cols)
                    
                    new_q = f"CREATE TABLE {table_name}_new AS SELECT {select_cols}, s - {case_statement} AS s FROM {table_name}; DROP TABLE {table_name}; ALTER TABLE {table_name}_new RENAME TO {table_name};"
                    return self.conn.execute(new_q)
                except Exception as e:
                    logger.error(f"CTAS optimization failed: {e}")
        
        return super()._execute_query(q)

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
        self.executor = OptimizedDuckdbExecutor(self.con)

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
        t_start = time.perf_counter()
        
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
        t_prepare = time.perf_counter()
        
        # Train a model for each target column
        for target in self.target_cols:
            logger.info(f"Training JoinBoost model for target {target} with features {self.feature_cols_used}")
            
            # Create JoinGraph
            dataset = JoinGraph(state.executor)
            
            dataset.add_relation(table_name, self.feature_cols_used, y=target)
            
            # Train
            # Using GradientBoosting
            reg = GradientBoosting(learning_rate=LEARNING_RATE, max_depth=DEPTH, n_estimators=ITERATIONS, num_leaves=NUM_LEAVES)
            reg.fit(dataset, skip_preprocess=True)
            
            self.models[target] = reg
        
        t_train = time.perf_counter()
        logger.info(f"JoinBoost Fit complete: prepare={t_prepare-t_start:.4f}s, train={t_train-t_prepare:.4f}s")
            
        # Store last window for prediction
        # We need the last 'lags' rows of the original dataframe
        self.last_window = df.iloc[-self.lags:].copy()

    def predict(self, n_periods: int):
        t_start = time.perf_counter()
        if not self.target_cols:
            return []
        
        # Optimization: use recursive CTE for efficiency
        # Combine all targets' SQLs
        target_case_sqls = {}
        all_lags = {}  # col -> max_lag
        
        for target, model in self.models.items():
            pred_agg = model.get_prediction_aggregate()
            # DuckDB is fine with AS DOUBLE (which JoinBoost emits by default)
            case_sql = agg_to_sql(pred_agg, qualified=False)
            target_case_sqls[target] = case_sql
            
            for feat in re.findall(r'\b[a-z0-9_]+_lag_\d+\b', case_sql):
                parts = feat.split('_lag_')
                col_name = parts[0]
                lag_num = int(parts[1])
                all_lags[col_name] = max(all_lags.get(col_name, 0), lag_num)
        
        required_features = sorted([f"{c}_lag_{i}" for c, m in all_lags.items() for i in range(1, m + 1)])
        
        # seed data from self.last_window
        df_seed = self.last_window.copy()
        df_seed.columns = [c.lower() for c in df_seed.columns]
        
        # Construct seed SQL
        last_row = df_seed.iloc[-1]
        seed_vals = [f"CAST('{last_row['date']}' AS TIMESTAMP) as date"]
        for target in self.target_cols:
             seed_vals.append(f"CAST({last_row[target]} AS DOUBLE) as {target}")
        
        for feat in required_features:
            parts = feat.split('_lag_')
            col = parts[0]
            lag = int(parts[1])
            val = df_seed.iloc[-1 - lag][col] if lag < len(df_seed) else 0.0
            seed_vals.append(f"CAST({val} AS DOUBLE) as {feat}")
        
        seed_sql = f"SELECT {', '.join(seed_vals)}, 0 as step"
        
        if len(df_seed) >= 2:
            step_interval = pd.to_datetime(df_seed.iloc[-1]['date']) - pd.to_datetime(df_seed.iloc[-2]['date'])
            interval_str = f"INTERVAL '{int(step_interval.total_seconds())} seconds'"
        else:
            interval_str = "INTERVAL '1 day'"
            
        recursive_cols = [f"date + {interval_str}"]
        for target in self.target_cols:
             recursive_cols.append(f"({target_case_sqls[target]})")
        
        for feat in required_features:
            parts = feat.split('_lag_')
            col = parts[0]
            lag = int(parts[1])
            if col in self.target_cols:
                if lag == 1:
                    recursive_cols.append(col)
                else:
                    recursive_cols.append(f"{col}_lag_{lag-1}")
            else:
                recursive_cols.append(feat)
        
        recursive_cols.append("step + 1")
        
        col_names = ["date"] + self.target_cols + required_features + ["step"]
        recursive_select = ", ".join([f"{c} as {n}" for c, n in zip(recursive_cols, col_names)])
        
        full_sql = f"""
        WITH RECURSIVE forecast_cte AS (
            ({seed_sql})
            UNION ALL
            SELECT {recursive_select}
            FROM forecast_cte
            WHERE step < {n_periods}
        )
        SELECT * FROM forecast_cte WHERE step > 0 ORDER BY step ASC;
        """
        
        try:
            res = self.con.execute(full_sql).df()
            
            preds = []
            for _, row in res.iterrows():
                step_preds = {}
                for target in self.target_cols:
                    step_preds[target] = float(row[target])
                preds.append(step_preds)
            
            t_end = time.perf_counter()
            logger.info(f"Recursive prediction complete in {t_end-t_start:.4f}s")
            return preds
        except Exception as e:
            logger.error(f"Recursive prediction failed: {e}")
            logger.error(f"SQL was: {full_sql}")
            raise e


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
    index: Optional[int] = None
    timestamp: Optional[datetime] = None
    values: Optional[Dict[str, Any]] = None
    
    # Univariate fields from Monash dataset
    time_index: Optional[int] = None
    global_index: Optional[int] = None
    start_timestamp: Optional[datetime] = None
    value: Optional[Any] = None

    @model_validator(mode='before')
    @classmethod
    def validate_record(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # If it has 'value' but not 'values', it's univariate
            if 'value' in data and 'values' not in data:
                if 'index' not in data:
                    data['index'] = data.get('global_index') or data.get('time_index') or 0
                if 'timestamp' not in data:
                    data['timestamp'] = data.get('start_timestamp')
                if 'values' not in data:
                    data['values'] = {'value': data['value']}
        return data
    
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
    # Try to parse as BatchRecords if it's a dict
    if "records" in data:
        recs = [Record.model_validate(r) for r in data["records"]]
        _add_records(recs)
    else:
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
            row = r.values.copy() if r.values else {}
            row['date'] = r.timestamp
        else:
            row = r['values'].copy() if 'values' in r else ({'value': r['value']} if 'value' in r else {})
            row['date'] = r.get('timestamp') or r.get('start_timestamp')
        
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

def _cleanup_temp_tables():
    # JoinBoost creates many temporary tables/views with 'joinboost_tmp_' prefix
    # and some others like 'train_data'
    try:
        res = state.con.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'joinboost_tmp_%' OR table_name LIKE 'train_data%'").fetchall()
        for row in res:
            state.con.execute(f"DROP TABLE IF EXISTS {row[0]} CASCADE")
            state.con.execute(f"DROP VIEW IF EXISTS {row[0]} CASCADE")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

def _train():
    _cleanup_temp_tables()
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

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

from joinboost.joingraph import JoinGraph
from joinboost.app import DecisionTree, GradientBoosting
from joinboost.aggregator import agg_to_sql, SelectionExpression, SELECTION, QualifiedAttribute

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
INCREMENTAL_ITERATIONS = 10

class State:
    def __init__(self):
        self.data = pd.DataFrame()
        self.model = None
        self.mode = "naive"
        self.n_lags = 5
        self.n_features = 50
        self.single_target = False
        self.incremental = False
        # DuckDB connection
        self.con = duckdb.connect(database=':memory:')

state = State()

# -----------------------------
# JoinBoost Model Wrapper
# -----------------------------

class JoinBoostTimeSeriesModel:
    def __init__(self, lags: int, max_features: int, con, single_target: bool = False, incremental: bool = False):
        self.lags = lags
        self.max_features = max_features
        self.con = con
        self.single_target = single_target
        self.incremental = incremental
        self.models = {}
        self.last_window = None
        self.feature_cols_used = []
        self.target_cols = []
        self.trained = False
        self.last_processed_date = None
        self.table_name = "train_data"
        self.base_table = "base_data"

    def fit(self, df: pd.DataFrame):
        # Prepare data with lags
        
        # Identify feature columns (everything except date)
        all_targets = [c for c in df.columns if c != 'date']
        
        logger.debug(f"JoinBoost fit: incremental={self.incremental}, single_target={self.single_target}, n_lags={self.lags}, max_features={self.max_features}, all_targets_len={len(all_targets)}")

        if not self.trained:
            if self.single_target:
                self.target_cols = all_targets[:1]
            else:
                n_raw = len(all_targets)
                all_lag_cols_len = n_raw * self.lags
                features_len = min(all_lag_cols_len, self.max_features)
                n_targets = math.ceil(features_len / self.lags)
                self.target_cols = all_targets[:n_targets]
            
            logger.debug(f"JoinBoost target_cols: {self.target_cols}")
            
            # Register base data
            self.con.execute(f"DROP TABLE IF EXISTS {self.base_table}")
            self.con.register("df_view", df[['date'] + self.target_cols])
            self.con.execute(f"CREATE TABLE {self.base_table} AS SELECT * FROM df_view")
        else:
            # Append new data to base table
            df['date'] = pd.to_datetime(df['date'])
            new_data = df[df['date'] > pd.to_datetime(self.last_processed_date)][['date'] + self.target_cols]
            if not new_data.empty:
                self.con.register("new_df_view", new_data)
                self.con.execute(f"INSERT INTO {self.base_table} SELECT * FROM new_df_view")
            else:
                logger.info("No new data for incremental fit, skipping.")
                return

        # Update last processed date
        res = self.con.execute(f"SELECT MAX(date) FROM {self.base_table}").fetchone()
        current_max_date = res[0]

        # Prepare training table with window functions for lags
        select_clause = ["date"]
        cols_to_train = []
        for col in self.target_cols:
            select_clause.append(col)
            for i in range(1, self.lags + 1):
                col_name = f"{col}_lag_{i}"
                select_clause.append(f"LAG({col}, {i}) OVER (ORDER BY date) as {col_name}")
                cols_to_train.append(col_name)
        
        if self.max_features and len(cols_to_train) > self.max_features:
            cols_to_train = cols_to_train[:self.max_features]
        self.feature_cols_used = cols_to_train

        conds = [f"{c} IS NOT NULL" for c in cols_to_train]
        
        if not self.trained or not self.incremental:
            self.con.execute(f"DROP TABLE IF EXISTS {self.table_name}")
            self.con.execute(f"""
                CREATE TABLE {self.table_name} AS 
                SELECT * FROM (SELECT {', '.join(select_clause)} FROM {self.base_table}) t
                WHERE {' AND '.join(conds)}
            """)
        else:
            # For incremental, we could just refresh the whole table if it's faster in DuckDB
            # but let's try to be consistent with Postgres approach if possible.
            # DuckDB doesn't have UNLOGGED or UNIQUE INDEX constraints the same way, 
            # so we just recreate/refresh for now or append.
            # Appending with window functions is tricky because LAG needs context.
            # Re-materializing is probably safest for now given DuckDB's speed.
            self.con.execute(f"DROP TABLE IF EXISTS {self.table_name}")
            self.con.execute(f"""
                CREATE TABLE {self.table_name} AS 
                SELECT * FROM (SELECT {', '.join(select_clause)} FROM {self.base_table}) t
                WHERE {' AND '.join(conds)}
            """)

        # Train a model for each target column
        for target in self.target_cols:
            dataset = JoinGraph(self.con)
            dataset.add_relation(self.table_name, self.feature_cols_used, y=target)
            
            if not self.trained or not self.incremental:
                logger.info(f"Training JoinBoost model for target {target} (Full)")
                reg = GradientBoosting(
                    learning_rate=LEARNING_RATE, 
                    max_depth=DEPTH, 
                    n_estimators=ITERATIONS, 
                    num_leaves=NUM_LEAVES,
                    incremental_estimators=INCREMENTAL_ITERATIONS
                )
                reg.fit(dataset, warm_start=False, skip_preprocess=True)
                self.models[target] = reg
            else:
                logger.info(f"Training JoinBoost model for target {target} (Incremental)")
                reg = self.models[target]
                filter_expression = None
                if self.last_processed_date:
                    filter_expression = SelectionExpression(
                        SELECTION.GREATER, 
                        (QualifiedAttribute(dataset.target_relation, 'date'), 
                         f"CAST('{self.last_processed_date}' AS TIMESTAMP)")
                    )
                reg.fit(dataset, warm_start=True, filter_expression=filter_expression)
            
        self.trained = True
        self.last_processed_date = current_max_date
        
        # Store last window for prediction
        # We need the last 'lags' rows of the base table
        self.last_window = pd.read_sql(f"SELECT * FROM {self.base_table} ORDER BY date DESC LIMIT {self.lags}", self.con).sort_values("date")

    def forecast(self, horizon: int) -> List[Dict[str, float]]:
        if not self.target_cols: return []
        target = self.target_cols[0]
        model = self.models.get(target)
        if not model: return []

        pred_agg = model.get_prediction_aggregate()
        # DuckDB uses DOUBLE
        case_sql = agg_to_sql(pred_agg, qualified=False)

        import re
        feature_map = {}
        for feat in re.findall(r'\b[a-z0-9_]+_lag_\d+\b', case_sql):
            parts = feat.split('_lag_')
            feature_map[parts[0]] = max(feature_map.get(parts[0], 0), int(parts[1]))
        
        required_features = sorted([f"{c}_lag_{i}" for c, m in feature_map.items() for i in range(1, m + 1)])
        
        # Get seed data
        df_seed = pd.read_sql(f"SELECT * FROM {self.base_table} ORDER BY date DESC LIMIT {self.lags}", self.con).sort_values("date")
        
        if len(df_seed) < 2:
             # Fallback if not enough data for interval
             interval_str = "INTERVAL '10 minutes'"
        else:
            step_interval = df_seed.iloc[-1]["date"] - df_seed.iloc[-2]["date"]
            interval_str = f"INTERVAL '{int(step_interval.total_seconds())} seconds'"

        last_date = df_seed.iloc[-1]["date"]
        
        seed_vals = [f"CAST('{last_date}' AS TIMESTAMP) as date", f"CAST({df_seed.iloc[-1][target]} AS DOUBLE) as {target}"]
        for feat in required_features:
            parts = feat.split('_lag_')
            val = df_seed.iloc[-int(parts[1])][parts[0]] if parts[0] in df_seed.columns and int(parts[1]) <= len(df_seed) else 0.0
            seed_vals.append(f"CAST({val} AS DOUBLE) as {feat}")
        
        seed_sql = f"SELECT {', '.join(seed_vals)}, 0 as step"
        
        recursive_cols = [f"date + {interval_str}", f"({case_sql})"]
        for feat in required_features:
            parts = feat.split('_lag_')
            if parts[0] == target:
                recursive_cols.append(f"{target}" if int(parts[1]) == 1 else f"{parts[0]}_lag_{int(parts[1])-1}")
            else:
                recursive_cols.append(f"{feat}")
        recursive_cols.append("step + 1")
        
        col_names = ["date", target] + required_features + ["step"]
        recursive_select = ", ".join([f"{c} as {n}" for c, n in zip(recursive_cols, col_names)])

        full_recursive_sql = f"""
        WITH RECURSIVE forecast_cte AS (
            ({seed_sql})
            UNION ALL
            SELECT {recursive_select}
            FROM forecast_cte
            WHERE step < {horizon}
        )
        SELECT date, {target} as value FROM forecast_cte WHERE step > 0;
        """
        
        results = self.con.execute(full_recursive_sql).fetchall()

        return [{"forecast_date": r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]), 
                 "forecast_value": {target: float(r[1])}} for r in results]

    def get_loss(self) -> float:
        total_sse = 0.0
        for target, model in self.models.items():
            pred_agg = model.get_prediction_aggregate()
            case_sql = agg_to_sql(pred_agg, qualified=False)
            sql = f"SELECT SUM(POWER({target} - ({case_sql}), 2)) FROM {self.table_name}"
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
    incremental: bool = False

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
    state.incremental = config.incremental
    return {"status": "ok", "mode": state.mode, "n_lags": state.n_lags, "n_features": state.n_features, "single_target": state.single_target, "incremental": state.incremental}

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
    return state.model.forecast(req.horizon)

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
    # Using 'ffill' method instead of fillna(method='ffill') for newer pandas
    df.ffill(inplace=True)

    # Fill any remaining NaNs (at the start) with 0
    df.fillna(0.0, inplace=True)

    state.data = pd.concat([state.data, df], ignore_index=True)

def _train():
    logger.info(f"Training JoinBoost on {len(state.data)} rows")
    
    if state.incremental and state.model:
        state.model.fit(state.data)
    else:
        model = JoinBoostTimeSeriesModel(state.n_lags, state.n_features, state.con, state.single_target, state.incremental)
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

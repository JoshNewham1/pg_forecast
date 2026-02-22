import sys
import os
import logging
import pandas as pd
import numpy as np
import time
import math
import re
from datetime import datetime, timedelta
from typing import Dict, List
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add JoinBoost to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eval/JoinBoost/src")))

from joinboost.aggregator import agg_to_sql, SelectionExpression, SELECTION, QualifiedAttribute
from joinboost.joingraph import JoinGraph
from joinboost.app import GradientBoosting

from joinboost_adapter import PostgresExecutor

logger = logging.getLogger(__name__)

LEARNING_RATE = 0.1
NUM_LEAVES = 8
DEPTH = 3
ITERATIONS = 50
INCREMENTAL_ITERATIONS = 50  # Number of trees to add per incremental update

class JoinBoostSUT:
    def __init__(self, model_name: str = "ts_joinboost", lags: int = 5, n_features: int = 50, predict_single_target: bool = False, is_incremental: bool = True):
        self.model_name = model_name
        self.lags = lags
        self.n_features = n_features
        self.predict_single_target = predict_single_target
        self.is_incremental = is_incremental

        load_dotenv()

        TEST_DB_USERNAME = os.getenv("TEST_DB_USERNAME")
        TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD")
        TEST_DB_HOST = os.getenv("TEST_DB_HOST", "localhost")
        TEST_DB_PORT = os.getenv("TEST_DB_PORT", "5432")
        TEST_DB_NAME = os.getenv("TEST_DB_NAME")
        self.base_table = os.getenv("TEST_BASE_TABLE", "ts_joinboost_base")
        self.train_table = f"{self.base_table}_train"

        if not all([TEST_DB_USERNAME, TEST_DB_PASSWORD, TEST_DB_NAME]):
             raise RuntimeError("TEST_DB env vars must be set")

        # Use AUTOCOMMIT isolation level to avoid transaction block issues during DDL and frequent queries
        self.engine = create_engine(
            f"postgresql+psycopg2://{TEST_DB_USERNAME}:{TEST_DB_PASSWORD}@"
            f"{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}",
            isolation_level="AUTOCOMMIT"
        )
        self.conn = self.engine.connect()
        self.executor = PostgresExecutor(self.conn)
        self.target_cols = []
        self.models = {}
        self.trained = False

        # NOTE: Settings optimised for 16 threads, 32GB RAM. They might want tweaking
        self.conn.execute(text("SET jit = off;"))
        self.conn.execute(text("SET max_parallel_workers_per_gather = 8;"))
        self.conn.execute(text("SET max_parallel_workers = 12;"))
        self.conn.execute(text("SET work_mem = '512MB';"))
        self.conn.execute(text("SET temp_buffers = '1GB';"))

    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def setup_single(self, seed_records: List[Dict[str, float]]):
        self.setup_batch(seed_records)

    def setup_batch(self, seed_records: List[Dict[str, float]]):
        if not seed_records: return
        first = seed_records[0]
        if "values" in first and isinstance(first["values"], dict):
            all_targets = [k.lower() for k in first["values"].keys()]
        elif "timestamp" in first:
            all_targets = [k.lower() for k in first.keys() if k not in ["index", "timestamp"]]
        else:
            all_targets = ['value']

        if self.predict_single_target:
            self.target_cols = all_targets[:1]
        else:
            n_targets = math.ceil(min(len(all_targets) * self.lags, self.n_features) / self.lags)
            self.target_cols = all_targets[:n_targets]
        
        logger.info(f"JoinBoost Target columns: {self.target_cols}")

        cols_ddl = ", ".join([f"{k} DOUBLE PRECISION" for k in self.target_cols])
        ddl = f"""
        DROP TABLE IF EXISTS {self.base_table} CASCADE;
        CREATE TABLE {self.base_table} (date TIMESTAMP NOT NULL, {cols_ddl});
        
        -- Indexing the base table for fast window function materialization
        CREATE INDEX idx_{self.base_table}_date ON {self.base_table} (date DESC);
        
        DROP TABLE IF EXISTS {self.train_table} CASCADE;
        """
        self.conn.execute(text(ddl))

        self.add_batch(seed_records)

    def add_single(self, record: Dict[str, float]):
        self.add_batch([record])

    def add_batch(self, records: List[Dict[str, float]]):
        if not records: return
        rows = []
        for r in records:
            if "timestamp" in r:
                row = {"date": r["timestamp"]}
                for k, v in r.items():
                    if k not in ["index", "timestamp"]:
                        row[k.lower()] = v if isinstance(v, float) else None
                rows.append(row)
            else:
                idx = r.get('global_index', r.get('time_index', 0))
                ts = (r.get('start_timestamp') or datetime(1970, 1, 1)) + timedelta(seconds=idx)
                rows.append({"date": ts, "value": r.get('value', 0.0)})
        
        df = pd.DataFrame(rows)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.replace("NULL", np.nan, inplace=True)
        df.ffill(inplace=True)
        df.fillna(0.0, inplace=True)
        df.columns = [c.lower() if c != 'date' else c for c in df.columns]
        df = df[['date'] + self.target_cols]
        df.to_sql(self.base_table, self.conn, if_exists='append', index=False)

    def _train(self):
        start_total = time.perf_counter()
        
        # Incremental logic
        last_processed_date = None
        try:
            res = self.conn.execute(text(f"SELECT MAX(date) FROM {self.train_table}")).fetchone()
            if res: last_processed_date = res[0]
        except Exception:
            pass

        select_clause = ["date"]
        cols_to_train = []
        for col in self.target_cols:
            select_clause.append(col)
            for i in range(1, self.lags + 1):
                select_clause.append(f"LAG({col}, {i}) OVER (ORDER BY date) as {col}_lag_{i}")
                cols_to_train.append(f"{col}_lag_{i}")
        if len(cols_to_train) > self.n_features:
            cols_to_train = cols_to_train[:self.n_features]

        conds = [f"{c} IS NOT NULL" for c in cols_to_train]
        
        training_table = self.train_table

        if last_processed_date is None:  # non-incremental or first run
            materialize_sql = f"""
            DROP TABLE IF EXISTS {self.train_table} CASCADE;
            CREATE UNLOGGED TABLE {self.train_table} AS
            SELECT * FROM (SELECT {', '.join(select_clause)} FROM {self.base_table}) t
            WHERE {' AND '.join(conds)};
            CREATE UNIQUE INDEX idx_{self.train_table}_date ON {self.train_table} (date DESC);
            """
            self.conn.execute(text(materialize_sql))
        else:
            inc_sql = f"""
            INSERT INTO {self.train_table}
            SELECT * FROM (
                SELECT {', '.join(select_clause)} 
                FROM {self.base_table} 
            ) t
            WHERE date > '{last_processed_date}'::timestamp AND {' AND '.join(conds)}
            ON CONFLICT (date) DO NOTHING;
            """
            self.conn.execute(text(inc_sql))

        for target in self.target_cols:
            # Re-compress
            try:
                self.conn.execute(text(f"ANALYZE {training_table};"))
            except Exception:
                pass

            dataset = JoinGraph(self.executor)
            dataset.add_relation(training_table, cols_to_train, y=target)
            
            if not self.trained or not self.is_incremental:
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
                reg = self.models[target]
                filter_expression = None
                if last_processed_date:
                    filter_expression = SelectionExpression(
                        SELECTION.GREATER, 
                        (QualifiedAttribute(dataset.target_relation, 'date'), 
                         f"'{last_processed_date}'::timestamp")
                    )
                reg.fit(dataset, warm_start=True, filter_expression=filter_expression)
            
            self.last_training_view = self.train_table
            
        self.trained = True
        logger.info(f"JoinBoost Fit complete in {time.perf_counter()-start_total:.4f}s")

        # Cleanup jb_ tables to free up temp_buffers and catalog space
        self._cleanup_temp_tables()

    def _cleanup_temp_tables(self):
        res = self.conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'jb_%'"
        )).fetchall()
        for row in res:
            self.conn.execute(text(f"DROP TABLE IF EXISTS {row[0]} CASCADE"))

    def forecast(self, horizon: int) -> List[Dict[str, float]]:
        self._train()
        if not self.target_cols: return []
        target = self.target_cols[0]
        model = self.models.get(target)
        if not model: return []

        pred_agg = model.get_prediction_aggregate()
        case_sql = agg_to_sql(pred_agg, qualified=False).replace("AS DOUBLE", "AS DOUBLE PRECISION")

        feature_map = {}
        for feat in re.findall(r'\b[a-z0-9_]+_lag_\d+\b', case_sql):
            parts = feat.split('_lag_')
            feature_map[parts[0]] = max(feature_map.get(parts[0], 0), int(parts[1]))
        
        required_features = sorted([f"{c}_lag_{i}" for c, m in feature_map.items() for i in range(1, m + 1)])
        df_seed = pd.read_sql(f"SELECT * FROM {self.base_table} ORDER BY date DESC LIMIT {self.lags}", self.conn).sort_values("date")
        df_seed.columns = [c.lower() for c in df_seed.columns]
        
        last_date = df_seed.iloc[-1]["date"]
        step_interval = df_seed.iloc[-1]["date"] - df_seed.iloc[-2]["date"] if len(df_seed) >= 2 else timedelta(seconds=600)
        interval_str = f"'{step_interval.total_seconds()} seconds'::interval"

        seed_vals = [f"'{last_date}'::timestamp as date", f"{df_seed.iloc[-1][target]}::double precision as {target}"]
        for feat in required_features:
            parts = feat.split('_lag_')
            val = df_seed.iloc[-int(parts[1])][parts[0]] if parts[0] in df_seed.columns and int(parts[1]) <= len(df_seed) else 0.0
            seed_vals.append(f"{val}::double precision as {feat}")
        
        seed_sql = f"SELECT {', '.join(seed_vals)}, 0 as step"
        recursive_cols = [f"date + {interval_str}", f"({case_sql})"]
        for feat in required_features:
            parts = feat.split('_lag_')
            if parts[0] == target:
                recursive_cols.append(f"{target}" if int(parts[1]) == 1 else f"{parts[0]}_lag_{int(parts[1])-1}")
            else:
                recursive_cols.append(feat)
        recursive_cols.append("step + 1")
        
        col_names = ["date", target] + required_features + ["step"]
        recursive_select = ", ".join([f"{c} as {n}" for c, n in zip(recursive_cols, col_names)])

        full_recursive_sql = f"WITH RECURSIVE forecast_cte AS (({seed_sql}) UNION ALL SELECT {recursive_select} FROM forecast_cte WHERE step < {horizon}) SELECT date, {target} as value FROM forecast_cte WHERE step > 0;"
        results = self.conn.execute(text(full_recursive_sql)).fetchall()

        return [{"forecast_date": r[0].isoformat() if isinstance(r[0], datetime) else str(r[0]), "forecast_value": {target: float(r[1])}} for r in results]

    def get_loss(self) -> float:
        if not self.models or not hasattr(self, 'last_training_view'): return 0.0
        total_sse = 0.0
        for target, model in self.models.items():
            pred_agg = model.get_prediction_aggregate()
            case_sql = agg_to_sql(pred_agg, qualified=False).replace("AS DOUBLE", "AS DOUBLE PRECISION")
            res = self.conn.execute(text(f"SELECT SUM(POWER({target} - ({case_sql}), 2)) FROM {self.last_training_view}")).fetchone()
            if res and res[0] is not None: total_sse += float(res[0])
        return total_sse

    def teardown_single(self):
        self.conn.execute(text(f"DROP TABLE IF EXISTS {self.base_table} CASCADE"))

    def teardown_batch(self):
        self.teardown_single()
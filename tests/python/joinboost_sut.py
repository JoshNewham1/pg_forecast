import sys
import os
import logging
import pandas as pd
import numpy as np
import time
import math
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any
from decimal import Decimal
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add JoinBoost to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eval/JoinBoost/src")))

from joinboost.executor import Executor, SPJAData, ExecuteMode, DuckdbExecutor
from joinboost.aggregator import value_to_sql, agg_to_sql, selection_to_sql, is_agg, Aggregator, AggExpression
from joinboost.joingraph import JoinGraph
from joinboost.app import GradientBoosting

logger = logging.getLogger(__name__)

class PostgresExecutor(DuckdbExecutor):
    def __init__(self, conn, debug=False):
        super().__init__(conn, debug)
        self.prefix = "joinboost_tmp_"

    def get_schema(self, table: str) -> list:
        # Get all columns in training table
        sql = f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table}'
        """
        result = self._execute_query(sql)
        return [row[0] for row in result]

    def add_table(self, table_name: str, df):
        if isinstance(df, pd.DataFrame):
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)
        else:
            raise Exception("PostgresExecutor.add_table only supports pandas DataFrame as table_address")

    def delete_table(self, table: str):
        self.check_table(table)
        sql = "DROP TABLE IF EXISTS " + table + " CASCADE;"
        self._execute_query(sql)

    def _execute_query(self, q):
        q = q.replace("AS DOUBLE", "AS DOUBLE PRECISION")
        if self.debug:
            print(q)
        result = self.conn.execute(text(q))
        
        # If it's a SELECT or WITH, return results
        if q.strip().upper().startswith("SELECT") or q.strip().upper().startswith("WITH") or "RETURNING" in q.strip().upper():
            rows = result.fetchall()
            # Convert Decimal results to float as JoinBoost expects floats
            return [tuple(float(v) if isinstance(v, Decimal) else v for v in row) for row in rows]
        else:
            if not self.conn.in_transaction():
                 self.conn.commit()
            return None

    def spja_query(self, spja_data: SPJAData, parenthesize: bool = True):
        # Override to handle Postgres specific syntax if needed
        # Mostly same as DuckDB, but SAMPLE is different
        
        parsed_aggregate_expressions = []
        for target_col, aggExp in spja_data.aggregate_expressions.items():
            window_clause = " OVER joinboost_window " if spja_data.window_by and is_agg(aggExp.agg) else ""
            rename_expr = (" AS " + value_to_sql(target_col, qualified=False)) if target_col is not None else ""
            parsed_aggregate_expressions.append(agg_to_sql(aggExp, qualified=spja_data.qualified) + window_clause + rename_expr)

        sql = "SELECT " + ", ".join(parsed_aggregate_expressions) + "\n"
        sql += "FROM " + ",".join(spja_data.from_tables) + "\n"

        # Postgres TAMPLESAMPLE syntax: FROM table TABLESAMPLE BERNOULLI(percent)
        # But JoinBoost supports sampling on the result of joins?
        # If from_tables has multiple tables, we can't easily use TABLESAMPLE on the join result in Postgres directly
        if spja_data.sample_rate is not None:
             pass

        if len(spja_data.select_conds) > 0 or len(spja_data.join_conds) > 0 or spja_data.sample_rate is not None:
            should_qualify = spja_data.qualified
            if len(spja_data.from_tables) == 1:
                should_qualify = False
            
            conds = [selection_to_sql(cond, qualified=should_qualify) for cond in spja_data.select_conds + spja_data.join_conds]
            
            if spja_data.sample_rate is not None:
                conds.append(f"random() < {spja_data.sample_rate}")

            sql += "WHERE " + " AND ".join(conds) + "\n"

        if len(spja_data.window_by) > 0:
            sql += ("WINDOW joinboost_window AS (ORDER BY " + ",".join([value_to_sql(att, qualified=False) for att in spja_data.window_by]) + ")\n")
        
        if len(spja_data.group_by) > 0:
            sql += "GROUP BY " + ",".join([value_to_sql(att) for att in spja_data.group_by]) + "\n"
        
        if len(spja_data.order_by) > 0:
            sql += ("ORDER BY " + ",".join([f"{col} {order}" for (col, order) in spja_data.order_by])+ "\n")
        
        if spja_data.limit is not None:
            sql += "LIMIT " + str(spja_data.limit) + "\n"
        
        # sample_rate handling moved to WHERE clause for Postgres

        if parenthesize:
            sql = f"({sql})"

        return sql

    def _spja_query_to_table(self, spja_data: SPJAData) -> str:
        spja = self.spja_query(spja_data, parenthesize=False)
        name_ = self.get_next_name()
        entity_type_ = "TABLE "
        
        if spja_data.replace:
            self._execute_query(f"DROP TABLE IF EXISTS {name_} CASCADE")
        
        sql = (
            "CREATE "
            + entity_type_
            + name_
            + " AS "
        )
        sql += spja
        self._execute_query(sql)
        return name_

    def _spja_query_as_view(self, spja_data: SPJAData):
        spja = self.spja_query(spja_data, parenthesize=False)
        name_ = self.get_next_name()
        entity_type_ = "VIEW "
        
        sql = (
            "CREATE OR REPLACE "
            + entity_type_
            + name_
            + " AS "
        )
        sql += spja
        self._execute_query(sql)
        return name_


class JoinBoostSUT:
    def __init__(self, model_name: str = "joinboost", lags: int = 5, n_features: int = 50, predict_single_target: bool = False):
        self.model_name = model_name
        self.lags = lags
        self.n_features = n_features
        self.predict_single_target = predict_single_target
        load_dotenv()

        TEST_DB_USERNAME = os.getenv("TEST_DB_USERNAME")
        TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD")
        TEST_DB_HOST = os.getenv("TEST_DB_HOST", "localhost")
        TEST_DB_PORT = os.getenv("TEST_DB_PORT", "5432")
        TEST_DB_NAME = os.getenv("TEST_DB_NAME")
        self.base_table = os.getenv("TEST_BASE_TABLE", "time_series_performance_test")

        if not all([TEST_DB_USERNAME, TEST_DB_PASSWORD, TEST_DB_NAME]):
             raise RuntimeError("TEST_DB env vars must be set")

        self.engine = create_engine(
            f"postgresql+psycopg2://{TEST_DB_USERNAME}:{TEST_DB_PASSWORD}@"
            f"{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"
        )
        self.conn = self.engine.connect()
        self.executor = PostgresExecutor(self.conn)
        
        # Keep track of target columns for multivariate support
        self.target_cols = []
        self.models = {} # target -> model

    def __del__(self):
        if self.conn:
            self.conn.close()

    def setup_single(self, seed_records: List[Dict[str, float]]):
        self.setup_batch(seed_records)

    def setup_batch(self, seed_records: List[Dict[str, float]]):
        # 1. Determine schema from seed_records
        if not seed_records:
            return

        # Flatten records to dataframe to infer schema
        first = seed_records[0]
        # Check if univariate or multivariate
        if "values" in first and isinstance(first["values"], dict):
            # Multivariate (transformed format)
            keys = first["values"].keys()
            all_targets = [k.lower() for k in keys]
        elif "timestamp" in first:
            # Multivariate (stream_tsf_aligned_series format)
            keys = [k for k in first.keys() if k not in ["index", "timestamp"]]
            all_targets = [k.lower() for k in keys]
        else:
            # Univariate
            all_targets = ['value']

        if self.predict_single_target:
            self.target_cols = all_targets[:1]
        else:
            n_raw = len(all_targets)
            all_lag_cols_len = n_raw * self.lags
            features_len = min(all_lag_cols_len, self.n_features)
            n_targets = math.ceil(features_len / self.lags)
            self.target_cols = all_targets[:n_targets]
        
        logger.info(f"Target columns: {self.target_cols}")

        # Prepare DDL
        cols_ddl = ", ".join([f"{k} DOUBLE PRECISION" for k in self.target_cols])
        
        ddl = f"""
        DROP TABLE IF EXISTS {self.base_table} CASCADE;
        CREATE TABLE {self.base_table} (
            date TIMESTAMP NOT NULL,
            {cols_ddl}
        );
        """

        with self.conn.begin():
            self.conn.execute(text(ddl))

        # Insert seed records
        self.add_batch(seed_records)
        
        # Initial training
        self._train()

    def add_single(self, record: Dict[str, float]):
        self.add_batch([record])

    def add_batch(self, records: List[Dict[str, float]]):
        if not records:
            return
        
        # Convert to DataFrame
        rows = []
        for r in records:
            if "timestamp" in r:
                row = {"date": r["timestamp"]}
                for k, v in r.items():
                    if k not in ["index", "timestamp"]:
                        row[k.lower()] = v if isinstance(v, float) else None
                rows.append(row)
            else:
                # Univariate
                idx = r.get('global_index', r.get('time_index', 0))
                if 'start_timestamp' in r:
                    ts = r['start_timestamp'] + timedelta(seconds=idx)
                else:
                    # Fallback if start_timestamp is missing
                    ts = datetime(1970, 1, 1) + timedelta(seconds=idx)
                rows.append({"date": ts, "value": r.get('value', 0.0)})
        
        df = pd.DataFrame(rows)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.replace("NULL", np.nan, inplace=True)

        # Forward fill (Last Observation Carried Forward)
        df.fillna(method='ffill', inplace=True)

        # Fill any remaining NaNs (at the start) with 0
        df.fillna(0.0, inplace=True)

        # Lowercase all columns except 'date'
        df.columns = [c.lower() if c != 'date' else c for c in df.columns]
        # ONLY KEEP TARGET COLS AND DATE
        keep_cols = ['date'] + self.target_cols
        df = df[[c for c in df.columns if c in keep_cols]]
        # Use to_sql for fast insertion
        df.to_sql(self.base_table, self.conn, if_exists='append', index=False)
        self.conn.commit()

    def _train(self):
        start_total = time.perf_counter()
        # Create lagged view for training
        # We need to create a view that has lags for all targets
        
        view_name = f"{self.base_table}_lagged"
        
        select_clause = ["date"]
        cols_to_train = []
        
        for col in self.target_cols:
            select_clause.append(col)
            # Add lags
            for i in range(1, self.lags + 1):
                select_clause.append(f"LAG({col}, {i}) OVER (ORDER BY date) as {col}_lag_{i}")
                cols_to_train.append(f"{col}_lag_{i}")
        
        # Limit features if needed (simple truncation of list)
        if len(cols_to_train) > self.n_features:
            cols_to_train = cols_to_train[:self.n_features]

        sql = f"""
        DROP VIEW IF EXISTS {view_name} CASCADE;
        CREATE VIEW {view_name} AS
        SELECT {', '.join(select_clause)}
        FROM {self.base_table};
        """
        
        # Execute DDL
        start_ddl = time.perf_counter()
        self.conn.execute(text(sql))
        self.conn.commit()
        
        # Train model for each target
        self.models = {}
        for target in self.target_cols:
            # "Clean view" - filter out NULLs caused by LAG (first few rows)
            clean_view = f"{view_name}_clean"
            conds = [f"{c} IS NOT NULL" for c in cols_to_train]
            
            clean_sql = f"""
            DROP VIEW IF EXISTS {clean_view} CASCADE;
            CREATE VIEW {clean_view} AS
            SELECT * FROM {view_name}
            WHERE {' AND '.join(conds)};
            """
            self.conn.execute(text(clean_sql))
            self.conn.commit()
            end_ddl = time.perf_counter()

            dataset = JoinGraph(self.executor)
            dataset.add_relation(clean_view, cols_to_train, y=target)
            
            # Using GradientBoosting
            start_jb_fit = time.perf_counter()
            reg = GradientBoosting(learning_rate=0.05, max_depth=4, n_estimators=100)
            reg.fit(dataset)
            end_jb_fit = time.perf_counter()
            self.models[target] = reg
            self.last_training_view = clean_view
            
            logger.info(f"JoinBoost Fit for {target}: DDL={end_ddl-start_ddl:.4f}s, Fit={end_jb_fit-start_jb_fit:.4f}s")

        end_total = time.perf_counter()
        logger.info(f"Total _train time: {end_total-start_total:.4f}s")

    def forecast(self, horizon: int) -> List[Dict[str, float]]:
        start_forecast_total = time.perf_counter()
        
        # Train on the latest data
        self._train()
        
        if not self.target_cols:
            return []
        
        target = self.target_cols[0]
        model = self.models.get(target)
        if not model:
            return []

        # Get the CASE statement for the tree model
        pred_agg = model.get_prediction_aggregate()
        case_sql = agg_to_sql(pred_agg, qualified=False)
        # Patch Postgres types
        case_sql = case_sql.replace("AS DOUBLE", "AS DOUBLE PRECISION")

        feature_map = {} # src_col -> max_lag
        for feat in re.findall(r'\b[a-z0-9_]+_lag_\d+\b', case_sql):
            parts = feat.split('_lag_')
            src_col = parts[0]
            lag_idx = int(parts[1])
            feature_map[src_col] = max(feature_map.get(src_col, 0), lag_idx)
        
        # Also ensure target lags up to self.lags are included if they are needed for shifting
        # Actually, let's just include all lags up to the max detected for each col
        required_features = []
        for src_col, max_lag in feature_map.items():
            for i in range(1, max_lag + 1):
                required_features.append(f"{src_col}_lag_{i}")
        
        required_features = sorted(list(set(required_features)))
        logger.debug(f"Detected and expanded required features for CTE: {required_features}")

        # Get the last 'lags' values to seed the recursion
        query_seed = f"SELECT * FROM {self.base_table} ORDER BY date DESC LIMIT {self.lags}"
        df_seed = pd.read_sql(query_seed, self.conn)
        df_seed = df_seed.sort_values("date")
        df_seed.columns = [c.lower() for c in df_seed.columns]
        
        last_date = df_seed.iloc[-1]["date"]
        if len(df_seed) >= 2:
            step_interval = df_seed.iloc[-1]["date"] - df_seed.iloc[-2]["date"]
        else:
            step_interval = timedelta(seconds=600)
        
        # Format interval for Postgres
        interval_str = f"'{step_interval.total_seconds()} seconds'::interval"

        # Build the Recursive CTE to forecast multiple features
        # Seed row
        seed_vals = [f"'{last_date}'::timestamp as date", f"{df_seed.iloc[-1][target]}::double precision as {target}"]
        
        # We need to map available data to required features in the seed
        for feat in required_features:
            # Parse feature name to get source col and lag index
            # format: {col}_lag_{i}
            parts = feat.split('_lag_')
            src_col = parts[0]
            lag_idx = int(parts[1])
            
            val = 0.0
            if src_col in df_seed.columns:
                if lag_idx <= len(df_seed):
                    val = df_seed.iloc[-lag_idx][src_col]
            
            seed_vals.append(f"{val}::double precision as {feat}")
        
        seed_sql = f"SELECT {', '.join(seed_vals)}, 0 as step"

        # Recursive part:
        recursive_cols = [
            f"date + {interval_str}",
            f"({case_sql})", # New value for target
        ]
        
        for feat in required_features:
            parts = feat.split('_lag_')
            src_col = parts[0]
            lag_idx = int(parts[1])
            
            if src_col == target:
                if lag_idx == 1:
                    # target_lag_1 comes from current target prediction
                    recursive_cols.append(f"{target}")
                else:
                    # target_lag_N comes from target_lag_{N-1}
                    recursive_cols.append(f"{src_col}_lag_{lag_idx-1}")
            else:
                recursive_cols.append(feat)
            
        recursive_cols.append("step + 1")
        
        # Column names for the recursive SELECT
        col_names = ["date", target] + required_features + ["step"]
        recursive_select = ", ".join([f"{col} as {name}" for col, name in zip(recursive_cols, col_names)])

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
        
        start_sql_exec = time.perf_counter()
        results = self.conn.execute(text(full_recursive_sql)).fetchall()
        end_sql_exec = time.perf_counter()
        
        end_forecast_total = time.perf_counter()
        logger.info(f"Recursive CTE Forecast: {end_forecast_total-start_forecast_total:.4f}s (DB Exec={end_sql_exec-start_sql_exec:.4f}s)")

        preds = []
        for row in results:
            preds.append({
                "forecast_date": row[0].isoformat() if isinstance(row[0], datetime) else str(row[0]),
                "forecast_value": {target: float(row[1])}
            })
            
        return preds

    def get_loss(self) -> float:
        if not self.models or not hasattr(self, 'last_training_view'):
            return 0.0
        
        # Check row count in training view
        try:
            count_res = self.conn.execute(text(f"SELECT COUNT(*) FROM {self.last_training_view}")).fetchone()
            row_count = count_res[0] if count_res else 0
            logger.info(f"Calculating loss on {row_count} rows in {self.last_training_view}")
        except Exception as e:
            logger.error(f"Error counting rows in {self.last_training_view}: {e}")
            return 0.0

        total_sse = 0.0
        # Calculate training SSE for each target
        for target, model in self.models.items():
            pred_agg = model.get_prediction_aggregate()
            case_sql = agg_to_sql(pred_agg, qualified=False)
            # Patch Postgres types
            case_sql = case_sql.replace("AS DOUBLE", "AS DOUBLE PRECISION")
            
            sql = f"""
            SELECT SUM(POWER({target} - ({case_sql}), 2))
            FROM {self.last_training_view}
            """
            
            try:
                res = self.conn.execute(text(sql)).fetchone()
                if res and res[0] is not None:
                    sse = float(res[0])
                    total_sse += sse
                    logger.debug(f"Target {target} SSE: {sse:.4f}")
            except Exception as e:
                logger.error(f"Error calculating loss for {target}: {e}")
        
        logger.info(f"Total SSE: {total_sse:.4f}")
        return total_sse

    def teardown_single(self):
        self.conn.execute(text(f"DROP TABLE IF EXISTS {self.base_table} CASCADE"))
        self.conn.commit()

    def teardown_batch(self):
        self.teardown_single()
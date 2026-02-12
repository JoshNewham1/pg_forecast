import sys
import os
import logging
import pandas as pd
import numpy as np
import time
import math
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
        self.query_stats = []

    def get_schema(self, table: str) -> list:
        sql = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'"
        result = self._execute_query(sql)
        return [row[0] for row in result]

    def add_table(self, table: str, table_address):
        if isinstance(table_address, pd.DataFrame):
            table_address.to_sql(table, self.conn, if_exists='replace', index=False)
        else:
            raise Exception("PostgresExecutor.add_table only supports pandas DataFrame")

    def delete_table(self, table: str):
        self.check_table(table)
        sql = "DROP TABLE IF EXISTS " + table + " CASCADE;"
        self._execute_query(sql)

    def _execute_query(self, q):
        q = q.replace("AS DOUBLE", "AS DOUBLE PRECISION")
        start = time.perf_counter()
        
        # We assume the connection is in AUTOCOMMIT mode to avoid transaction blocks
        result = self.conn.execute(text(q))
        
        duration = time.perf_counter() - start
        self.query_stats.append({"query": q[:100].replace("\n", " "), "duration": duration})
        
        if q.strip().upper().startswith("SELECT") or q.strip().upper().startswith("WITH") or "RETURNING" in q.strip().upper():
            rows = result.fetchall()
            return [tuple(float(v) if isinstance(v, Decimal) else v for v in row) for row in rows]
        else:
            return None

    def log_stats(self):
        if not self.query_stats:
            return
        total_time = sum(s["duration"] for s in self.query_stats)
        logger.info(f"--- Query Profiling (Total: {total_time:.4f}s, Count: {len(self.query_stats)}) ---")
        sorted_stats = sorted(self.query_stats, key=lambda x: x["duration"], reverse=True)
        for i, s in enumerate(sorted_stats[:5]):
            logger.info(f"Slow Query #{i+1} ({s['duration']:.4f}s): {s['query']}...")
        self.query_stats = []

    def spja_query(self, spja_data: SPJAData, parenthesize: bool = True):
        parsed_aggregate_expressions = []
        for target_col, aggExp in spja_data.aggregate_expressions.items():
            window_clause = " OVER joinboost_window " if spja_data.window_by and is_agg(aggExp.agg) else ""
            rename_expr = (" AS " + value_to_sql(target_col, qualified=False)) if target_col is not None else ""
            parsed_aggregate_expressions.append(agg_to_sql(aggExp, qualified=spja_data.qualified) + window_clause + rename_expr)

        sql = "SELECT " + ", ".join(parsed_aggregate_expressions) + "\n"
        sql += "FROM " + ",".join(spja_data.from_tables) + "\n"

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
        if parenthesize:
            sql = f"({sql})"
        return sql

    def _spja_query_to_table(self, spja_data: SPJAData) -> str:
        spja = self.spja_query(spja_data, parenthesize=False)
        name_ = self.get_next_name()
        if spja_data.replace:
            self._execute_query(f"DROP TABLE IF EXISTS {name_} CASCADE")
        sql = "CREATE TABLE " + name_ + " AS " + spja
        self._execute_query(sql)
        return name_

    def _spja_query_as_view(self, spja_data: SPJAData):
        spja = self.spja_query(spja_data, parenthesize=False)
        name_ = self.get_next_name()
        sql = "CREATE OR REPLACE VIEW " + name_ + " AS " + spja
        self._execute_query(sql)
        return name_


class TSJoinBoostSUT:
    def __init__(self, model_name: str = "ts_joinboost", lags: int = 5, n_features: int = 50, predict_single_target: bool = False):
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
        
        logger.info(f"TSJoinBoost Target columns: {self.target_cols}")

        cols_ddl = ", ".join([f"{k} DOUBLE PRECISION" for k in self.target_cols])
        ddl = f"""
        DROP TABLE IF EXISTS {self.base_table} CASCADE;
        CREATE TABLE {self.base_table} (date TIMESTAMP PRIMARY KEY, {cols_ddl});
        SELECT create_hypertable('{self.base_table}', 'date', if_not_exists => TRUE);
        ALTER TABLE {self.base_table} SET (timescaledb.compress, timescaledb.compress_orderby = 'date DESC');
        
        DROP TABLE IF EXISTS {self.train_table} CASCADE;
        """
        self.conn.execute(text(ddl))

        self.add_batch(seed_records)
        self._train()

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
        
        if last_processed_date is None:
            # Full materialization
            materialize_sql = f"""
            DROP TABLE IF EXISTS {self.train_table} CASCADE;
            CREATE TABLE {self.train_table} AS
            SELECT * FROM (SELECT {', '.join(select_clause)} FROM {self.base_table}) t
            WHERE {' AND '.join(conds)};
            SELECT create_hypertable('{self.train_table}', 'date', if_not_exists => TRUE, migrate_data => true);
            ALTER TABLE {self.train_table} ADD PRIMARY KEY (date);
            """
            self.conn.execute(text(materialize_sql))
        else:
            # Incremental append
            interval_res = self.conn.execute(text(f"SELECT date FROM {self.base_table} ORDER BY date DESC LIMIT 2")).fetchall()
            lookback = (interval_res[0][0] - interval_res[1][0]) * (self.lags + 1) if len(interval_res) == 2 else timedelta(hours=1)

            inc_sql = f"""
            INSERT INTO {self.train_table}
            SELECT * FROM (
                SELECT {', '.join(select_clause)} 
                FROM {self.base_table} 
                WHERE date > '{last_processed_date - lookback}'::timestamp
            ) t
            WHERE date > '{last_processed_date}'::timestamp AND {' AND '.join(conds)}
            ON CONFLICT (date) DO NOTHING;
            """
            self.conn.execute(text(inc_sql))

        self.models = {}
        for target in self.target_cols:
            # Re-compress
            try:
                compress_sql = f"""
                ALTER TABLE {self.train_table} SET (
                    timescaledb.compress, 
                    timescaledb.compress_orderby = 'date DESC',
                    timescaledb.compress_segmentby = '{target}'
                );
                SELECT compress_chunk(i) FROM show_chunks('{self.train_table}') i WHERE NOT is_compressed(i);
                ANALYZE {self.train_table};
                """
                self.conn.execute(text(compress_sql))
            except Exception:
                pass

            dataset = JoinGraph(self.executor)
            dataset.add_relation(self.train_table, cols_to_train, y=target)
            reg = GradientBoosting(learning_rate=0.05, max_depth=4, n_estimators=100)
            reg.fit(dataset)
            self.models[target] = reg
            self.last_training_view = self.train_table
            
        logger.info(f"TSJoinBoost Fit complete in {time.perf_counter()-start_total:.4f}s")
        self.executor.log_stats()

    def forecast(self, horizon: int) -> List[Dict[str, float]]:
        self._train()
        if not self.target_cols: return []
        target = self.target_cols[0]
        model = self.models.get(target)
        if not model: return []

        pred_agg = model.get_prediction_aggregate()
        case_sql = agg_to_sql(pred_agg, qualified=False).replace("AS DOUBLE", "AS DOUBLE PRECISION")

        import re
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
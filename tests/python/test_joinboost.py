import time
import math
import pandas as pd
import numpy as np
import duckdb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import sys
import os
import gc
import pyarrow.parquet as pq
import csv

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Add JoinBoost to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eval/JoinBoost/src")))

from joinboost.executor import DuckdbExecutor
from joinboost.joingraph import JoinGraph
from joinboost.app import GradientBoosting
from joinboost.aggregator import agg_to_sql

from joinboost_adapter import PostgresExecutor

RESULTS_FILE = "joinboost_results.csv"
LIMIT = 80_000_000
LEARNING_RATE = 0.1
NUM_LEAVES = 8
DEPTH = 3
ORDER_BY = "id"
XGB_EXPORT_PATH = "xgb_train.parquet"
CSV_EXPORT_PATH = "train.csv"

# DuckDB/XGBoost/Sklearn
df = None
con = None
# JoinBoost
engine = None

def setup_duckdb():
    global con
    con = duckdb.connect("favorita.db")

    con.execute("SET memory_limit = '30GB';")

    # Base path for data
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eval/favorita/data"))

    sales_path = os.path.join(base_path, "sales_80m.csv")
    items_path = os.path.join(base_path, "items.csv")
    holiday_path = os.path.join(base_path, "holidays.csv")
    oil_path = os.path.join(base_path, "oil.csv")
    trans_path = os.path.join(base_path, "trans.csv")
    stores_path = os.path.join(base_path, "stores.csv")

    # Load tables into DuckDB
    con.execute(f"CREATE TABLE IF NOT EXISTS sales AS SELECT * FROM '{sales_path}' ORDER BY {ORDER_BY} LIMIT {LIMIT}")
    con.execute(f"CREATE TABLE IF NOT EXISTS items AS SELECT * FROM '{items_path}'")
    con.execute(f"CREATE TABLE IF NOT EXISTS holidays AS SELECT * FROM '{holiday_path}'")
    con.execute(f"CREATE TABLE IF NOT EXISTS oil AS SELECT * FROM '{oil_path}'")
    con.execute(f"CREATE TABLE IF NOT EXISTS trans AS SELECT * FROM '{trans_path}'")
    con.execute(f"CREATE TABLE IF NOT EXISTS stores AS SELECT * FROM '{stores_path}'")

    # Create the full join table for validation and scikit-learn
    # We use LEFT JOIN to keep all rows from sales
    view_sql = f"""
    CREATE OR REPLACE VIEW train AS (
    SELECT
        s.id,
        s.target,
        h.htype, h.locale, h.locale_name, h.transferred, h.f2,
        o.dcoilwtico, o.f3,
        t.transactions, t.f5,
        st.city, st.state, st.stype, st.cluster, st.f4,
        i.family, i.class, i.perishable, i.f1
    FROM sales AS s
    INNER JOIN items i ON s.item_nbr = i.item_nbr
    INNER JOIN trans t ON s.date = t.date AND s.store_nbr = t.store_nbr
    INNER JOIN stores st ON t.store_nbr = st.store_nbr
    LEFT JOIN holidays h ON t.date = h.date
    LEFT JOIN oil o ON h.date = o.date
    ORDER BY s.id
    )
    """
    con.execute(view_sql)

def setup_postgres():
    global engine
    load_dotenv()
    TEST_DB_USERNAME = os.getenv("TEST_DB_USERNAME")
    TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD")
    TEST_DB_HOST = os.getenv("TEST_DB_HOST", "localhost")
    TEST_DB_PORT = os.getenv("TEST_DB_PORT", "5432")
    TEST_DB_NAME = os.getenv("TEST_DB_NAME")
    engine = create_engine(
        f"postgresql+psycopg2://{TEST_DB_USERNAME}:{TEST_DB_PASSWORD}@"
        f"{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}?options=-csearch_path%3Dfav,public"
    )

def set_postgres_optimisations(conn):
    # Enable JIT and Parallelism for aggregates
    # NOTE: Settings optimised for 16 threads, 32GB RAM. They might want tweaking
    conn.execute(text("SET jit = off;"))
    conn.execute(text("SET max_parallel_workers_per_gather = 8;"))
    conn.execute(text("SET max_parallel_workers = 12;"))
    conn.execute(text("SET parallel_setup_cost = 0;"))
    conn.execute(text("SET parallel_tuple_cost = 0;"))
    conn.execute(text("SET min_parallel_table_scan_size = '1MB';"))
    conn.execute(text("SET work_mem = '2GB';"))
    conn.execute(text("SET temp_buffers = '1GB';"))
    

def setup_timescale_first_time():
    global engine
    with engine.connect() as conn:
        print("Converting to Hypertable and Compressing...")
        conn.execute(text("SELECT create_hypertable('sales', 'date', migrate_data => true, if_not_exists => true);"))
        
        # Crucial: Segment by join keys (store_nbr, item_nbr) to co-locate data
        conn.execute(text("""
            ALTER TABLE sales SET (
                timescaledb.compress,
                timescaledb.compress_orderby = 'date DESC',
                timescaledb.compress_segmentby = 'item_nbr, store_nbr' 
            );
        """))
        
        # Compress existing data immediately
        conn.execute(text("SELECT compress_chunk(i) FROM show_chunks('sales') i;"))
        conn.execute(text("ANALYZE sales;")) # Update stats for planner
        conn.commit()

        # Create the full join table for RMSE calculation
        view_sql = f"""
        CREATE VIEW fav.train AS (
            SELECT
                s.id,
                s.target,
                h.htype, h.locale, h.locale_name, h.transferred, h.f2,
                o.dcoilwtico, o.f3,
                t.transactions, t.f5,
                st.city, st.state, st.stype, st.cluster, st.f4,
                i.family, i.class, i.perishable, i.f1
            FROM fav.sales AS s
            INNER JOIN fav.items i ON s.item_nbr = i.item_nbr
            INNER JOIN fav.trans t ON s.date = t.date AND s.store_nbr = t.store_nbr
            INNER JOIN fav.stores st ON t.store_nbr = st.store_nbr
            LEFT JOIN fav.holidays h ON t.date = h.date
            LEFT JOIN fav.oil o ON h.date = o.date
            ORDER BY s.id
        );
        """
        conn.execute(view_sql)

def setup_sklearn():
    global df
    global con

    start_time = time.perf_counter()
    # Get the train data as dataframe for scikit-learn
    # We explicitly ORDER BY id to ensure consistent ordering
    if not os.path.exists(os.path.join(os.getcwd(), CSV_EXPORT_PATH)):
        con.execute(f"""
            COPY (
                SELECT *
                FROM train
                ORDER BY id
            )
            TO '{CSV_EXPORT_PATH}' (FORMAT CSV);
        """)

        print(f"Sklearn join materialisation time: ", time.perf_counter() - start_time)

def setup_xgboost():
    global con

    y_col = "target"
    x_cols = [
        "htype",
        "locale",
        "locale_name",
        "transferred",
        "f2",
        "dcoilwtico",
        "f3",
        "transactions",
        "f5",
        "city",
        "state",
        "stype",
        "cluster",
        "f4",
        "family",
        "class",
        "perishable",
        "f1",
    ]

    if os.path.exists(XGB_EXPORT_PATH):
        return

    start_time = time.perf_counter()

    con.execute(f"""
        COPY (
            SELECT {y_col}, {", ".join(x_cols)}
            FROM train
            ORDER BY id
        )
        TO '{XGB_EXPORT_PATH}' (FORMAT PARQUET);
    """)

    print(f"XGBoost export time: {time.perf_counter() - start_time}")

def test_duckdb(iterations=1, predict=False):
    global con
    results = {}

    exe = DuckdbExecutor(con, debug=False)
    dataset = JoinGraph(exe=exe)

    dataset.add_relation("sales", [], y = 'target')
    dataset.add_relation("holidays", ["htype", "locale", "locale_name", "transferred","f2"])
    dataset.add_relation("oil", ["dcoilwtico","f3"])
    dataset.add_relation("trans", ["transactions","f5"])
    dataset.add_relation("stores", ["city","state","stype","cluster","f4"])
    dataset.add_relation("items", ["family","class","perishable","f1"])

    dataset.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
    dataset.add_join("sales", "trans", ["date", "store_nbr"], ["date", "store_nbr"])
    dataset.add_join("trans", "stores", ["store_nbr"], ["store_nbr"])
    dataset.add_join("trans", "holidays", ["date"], ["date"])
    dataset.add_join("holidays", "oil", ["date"], ["date"])

    reg = GradientBoosting(
        learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES, max_depth=DEPTH, n_estimators=iterations, partition_early = True
    )

    start_time = time.perf_counter()
    reg.fit(dataset)
    results["joinboost_fit"] = time.perf_counter() - start_time
    print(f"DuckDB JoinBoost fit time: ", time.perf_counter() - start_time)

    if predict:
        start_time = time.perf_counter()
        pred_agg = reg.get_prediction_aggregate()
        pred_sql = agg_to_sql(pred_agg, qualified=False)
        reg_prediction = con.execute(f"SELECT {pred_sql} FROM train ORDER BY id").fetchall()
        reg_prediction = np.array(reg_prediction)[:, 0]
        results["joinboost_predict"] = time.perf_counter() - start_time
        print(f"DuckDB JoinBoost predict time: ", time.perf_counter() - start_time)
        gc.collect()

    _reg_rmse = reg.compute_rmse("train")[0]

    print(f"JoinBoost RMSE: {_reg_rmse}")
    results["joinboost_rmse"] = _reg_rmse

    return results

def test_postgres(iterations=1, predict=False):
    global engine
    global exe
    global dataset
    global reg
    results = {}

    with engine.connect() as conn:
        set_postgres_optimisations(conn)

        exe = PostgresExecutor(conn, debug=False)
        dataset = JoinGraph(exe=exe)

        # Define schema
        dataset.add_relation("fav.pg_sales", [], y='target')
        dataset.add_relation("fav.holidays", ["htype", "locale", "locale_name", "transferred", "f2"])
        dataset.add_relation("fav.oil", ["dcoilwtico", "f3"])
        dataset.add_relation("fav.trans", ["transactions", "f5"])
        dataset.add_relation("fav.stores", ["city", "state", "stype", "cluster", "f4"])
        dataset.add_relation("fav.items", ["family", "class", "perishable", "f1"])

        # Define joins
        dataset.add_join("fav.pg_sales", "fav.items", ["item_nbr"], ["item_nbr"])
        dataset.add_join("fav.pg_sales", "fav.trans", ["date", "store_nbr"], ["date", "store_nbr"])
        dataset.add_join("fav.trans", "fav.stores", ["store_nbr"], ["store_nbr"])
        dataset.add_join("fav.trans", "fav.holidays", ["date"], ["date"])
        dataset.add_join("fav.holidays", "fav.oil", ["date"], ["date"])

        reg = GradientBoosting(
            learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES, max_depth=DEPTH, n_estimators=iterations
        )

        # FIT
        start_time = time.perf_counter()
        reg.fit(dataset)
        results["postgres_fit"] = time.perf_counter() - start_time
        print(f"PostgreSQL JoinBoost fit time: {results['postgres_fit']}")

        # PREDICT (In-Database)
        if predict:
            start_pred = time.perf_counter()
            pred_agg = reg.get_prediction_aggregate()
            pred_sql = agg_to_sql(pred_agg, qualified=False)
            reg_prediction = exe._execute_query(f"SELECT {pred_sql} FROM train ORDER BY id")
            print(f"Prediction shape: {len(reg_prediction)}")
            results["pred_time"] = time.perf_counter() - start_pred
            print(f"PostgreSQL Predict Time: {results['pred_time']}")

        # RMSE Calculation
        rmse = reg.compute_rmse("fav.train")[0]
        print(f"PostgreSQL RMSE: {rmse}")
        results["postgres_rmse"] = rmse
        
        # CLEANUP: Delete all jb_* temp tables
        print("Cleaning up intermediate JoinBoost tables...")
        cleanup_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE 'jb_%';
        """
        temp_tables = exe._execute_query(cleanup_query)
        if temp_tables:
            for (tbl,) in temp_tables:
                exe._execute_query(f"DROP TABLE IF EXISTS {tbl} CASCADE")
            print(f"Removed {len(temp_tables)} temporary tables.")
    
    return results

def test_timescale(iterations=1, predict=False):
    global engine
    global exe
    global dataset
    global reg
    results = {}
    
    # Use the optimized PostgresExecutor
    with engine.connect() as conn:
        # Enable JIT and Parallelism for aggregates
        # NOTE: Settings optimised for 16 threads, 32GB RAM. They might want tweaking
        set_postgres_optimisations(conn)
        
        exe = PostgresExecutor(conn, debug=False)
        dataset = JoinGraph(exe=exe)

        # Define schema
        dataset.add_relation("fav.sales", [], y='target')
        dataset.add_relation("fav.holidays", ["htype", "locale", "locale_name", "transferred", "f2"])
        dataset.add_relation("fav.oil", ["dcoilwtico", "f3"])
        dataset.add_relation("fav.trans", ["transactions", "f5"])
        dataset.add_relation("fav.stores", ["city", "state", "stype", "cluster", "f4"])
        dataset.add_relation("fav.items", ["family", "class", "perishable", "f1"])

        # Define joins
        dataset.add_join("fav.sales", "fav.items", ["item_nbr"], ["item_nbr"])
        dataset.add_join("fav.sales", "fav.trans", ["date", "store_nbr"], ["date", "store_nbr"])
        dataset.add_join("fav.trans", "fav.stores", ["store_nbr"], ["store_nbr"])
        dataset.add_join("fav.trans", "fav.holidays", ["date"], ["date"])
        dataset.add_join("fav.holidays", "fav.oil", ["date"], ["date"])

        reg = GradientBoosting(
            learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES, max_depth=DEPTH, n_estimators=iterations
        )

        # FIT
        start_time = time.perf_counter()
        reg.fit(dataset)
        results["timescale_fit"] = time.perf_counter() - start_time
        print(f"Timescale JoinBoost fit time: {results['timescale_fit']}")

        # PREDICT (In-Database)
        if predict:
            start_pred = time.perf_counter()
            pred_agg = reg.get_prediction_aggregate()
            pred_sql = agg_to_sql(pred_agg, qualified=False)
            reg_prediction = exe._execute_query(f"SELECT {pred_sql} FROM train ORDER BY id")
            print(f"Prediction shape: {len(reg_prediction)}")
            results["pred_time"] = time.perf_counter() - start_pred
            print(f"Timescale Predict Time: {results['pred_time']}")

        # RMSE Calculation
        rmse = reg.compute_rmse("fav.train")[0]
        print(f"Timescale RMSE: {rmse}")
        results["timescale_rmse"] = rmse
        
        # CLEANUP: Delete all jb_* temp tables
        print("Cleaning up intermediate JoinBoost tables...")
        cleanup_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE 'jb_%';
        """
        temp_tables = exe._execute_query(cleanup_query)
        if temp_tables:
            for (tbl,) in temp_tables:
                exe._execute_query(f"DROP TABLE IF EXISTS {tbl} CASCADE")
            print(f"Removed {len(temp_tables)} temporary tables.")
    
    return results

def test_citus(iterations=1, predict=False):
    global engine
    results = {}
    
    # Use the optimized PostgresExecutor
    with engine.connect() as conn:
        set_postgres_optimisations(conn)
        conn.execute(text("SET columnar.compression_decode_buffer_size = '1GB';"))
        
        exe = PostgresExecutor(conn, debug=False)
        dataset = JoinGraph(exe=exe)

        # Define schema
        dataset.add_relation("fav.citus_sales", [], y='target')
        dataset.add_relation("fav.citus_holidays", ["htype", "locale", "locale_name", "transferred", "f2"])
        dataset.add_relation("fav.citus_oil", ["dcoilwtico", "f3"])
        dataset.add_relation("fav.citus_trans", ["transactions", "f5"])
        dataset.add_relation("fav.citus_stores", ["city", "state", "stype", "cluster", "f4"])
        dataset.add_relation("fav.citus_items", ["family", "class", "perishable", "f1"])

        # Define joins
        dataset.add_join("fav.citus_sales", "fav.citus_items", ["item_nbr"], ["item_nbr"])
        dataset.add_join("fav.citus_sales", "fav.citus_trans", ["date", "store_nbr"], ["date", "store_nbr"])
        dataset.add_join("fav.citus_trans", "fav.citus_stores", ["store_nbr"], ["store_nbr"])
        dataset.add_join("fav.citus_trans", "fav.citus_holidays", ["date"], ["date"])
        dataset.add_join("fav.citus_holidays", "fav.citus_oil", ["date"], ["date"])

        reg = GradientBoosting(
            learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES, max_depth=DEPTH, n_estimators=iterations
        )

        # FIT
        start_time = time.perf_counter()
        reg.fit(dataset)
        results["timescale_fit"] = time.perf_counter() - start_time
        print(f"Citus JoinBoost fit time: {results['timescale_fit']}")

        # PREDICT (In-Database)
        if predict:
            start_pred = time.perf_counter()
            pred_agg = reg.get_prediction_aggregate()
            pred_sql = agg_to_sql(pred_agg, qualified=False)
            reg_prediction = exe._execute_query(f"SELECT {pred_sql} FROM train ORDER BY id")
            print(f"Prediction shape: {len(reg_prediction)}")
            results["pred_time"] = time.perf_counter() - start_pred
            print(f"Citus Predict Time: {results['pred_time']}")

        # RMSE Calculation
        rmse = reg.compute_rmse("fav.train")[0]
        print(f"Citus RMSE: {rmse}")
        results["rmse"] = rmse
        
        # CLEANUP: Delete all jb_* temp tables
        print("Cleaning up intermediate JoinBoost tables...")
        cleanup_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE 'jb_%';
        """
        temp_tables = exe._execute_query(cleanup_query)
        if temp_tables:
            for (tbl,) in temp_tables:
                exe._execute_query(f"DROP TABLE IF EXISTS {tbl} CASCADE")
            print(f"Removed {len(temp_tables)} temporary tables.")
    
    return results

def test_sklearn(iterations=1, predict=False):
    # Preprocess for scikit-learn (handle nulls and categorical)
    df = pd.read_csv(CSV_EXPORT_PATH)
    results = {}

    if df is None:
        print("Run setup_sklearn() first to test XGBoost in memory")
        return
    
    df = df.fillna(0)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]

    y = "target"
    x = [
        "htype",
        "locale",
        "locale_name",
        "transferred",
        "f2",
        "dcoilwtico",
        "f3",
        "transactions",
        "f5",
        "city",
        "state",
        "stype",
        "cluster",
        "f4",
        "family",
        "class",
        "perishable",
        "f1",
    ]

    clf = GradientBoostingRegressor(
        max_depth=DEPTH, learning_rate=LEARNING_RATE, n_estimators=iterations, max_leaf_nodes=NUM_LEAVES
    )
    start_time = time.perf_counter()
    clf = clf.fit(df[x], df[y])
    results["sklearn_fit"] = time.perf_counter() - start_time
    print(f"Sklearn fit time: ", time.perf_counter() - start_time)

    if predict:
        start_time = time.perf_counter()
        clf_prediction = clf.predict(df[x])
        results["joinboost_predict"] = time.perf_counter() - start_time
        print(f"Sklearn predict time: ", time.perf_counter() - start_time)

    mse = mean_squared_error(df[y], clf_prediction)
    print(f"Sklearn RMSE: {math.sqrt(mse)}")
    results["sklearn_rmse"] = math.sqrt(mse)

    return results

def test_xgboost_in_memory(iterations=1, predict=False):
    # Preprocess for scikit-learn (handle nulls and categorical)
    df = pd.read_csv(CSV_EXPORT_PATH)
    results = {}

    if df is None:
        print("Run setup_sklearn() first to test XGBoost in memory")
        return
    
    df = df.fillna(0)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]
    print("XGBoost loaded CSV...")

    y_col = "target"
    # Filter features to match the other benchmarks
    x_cols = [col for col in df.columns if col not in ["id", "target"]]

    # We use the already materialized and factorized 'df' from setup_sklearn
    start_time = time.perf_counter()
    dtrain = xgb.DMatrix(df[x_cols], label=df[y_col])
    results["xgboost_mem_prep"] = time.perf_counter() - start_time

    # TRAIN
    params = {
        "objective": "reg:squarederror",
        "learning_rate": LEARNING_RATE,
        "max_depth": DEPTH,
        "tree_method": "hist", # Matches the out-of-core version
    }

    start_train = time.perf_counter()
    model = xgb.train(params, dtrain, num_boost_round=iterations)
    results["xgboost_mem_fit"] = time.perf_counter() - start_train
    print(f"XGBoost In-Memory fit time: {results['xgboost_mem_fit']}")

    # RMSE
    preds = model.predict(dtrain)
    rmse = np.sqrt(mean_squared_error(df[y_col], preds))
    print(f"XGBoost In-Memory RMSE: {rmse}")
    results["xgboost_mem_rmse"] = rmse

    return results

def test_xgboost_out_of_core(iterations=1, predict=False):
    results = {}

    if not os.path.exists(XGB_EXPORT_PATH):
        raise RuntimeError("Run setup_xgboost() first.")

    # 1. Define the Iterator following the XGBoost DataIter interface
    class ParquetIterator(xgb.DataIter):
        def __init__(self, path, batch_size=100_000):
            self.path = path
            self.batch_size = batch_size
            self._it = None
            super().__init__()

        def next(self, input_data):
            # Initialize the pyarrow iterator on first call
            if self._it is None:
                parquet_file = pq.ParquetFile(self.path)
                # iter_batches returns an iterator of RecordBatches
                self._it = parquet_file.iter_batches(batch_size=self.batch_size)
            
            try:
                # Get next batch and convert to Pandas for XGBoost
                batch = next(self._it)
                df_batch = batch.to_pandas()
                
                y = df_batch["target"]
                X = df_batch.drop(columns=["target"])
                
                # Important: Ensure categorical columns are typed correctly if present
                # for col in X.select_dtypes("object").columns:
                #    X[col] = X[col].astype("category")

                input_data(data=X, label=y)
                return 1
            except StopIteration:
                return 0

        def reset(self):
            self._it = None

    # Define QuantileDMatrix (for out-of-core execution)
    # with ParquetIterator above, then train
    start_time = time.perf_counter()
    it = ParquetIterator(XGB_EXPORT_PATH)
    dtrain = xgb.QuantileDMatrix(it, enable_categorical=True)

    params = {
        "objective": "reg:squarederror",
        "eta": LEARNING_RATE,
        "max_depth": DEPTH,
        "tree_method": "hist", 
        "eval_metric": "rmse",
    }

    model = xgb.train(params, dtrain, num_boost_round=iterations)
    
    results["xgboost_fit"] = time.perf_counter() - start_time
    print(f"XGBoost fit time: {results['xgboost_fit']}")

    # Prediction for RMSE
    start_time = time.perf_counter()
    preds = model.predict(dtrain)
    results["xgboost_pred"] = time.perf_counter() - start_time
    print(f"XGBoost prediction time: {results['xgboost_pred']}")

    rmse = np.sqrt(np.mean((preds - dtrain.get_label()) ** 2))
    print(f"XGBoost RMSE: {rmse}")
    
    return results

def cleanup_artifacts():
    files_to_delete = ["favorita.db", XGB_EXPORT_PATH, CSV_EXPORT_PATH]

    for file in files_to_delete:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Deleted existing file: {file}")
            except Exception as e:
                print(f"Failed to delete {file}: {e}")

def append_results_to_file(model_name, iteration, result_dict):
    file_exists = os.path.exists(RESULTS_FILE)

    row = {
        "model": model_name,
        "iterations": iteration,
        **result_dict
    }

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python benchmark.py <model> <iterations>")
        print("Available models: duckdb, postgres, timescale, citus, sklearn, xgb_mem, xgb_ooc")
        sys.exit(1)

    model = sys.argv[1].lower()
    iterations = int(sys.argv[2])

    valid_models = {
        "duckdb",
        "postgres",
        "timescale",
        "citus",
        "sklearn",
        "xgb_mem",
        "xgb_ooc",
    }

    if model not in valid_models:
        print(f"Invalid model '{model}'")
        sys.exit(1)

    print(f"Running benchmark for model: {model}")
    print("=" * 60)

    # Setup only what is required
    if model == "duckdb":
        setup_duckdb()

    elif model in {"postgres", "timescale", "citus"}:
        setup_postgres()

    elif model == "sklearn":
        setup_duckdb()
        setup_sklearn()

    elif model == "xgb_mem":
        setup_duckdb()
        setup_sklearn()

    elif model == "xgb_ooc":
        setup_duckdb()
        setup_xgboost()
    
    print("Finished set up, waiting to start full test...")
    time.sleep(30)

    # Run selected model
    for _ in range(iterations):
        for i in [1, 5, 10, 15, 20, 25, 30]:
            print(f"\nTesting {model} for {i} iterations")
            print("-" * 60)

            if model == "duckdb":
                result = test_duckdb(i)

            elif model == "postgres":
                result = test_postgres(i)

            elif model == "timescale":
                result = test_timescale(i)

            elif model == "citus":
                result = test_citus(i)

            elif model == "sklearn":
                result = test_sklearn(i)

            elif model == "xgb_mem":
                result = test_xgboost_in_memory(i)

            elif model == "xgb_ooc":
                result = test_xgboost_out_of_core(i)

            append_results_to_file(model, i, result)
            gc.collect()
            time.sleep(60)
        gc.collect()
        time.sleep(300)

    print("\nBenchmark complete.")

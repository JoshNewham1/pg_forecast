import time
import math
import pandas as pd
import numpy as np
import duckdb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import sys
import os
import gc

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Add JoinBoost to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eval/JoinBoost/src")))

from joinboost.executor import DuckdbExecutor
from joinboost.joingraph import JoinGraph
from joinboost.app import GradientBoosting
from joinboost.aggregator import agg_to_sql

from joinboost_adapter import PostgresExecutor

LIMIT = 80_000_000
LEARNING_RATE = 0.1
NUM_LEAVES = 8
DEPTH = 3
ORDER_BY = "id"
df = None
con = duckdb.connect("favorita.db")
engine = None

def setup_duckdb():
    global con

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
    CREATE VIEW train AS (
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

def setup_timescale():
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
    df = con.execute("SELECT * FROM train").df()

    # Preprocess for scikit-learn (handle nulls and categorical)
    df = df.fillna(0)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]

    print(f"Sklearn join materialisation time: ", time.perf_counter() - start_time)

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

def test_timescale(iterations=1, predict=False):
    global engine
    results = {}
    
    # Use the optimized PostgresExecutor
    with engine.connect() as conn:
        # Enable JIT and Parallelism for aggregates
        # NOTE: Settings optimised for 16 threads, 32GB RAM. They might want tweaking
        conn.execute(text("SET jit = on;"))
        conn.execute(text("SET max_parallel_workers_per_gather = 8;"))
        conn.execute(text("SET max_parallel_workers = 12;"))
        conn.execute(text("SET work_mem = '512MB';"))
        conn.execute(text("SET temp_buffers = '1GB';"))
        
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
    global df
    results = {}

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

if __name__ == "__main__":
    results = []
    setup_duckdb()
    setup_sklearn()
    # setup_timescale_first_time()
    # setup_timescale()
    for i in [1, 5, 10, 15, 20, 25, 30]:
        print(f"Testing gradient boosting for {i} iterations")
        print(f"=============================================")
        # results.append(test_duckdb(i))
        # results.append(test_timescale(i))
        results.append(test_sklearn(i))

    print(results)
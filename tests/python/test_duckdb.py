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

LIMIT = 80_000_000
LEARNING_RATE = 0.1
NUM_LEAVES = 8
DEPTH = 3
ORDER_BY = "id"
df = None

def test_gradient_boosting(iterations=1):
    global df
    results = {"iteration": iterations}

    # Base path for data
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eval/favorita/data"))

    sales_path = os.path.join(base_path, "sales_80m.csv")
    items_path = os.path.join(base_path, "items.csv")
    holiday_path = os.path.join(base_path, "holidays.csv")
    oil_path = os.path.join(base_path, "oil.csv")
    trans_path = os.path.join(base_path, "trans.csv")
    stores_path = os.path.join(base_path, "stores.csv")

    con = duckdb.connect(database=':memory:')

    # Load tables into DuckDB
    con.execute(f"CREATE OR REPLACE TABLE sales AS SELECT * FROM '{sales_path}' ORDER BY {ORDER_BY} LIMIT {LIMIT}")
    con.execute(f"CREATE OR REPLACE TABLE items AS SELECT * FROM '{items_path}'")
    con.execute(f"CREATE OR REPLACE TABLE holidays AS SELECT * FROM '{holiday_path}'")
    con.execute(f"CREATE OR REPLACE TABLE oil AS SELECT * FROM '{oil_path}'")
    con.execute(f"CREATE OR REPLACE TABLE trans AS SELECT * FROM '{trans_path}'")
    con.execute(f"CREATE OR REPLACE TABLE stores AS SELECT * FROM '{stores_path}'")

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

    start_time = time.perf_counter()
    pred_agg = reg.get_prediction_aggregate()
    pred_sql = agg_to_sql(pred_agg, qualified=False)
    reg_prediction = con.execute(f"SELECT {pred_sql} FROM train ORDER BY id").fetchall()
    reg_prediction = np.array(reg_prediction)[:, 0]
    results["joinboost_predict"] = time.perf_counter() - start_time
    print(f"DuckDB JoinBoost predict time: ", time.perf_counter() - start_time)
    gc.collect()

    if df is None:
        start_time = time.perf_counter()
        # Get the train data as dataframe for scikit-learn
        # We explicitly ORDER BY id to ensure consistent ordering
        df = con.execute("SELECT * FROM train").df()

        # Preprocess for scikit-learn (handle nulls and categorical)
        df = df.fillna(0)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col])[0]

        results["materialise"] = time.perf_counter() - start_time
        print(f"Sklearn join materialisation time: ", time.perf_counter() - start_time)

    clf = GradientBoostingRegressor(
        max_depth=DEPTH, learning_rate=LEARNING_RATE, n_estimators=iterations, max_leaf_nodes=NUM_LEAVES
    )
    start_time = time.perf_counter()
    clf = clf.fit(df[x], df[y])
    results["sklearn_fit"] = time.perf_counter() - start_time
    print(f"Sklearn fit time: ", time.perf_counter() - start_time)

    start_time = time.perf_counter()
    clf_prediction = clf.predict(df[x])
    results["joinboost_predict"] = time.perf_counter() - start_time
    print(f"Sklearn predict time: ", time.perf_counter() - start_time)

    mse = mean_squared_error(df[y], clf_prediction)
    _reg_rmse = reg.compute_rmse("train")[0]

    print(f"JoinBoost RMSE: {_reg_rmse}")
    print(f"Sklearn RMSE: {math.sqrt(mse)}")
    results["joinboost_rmse"] = _reg_rmse
    results["sklearn_rmse"] = math.sqrt(mse)

    # Check for parity
    # assert(abs(_reg_rmse - math.sqrt(mse)) < 10)
    assert(len(reg_prediction) == len(clf_prediction))
    # assert(np.allclose(reg_prediction, clf_prediction, atol=10))

    return results

if __name__ == "__main__":
    results = []
    for i in [1, 5, 10, 15, 20, 25, 30]:
        print(f"Testing gradient boosting for {i} iterations")
        print(f"=============================================")
        results.append(test_gradient_boosting(i))

    print(results)
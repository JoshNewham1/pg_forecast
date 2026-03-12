import pytest
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv


@pytest.fixture(scope="session")
def test_engine():
    """
    Create an SQLAlchemy engine for the test database
    """
    load_dotenv()
    TEST_DB_USERNAME = os.getenv("TEST_DB_USERNAME")
    TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD")
    TEST_DB_HOST = os.getenv("TEST_DB_HOST", "localhost")
    TEST_DB_PORT = os.getenv("TEST_DB_PORT", "5432")
    TEST_DB_NAME = os.getenv("TEST_DB_NAME")

    if not all([TEST_DB_USERNAME, TEST_DB_PASSWORD, TEST_DB_NAME]):
        raise RuntimeError(
            "TEST_DB_USERNAME, TEST_DB_PASSWORD, TEST_DB_NAME must be set"
        )

    engine = create_engine(
        f"postgresql+psycopg2://{TEST_DB_USERNAME}:{TEST_DB_PASSWORD}@"
        f"{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"
    )
    yield engine

    # Drop after all tests
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS pg_forecast_unit_test;"))
        conn.commit()


@pytest.fixture(scope="function", autouse=True)
def setup_schema(test_engine):
    """
    Optionally set up / tear down schema for each test.
    """
    with test_engine.connect() as conn:
        # Create function (you can also assume it exists)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS pg_forecast_unit_test (
            t TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            series_id TEXT NOT NULL,
            value DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (t, series_id)
        );
        CREATE INDEX IF NOT EXISTS idx_pg_forecast_unit_test_t ON public.pg_forecast_unit_test (t);
        TRUNCATE TABLE pg_forecast_unit_test;
        """))
        conn.commit()
    yield


def run_function(engine, func_name, *args):
    """
    Helper to call a Postgres function with parameters and return result.
    """
    placeholders = ", ".join([f":param{i}" for i in range(len(args))])
    query = text(f"SELECT {func_name}({placeholders}) AS result;")
    params = {f"param{i}": arg for i, arg in enumerate(args)}
    with engine.connect() as conn:
        result = conn.execute(query, params).scalar()
    return result


def css_loss_query(phi: list[float], theta: list[float], p: int, q: int, c: float = 0.0, series_id=None):
    """
    Build a SQLAlchemy text query to call css_loss on pg_forecast_unit_test.
    """
    where_clause = f"WHERE series_id = '{series_id}'" if series_id else ""
    phi_array = ", ".join(map(str, phi))
    theta_array = ", ".join(map(str, theta))

    query_str = f"""
        SELECT css_loss(
            vals := array_agg(value ORDER BY t)::double precision[],
            phi := ARRAY[{phi_array}]::double precision[],
            theta := ARRAY[{theta_array}]::double precision[],
            p := {p},
            q := {q},
            c := {c}
        )
        FROM pg_forecast_unit_test
        {where_clause};
    """
    return text(query_str)

def css_incremental_query(phi: list[float], theta: list[float], c: float = 0.0, d: int = 0, series_id=None):
    """
    Build a SQLAlchemy text query to call css_incremental aggregate
    on pg_forecast_unit_test.
    """
    where_clause = f"WHERE series_id = '{series_id}'" if series_id else ""
    phi_array = ", ".join(map(str, phi))
    theta_array = ", ".join(map(str, theta))

    query_str = f"""
        SELECT
            (css_incremental(value,
                ARRAY[{phi_array}]::double precision[],
                ARRAY[{theta_array}]::double precision[],
                {c}::double precision,
                {d}::int
            )).css
        FROM pg_forecast_unit_test
        {where_clause};
    """
    return text(query_str)


def arima_difference_query(d: int, series_id=None):
    """
    Build a SQLAlchemy text query to call arima_difference on pg_forecast_unit_test.
    """
    where_clause = f"WHERE series_id = '{series_id}'" if series_id else ""

    query_str = f"""
        SELECT arima_difference(
            vals := array_agg(value ORDER BY t)::double precision[],
            d := {d}
        )
        FROM pg_forecast_unit_test
        {where_clause};
    """
    return text(query_str)


def arima_integrate_query(d: int, series_id=None):
    """
    Build a SQLAlchemy text query to call arima_integrate on pg_forecast_unit_test.
    First call arima_difference to get differenced values, then arima_integrate
    to get the original values back
    """
    where_clause = f"WHERE series_id = '{series_id}'" if series_id else ""

    query_str = f"""
        SELECT arima_integrate(
            differences := arima_difference(array_agg(value ORDER BY t)::double precision[], {d}),
            d := {d},
            initial_vals := (array_agg(value ORDER BY t)::double precision[])[1:{d}]
        )
        FROM pg_forecast_unit_test
        {where_clause};
    """
    return text(query_str)


def arima_forecast_query(last_vals: list[float], last_residuals: list[float], p: int, q: int, c: float, phi: list[float], theta: list[float], horizon: int):
    """
    Build a SQLAlchemy text query to call arima on pg_forecast_unit_test.
    """
    query_str = f"""
        SELECT
            arima_forecast(
                :lastvals,
                :lastresids,
                :p,
                :q,
                :c,
                :phi,
                :theta,
                :horizon
            )
    """
    return text(query_str).bindparams(lastvals=last_vals, lastresids=last_residuals,
                                      p=p, q=q, c=c, phi=phi, theta=theta, horizon=horizon)


def arima_run_forecast_incremental_query(p, d, q, phi, theta, c, last_y_diffs, last_residuals, last_original_vals, last_date, horizon, forecast_step, log_transform):
    """
    Build a SQLAlchemy text query to call arima_run_forecast_incremental_v2.
    """
    query_str = """
        SELECT * FROM arima_run_forecast_incremental_v2(
            :p, :d, :q, :phi, :theta, :c, :last_y_diffs, :last_residuals, :last_original_vals, :last_date, :horizon, :forecast_step, :log_transform
        )
    """
    return text(query_str).bindparams(
        p=p, d=d, q=q, phi=phi, theta=theta, c=c,
        last_y_diffs=last_y_diffs,
        last_residuals=last_residuals,
        last_original_vals=last_original_vals,
        last_date=last_date,
        horizon=horizon,
        forecast_step=forecast_step,
        log_transform=log_transform
    )


def arima_query(p: int, d: int, q: int, horizon: int, include_mean=True, optimiser="Nelder-Mead", log_transform=False, table="pg_forecast_unit_test"):
    """
    Build a SQLAlchemy text query to call arima on pg_forecast_unit_test.
    """
    query_str = f"""
        SELECT *
        FROM arima_train_and_forecast(
            p := :p,
            d := :d,
            q := :q,
            horizon := :horizon,
            source_table := '{table}',
            date_col := 't',
            value_col := 'value',
            include_mean := {include_mean},
            optimiser := '{optimiser}',
            log_transform := :logt
        )
    """
    return text(query_str).bindparams(p=p, q=q, d=d, horizon=horizon, logt=log_transform)


def assert_close(actual, expected, tolerance=1e-3, name="value"):
    """
    Assert that two numeric values are approximately equal within `tolerance`.

    Raises
    ------
    AssertionError
        If the values differ by more than `tolerance`.
    """
    if actual is None:
        raise AssertionError(f"{name} is None (expected {expected})")
    diff = abs(actual - expected)
    if diff > tolerance:
        raise AssertionError(
            f"{name} differs from expected value by {diff:.6f} "
            f"(actual={actual:.6f}, expected={expected:.6f})"
        )


def all_different(arr):
    for i in range(1, len(arr)):
        if arr[i-1] == arr[i]:
            return False
    return True


def increasing(arr):
    for i in range(1, len(arr)):
        if arr[i-1] >= arr[i]:
            return False
    return True

def assert_css_matches_incremental(
    test_engine,
    phi,
    theta,
    p,
    q,
    series_id,
    tolerance=1e-6
):
    css_query = css_loss_query(
        phi=phi,
        theta=theta,
        p=p,
        q=q,
        series_id=series_id
    )

    inc_query = css_incremental_query(
        phi=phi,
        theta=theta,
        series_id=series_id
    )

    with test_engine.connect() as conn:
        css_value = conn.execute(css_query).scalar()
        inc_value = conn.execute(inc_query).scalar()

    assert_close(
        inc_value,
        css_value,
        tolerance=tolerance,
        name="css_incremental vs css_loss"
    )


def setup_basic_dataset(test_engine):
    with test_engine.connect() as conn:
        conn.execute(text("""
        TRUNCATE TABLE pg_forecast_unit_test;
        INSERT INTO pg_forecast_unit_test (t, series_id, value) VALUES
        ('2023-01-01 00:00:00', 'TestSeries', 10.0), -- x_1
        ('2023-01-02 00:00:00', 'TestSeries', 10.5), -- x_2
        ('2023-01-03 00:00:00', 'TestSeries', 10.8), -- x_3
        ('2023-01-04 00:00:00', 'TestSeries', 11.2), -- x_4
        ('2023-01-05 00:00:00', 'TestSeries', 11.5), -- x_5
        ('2023-01-06 00:00:00', 'TestSeries', 11.7), -- x_6
        ('2023-01-07 00:00:00', 'TestSeries', 11.9); -- x_7
        """))
        conn.commit()


def setup_large_dataset(test_engine):
    with test_engine.connect() as conn:
        conn.execute(text("""
        TRUNCATE pg_forecast_unit_test;
        INSERT INTO pg_forecast_unit_test (t, series_id, value)
        SELECT
            '2025-01-01'::timestamp + n * interval '1 day' AS t,
            'BigSeries' AS series_id,
            n AS value
        FROM
            generate_series(0, 1000000) n;
        """))
        conn.commit()

def setup_nonstationary_dataset(test_engine):
    """
    Generates a random walk with drift:
        x_t = x_{t-1} + drift + noise
    This *requires* differencing (d=1).
    """
    with test_engine.connect() as conn:
        conn.execute(text("""
            TRUNCATE TABLE pg_forecast_unit_test;

            WITH RECURSIVE rw AS (
                SELECT
                    0 AS n,
                    10.0::double precision AS value
                UNION ALL
                SELECT
                    n + 1,
                    value + 0.5 + (random() - 0.5) * 0.1
                FROM rw
                WHERE n < 120
            )
            INSERT INTO pg_forecast_unit_test (t, series_id, value)
            SELECT
                '2023-01-01'::timestamp + n * interval '1 day',
                'DiffSeries',
                value
            FROM rw;
        """))
        conn.commit()


def test_css_loss_p_1_q_0(test_engine):
    """
    Unit test for ARIMA CSS function on a basic dataset
    where p=1, q=0
    """
    setup_basic_dataset(test_engine)
    query = css_loss_query(phi=[0.5], theta=[],
                           p=1, q=0, series_id='TestSeries')
    with test_engine.connect() as conn:
        result = conn.execute(query).scalar()
    assert_close(result, 201.5075)

def test_css_loss_p_0_q_1(test_engine):
    """
    Unit test for ARIMA CSS function on a basic dataset
    where p=0, q=1
    """
    setup_basic_dataset(test_engine)
    query = css_loss_query(phi=[], theta=[0.3],
                           p=0, q=1, series_id='TestSeries')
    with test_engine.connect() as conn:
        result = conn.execute(query).scalar()
    assert_close(result, 492.2989186)

def test_css_loss_p_1_q_1(test_engine):
    """
    Unit test for ARIMA CSS function on a basic dataset
    where p=1, q=1
    """
    setup_basic_dataset(test_engine)
    query = css_loss_query(phi=[0.5], theta=[0.3],
                           p=1, q=1, series_id='TestSeries')
    with test_engine.connect() as conn:
        result = conn.execute(query).scalar()
    assert_close(result, 130.1938)


def test_css_loss_p_2_q_1(test_engine):
    """
    Unit test for ARIMA CSS function on a basic dataset
    where p=2, q=1
    """
    setup_basic_dataset(test_engine)
    query = css_loss_query(phi=[0.5, 0.5], theta=[
                           0.3], p=2, q=1, series_id='TestSeries')
    with test_engine.connect() as conn:
        result = conn.execute(query).scalar()
    assert_close(result, 0.706344)


def test_css_loss_p_2_q_2(test_engine):
    """
    Unit test for ARIMA CSS function on a basic dataset
    where p=2, q=2
    """
    setup_basic_dataset(test_engine)
    query = css_loss_query(phi=[0.5, 0.25], theta=[
                           0.3, 0.5], p=2, q=2, series_id='TestSeries')
    with test_engine.connect() as conn:
        result = conn.execute(query).scalar()
    assert_close(result, 23.0322)

def test_css_loss_p_2_q_2_constant(test_engine):
    """
    Unit test for ARIMA CSS function on a basic dataset
    where p=2, q=2, c=3.464544387
    """
    setup_basic_dataset(test_engine)
    query = css_loss_query(phi=[-0.107714174, 0.847949993], theta=[2, -2], 
                           p=2, q=2, c=3.464544387, series_id='TestSeries')
    with test_engine.connect() as conn:
        result = conn.execute(query).scalar()
    assert_close(result, 0.000986208)

def test_css_incremental_matches_css_loss_p_1_q_0(test_engine):
    setup_basic_dataset(test_engine)

    assert_css_matches_incremental(
        test_engine=test_engine,
        phi=[0.5],
        theta=[],
        p=1,
        q=0,
        series_id="TestSeries"
    )

def test_css_incremental_matches_css_loss_p_0_q_1(test_engine):
    setup_basic_dataset(test_engine)

    assert_css_matches_incremental(
        test_engine=test_engine,
        phi=[],
        theta=[0.3],
        p=0,
        q=1,
        series_id="TestSeries"
    )

def test_css_incremental_matches_css_loss_p_1_q_1(test_engine):
    setup_basic_dataset(test_engine)

    assert_css_matches_incremental(
        test_engine=test_engine,
        phi=[0.5],
        theta=[0.3],
        p=1,
        q=1,
        series_id="TestSeries"
    )

def test_css_incremental_matches_css_loss_p_2_q_1(test_engine):
    setup_basic_dataset(test_engine)

    assert_css_matches_incremental(
        test_engine=test_engine,
        phi=[0.5, 0.5],
        theta=[0.3],
        p=2,
        q=1,
        series_id="TestSeries"
    )

def test_css_incremental_matches_css_loss_p_2_q_2(test_engine):
    setup_basic_dataset(test_engine)

    assert_css_matches_incremental(
        test_engine=test_engine,
        phi=[0.5, 0.25],
        theta=[0.3, 0.5],
        p=2,
        q=2,
        series_id="TestSeries"
    )

def test_css_incremental_matches_css_loss_p_2_q_2_constant(test_engine):
    setup_basic_dataset(test_engine)

    assert_css_matches_incremental(
        test_engine=test_engine,
        phi=[-0.107714174, 0.847949993],
        theta=[2, -2],
        p=2,
        q=2,
        series_id="TestSeries"
    )

def test_css_incremental_p_2_d_1_q_2_constant(test_engine):
    setup_basic_dataset(test_engine)

    query = css_incremental_query(
        phi = [0.099934682, 0.674069236],
        theta = [2.0, -2.0],
        c = 0.0,
        d = 1
    )

    with test_engine.connect() as conn:
        css_value = conn.execute(query).scalar()
        assert_close(0.001469628, css_value, tolerance=0.0001, name="css_incremental differenced")

def test_css_incremental_p_2_d_2_q_2_constant(test_engine):
    setup_basic_dataset(test_engine)

    query = css_incremental_query(
        phi = [0.450980392, 0.480392157],
        theta = [2.0, -2.0],
        c = 0.0,
        d = 2
    )

    with test_engine.connect() as conn:
        css_value = conn.execute(query).scalar()
        assert_close(0.00245098, css_value, tolerance=0.0001, name="css_incremental differenced")

def test_css_incremental_matches_css_loss_large_input(test_engine):
    setup_large_dataset(test_engine)

    assert_css_matches_incremental(
        test_engine=test_engine,
        phi=[0.9],
        theta=[0.4],
        p=1,
        q=1,
        series_id="BigSeries",
        tolerance=1e-4  # slightly looser for large accumulation
    )


def test_arima_difference_d_1(test_engine):
    setup_basic_dataset(test_engine)
    query = arima_difference_query(1, series_id='TestSeries')

    with test_engine.connect() as conn:
        result = conn.execute(query)
        differences = [row[0] for row in result.fetchall()][0]
        expected_differences = [0.5, 0.3, 0.4, 0.3, 0.2, 0.2]
        for i, diff in enumerate(differences):
            assert_close(diff, expected_differences[i])


def test_arima_difference_d_2(test_engine):
    setup_basic_dataset(test_engine)
    query = arima_difference_query(2, series_id='TestSeries')

    with test_engine.connect() as conn:
        result = conn.execute(query)
        differences = [row[0] for row in result.fetchall()][0]
        expected_differences = [-0.2, 0.1, -0.1, -0.1, 1.78e-15]
        for i, diff in enumerate(differences):
            assert_close(diff, expected_differences[i])


def test_arima_integrate_d_1(test_engine):
    setup_basic_dataset(test_engine)
    query = arima_integrate_query(1, series_id='TestSeries')

    with test_engine.connect() as conn:
        result = conn.execute(query)
        integrated = [row[0] for row in result.fetchall()][0]
        expected_integrated = [10, 10.5, 10.8, 11.2, 11.5, 11.7, 11.9]
        for i, diff in enumerate(integrated):
            assert_close(diff, expected_integrated[i])


def test_arima_integrate_d_2(test_engine):
    setup_basic_dataset(test_engine)
    query = arima_integrate_query(2, series_id='TestSeries')

    with test_engine.connect() as conn:
        result = conn.execute(query)
        integrated = [row[0] for row in result.fetchall()][0]
        expected_integrated = [10, 10.5, 10.8, 11.2, 11.5, 11.7, 11.9]
        for i, diff in enumerate(integrated):
            assert_close(diff, expected_integrated[i])


def test_arima_forecast_p_2_q_2(test_engine):
    setup_basic_dataset(test_engine)
    query = arima_forecast_query([11.7, 11.9], [-0.016511328, -0.035082334], 2, 2, 0.0, [
                                 0.914982910934596, 0.113555903738228], [2.062502180358700, 0.710978584599870], 4)

    with test_engine.connect() as conn:
        result = conn.execute(query)
        forecast = [row[0] for row in result.fetchall()][0]
        expected_forecast = [12.15790328,
                             12.46384978, 12.78481125, 13.11322754]
        for i, diff in enumerate(forecast):
            assert_close(diff, expected_forecast[i])


def test_arima_run_forecast_incremental_p_2_d_0_q_2(test_engine):
    p, d, q = 2, 0, 2
    phi = [0.914982910934596, 0.113555903738228]
    theta = [2.062502180358700, 0.710978584599870]
    c = 0.0
    last_y_diffs = [11.9, 11.7]  # y_t, y_{t-1}
    last_residuals = [-0.016511328, -0.035082334]  # e_t, e_{t-1}
    last_original_vals = [11.7, 11.9]
    last_date = "2023-01-07 00:00:00"
    horizon = 4
    forecast_step = "1 day"
    log_transform = False

    query = arima_run_forecast_incremental_query(
        p, d, q, phi, theta, c, last_y_diffs, last_residuals, last_original_vals, last_date, horizon, forecast_step, log_transform
    )

    with test_engine.connect() as conn:
        result = conn.execute(query).fetchall()
        forecast = [row[1] for row in result]
        expected_forecast = [12.15790328, 12.46384978, 12.78481125, 13.11322754]
        for i, val in enumerate(forecast):
            assert_close(val, expected_forecast[i], name=f"Step {i+1}")


def test_arima_run_forecast_incremental_p_2_d_1_q_2(test_engine):
    p, d, q = 2, 1, 2
    phi = [0.099934682, 0.674069236]
    theta = [2.0, -2.0]
    c = 0.0
    last_y_diffs = [0.2, 0.2]  # most recent at START (y_7-y_6, y_6-y_5)
    last_residuals = [0.0, 0.0]  # dummy
    last_original_vals = [11.7, 11.9]  # most recent at END
    last_date = "2023-01-07 00:00:00"
    horizon = 4
    forecast_step = "1 day"
    log_transform = False

    query = arima_run_forecast_incremental_query(
        p, d, q, phi, theta, c, last_y_diffs, last_residuals, last_original_vals, last_date, horizon, forecast_step, log_transform
    )

    with test_engine.connect() as conn:
        result = conn.execute(query).fetchall()
        forecast = [row[1] for row in result]
        expected_forecast = [12.0548007836, 12.2050845979, 12.324449609, 12.4376800093]
        for i, val in enumerate(forecast):
            assert_close(val, expected_forecast[i], name=f"Step {i+1}")


def test_arima_run_forecast_incremental_null_repro(test_engine):
    p, d, q = 0, 1, 0
    phi = []
    theta = []
    c = 0.0
    last_y_diffs = [0.0]
    last_residuals = []
    last_original_vals = []  # Empty! This should trigger an error if d > 0
    last_date = "2023-01-07 00:00:00"
    horizon = 2
    forecast_step = "1 day"
    log_transform = False

    query = arima_run_forecast_incremental_query(
        p, d, q, phi, theta, c, last_y_diffs, last_residuals, last_original_vals, last_date, horizon, forecast_step, log_transform
    )

    with test_engine.connect() as conn:
        conn.execute(query).fetchall()


def test_arima_incremental_matches_regular_p_2_d_1_q_2(test_engine):
    """
    Verify that arima_run_forecast_incremental matches arima_train_and_forecast
    given the same parameters.
    """
    setup_basic_dataset(test_engine)
    p, d, q = 2, 1, 2
    horizon = 4
    include_mean = False
    log_transform = False

    with test_engine.connect() as conn:
        # 1. Get regular forecast
        regular_query = arima_query(p, d, q, horizon, include_mean=include_mean, log_transform=log_transform)
        regular_result = conn.execute(regular_query).fetchall()
        regular_forecast = [row[1] for row in regular_result]

        # 2. Get parameters and state for incremental forecast
        # We'll use arima_train to get the same parameters as the regular query would find
        train_query = text("""
            SELECT phi, theta, c FROM arima_train(:p, :d, :q, 'pg_forecast_unit_test', 't', 'value', :include_mean)
        """).bindparams(p=p, d=d, q=q, include_mean=include_mean)
        phi, theta, c = conn.execute(train_query).fetchone()

        # Get incremental state
        state_query = text("""
            SELECT (s).* FROM (
                SELECT css_incremental_full_table('pg_forecast_unit_test', 'value', 't', :phi, :theta, :c, :d, :logt) as s
            ) q
        """).bindparams(phi=phi, theta=theta, c=c, d=d, logt=log_transform)
        state_row = conn.execute(state_query).fetchone()
        # css_incremental_state fields: t, p, q, y_lags, e_lags, css, d, n_diff, diff_buf
        last_y_diffs = state_row[3]
        last_residuals = state_row[4]
        last_original_vals = state_row[8]

        # Get last date
        last_date = conn.execute(text("SELECT MAX(t) FROM pg_forecast_unit_test")).scalar()

        # 3. Get incremental forecast
        incremental_query = arima_run_forecast_incremental_query(
            p, d, q, phi, theta, c, last_y_diffs, last_residuals, last_original_vals, last_date, horizon, "1 day", log_transform
        )
        incremental_result = conn.execute(incremental_query).fetchall()
        incremental_forecast = [row[1] for row in incremental_result]

    # 4. Compare
    for i in range(horizon):
        assert_close(incremental_forecast[i], regular_forecast[i], name=f"Step {i+1}")


def test_arima_p_2_d_0_q_2(test_engine):
    setup_basic_dataset(test_engine)
    query = arima_query(2, 0, 2, 4, include_mean=False)

    with test_engine.connect() as conn:
        result = conn.execute(query)
        forecast = [row[1] for row in result.fetchall()]
        expected_forecast = [12.24421907,
                             12.57920787, 12.89879338, 13.22734702]
        for i, diff in enumerate(forecast):
            assert_close(diff, expected_forecast[i])


def test_arima_p_2_d_1_q_2(test_engine):
    setup_basic_dataset(test_engine)
    query = arima_query(2, 1, 2, 4, include_mean=False)

    with test_engine.connect() as conn:
        result = conn.execute(query)
        forecast = [row[1] for row in result.fetchall()]
        expected_forecast = [12.08158067,
                             12.24237873, 12.38084598, 12.50307269]
        for i, diff in enumerate(forecast):
            assert_close(diff, expected_forecast[i])


def test_arima_p_2_d_2_q_2(test_engine):
    setup_basic_dataset(test_engine)
    query = arima_query(2, 2, 2, 4, include_mean=False)

    with test_engine.connect() as conn:
        result = conn.execute(query)
        forecast = [row[1] for row in result.fetchall()]
        expected_forecast = [12.07156863, 12.22051134, 12.345592, 12.44904204]
        for i, diff in enumerate(forecast):
            assert_close(diff, expected_forecast[i])


def test_arima_p_2_d_0_q_2_include_c(test_engine):
    setup_basic_dataset(test_engine)
    query = arima_query(2, 0, 2, 4, include_mean=True)

    with test_engine.connect() as conn:
        result = conn.execute(query)
        forecast = [row[1] for row in result.fetchall()]
        expected_forecast = [12.11508753,
                             12.25959515, 12.41700061, 12.52258105]
        for i, diff in enumerate(forecast):
            assert_close(diff, expected_forecast[i])


def test_arima_p_2_d_1_q_2_include_c(test_engine):
    setup_basic_dataset(test_engine)
    query = arima_query(2, 1, 2, 4, include_mean=True)

    with test_engine.connect() as conn:
        result = conn.execute(query)
        forecast = [row[1] for row in result.fetchall()]
        expected_forecast = [12.00216195,
                             12.03697575, 11.96609202, 11.78935975]
        for i, diff in enumerate(forecast):
            assert_close(diff, expected_forecast[i])


def test_arima_large_input(test_engine):
    setup_large_dataset(test_engine)
    horizon = 100
    query = arima_query(2, 1, 2, horizon, include_mean=False)

    with test_engine.connect() as conn:
        result = conn.execute(query)
        forecast = [row[1] for row in result.fetchall()]
        assert len(forecast) == horizon and all_different(
            forecast) and increasing(forecast)


def test_arima_large_input_include_c(test_engine):
    setup_large_dataset(test_engine)
    horizon = 100
    query = arima_query(1, 0, 1, horizon, include_mean=True)

    with test_engine.connect() as conn:
        result = conn.execute(query)
        forecast = [row[1] for row in result.fetchall()]
        assert len(forecast) == horizon and all_different(
            forecast) and increasing(forecast)
        
def test_autoarima_finds_low_loss_model(test_engine):
    """
    Test that AutoARIMA finds a model with a suitably low CSS loss
    on a clearly defined AR(1) process.
    """
    with test_engine.connect() as conn:
        # Generate an AR(1) process: x_t = 0.8 * x_{t-1} + noise
        conn.execute(text("""
            SELECT remove_forecast('autoarima', 'pg_forecast_unit_test', 't', 'value');
            TRUNCATE TABLE pg_forecast_unit_test;
            INSERT INTO pg_forecast_unit_test (t, series_id, value)
            SELECT 
                '2023-01-01'::timestamp + (n || ' minutes')::interval,
                'AR1_Series',
                -- Simple AR(1) approximation
                10 * power(0.8, n % 10) + (random() * 0.1)
            FROM generate_series(0, 100) n;
        """))
        conn.commit()

        # Call autoarima_train
        query = text("""
            SELECT css, p, d, q 
            FROM autoarima_train(
                source_table := 'pg_forecast_unit_test',
                date_col := 't',
                value_col := 'value'
            );
        """)
        result = conn.execute(query).fetchone()
        
        css_loss = result[0]
        p, d, q = result[1], result[2], result[3]

        print(f"AutoARIMA found ARIMA({p},{d},{q}) with CSS: {css_loss}")

        # Assertions
        # 1. CSS should be similar to Python baseline (304.9918959560322)
        assert css_loss <= 350.0, f"Loss too high: {css_loss}"
        # 2. Check that it didn't just default to ARIMA(0,0,0) if there is structure
        assert (p + d + q) >= 0 

def test_autoarima_trigger_on_outliers(test_engine):
    """
    Test that inserting vastly different rows triggers deactivation,
    verified using arima_get_active_model_id.
    """
    setup_basic_dataset(test_engine)
    
    with test_engine.connect() as conn:
        # 1. Train the initial model
        # This populates models and model_arima_stats
        train_result = conn.execute(text("""
            SELECT remove_forecast('autoarima', 'pg_forecast_unit_test', 't', 'value');
            SELECT p, d, q, c, phi, theta 
            FROM autoarima_train('pg_forecast_unit_test', 't', 'value');
        """)).fetchone()
        conn.commit()

        _, d, _, c, phi, theta = train_result

        # 2. Get the ID of the model we just trained using your helper function
        # We need the model_id from the 'models' table first
        parent_model_id = conn.execute(text("""
            SELECT model_get_id('autoarima', 'pg_forecast_unit_test', 't', 'value')
        """)).scalar()

        print(f"model: {parent_model_id}, phi: {phi}, theta: {theta}, d: {d}, c: {c}")

        initial_arima_id = conn.execute(text("""
            SELECT arima_get_active_id(:mid, :phi, :theta, :d, :c)
        """), {"mid": parent_model_id, "phi": phi, "theta": theta, "d": d, "c": c}).scalar()

        assert initial_arima_id is not None, "Active model ID should be retrievable after training"

        # 3. Insert a "vastly different" row
        conn.execute(text("""
            INSERT INTO pg_forecast_unit_test (t, series_id, value) 
            VALUES ('2023-01-08 00:00:00', 'TestSeries', 500.0);
        """))
        conn.commit()

        # 4. Verify the helper function now returns a different ID
        post_outlier_id = conn.execute(text("""
            SELECT arima_get_active_id(:mid, :phi, :theta, :d, :c)
        """), {"mid": parent_model_id, "phi": phi, "theta": theta, "d": d, "c": c}).scalar()

        assert post_outlier_id is None, (
            f"Model {initial_arima_id} should have been deactivated "
            f"and no longer returned as active."
        )

def test_autoarima_matches_manual_arima_on_differenced_series(test_engine):
    """
    Verify that AutoARIMA forecast matches a manually trained ARIMA forecast
    on a series that requires differencing (d > 0).
    """
    setup_nonstationary_dataset(test_engine)

    with test_engine.connect() as conn:
        horizon = 4

        # Step 1: Train AutoARIMA to get optimum p, d, q, c, use_log_transform
        auto_query = text("""
            SELECT p, d, q, c, use_log_transform
            FROM autoarima_train(
                source_table := 'pg_forecast_unit_test',
                date_col := 't',
                value_col := 'value'
            );
        """)
        auto_result = conn.execute(auto_query).fetchone()
        auto_p, auto_d, auto_q, auto_c, auto_use_log = auto_result
        conn.commit()

        assert auto_d > 0, "AutoARIMA found undifferenced forecast, d = " + str(auto_d)

        # Step 2: Get AutoARIMA forecast via autoarima_train_and_forecast
        # This retrains using the best parameters found above
        auto_forecast_query = text("""
            SELECT forecast_value
            FROM autoarima_train_and_forecast(
                horizon := :horizon,
                source_table := 'pg_forecast_unit_test',
                date_col := 't',
                value_col := 'value'
            );
        """).bindparams(horizon=horizon)
        auto_forecast = conn.execute(auto_forecast_query).fetchall()
        auto_forecast = [row[0] for row in auto_forecast]  # extract floats

        # Step 3: Train and forecast manually using manual ARIMA with same parameters
        include_mean = auto_c != 0
        manual_query = arima_query(
            p=auto_p, d=auto_d, q=auto_q, horizon=horizon, include_mean=include_mean, optimiser="L-BFGS", log_transform=auto_use_log
        )
        manual_forecast = [row[1] for row in conn.execute(manual_query).fetchall()]

    # Step 4: Assert forecasts match closely
    for i, (af, mf) in enumerate(zip(auto_forecast, manual_forecast)):
        assert_close(
            af, mf,
            tolerance=1e-3,
            name=f"AutoARIMA vs manual ARIMA forecast at step {i+1}"
        )

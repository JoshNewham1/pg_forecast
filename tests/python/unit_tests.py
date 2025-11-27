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


def css_loss_query(phi: list[float], theta: list[float], p: int, q: int, series_id=None):
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
            q := {q}
        )
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


def arima_forecast_query(last_vals: list[float], last_residuals: list[float], p: int, q: int, phi: list[float], theta: list[float], horizon: int):
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
                :phi,
                :theta,
                :horizon
            )
    """
    return text(query_str).bindparams(lastvals=last_vals, lastresids=last_residuals,
                                      p=p, q=q, phi=phi, theta=theta, horizon=horizon)


def arima_query(p: int, d: int, q: int, horizon: int, table="pg_forecast_unit_test"):
    """
    Build a SQLAlchemy text query to call arima on pg_forecast_unit_test.
    """
    query_str = f"""
        SELECT *
        FROM arima(
            p := :p,
            d := :d,
            q := :q,
            horizon := :horizon,
            source_table := '{table}',
            date_col := 't',
            value_col := 'value'
        )
    """
    return text(query_str).bindparams(p=p, q=q, d=d, horizon=horizon)


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


def setup_basic_dataset(test_engine):
    with test_engine.connect() as conn:
        conn.execute(text("""
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
    query = arima_forecast_query([11.7, 11.9], [-0.016511328, -0.035082334], 2, 2, [
                                 0.914982910934596, 0.113555903738228], [2.062502180358700, 0.710978584599870], 4)

    with test_engine.connect() as conn:
        result = conn.execute(query)
        forecast = [row[0] for row in result.fetchall()][0]
        expected_forecast = [12.15790328,
                             12.46384978, 12.78481125, 13.11322754]
        for i, diff in enumerate(forecast):
            assert_close(diff, expected_forecast[i])


def test_arima_p_2_d_0_q_2(test_engine):
    setup_basic_dataset(test_engine)
    query = arima_query(2, 0, 2, 4)

    with test_engine.connect() as conn:
        result = conn.execute(query)
        forecast = [row[1] for row in result.fetchall()]
        expected_forecast = [12.24421907,
                             12.57920787, 12.89879338, 13.22734702]
        for i, diff in enumerate(forecast):
            assert_close(diff, expected_forecast[i])


def test_arima_p_2_d_1_q_2(test_engine):
    setup_basic_dataset(test_engine)
    query = arima_query(2, 1, 2, 4)

    with test_engine.connect() as conn:
        result = conn.execute(query)
        forecast = [row[1] for row in result.fetchall()]
        expected_forecast = [12.08158067,
                             12.24237873, 12.38084598, 12.50307269]
        for i, diff in enumerate(forecast):
            assert_close(diff, expected_forecast[i])


def test_arima_p_2_d_2_q_2(test_engine):
    setup_basic_dataset(test_engine)
    query = arima_query(2, 2, 2, 4)

    with test_engine.connect() as conn:
        result = conn.execute(query)
        forecast = [row[1] for row in result.fetchall()]
        expected_forecast = [12.07156863, 12.22051134, 12.345592, 12.44904204]
        for i, diff in enumerate(forecast):
            assert_close(diff, expected_forecast[i])


def test_arima_large_input(test_engine):
    setup_large_dataset(test_engine)
    horizon = 100
    query = arima_query(2, 1, 2, horizon)

    with test_engine.connect() as conn:
        result = conn.execute(query)
        forecast = [row[1] for row in result.fetchall()]
        assert len(forecast) == horizon and all_different(
            forecast) and increasing(forecast)

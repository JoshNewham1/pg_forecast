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


def test_arima_css_p_1_q_1(test_engine):
    """
    Unit test for ARIMA CSS function on a basic dataset
    where p=1, q=1
    """
    setup_basic_dataset(test_engine)
    result = run_function(test_engine, "arima_css", "pg_forecast_unit_test WHERE series_id = 'TestSeries'",
                          't', 'value', 1, 1, [0.5], [0.3])
    assert_close(result, 130.1938)


def test_arima_css_p_2_q_1(test_engine):
    """
    Unit test for ARIMA CSS function on a basic dataset
    where p=2, q=1
    """
    setup_basic_dataset(test_engine)
    result = run_function(test_engine, "arima_css", "pg_forecast_unit_test WHERE series_id = 'TestSeries'",
                          't', 'value', 2, 1, [0.5, 0.5], [0.3])
    assert_close(result, 0.706344)


def test_arima_css_p_2_q_2(test_engine):
    """
    Unit test for ARIMA CSS function on a basic dataset
    where p=2, q=2
    """
    setup_basic_dataset(test_engine)
    result = run_function(test_engine, "arima_css", "pg_forecast_unit_test",
                          't', 'value', 2, 2, [0.5, 0.25], [0.3, 0.5])
    assert_close(result, 23.0322)

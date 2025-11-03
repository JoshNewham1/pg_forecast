from monash.utils import stream_tsf_series, stream_tsf_values
import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from utils import MODEL_FACTORIES
import os
import time
from datetime import timedelta

BASE_TABLE = "pg_forecast_tfb_performance"


@pytest.fixture(scope="session")
def test_engine():
    """
    Create an SQLAlchemy engine for the test database using TEST_ env vars.
    """
    load_dotenv()
    TEST_DB_USERNAME = os.getenv("TEST_DB_USERNAME")
    TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD")
    TEST_DB_HOST = os.getenv("TEST_DB_HOST", "localhost")
    TEST_DB_PORT = os.getenv("TEST_DB_PORT", "5432")
    TEST_DB_NAME = os.getenv("TEST_DB_NAME")

    if not all([TEST_DB_USERNAME, TEST_DB_PASSWORD, TEST_DB_NAME]):
        raise RuntimeError(
            "TEST_DB_USERNAME, TEST_DB_PASSWORD, TEST_DB_NAME must be set")

    engine = create_engine(
        f"postgresql+psycopg2://{TEST_DB_USERNAME}:{TEST_DB_PASSWORD}@"
        f"{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"
    )
    yield engine


# Datasets can be downloaded from here
# https://huggingface.co/datasets/Monash-University/monash_tsf/tree/main/data
@pytest.fixture
def wind_farms_dataset():
    path = os.path.abspath(os.path.join(os.path.dirname(
        __file__), "../../eval/monash/data/wind_farms_minutely_dataset_with_missing_values.tsf"))
    return stream_tsf_values(path)


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_wind_farms_individual(model_factory, test_engine, wind_farms_dataset):
    timings = []
    inserts = 0
    MAXIMUM_INSERTS = 10_000
    with test_engine.connect() as conn:
        # TODO: Set up incremental forecasting model
        trans = conn.begin()
        conn.execute(
            text(f"""
            CREATE TABLE IF NOT EXISTS {BASE_TABLE} (
                date TIMESTAMP,
                value DOUBLE PRECISION
            );
            """)
        )
        conn.execute(text(f"TRUNCATE TABLE {BASE_TABLE};"))
        trans.commit()
        # Go through each time-series value in the first series and insert & commit
        for series_val in wind_farms_dataset:
            if inserts == MAXIMUM_INSERTS:
                break
            trans = conn.begin()
            timestamp = series_val['start_timestamp'] + \
                timedelta(seconds=series_val['time_index'])
            start = time.perf_counter()
            conn.execute(text(
                f"INSERT INTO {BASE_TABLE}(date, value) VALUES ('{timestamp}', {series_val['value']})"))
            trans.commit()
            end = time.perf_counter()
            timings.append(end - start)
            inserts += 1

    avg_insert = sum(timings) / len(timings)
    assert avg_insert < 1, f"Average insert time ({avg_insert} s) is over an acceptable threshold"


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_wind_farms_batch(model_factory, test_engine, wind_farms_dataset):
    timings = []
    inserts = 0
    PAGE_SIZE = 10_000
    MAXIMUM_INSERTS = 1_000_000
    with test_engine.connect() as conn:
        # TODO: Set up incremental forecasting model
        trans = conn.begin()
        conn.execute(
            text(f"""
            CREATE TABLE IF NOT EXISTS {BASE_TABLE} (
                date TIMESTAMP,
                value DOUBLE PRECISION
            );
            """)
        )
        conn.execute(text(f"TRUNCATE TABLE {BASE_TABLE};"))
        trans.commit()
        # Go through each time-series value in all series, bulk inserting every PAGE_SIZE records
        values = []
        for series_val in wind_farms_dataset:
            if inserts == MAXIMUM_INSERTS:
                break

            # Parse individual value
            timestamp = series_val['start_timestamp'] + \
                timedelta(seconds=series_val['time_index'])
            values.append(f"('{timestamp}', {series_val['value']})")
            inserts += 1

            # Every PAGE_SIZE values, bulk insert them and time it
            if inserts % PAGE_SIZE == 0:
                trans = conn.begin()
                start = time.perf_counter()
                conn.execute(text(
                    f"INSERT INTO {BASE_TABLE}(date, value) VALUES {','.join(values)}"))
                trans.commit()
                end = time.perf_counter()
                timings.append(end - start)
                values = []

    avg_insert = sum(timings) / len(timings) / PAGE_SIZE
    assert avg_insert < 0.01, f"Average insert time ({avg_insert} s) is over an acceptable threshold"

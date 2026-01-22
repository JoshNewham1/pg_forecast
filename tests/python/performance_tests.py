from monash.utils import stream_tsf_series, stream_tsf_values
import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os
import time
from datetime import timedelta
import logging

BASE_TABLE = "pg_forecast_tfb_performance"
MODELS_TO_TEST = ["autoarima"]


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


@pytest.fixture
def solar_dataset():
    path = os.path.abspath(os.path.join(os.path.dirname(
        __file__), "../../eval/monash/data/solar_10_minutes_dataset.tsf"))
    return stream_tsf_values(path)

# Utility functions


def setup_forecast_model(conn, model_name, table_name):
    conn.execute(text(
        f"SELECT create_forecast('{model_name}', '{table_name}', 'date', 'value', 10)"
    ))
    conn.commit()
    return model_name

def tear_down_forecast_model(conn, model_name, table_name):
    conn.execute(text(
        f"SELECT remove_forecast('{model_name}', '{table_name}', 'date', 'value')"
    ))
    conn.commit()
    return model_name


def single_inserts_test(test_engine, dataset, num_inserts, max_avg_insert, model_name):
    timings = []
    inserts = 0
    train_size = 100

    with test_engine.connect() as conn:
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

        # Insert training data
        for _ in range(train_size):
            try:
                series_val = next(dataset)
                timestamp = series_val['start_timestamp'] + \
                    timedelta(seconds=series_val['time_index'])
                trans = conn.begin()
                conn.execute(text(
                    f"INSERT INTO {BASE_TABLE}(date, value) VALUES ('{timestamp}', {series_val['value']})"))
                trans.commit()
            except StopIteration:
                break

        setup_forecast_model(conn, model_name, BASE_TABLE)

        try:
            # Go through each time-series value in the first series and insert & commit
            for series_val in dataset:
                if inserts == num_inserts:
                    break
                
                timestamp = series_val['start_timestamp'] + \
                    timedelta(seconds=series_val['time_index'])
                
                start = time.perf_counter()
                with conn.begin():
                    conn.execute(text(
                        f"INSERT INTO {BASE_TABLE}(date, value) VALUES ('{timestamp}', {series_val['value']})"))
                end = time.perf_counter()
                
                timings.append(end - start)
                inserts += 1
        finally:
            if conn.in_transaction():
                conn.rollback()
            tear_down_forecast_model(conn, model_name, BASE_TABLE)

    avg_insert = sum(timings) / len(timings)
    logging.info(f"Average insert time (s): {avg_insert}")
    assert avg_insert < max_avg_insert, f"Average insert time ({avg_insert} s) is over an acceptable threshold"


def bulk_inserts_test(test_engine, dataset, num_inserts, page_size, max_avg_insert, model_name):
    timings = []
    inserts = 0
    train_size = 100

    with test_engine.connect() as conn:
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

        # Insert training data
        values = []
        for _ in range(train_size):
            try:
                series_val = next(dataset)
                timestamp = series_val['start_timestamp'] + \
                    timedelta(seconds=series_val['time_index'])
                values.append(f"('{timestamp}', {series_val['value']})")
            except StopIteration:
                break
        
        if values:
            with conn.begin():
                conn.execute(text(
                    f"INSERT INTO {BASE_TABLE}(date, value) VALUES {','.join(values)}"))
            values = []

        setup_forecast_model(conn, model_name, BASE_TABLE)

        try:
            # Go through each time-series value in all series, bulk inserting every PAGE_SIZE records
            values = []
            for series_val in dataset:
                if inserts == num_inserts:
                    break

                # Parse individual value
                timestamp = series_val['start_timestamp'] + \
                    timedelta(seconds=series_val['time_index'])
                values.append(f"('{timestamp}', {series_val['value']})")
                inserts += 1

                # Every page_size values, bulk insert them and time it
                if inserts % page_size == 0:
                    start = time.perf_counter()
                    with conn.begin():
                        conn.execute(text(
                            f"INSERT INTO {BASE_TABLE}(date, value) VALUES {','.join(values)}"))
                    end = time.perf_counter()
                    timings.append(end - start)
                    values = []
                    logging.info(f"Page {len(timings)}/{num_inserts//page_size}: inserted {page_size} records in {end - start} seconds")
        finally:
            if conn.in_transaction():
                conn.rollback()
            tear_down_forecast_model(conn, model_name, BASE_TABLE)

    avg_insert = sum(timings) / len(timings) / page_size
    logging.info(f"Average insert time (s): {avg_insert}")
    assert avg_insert < max_avg_insert, f"Average insert time ({avg_insert} s) is over an acceptable threshold"


# Test in-DB models
@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_wind_farms_individual(model_name, test_engine, wind_farms_dataset):
    single_inserts_test(test_engine, wind_farms_dataset,
                        num_inserts=10_000, max_avg_insert=1, model_name=model_name)


@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_wind_farms_batch(model_name, test_engine, wind_farms_dataset):
    bulk_inserts_test(test_engine, wind_farms_dataset,
                      num_inserts=1_000_000, page_size=10_000, max_avg_insert=0.01, model_name=model_name)


@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_solar_individual(model_name, test_engine, solar_dataset):
    single_inserts_test(test_engine, solar_dataset,
                        num_inserts=10_000, max_avg_insert=1, model_name=model_name)


@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_solar_batch(model_name, test_engine, solar_dataset):
    bulk_inserts_test(test_engine, solar_dataset,
                      num_inserts=1_000_000, page_size=10_000, max_avg_insert=0.01, model_name=model_name)

# Test other solutions
import time
from typing import Dict, List, Any
import csv
from datetime import datetime, timedelta, timezone
import os
import math

from sqlalchemy import create_engine, text
from monash.utils import stream_tsf_values, stream_tsf_aligned_series
from itertools import islice
import itertools
import logging
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import pytest
import requests
import subprocess
from pg_joinboost_sut import JoinBoostSUT
from ts_joinboost_sut import TSJoinBoostSUT

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

class BenchmarkResult:
    def __init__(self, test_name: str, insert_timings: list[float], forecast_timings: list[float], rmse_losses: list[float], msmape_losses: list[float], write_to_file = True, csv_prefix = "performance_"):
        self.insert_timings = insert_timings
        self.forecast_timings = forecast_timings
        self.rmse_losses = rmse_losses
        self.msmape_losses = msmape_losses
        self.metrics = self.calculate_metrics(insert_timings, forecast_timings)
        self.test_name = test_name
        self.csv_prefix = csv_prefix
        if write_to_file:
            self._write_to_file()
        logging.info(f"Time-series benchmark {self.test_name}: {self.metrics}")
    
    def _write_to_file(self):
        if not isinstance(self.metrics, dict):
            logger.error("_write_to_file called before metrics were calculated")
            raise ValueError("_write_to_file called before instantiation")
        
        timestamp = datetime.now(timezone.utc)
        timestamp_text = timestamp.isoformat()
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        timestamp_unix = (timestamp - epoch).total_seconds()

        summary_fields = ["timestamp", "test_name", *self.metrics.keys()]
        summary_path = f"{self.csv_prefix}summary.csv"
        summary_exists = os.path.isfile(summary_path)
        with open(summary_path, mode="a", newline="") as f:
            logging.info("Writing benchmark summary results to %s", summary_path)

            writer = csv.DictWriter(f, fieldnames=summary_fields)
            if not summary_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "timestamp": timestamp_text,
                    "test_name": self.test_name,
                    **self.metrics,
                }
            )
        
        full_fields = ["type", "time"]
        full_path = f"{self.csv_prefix}{timestamp_unix}.csv"
        with open(full_path, mode="w", newline="") as f:
            logging.info("Writing full benchmark results to %s", full_path)

            writer = csv.DictWriter(f, fieldnames=full_fields)
            writer.writeheader()

            insert_rows = map(lambda t: { "type": "insert", "time": t }, self.insert_timings)
            forecast_rows = map(lambda t: { "type": "forecast", "time": t }, self.forecast_timings)
            rmse_rows = map(lambda l: { "type": "rmse", "time": l }, self.rmse_losses)
            msmape_rows = map(lambda l: { "type": "msmape", "time": l }, self.msmape_losses)
            writer.writerows(insert_rows)
            writer.writerows(forecast_rows)
            writer.writerows(rmse_rows)
            writer.writerows(msmape_rows)

    def calculate_metrics(self, insert_timings: list[float], forecast_timings: list[float]):
        avg_forecast = None
        avg_insert = None
        sum_rmse = None
        avg_msmape = None

        if len(insert_timings) > 0:
            avg_insert = sum(insert_timings) / len(insert_timings)
        if len(forecast_timings) > 0:
            avg_forecast = sum(forecast_timings) / len(forecast_timings)
        if len(self.rmse_losses) > 0:
            sum_rmse = sum(self.rmse_losses)
        if len(self.msmape_losses) > 0:
            avg_msmape = sum(self.msmape_losses) / len(self.msmape_losses)

        return {"avg_insert": avg_insert, "avg_forecast": avg_forecast, "sum_rmse": sum_rmse, "avg_msmape": avg_msmape}
    
class Dataset(ABC):
    def __init__(self, file_name: str, path: str = "../../eval/monash/data"):
        raise NotImplementedError("Please use an implementation of Dataset")
    
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def get_n(self, n_records: int) -> List[Dict[str, float]]:
        pass

    @abstractmethod
    def get_file_name(self) -> str:
        return self.file_name


class UnivariateDataset(Dataset):
    """
    Loads each time-series sequentially, assuming they're all independent
    """
    def __init__(self, file_name: str, path: str = "../../eval/monash/data"):
        self.file_name = file_name
        self.path = os.path.join(os.path.dirname(__file__), path, file_name)

        logger.info("Loading univariate dataset: %s", self.path)

        if not os.path.exists(self.path):
            raise RuntimeError(f"The path {self.path} does not exist for dataset {file_name}, could not instantiate object")
        
    def __iter__(self):
        logger.debug("Creating univariate dataset iterator for %s", self.file_name)
        return stream_tsf_values(self.path)
    
    def get_n(self, n_records: int) -> List[Dict[str, float]]:
        logger.debug("Fetching %d records from dataset", n_records)
        return list(islice(self, n_records))
    
    def get_file_name(self) -> str:
        return self.file_name

class MultivariateDataset(Dataset):
    """
    Loads all time series **with the same start date** together, each as a variable
    """
    def __init__(self, file_name: str, path: str = "../../eval/monash/data"):
        self.file_name = file_name
        self.path = os.path.join(os.path.dirname(__file__), path, file_name)

        logger.info("Loading multivariate dataset: %s", self.path)

        if not os.path.exists(self.path):
            raise RuntimeError(f"The path {self.path} does not exist for dataset {file_name}, could not instantiate object")
        
    def __iter__(self):
        logger.debug("Creating multivariate dataset iterator for %s", self.file_name)
        return stream_tsf_aligned_series(self.path)
    
    def get_n(self, n_records: int) -> List[Dict[str, float]]:
        logger.debug("Fetching %d records from dataset", n_records)
        return list(islice(self, n_records))
    
    def get_file_name(self) -> str:
        return self.file_name
    
class SystemUnderTest(ABC):
    def __init__():
        raise NotImplementedError("Please use an implementation of SystemUnderTest")

    @abstractmethod
    def setup_single(self, seed_records: List[Dict[str, float]]):
        pass
    
    @abstractmethod
    def setup_batch(self, seed_records: List[Dict[str, float]]):
        pass

    @abstractmethod
    def add_single(self, record: Dict[str, float]):
        pass

    @abstractmethod
    def add_batch(self, records: List[Dict[str, float]]):
        pass

    @abstractmethod
    def forecast(self, horizon: int) -> List[Dict[str, float]]:
        pass

    @abstractmethod
    def get_loss(self) -> float:
        pass

    @abstractmethod
    def teardown_single(self):
        pass

    @abstractmethod
    def teardown_batch(self):
        pass

class BenchmarkRunner:
    def __init__(self, sut: SystemUnderTest, dataset: Dataset, test_name: str, forecast_horizon: int = 365):
        self.sut = sut
        self.dataset = dataset
        self.test_name = test_name
        self.forecast_horizon = forecast_horizon
        self.insert_timings = []
        self.forecast_timings = []
        self.rmse_losses = []
        self.msmape_losses = []

    def get_forecast_values_from_response(self, f_entry, a_entry) -> List[tuple[float, float]]:
        """
        Extracts (actual, forecast) pairs from entries. 
        Supports both univariate and multivariate data.
        """

        if isinstance(f_entry, dict):
            # PythonWebServer or JoinBoostSUT
            fv = f_entry.get("forecast_value")
            if isinstance(fv, dict):
                # JoinBoostSUT: {"forecast_value": {"T1": 1.23}}
                if fv:
                    target_key = next(iter(fv.keys()))
                    f_val = float(fv[target_key])
                    # Try to match the same key in actuals
                    val = a_entry.get(target_key, a_entry.get(target_key.upper(), 0.0))
                    a_val = float(val) if val != "NULL" else 0.0
            else:
                # PythonWebServer: {"forecast_value": 1.23}
                f_val = float(fv) if fv is not None else 0.0
                if "value" in a_entry:
                    val = a_entry["value"]
                    a_val = float(val) if val != "NULL" else 0.0
                else:
                    # Guess first T* key for multivariate
                    t_keys = [k for k in a_entry.keys() if k.startswith("T")]
                    if t_keys:
                        val = a_entry[t_keys[0]]
                        a_val = float(val) if val != "NULL" else 0.0
                    else:
                        a_val = 0.0
        elif hasattr(f_entry, "__getitem__") and not isinstance(f_entry, (str, bytes)):
            # Sequence (tuple, list, Row, etc.)
            # PgForecast returns (date, value)
            if len(f_entry) > 1:
                f_val = float(f_entry[1])
            else:
                f_val = float(f_entry[0])
            a_val = float(a_entry.get("value", 0.0))
        else:
            try:
                f_val = float(f_entry)
            except (TypeError, ValueError):
                f_val = 0.0
            a_val = float(a_entry.get("value", 0.0))

        return a_val, f_val

    def calculate_rmse_loss(self, forecast: List[Any], actuals: List[Dict[str, float]]) -> float:
        """
        Calculates the Root Mean Squared Error (RMSE) between the forecast and actual values.
        Standardised loss function across all systems.
        """
        if not forecast or not actuals:
            return 0.0
        
        n = min(len(forecast), len(actuals))
        if n == 0:
            return 0.0
            
        css = 0.0
        for i in range(n):
            f_entry = forecast[i]
            a_entry = actuals[i]
            
            # Extract forecast value
            f_val, a_val = self.get_forecast_values_from_response(f_entry, a_entry)
                
            css += (f_val - a_val) ** 2
            
        return math.sqrt(float(css) / n)
    
    def calculate_msmape_loss(self, forecast: List[Any], actuals: List[Dict[str, float]], epsilon: float = 1e-12) -> float:
        """
        Calculates stabilized MSMAPE between forecast and actual values.

        MSMAPE = (100 / h) * sum(
            |F - Y| / max((|Y| + |F|)/2 + eps, 0.5 + eps)
        )

        Standardised loss function across all systems.
        """
        if not forecast or not actuals:
            return 0.0

        n = min(len(forecast), len(actuals))
        if n == 0:
            return 0.0

        total = 0.0

        for i in range(n):
            f_entry = forecast[i]
            a_entry = actuals[i]

            f_val, a_val = self.get_forecast_values_from_response(f_entry, a_entry)

            numerator = abs(f_val - a_val)
            denom = max((abs(a_val) + abs(f_val)) / 2.0 + epsilon,
                        0.5 + epsilon)

            total += numerator / denom

        return 100.0 * total / n
    
    def run_single(self, num_records: int) -> BenchmarkResult:
        logger.info(
            "Starting single insert benchmark (%d records)", num_records
        )

        self.insert_timings = []
        self.forecast_timings = []
        self.rmse_losses = []
        self.msmape_losses = []

        it_main, it_peek = itertools.tee(iter(self.dataset))
        
        # Seed records
        seed_records = list(islice(it_main, 100))
        self.sut.setup_single(seed_records)
        # Keep it_peek in sync
        for _ in range(100): next(it_peek)

        num_inserted = 0
        for record in it_main:
            next(it_peek) # keep in sync
            
            if num_inserted == num_records:
                break
            
            if num_inserted % 100 == 0 and num_inserted > 0:
                start = time.perf_counter()
                forecast_result = self.sut.forecast(self.forecast_horizon)
                end = time.perf_counter()
                self.forecast_timings.append(end - start)
                
                # Calculate external loss
                peek_ahead_it, _ = itertools.tee(it_peek)
                actuals = list(islice(peek_ahead_it, self.forecast_horizon))
                rmse = self.calculate_rmse_loss(forecast_result, actuals)
                msmape = self.calculate_msmape_loss(forecast_result, actuals)
                self.rmse_losses.append(rmse)
                self.msmape_losses.append(msmape)
                
                logging.info(f"Inserted {num_inserted} records, forecasted in {end - start}s, rmse={rmse}, msmape={msmape}")

            start = time.perf_counter()
            self.sut.add_single(record)
            end = time.perf_counter()
            self.insert_timings.append(end - start)
            
            num_inserted += 1
            logging.debug(f"Inserted record #{num_inserted}")

        logger.info("Single insert benchmark complete")
        self.sut.teardown_single()
        result = BenchmarkResult(self.test_name,
                                 self.insert_timings,
                                 self.forecast_timings,
                                 self.rmse_losses,
                                 self.msmape_losses,
                                 write_to_file=True)
        return result
    
    def run_batch(self, page_size: int, num_records: int) -> BenchmarkResult:
        logger.info(
            "Starting batch insert benchmark (%d records, page_size=%d)", num_records, page_size
        )
        self.insert_timings = []
        self.forecast_timings = []
        self.rmse_losses = []
        self.msmape_losses = []

        it_main, it_peek = itertools.tee(iter(self.dataset))
        
        # Seed records
        seed_records = list(islice(it_main, 100))
        self.sut.setup_batch(seed_records)
        # Keep it_peek in sync
        for _ in range(100): next(it_peek)

        num_pages = num_records // page_size
        for page in range(0, num_pages):
            records = list(islice(it_main, page_size))
            if not records:
                break
            # Keep it_peek in sync
            for _ in range(len(records)): next(it_peek)

            start = time.perf_counter()
            self.sut.add_batch(records)
            end = time.perf_counter()
            self.insert_timings.append(end - start)

            start = time.perf_counter()
            forecast_result = self.sut.forecast(self.forecast_horizon)
            end = time.perf_counter()
            self.forecast_timings.append(end - start)

            # Calculate external loss
            peek_ahead_it, _ = itertools.tee(it_peek)
            actuals = list(islice(peek_ahead_it, self.forecast_horizon))
            rmse = self.calculate_rmse_loss(forecast_result, actuals)
            msmape = self.calculate_msmape_loss(forecast_result, actuals)
            self.rmse_losses.append(rmse)
            self.msmape_losses.append(msmape)

            logger.debug(
                "Page %d/%d processed (insert=%.6fs, forecast=%.6fs, rmse=%f, msmape=%f)",
                page + 1,
                num_pages,
                self.insert_timings[-1],
                self.forecast_timings[-1],
                rmse,
                msmape
            )

        logger.info("Batch insert benchmark complete")
        self.sut.teardown_batch()
        result = BenchmarkResult(self.test_name,
                                 self.insert_timings,
                                 self.forecast_timings,
                                 self.rmse_losses,
                                 self.msmape_losses,
                                 write_to_file=True)
        return result
    
class PgForecast(SystemUnderTest):
    def __init__(self, model_name: str = "autoarima", with_timescale: bool = False):
        self.model_name = model_name
        self.with_timescale = with_timescale
        load_dotenv()

        TEST_DB_USERNAME = os.getenv("TEST_DB_USERNAME")
        TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD")
        TEST_DB_HOST = os.getenv("TEST_DB_HOST", "localhost")
        TEST_DB_PORT = os.getenv("TEST_DB_PORT", "5432")
        TEST_DB_NAME = os.getenv("TEST_DB_NAME")
        self.base_table = os.getenv("TEST_BASE_TABLE", "time_series_performance_test")

        if not all([TEST_DB_USERNAME, TEST_DB_PASSWORD, TEST_DB_NAME]):
            raise RuntimeError(
                "TEST_DB_USERNAME, TEST_DB_PASSWORD, TEST_DB_NAME must be set")

        self.engine = create_engine(
            f"postgresql+psycopg2://{TEST_DB_USERNAME}:{TEST_DB_PASSWORD}@"
            f"{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"
        )
        self.conn = self.engine.connect()

    def __del__(self):
        self.conn.close()

    def setup_single(self, seed_records):
        trans = self.conn.begin()
        self.conn.execute(
            text(f"""
            SELECT remove_forecast('{self.model_name}', '{self.base_table}', 'date', 'value');
            CREATE TABLE IF NOT EXISTS {self.base_table} (
                date TIMESTAMP,
                value DOUBLE PRECISION
            ) {"WITH (tsdb.hypertable)" if self.with_timescale else ""};
            TRUNCATE TABLE {self.base_table};
            """)
        )
        trans.commit()
        self.add_batch(seed_records)
        trans = self.conn.begin()
        self.conn.execute(text(f"""SELECT create_forecast('{self.model_name}', 
                                                          '{self.base_table}', 
                                                          'date', 
                                                          'value', 
                                                          '{'{"is_incremental": "true"}' if self.model_name == "pyautoarima" else '{}'}'
                                                        );"""))
        trans.commit()
        logging.debug(f"Setup completed with {len(seed_records)} seed records")
    
    def setup_batch(self, seed_records):
        return self.setup_single(seed_records)
    
    def add_single(self, record):
        index = record.get('global_index', record['time_index'])
        timestamp = record['start_timestamp'] + timedelta(seconds=index)
        self.conn.execute(text(f"INSERT INTO {self.base_table}(date, value) VALUES ('{timestamp}', {record['value']})"))
        self.conn.commit()

    def add_batch(self, records):
        values = []
        for record in records:
            index = record.get('global_index', record['time_index'])
            timestamp = record['start_timestamp'] + timedelta(seconds=index)
            values.append(f"('{timestamp}', {record['value']})")
        
        self.conn.execute(text(f"INSERT INTO {self.base_table}(date, value) VALUES {','.join(values)};"))
        self.conn.commit()

    def forecast(self, horizon):
        sql = f"""
        SELECT forecast_date, forecast_value
        FROM run_forecast('{self.model_name}', '{self.base_table}', 'date', 'value', {horizon});
        """
        result = self.conn.execute(text(sql)).fetchall()
        return result
        
    def get_loss(self):
        css = self.conn.execute(text(f"SELECT model_get_loss('{self.model_name}', '{self.base_table}', 'date', 'value');")).fetchone()[0]
        if css is None or css == float("NaN"):
            logging.warning(f"Invalid CSS returned on run {len(self.insert_timings)}: {css}")
        return css
    
    def teardown_single(self):
        self.conn.execute(text(f"SELECT remove_forecast('{self.model_name}', '{self.base_table}', 'date', 'value');"))
        self.conn.execute(text(f"TRUNCATE TABLE {self.base_table};"))
        self.conn.commit()

    def teardown_batch(self):
        return self.teardown_single()
    
class PythonWebServer(SystemUnderTest):
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def _post(self, endpoint: str, data: Any = None):
        response = requests.post(f"{self.base_url}/{endpoint}", json=data)
        response.raise_for_status()
        return response.json()

    def set_mode(self, mode: str):
        self._post("config", {"mode": mode})

    def set_n_features(self, n_lags: int, n_features: int):
        self._post("config", {"mode": "naive", "n_lags": n_lags, "n_features": n_features})

    def setup_single(self, seed_records: List[Dict[str, float]]):
        records = [self._transform_record(r) for r in seed_records]
        self._post("setup_single", records)
    
    def setup_batch(self, seed_records: List[Dict[str, float]]):
        records = [self._transform_record(r) for r in seed_records]
        self._post("setup_batch", records)

    def add_single(self, record: Dict[str, float]):
        self._post("add_single", self._transform_record(record))

    def add_batch(self, records: List[Dict[str, float]]):
        batch = {"records": [self._transform_record(r) for r in records]}
        self._post("add_batch", batch)

    def forecast(self, horizon: int) -> List[Dict[str, float]]:
        return self._post("forecast", {"horizon": horizon})
    
    def get_loss(self) -> float:
        response = requests.get(f"{self.base_url}/loss")
        response.raise_for_status()
        return float(response.json())
    
    def teardown_single(self):
        self._post("teardown")

    def teardown_batch(self):
        self._post("teardown")

    def _transform_record(self, record: Dict[str, float]) -> Dict[str, Any]:
        if not "T1" in record: # univariate
            r = record.copy()
            if isinstance(r.get('start_timestamp'), datetime):
                r['start_timestamp'] = r['start_timestamp'].isoformat()
        else: # multivariate
            r = {"index": record["index"], "timestamp": str(record["timestamp"]), "values": {}}

            for k, v in record.items():
                if k.startswith("T"):
                    r["values"][k] = v
        return r
    
@pytest.fixture
def competitor_server(request):
    server_name = request.param

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    venv_python = os.path.join(root, ".venv/bin/python")
    server_script = os.path.join(root, f"eval/competitors/{server_name}/main.py")
    
    logger.info(f"Starting competitor server from {server_script}")
    
    proc = subprocess.Popen(
        [venv_python, server_script],
        cwd=root,
        env={**os.environ, "PYTHONPATH": root}
    )
    
    max_retries = 10
    started = False
    for i in range(max_retries):
        try:
            resp = requests.get("http://localhost:8000/health", timeout=1)
            if resp.status_code == 200:
                started = True
                break
        except Exception as e:
            logging.debug(e)
            pass
        time.sleep(1)
        
    if not started:
        proc.terminate()
        raise RuntimeError("Competitor server failed to start")
        
    yield
    
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

@pytest.fixture(scope="session")
def univar_dataset():
    return UnivariateDataset("solar_10_minutes_dataset.tsf")

@pytest.fixture(scope="session")
def multivar_dataset():
    return MultivariateDataset("wind_farms_minutely_dataset_with_missing_values.tsf")

@pytest.fixture
def pgforecast_sut():
    sut = PgForecast()
    yield sut
    try:
        sut.teardown_single()
    except Exception:
        pass

@pytest.fixture
def pgforecast_timescale_sut():
    sut = PgForecast("autoarima", with_timescale=True)
    yield sut
    try:
        sut.teardown_single()
    except Exception:
        pass

@pytest.fixture
def python_sut(competitor_server):
    sut = PythonWebServer()
    
    sut.set_mode("naive")

    yield sut
    try:
        sut.teardown_single()
    except Exception:
        pass

@pytest.fixture
def python_geometric_sut(competitor_server):
    sut = PythonWebServer()
    sut.set_mode("geometric")
    yield sut
    try:
        sut.teardown_single()
    except Exception:
        pass

@pytest.fixture
def timescale_python_geometric_sut():
    sut = PgForecast(model_name="pyautoarima", with_timescale=True)
    yield sut
    try:
        sut.teardown_single()
    except Exception:
        pass

@pytest.fixture
def python_xgboost_sut(competitor_server, n_lags, n_features, single_target):
    sut = PythonWebServer()
    
    sut._post("config", {"mode": "naive", "n_lags": n_lags, "n_features": n_features, "single_target": single_target, "incremental": False})

    yield sut
    try:
        sut.teardown_single()
    except Exception:
        pass

@pytest.fixture
def python_xgboost_incremental_sut(competitor_server, n_lags, n_features, single_target):
    sut = PythonWebServer()
    
    sut._post("config", {"mode": "naive", "n_lags": n_lags, "n_features": n_features, "single_target": single_target, "incremental": True})

    yield sut
    try:
        sut.teardown_single()
    except Exception:
        pass

@pytest.fixture
def pg_joinboost_sut_incremental(n_lags, n_features, single_target):
    sut = JoinBoostSUT(lags=n_lags, n_features=n_features, predict_single_target=single_target, is_incremental=True)
    yield sut
    try:
        sut.teardown_single()
    except Exception:
        pass

@pytest.fixture
def pg_joinboost_sut_non_incremental(n_lags, n_features, single_target):
    sut = JoinBoostSUT(lags=n_lags, n_features=n_features, predict_single_target=single_target, is_incremental=False)
    yield sut
    try:
        sut.teardown_single()
    except Exception:
        pass

@pytest.fixture
def ts_joinboost_sut(n_lags, n_features, single_target):
    sut = TSJoinBoostSUT(lags=n_lags, n_features=n_features, predict_single_target=single_target)
    yield sut
    try:
        sut.teardown_single()
    except Exception:
        pass

@pytest.fixture
def duckdb_joinboost_sut(competitor_server, n_lags, n_features, single_target):
    sut = PythonWebServer()
    sut._post("config", {"mode": "naive", "n_lags": n_lags, "n_features": n_features, "single_target": single_target})
    yield sut
    try:
        sut.teardown_single()
    except Exception:
        pass

@pytest.fixture
def sut(request):
    """
    Returns the SUT based on the parametrization.
    """
    return request.getfixturevalue(request.param)

@pytest.fixture
def univar_runner(sut, univar_dataset, request):
    return BenchmarkRunner(sut, univar_dataset, request.node.name)

@pytest.fixture
def multivar_runner(sut, multivar_dataset, request):
    return BenchmarkRunner(sut, multivar_dataset, request.node.name)

UNIVAR_SUTS = ["pgforecast_sut", "pgforecast_timescale_sut", "python_sut", "python_geometric_sut", "timescale_python_geometric_sut"]

@pytest.mark.parametrize("competitor_server", ["python_autoarima"], indirect=True)
@pytest.mark.parametrize("sut", UNIVAR_SUTS, indirect=True)
@pytest.mark.parametrize("num_records", [10_000])
def test_univar_single_insert(univar_runner, num_records, competitor_server):
    """
    Verifies that single-insert benchmarking runs successfully
    and produces sane metrics.
    """
    result = univar_runner.run_single(num_records)

    # Type & structure
    assert isinstance(result, BenchmarkResult)
    assert isinstance(result.metrics, dict)

    # Metrics correctness
    assert "avg_insert" in result.metrics
    assert "avg_forecast" in result.metrics

    # Basic sanity checks
    assert len(result.insert_timings) == num_records
    assert result.metrics["avg_insert"] is not None
    assert result.metrics["avg_insert"] > 0.0
    assert result.metrics["avg_forecast"] is not None
    assert result.metrics["avg_forecast"] > 0.0


@pytest.mark.parametrize("competitor_server", ["python_autoarima"], indirect=True)
@pytest.mark.parametrize("sut", UNIVAR_SUTS, indirect=True)
@pytest.mark.parametrize("page_size,num_records", [(10_000, 1_000_000)])
def test_univar_batch_insert(univar_runner, page_size, num_records, competitor_server):
    """
    Verifies that batch-insert benchmarking runs successfully
    and produces sane metrics.
    """
    result = univar_runner.run_batch(page_size, num_records)

    # Type & structure
    assert isinstance(result, BenchmarkResult)
    assert isinstance(result.metrics, dict)

    # Derived expectations
    expected_pages = num_records // page_size

    assert len(result.insert_timings) == expected_pages

    assert result.metrics["avg_insert"] is not None
    assert result.metrics["avg_insert"] > 0.0
    assert result.metrics["avg_forecast"] is not None
    assert result.metrics["avg_forecast"] > 0.0

MULTIVAR_CASES = [
    ("python_xgboost", "python_xgboost_sut"),
    ("python_xgboost", "python_xgboost_incremental_sut"),
    ("python_xgboost", "pg_joinboost_sut_incremental"),
    ("python_xgboost", "pg_joinboost_sut_non_incremental"),
    ("python_xgboost", "ts_joinboost_sut"),
    ("python_joinboost", "duckdb_joinboost_sut"),
]

@pytest.mark.parametrize("competitor_server, sut", MULTIVAR_CASES, indirect=True)
@pytest.mark.parametrize("num_records,n_lags,n_features,single_target", [(10_000, 1, 10, True)])
def test_multivar_single_insert(multivar_runner, num_records, n_lags, n_features, single_target, competitor_server):
    result = multivar_runner.run_single(num_records)

    # Type & structure
    assert isinstance(result, BenchmarkResult)
    assert isinstance(result.metrics, dict)

    # Derived expectations
    assert len(result.insert_timings) == num_records

    assert result.metrics["avg_insert"] is not None
    assert result.metrics["avg_insert"] > 0.0
    assert result.metrics["avg_forecast"] is not None
    assert result.metrics["avg_forecast"] > 0.0

@pytest.mark.parametrize("competitor_server, sut", MULTIVAR_CASES, indirect=True)
@pytest.mark.parametrize("page_size,num_records,n_lags,n_features,single_target", [(10_000, 1_000_000, 5, 5, True)])
def test_univar_joinboost_batch_insert(univar_runner, page_size, num_records, n_lags, n_features, single_target, competitor_server):
    result = univar_runner.run_batch(page_size, num_records)

    # Type & structure
    assert isinstance(result, BenchmarkResult)
    assert isinstance(result.metrics, dict)

    # Derived expectations
    expected_pages = num_records // page_size

    assert len(result.insert_timings) == expected_pages

    assert result.metrics["avg_insert"] is not None
    assert result.metrics["avg_insert"] > 0.0
    assert result.metrics["avg_forecast"] is not None
    assert result.metrics["avg_forecast"] > 0.0

@pytest.mark.parametrize("competitor_server, sut", MULTIVAR_CASES, indirect=True)
@pytest.mark.parametrize("page_size,num_records,n_lags,n_features,single_target", [(10_000, 100_000, 5, 50, False)])
def test_multivar_batch_insert(multivar_runner, page_size, num_records, n_lags, n_features, single_target, competitor_server):
    result = multivar_runner.run_batch(page_size, num_records)

    # Type & structure
    assert isinstance(result, BenchmarkResult)
    assert isinstance(result.metrics, dict)

    # Derived expectations
    expected_pages = num_records // page_size

    assert len(result.insert_timings) == expected_pages

    assert result.metrics["avg_insert"] is not None
    assert result.metrics["avg_insert"] > 0.0
    assert result.metrics["avg_forecast"] is not None
    assert result.metrics["avg_forecast"] > 0.0
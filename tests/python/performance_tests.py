import time
from typing import Dict, List, Any
import csv
from datetime import datetime, timedelta, timezone
import os

from sqlalchemy import create_engine, text
from monash.utils import stream_tsf_values, stream_tsf_aligned_series
from itertools import islice
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
    def __init__(self, test_name: str, insert_timings: list[float], forecast_timings: list[float], losses: list[float], write_to_file = True, csv_prefix = "performance_"):
        self.insert_timings = insert_timings
        self.forecast_timings = forecast_timings
        self.losses = losses
        self.metrics = self.calculate_metrics(insert_timings, forecast_timings, losses)
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
            loss_rows = map(lambda l: { "type": "loss", "time": l }, self.losses)
            writer.writerows(insert_rows)
            writer.writerows(forecast_rows)
            writer.writerows(loss_rows)

    def calculate_metrics(self, insert_timings: list[float], forecast_timings: list[float], losses: list[float]):
        avg_forecast = None
        avg_insert = None
        final_loss = None

        if len(insert_timings) > 0:
            avg_insert = sum(insert_timings) / len(insert_timings)
        if len(forecast_timings) > 0:
            avg_forecast = sum(forecast_timings) / len(forecast_timings)
        if len(losses) > 0:
            final_loss = losses[-1]

        return {"avg_insert": avg_insert, "avg_forecast": avg_forecast, "final_loss": final_loss}
    
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
    def __init__(self, sut: SystemUnderTest, dataset: Dataset, test_name: str, forecast_horizon: int = 14):
        self.sut = sut
        self.dataset = dataset
        self.test_name = test_name
        self.forecast_horizon = forecast_horizon
        self.insert_timings = []
        self.forecast_timings = []
        self.losses = []
    
    def run_single(self, num_records: int) -> BenchmarkResult:
        logger.info(
            "Starting single insert benchmark (%d records)", num_records
        )

        self.insert_timings = []
        self.forecast_timings = []

        it = iter(self.dataset)
        self.sut.setup_single(list(islice(it, 100)))

        num_inserted = 0
        for record in it:
            if num_inserted == num_records:
                break
            if num_inserted % 100 == 0 and num_inserted > 0:
                start = time.perf_counter()
                _ = self.sut.forecast(self.forecast_horizon)
                end = time.perf_counter()
                self.forecast_timings.append(end - start)
                logging.info(f"Inserted {num_inserted} records, forecasted in {end - start}s")

            start = time.perf_counter()
            self.sut.add_single(record)
            end = time.perf_counter()
            self.insert_timings.append(end - start)
            self.losses.append(self.sut.get_loss())
            num_inserted += 1
            logging.debug(f"Inserted record #{num_inserted}")

        logger.info("Single insert benchmark complete")
        self.sut.teardown_single()
        result = BenchmarkResult(self.test_name,
                                 self.insert_timings,
                                 self.forecast_timings,
                                 self.losses,
                                 write_to_file=True)
        return result
    
    def run_batch(self, page_size: int, num_records: int) -> BenchmarkResult:
        logger.info(
            "Starting batch insert benchmark (%d records, page_size=%d)", num_records, page_size
        )
        self.insert_timings = []
        self.forecast_timings = []

        it = iter(self.dataset)
        self.sut.setup_batch(list(islice(it, 100)))

        num_pages = num_records // page_size
        for page in range(0, num_pages):
            records = list(islice(it, page_size))
            if not records:
                break

            start = time.perf_counter()
            self.sut.add_batch(records)
            end = time.perf_counter()
            self.insert_timings.append(end - start)

            start = time.perf_counter()
            _ = self.sut.forecast(self.forecast_horizon)
            end = time.perf_counter()
            self.forecast_timings.append(end - start)

            self.losses.append(self.sut.get_loss())

            logger.debug(
                "Page %d/%d processed (insert=%.6fs, forecast=%.6fs, loss=%f)",
                page + 1,
                num_pages,
                self.insert_timings[-1],
                self.forecast_timings[-1],
                self.losses[-1]
            )

        logger.info("Batch insert benchmark complete")
        self.sut.teardown_batch()
        result = BenchmarkResult(self.test_name,
                                 self.insert_timings,
                                 self.forecast_timings,
                                 self.losses,
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
def python_sut(competitor_server, n_lags, n_features, single_target):
    sut = PythonWebServer()
    
    if n_lags is not None and n_features is not None:
        sut._post("config", {"mode": "naive", "n_lags": n_lags, "n_features": n_features, "single_target": single_target})
    else:
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
def pg_joinboost_sut(n_lags, n_features, single_target):
    sut = JoinBoostSUT(lags=n_lags, n_features=n_features, predict_single_target=single_target)
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
    ("python_xgboost", "python_sut"),
    ("python_xgboost", "pg_joinboost_sut"),
    ("python_xgboost", "ts_joinboost_sut"),
    ("python_joinboost", "duckdb_joinboost_sut"),
]

@pytest.mark.parametrize("competitor_server, sut", MULTIVAR_CASES, indirect=True)
@pytest.mark.parametrize("page_size,num_records,n_lags,n_features,single_target", [(10_000, 1_000_000, 5, 5, True), (10_000, 50_000, 5, 50, False)])
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
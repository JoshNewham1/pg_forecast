from ts_benchmark.data.data_source import read_data
from ts_benchmark.evaluation.metrics import REGRESSION_METRICS
from sqlalchemy import create_engine
import pytest
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from inspect import signature
from sklearn.preprocessing import StandardScaler
from utils import MODEL_FACTORIES


# Accuracy tolerance vs TFB models
TOLERANCE = 0.2

# Override with test DB for all tests


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

# Sample training and evaluation data


@pytest.fixture
def seasonality_series():
    return read_data(os.path.join(os.path.dirname(__file__), "data", "seasonality.csv"))


@pytest.fixture
def shifting_series():
    return read_data(os.path.join(os.path.dirname(__file__), "data", "shifting.csv"))


@pytest.fixture
def stationarity_series():
    return read_data(os.path.join(os.path.dirname(__file__), "data", "stationarity.csv"))


@pytest.fixture
def transition_series():
    return read_data(os.path.join(os.path.dirname(__file__), "data", "transition.csv"))


@pytest.fixture
def trend_series():
    return read_data(os.path.join(os.path.dirname(__file__), "data", "trend.csv"))


@pytest.fixture
def dummy_series():
    return read_data(os.path.join(os.path.dirname(__file__), "data", "dummy.csv"))


@pytest.fixture
def tfb_results():
    return {
        "ARIMA": {
            "seasonality": {
                "mae": 3.3443343615047, "mse": 15.4125805235647, "rmse": 3.92588595396819, "mape": 16.8029260823427, "smape": 15.6803247578165, "mase": 28.379845533712, "wape": 15.2347014664729, "msmape": 15.6424793115566
            },
            "shifting": {
                "mae": 0.253362691420941, "mse": 0.0910771205573611, "rmse": 0.301789861588094, "mape": 30.2407592705918, "smape": 27.5007415134066, "mase": 0.442211375259662, "wape": 27.2677336058412, "msmape": 25.9665719851187
            },
            "stationarity": {
                "mae": 70.7891739080406, "mse": 6995.52247939511, "rmse": 83.6392400694501, "mape": 0.713667100315758, "smape": 0.716592852326448, "mase": 2.12231720719545, "wape": 0.717266970123747, "msmape": 0.716589224838155
            },
            "transition": {
                "mae": 0.0744705575580259, "mse": 0.00554586394300325, "rmse": 0.0744705575580259, "mape": float("inf"), "smape": 200, "mase": 0.0135846976036936, "wape": float("inf"), "msmape": 24.8235191860086
            },
            "trend": {
                "mae": 2.85524824621666, "mse": 13.5380907275609, "rmse": 3.67941445444257, "mape": 0.0209054844059947, "smape": 0.0209090609695868, "mase": 0.00759290392345704, "wape": 0.0212723027498237, "msmape": 0.0209089843903107
            }
        }
    }

# Inject test database engine into model factories


def make_test_model(model_factory, engine):
    """
    Override the model's engine to point to the test database.
    """
    model = model_factory()
    model.engine = engine
    return model


def generate_train_eval(series, horizon):
    train_series = series[:-horizon]
    eval_series = series[-horizon:]["channel_1"]
    return train_series, eval_series


def evaluate_series(train, eval, pred):
    metrics = {}
    for metric, func in REGRESSION_METRICS.items():
        sig = signature(func)
        args = [eval, pred]

        # For metrics requiring normalisation
        if "scaler" in sig.parameters:
            # Reshape inputs into 2D array for sklearn's scaler
            eval = np.asarray(eval).reshape(-1, 1)
            pred = np.asarray(pred).reshape(-1, 1)
            args = [eval, pred]
            args.append(StandardScaler().fit(eval))
        # For MASE
        if "hist_data" in sig.parameters:
            args.append(train)
        # TODO: Implement MASE
        if "seasonality" in sig.parameters:
            continue

        metrics[metric] = func(*args)
    return metrics


def dict_close(a: dict, b: dict, tolerance=TOLERANCE):
    """
    Compute percentage closeness of values for all matching keys in two dictionaries.

    Parameters
    ----------
    a : dict
        First dictionary with numeric values.
    b : dict
        Second dictionary with numeric values.
    tolerance : float, optional
        Maximum relative difference allowed.

    Returns
    -------
    string
        Failing key, or "" if all keys are within `tolerance`
    float
        Percentage closeness of a failing key, or -1 if all keys are within `tolerance`.
    """
    for key in a.keys():
        if key in b and a[key] <= b[key]*(1-tolerance):
            return key, (b[key] / a[key]) * 100
        elif key in b and a[key] > b[key]*(1+tolerance):
            return key, (a[key] / b[key]) * 100
    return "", -1


def benchmark_against_tfb(model_factory, test_engine, series, horizon, name, tfb_results):
    train_series, eval_series = generate_train_eval(
        series, horizon)

    model = make_test_model(model_factory, test_engine)
    model.forecast_fit(train_series)
    preds = model.forecast(horizon, train_series)

    metrics = evaluate_series(train_series, eval_series, preds)
    goal_metrics = tfb_results[model.model_name][name]

    fail_key, fail_tolerance = dict_close(metrics, goal_metrics)
    if fail_key != "":
        raise ValueError(
            f"{fail_key} is outside the tolerance range ({round(metrics[fail_key], 2)}, {round(fail_tolerance, 2)}% of baseline)")

# Parametrised tests for all models


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_forecast_length(model_factory, test_engine, dummy_series):
    horizon = 5
    train_series, eval_series = generate_train_eval(dummy_series, horizon)

    model = make_test_model(model_factory, test_engine)
    model.forecast_fit(train_series)
    preds = model.forecast(horizon, eval_series)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == horizon


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_seasonality(model_factory, test_engine, seasonality_series, tfb_results):
    benchmark_against_tfb(model_factory, test_engine, seasonality_series,
                          48, "seasonality", tfb_results)


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_shifting(model_factory, test_engine, shifting_series, tfb_results):
    benchmark_against_tfb(model_factory, test_engine, shifting_series,
                          48, "shifting", tfb_results)


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_stationarity(model_factory, test_engine, stationarity_series, tfb_results):
    benchmark_against_tfb(model_factory, test_engine, stationarity_series,
                          13, "stationarity", tfb_results)


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_transition(model_factory, test_engine, transition_series, tfb_results):
    benchmark_against_tfb(model_factory, test_engine, transition_series,
                          13, "transition", tfb_results)


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_trend(model_factory, test_engine, trend_series, tfb_results):
    benchmark_against_tfb(model_factory, test_engine, trend_series,
                          8, "trend", tfb_results)

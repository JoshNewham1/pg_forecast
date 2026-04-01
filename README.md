# PGForecast: In-Database Forecasting on Streams of IoT Data

`pg_forecast` is a PostgreSQL extension designed to provide efficient, in-database time series forecasting using ARIMA models. It is optimised for high-performance incremental updates, allowing models to be updated as new data arrives without full re-training. This project was developed as part of an undergraduate dissertation.

## System Architecture

The system consists of three main components:
1.  **PostgreSQL Extension (`src/pg_forecast`)**: A C-based extension implementing the ARIMA model logic, incremental updates, and SQL interface. It leverages `libnlopt` for initial model fitting.
2.  **JoinBoost Integration (`src/JoinBoost`)**: Modifications to Huang et al.'s JoinBoost that provide incremental updates and PostgreSQL support
3.  **Performance Benchmark (`tests/python/performance_tests.py`)**: A Python-based test suite for validating accuracy, performance, and scalability on IoT data.

## Installation

First, clone this repo and all its dependencies (in `eval/`):

```bash
git clone --recursive https://github.com/JoshNewham1/pg_forecast.git
```

### Prerequisites

Then, to build and run the project, ensure the following dependencies are installed:

- **Build Tools**: `gcc`, `make`
- **PostgreSQL**: `postgresql-server-dev-xx` (xx matches your version of PostgreSQL e.g., 18)
- **Libraries**: `libnlopt-dev`
- **Python**: Python 3.12.3 (recommended for TFB compatibility) and `venv`

```bash
sudo apt install gcc make postgresql-server-dev-18 libnlopt-dev
```

### Building the PostgreSQL Extension

```bash
# Combined command
sudo make install && sudo make clean && sudo -u postgres psql -d DB_NAME -c "DROP EXTENSION IF EXISTS pg_forecast CASCADE; CREATE EXTENSION pg_forecast;"

# Or individually
sudo make install
psql -d DB_NAME -c "CREATE EXTENSION pg_forecast;"
```

### Python Environment Setup

The Python environment is required for running benchmarks. The `install.sh` script will set up a virtual environment and symlink our modified JoinBoost and TFB code into their repositories in `eval/JoinBoost` and `eval/TFB` respectively.

```bash
# Use the provided installation script
./install.sh

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

This project utilises several datasets that are a few gigabytes each. Due to their size, they are not included in the repository and must be downloaded separately.

### 1. Time Series Forecasting Benchmark (TFB)
- **Source**: [TFB Dataset (Google Drive)](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link)
- **Target Location**: `eval/TFB/dataset/`
- **Usage**: The accuracy benchmark (RQ1)

### 2. Monash Time Series Repository
- **Source**: [Monash TSF (Hugging Face)](https://huggingface.co/datasets/Monash-University/monash_tsf/tree/main/data)
- **Required Files**: `solar_10_minutes_dataset`, `wind_farms_minutely_dataset_with_missing_values`.
- **Target Location**: `eval/monash/data/`
- **Usage**: The IoT performance benchmark (RQ2)

### 3. Favorita
- **Source**: Kaggle (Corporación Favorita Grocery Sales Forecasting)
- **Processing**: The raw Kaggle data is preprocessed into a relational schema using `eval/favorita/preprocess.sql` to test forecasting over joins.
- **Target Location**: `eval/favorita/data`
- **Usage**: The relational scalability benchmark (RQ3: `tests/python/test_joinboost.py`)

## Usage

### Basic Forecasting Flow

1.  **Initialise Model**:
    ```sql
    SELECT create_forecast(
        model_name   => 'autoarima',
        input_table  => 'time_series_data',
        date_column  => 'timestamp',
        value_column => 'value'
    );
    ```

2.  **Generate Forecast**:
    ```sql
    SELECT * FROM run_forecast(
        model_name    => 'autoarima',
        input_table   => 'time_series_data',
        date_column   => 'timestamp',
        value_column  => 'value',
        horizon       => 10,
        forecast_step => '1 hour'
    );
    ```

3.  **Incremental Update**:
    Inserting new data into the tracked table automatically triggers a state update for the associated forecast model (if configured via triggers) or can be updated manually.

## Evaluation & Testing
Run the accuracy and performance tests together:
```bash
./run_tests.sh
```

Run the unit tests:
```bash
pytest tests/python/unit_tests.py
```

### Accuracy Experiment (RQ1)
To run the full TFB suite against `pg_forecast`, use `./run_tests.sh`. To run an individual test:
```bash
# Example for daily frequency
python ./eval/TFB/scripts/run_benchmark.py --config-path fixed_forecast_config_daily.json --save-path daily --model-name "pg_forecast.ARIMA"
```

### Latency Experiment (RQ2)
To reproduce the performance results for a specific configuration, use the following command (e.g., for single-row insert latency). Put the SUT and either `single` or `batch`:
```bash
pytest "tests/python/performance_tests.py" -k "pgforecast and single"
```

### Scalability Experiment (RQ3)
To reproduce scalability results, run the following script with a model and number of iterations. For example:
```bash
python tests/python/test_joinboost.py duckdb 30
```
Supported models are `duckdb`, `postgres`, `timescale`, `citus`, `sklearn`, `xgb_mem` (in-memory XGBoost), `xgb_ooc` (disk XGBoost).


## Repository Structure

- `src/pg_forecast/`: Implementation of PGForecast - our incremental, in-database AutoARIMA model.
- `tests/python/`: Unit, performance and scalability tests.
- `eval/TFB/`: Cloned TFB repo, updated with our modifications through `install.sh`.
- `eval/JoinBoost/`: Cloned JoinBoost repo, updated with our modifications through `install.sh`.
- `eval/competitors/`: Baseline models (Python-based ARIMA, XGBoost).
- `eval/graphs`: All results and graphs from the Experiments chapter and Appendices
- `install.sh`: Automated setup script for environment and symlinks.
- `run_tests.sh`: Run accuracy and performance experiments (RQ1 & RQ2)
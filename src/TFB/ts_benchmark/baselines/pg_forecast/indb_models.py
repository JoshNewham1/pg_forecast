import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from io import StringIO
import functools
from typing import Dict

from ts_benchmark.models import ModelBase

IN_DB_MODEL_ARGS = {
    "model": "model"
}


class InDBModelAdapter(ModelBase):
    """
    Generic In-Database Model Adapter.

    Adapts in-database forecasting model from pg_forecast (ARIMA, ETS, etc.)
    that exposes SQL functions for training and forecasting.
    """

    def __init__(
        self,
        model_name: str,
        model_args: dict = None,
        train_fn: str = "pg_forecast_train",
        forecast_fn: str = "pg_forecast",
        base_table: str = "pg_forecast_tfb_eval",
        **kwargs
    ):
        super().__init__(**kwargs)

        self._model_name = model_name
        self.train_fn = train_fn
        self.forecast_fn = forecast_fn
        self.base_table = base_table
        self.model_args = model_args or {}

        # Load DB credentials
        load_dotenv()
        DB_USERNAME = os.getenv("DB_USERNAME")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST", "localhost")
        DB_PORT = os.getenv("DB_PORT", "5432")
        DB_NAME = os.getenv("DB_NAME")

        if not all((DB_USERNAME, DB_PASSWORD, DB_NAME)):
            raise RuntimeError(
                "DB_USERNAME, DB_PASSWORD, DB_NAME must be specified in .env")

        self.engine = create_engine(
            f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

    @property
    def model_name(self):
        return self._model_name

    def _copy_to_db(self, df: pd.DataFrame, table: str):
        """
        Efficiently uploads a DataFrame to PostgreSQL using COPY.
        """
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)

        conn = self.engine.raw_connection()
        try:
            cur = conn.cursor()
            cur.copy_expert(f"COPY {table} FROM STDIN WITH CSV", buffer)
            conn.commit()
            cur.close()
        finally:
            conn.close()

    def forecast_fit(self, train_data: pd.DataFrame, **kwargs) -> "ModelBase":
        """
        Uploads training data to DB and triggers model training.
        """
        date_col, value_col = "date", "value"

        with self.engine.begin() as conn:
            conn.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS {self.base_table} (
                    {date_col} TIMESTAMP,
                    {value_col} DOUBLE PRECISION
                );
                TRUNCATE TABLE {self.base_table};
                """)
            )

        # Bulk insert training data
        train_data = train_data.reset_index()
        train_data.columns = [date_col, value_col]
        self._copy_to_db(train_data, self.base_table)

        # Run in-DB training
        args_str = self._format_sql_args(self.model_args)
        with self.engine.begin() as conn:
            sql = f"SELECT {self.train_fn}('{self._model_name}', '{self.base_table}', '{date_col}', '{value_col}'{args_str});"
            conn.execute(text(sql))

        return self

    def forecast(self, horizon: int, series: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Executes in-database forecast and retrieves predictions.
        """
        date_col, value_col = "date", "value"
        temp_table = f"{self.base_table}_eval"

        with self.engine.begin() as conn:
            conn.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS {temp_table} (
                    {date_col} TIMESTAMP,
                    {value_col} DOUBLE PRECISION
                );
                TRUNCATE TABLE {temp_table};
                """)
            )

        # Bulk insert eval data
        series = series.reset_index()
        series.columns = [date_col, value_col]
        self._copy_to_db(series, temp_table)

        # Run forecast
        args_str = self._format_sql_args(self.model_args)
        with self.engine.begin() as conn:
            sql = f"""
                SELECT forecast_value
                FROM {self.forecast_fn}('{self._model_name}', '{temp_table}', '{date_col}', '{value_col}', {horizon}{args_str});
            """
            result = conn.execute(text(sql))
            preds = [row[0] for row in result.fetchall()]

        return np.array(preds)

    # Helpers
    def _format_sql_args(self, args: dict) -> str:
        """
        Format extra args into SQL-style argument string.
        """
        if not args:
            return ""
        return ", " + ", ".join(f"{k} => '{v}'" for k, v in args.items())


# Factory generator for different types of model
def _generate_model_factory(
    model_name: str,
    train_fn: str = "pg_forecast_train",
    forecast_fn: str = "pg_forecast",
    model_args: dict = None,
) -> Dict:
    """
    Generate a factory for creating in-database model adapters.
    """
    model_factory = functools.partial(
        InDBModelAdapter,
        model_name=model_name,
        train_fn=train_fn,
        forecast_fn=forecast_fn,
        model_args=model_args or {},
    )

    # Map of hyperparameters that benchmark may override
    required_hyper_params = {}

    return {"model_factory": model_factory, "required_hyper_params": required_hyper_params}


def _get_model_info(model_name: str, required_args: Dict, model_args: Dict) -> tuple:
    """
    Helper function to retrieve darts model information by name

    :param model_name: name of the model.
    :param required_args: arguments that the model requires from the pipeline.
    :param model_args: specified model arguments.
    :return: a tuple including model name, model_class, required args and model args.
    """
    return model_name, None, required_args, model_args


def in_db_model_adapter(model_class: type) -> Dict:
    """
    Adapts a Darts deep model class to OTB protocol

    :param model_class: a class of deep forecasting model from Darts library.
    :return: model factory that follows the OTB protocol.
    """
    return _generate_model_factory(
        model_class.__name__,
        "pg_forecast_train",
        "pg_forecast",
        {"model": model_class.__name__}
    )


# Register models for auto-discovery
ARIMA = _generate_model_factory(
    model_name="ARIMA",
    model_args={}
)

ETS = _generate_model_factory(
    model_name="ETS",
    model_args={}
)

IN_DB_MODELS = [
    _get_model_info("ARIMA", {}, {}),
    _get_model_info("ETS", {}, {})
]

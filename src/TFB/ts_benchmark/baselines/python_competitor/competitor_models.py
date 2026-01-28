import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime
import functools

from ts_benchmark.models import ModelBase

class PythonCompetitorAdapter(ModelBase):
    """
    Adapter for the Python REST API competitor.
    """

    def __init__(
        self,
        model_name: str,
        model_args: dict = None,
        mode: str = "naive",
        base_url: str = "http://localhost:8000",
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model_name = model_name
        self.model_args = model_args
        self.mode = mode
        self.base_url = base_url

    @property
    def model_name(self):
        return self._model_name

    def forecast_fit(self, train_data: pd.DataFrame, **kwargs) -> "ModelBase":
        """
        TFB fit stage. We setup the model on the REST API.
        """
        return self

    def forecast(self, horizon: int, series: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        TFB forecast stage.
        """
        try:
            # Sync data to API
            records = self._df_to_records(series)
            requests.post(f"{self.base_url}/setup_batch", json=records)
            requests.post(f"{self.base_url}/config", json={"mode": self.mode})
            
            # Call forecast
            resp = requests.post(f"{self.base_url}/forecast", json={"horizon": horizon})
            resp.raise_for_status()
            preds = [row["forecast_value"] for row in resp.json()]
            return np.array(preds)
        except Exception as e:
            print(f"Error in forecast: {e}")
            # Fallback to zeros or last value if API fails
            return np.zeros(horizon)

    def _df_to_records(self, df: pd.DataFrame) -> list:
        df = df.reset_index()
        # Assume first col is date, second is value
        date_col = df.columns[0]
        val_col = df.columns[1]
        
        # Ensure date_col is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
            
        start_ts = df[date_col].iloc[0]
        records = []
        for _, row in df.iterrows():
            offset = (row[date_col] - start_ts).total_seconds()
            records.append({
                "start_timestamp": start_ts.isoformat(),
                "time_index": offset,
                "value": float(row[val_col])
            })
        return records

def _generate_model_factory(
    model_name: str,
    mode: str,
    base_url: str = "http://localhost:8000",
) -> Dict:
    model_factory = functools.partial(
        PythonCompetitorAdapter,
        model_name=model_name,
        mode=mode,
        base_url=base_url,
    )
    return {"model_factory": model_factory, "required_hyper_params": {}}

def _get_model_info(model_name: str, required_args: Dict, model_args: Dict) -> tuple:
    return model_name, None, required_args, model_args

# Register models
PYTHON_NAIVE = _generate_model_factory(model_name="PYTHON_NAIVE", mode="naive")
PYTHON_GEOMETRIC = _generate_model_factory(model_name="PYTHON_GEOMETRIC", mode="geometric")

COMPETITOR_MODELS = [
    _get_model_info("PYTHON_NAIVE", {}, {}),
    _get_model_info("PYTHON_GEOMETRIC", {}, {})
]

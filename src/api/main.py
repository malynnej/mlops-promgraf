import logging
import datetime
from typing import Any, Optional

import pandas as pd
import requests
import io
import zipfile
import sys
import warnings
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from evidently import Report
from evidently.presets import DataDriftPreset

from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel, Field

from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, Gauge


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bike Sharing Predictor API",
    description="API for predicting bike sharing demand with MLOps monitoring.",
    version="1.0.0"
)

# --- Global Variables for Model and Data ---
TARGET = 'cnt'
PREDICTION = 'prediction'
NUM_FEATS = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
CAT_FEATS = ['season', 'holiday', 'workingday', 'weathersit']

DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DTEDAY_COL_NAME = "dteday"

MODEL: Optional[RandomForestRegressor] = None
REFERENCE_DATA: Optional[pd.DataFrame] = None  # January 2011 reference
PROM_REGISTRY = CollectorRegistry()

# --- Prometheus Metrics Definitions ---

API_REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status_code"],
    registry=PROM_REGISTRY,
)

API_REQUEST_DURATION_SECONDS = Histogram(
    "api_request_duration_seconds",
    "API request latency in seconds",
    ["endpoint", "method", "status_code"],
    registry=PROM_REGISTRY,
)

MODEL_RMSE_SCORE = Gauge(
    "model_rmse_score",
    "Model RMSE on evaluation data",
    registry=PROM_REGISTRY,
)

MODEL_MAE_SCORE = Gauge(
    "model_mae_score",
    "Model MAE on evaluation data",
    registry=PROM_REGISTRY,
)

MODEL_R2_SCORE = Gauge(
    "model_r2_score",
    "Model R2 score on evaluation data",
    registry=PROM_REGISTRY,
)

# Custom metric: MAPE (relevant for scale‑independent error monitoring)
MODEL_MAPE_SCORE = Gauge(
    "model_mape_score",
    "Model MAPE on evaluation data",
    registry=PROM_REGISTRY,
)

# data drift status 
DATA_DRIFT_DETECTED = Gauge(
    "evidently_data_drift_detected_status",
    "Data drift detected status (0/1)",
    registry=PROM_REGISTRY,
)


# --- Data Ingestion and Preparation Functions ---

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def _fetch_data() -> pd.DataFrame:
    """Fetch the bike sharing dataset from UCI."""
    logger.info("Fetching data from UCI archive...")
    try:
        content = requests.get(DATASET_URL, verify=False, timeout=60).content
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            df = pd.read_csv(
                z.open("hour.csv"),
                header=0,
                sep=",",
                parse_dates=[DTEDAY_COL_NAME],
            )
        logger.info("Data fetched successfully.")
        return df
    except Exception as e:
        logger.exception("Error fetching data")
        sys.exit(1)


def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Set datetime index and sort, same as in run_evaluation.py."""
    logger.info("Processing raw data...")
    raw_data["hr"] = raw_data["hr"].astype(int)
    raw_data.index = raw_data.apply(
        lambda row: datetime.datetime.combine(
            row[DTEDAY_COL_NAME].date(), datetime.time(row.hr)
        ),
        axis=1,
    )
    raw_data = raw_data.sort_index()
    logger.info("Data processed successfully.")
    return raw_data


def _train_and_prepare_reference_model():
    """Train RandomForestRegressor on January 2011 and store reference data."""
    global MODEL, REFERENCE_DATA

    full_data = _process_data(_fetch_data())

    jan_mask = (full_data.index >= "2011-01-01 00:00:00") & (
        full_data.index <= "2011-01-31 23:00:00"
    )
    reference = full_data.loc[jan_mask].copy()

    X_ref = reference[NUM_FEATS + CAT_FEATS]
    y_ref = reference[TARGET]

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_ref, y_ref)

    MODEL = rf
    REFERENCE_DATA = reference
    logger.info("Model trained on January 2011 data with %d rows.", len(reference))

def _compute_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    # Avoid division by zero: filter out zero targets
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(( (y_true[mask] - y_pred[mask]).abs() / y_true[mask].abs() ).mean() * 100.0)


# --- Pydantic Models for API Input/Output ---
class BikeSharingInput(BaseModel):
    temp: float = Field(..., example=0.24)
    atemp: float = Field(..., example=0.2879)
    hum: float = Field(..., example=0.81)
    windspeed: float = Field(..., example=0.0)
    mnth: int = Field(..., example=1)
    hr: int = Field(..., example=0)
    weekday: int = Field(..., example=6)
    season: int = Field(..., example=1)
    holiday: int = Field(..., example=0)
    workingday: int = Field(..., example=0)
    weathersit: int = Field(..., example=1)
    dteday: datetime.date = Field(..., example="2011-01-01", description="Date of the record in YYYY-MM-DD format.")

class PredictionOutput(BaseModel):
    predicted_count: float = Field(..., example=16.0)

class EvaluationData(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="List of data points, each containing features and the true target ('cnt').")
    evaluation_period_name: str = Field("unknown_period", description="Name of the period being evaluated (e.g., 'week1_february').")
    model_config = {'arbitrary_types_allowed': True}

class EvaluationReportOutput(BaseModel):
    message: str
    rmse: Optional[float]
    mape: Optional[float]
    mae: Optional[float]
    r2score: Optional[float]
    drift_detected: int
    evaluated_items: int

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Bike Sharing Predictor API. Use /predict to get bike counts or /evaluate to run drift reports."}

@app.on_event("startup")
async def startup_event():
    _train_and_prepare_reference_model()

@app.get("/metrics")
async def metrics():
    data = generate_latest(PROM_REGISTRY)
    return Response(content=data, media_type="text/plain; version=0.0.4")

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: BikeSharingInput, request: Request):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    endpoint = "/predict"
    method = request.method
    start_time = datetime.datetime.now()

    status_code = 200
    try:
        features = {
            "temp": input_data.temp,
            "atemp": input_data.atemp,
            "hum": input_data.hum,
            "windspeed": input_data.windspeed,
            "mnth": input_data.mnth,
            "hr": input_data.hr,
            "weekday": input_data.weekday,
            "season": input_data.season,
            "holiday": input_data.holiday,
            "workingday": input_data.workingday,
            "weathersit": input_data.weathersit,
        }
        X = pd.DataFrame([features])

        pred = float(MODEL.predict(X)[0])

        return PredictionOutput(predicted_count=pred)
    except Exception as e:
        logger.exception("Error in /predict")
        status_code = 500
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        duration = (datetime.datetime.now() - start_time).total_seconds()
        API_REQUESTS_TOTAL.labels(
            endpoint=endpoint, method=method, status_code=str(status_code)
        ).inc()
        API_REQUEST_DURATION_SECONDS.labels(
            endpoint=endpoint, method=method, status_code=str(status_code)
        ).observe(duration)


@app.post("/evaluate", response_model=EvaluationReportOutput)
async def evaluate(eval_data: EvaluationData, request: Request):
    if MODEL is None or REFERENCE_DATA is None:
        raise HTTPException(status_code=500, detail="Model or reference data not loaded.")

    endpoint = "/evaluate"
    method = request.method
    start_time = datetime.datetime.now()
    status_code = 200

    try:
        # Convert payload to DataFrame
        current_df = pd.DataFrame(eval_data.data)
        if current_df.empty:
            raise HTTPException(status_code=400, detail="No data provided for evaluation.")

        # Ensure required columns exist
        missing_cols = [c for c in NUM_FEATS + CAT_FEATS + [TARGET] if c not in current_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns in evaluation data: {missing_cols}",
            )

        # Predictions for current
        X_current = current_df[NUM_FEATS + CAT_FEATS]
        y_current = current_df[TARGET]
        y_current_pred = MODEL.predict(X_current)

        # 1) Regression metrics with sklearn
        rmse_value = float(mean_squared_error(y_current, y_current_pred) ** 0.5)
        mae_value = float(mean_absolute_error(y_current, y_current_pred))
        r2_value = float(r2_score(y_current, y_current_pred))
        mape_value = _compute_mape(y_current, pd.Series(y_current_pred))

        # 2) Data drift with Evidently 0.7.x
    #    DataDriftPreset will compute dataset-level drift between reference and current
        report = Report(metrics=[DataDriftPreset()])
        my_eval = report.run(
            reference_data=REFERENCE_DATA,
            current_data=current_df,
        )
        snapshot = my_eval.dict()

        drift_detected = 0
        for metric in snapshot.get("metrics", []):
            if metric.get("metric") == "DataDriftMetric":
                result = metric.get("result", {})
                if result.get("dataset_drift"):
                    drift_detected = 1
                break

        # 3) Update Prometheus Gauges
        MODEL_RMSE_SCORE.set(rmse_value)
        MODEL_MAE_SCORE.set(mae_value)
        MODEL_R2_SCORE.set(r2_value)
        MODEL_MAPE_SCORE.set(mape_value)
        DATA_DRIFT_DETECTED.set(drift_detected)

        evaluated_items = len(current_df)

        return EvaluationReportOutput(
        message=f"Evaluation for {eval_data.evaluation_period_name} completed.",
        rmse=rmse_value,
        mape=mape_value,
        mae=mae_value,
        r2score=r2_value,
        drift_detected=drift_detected,
        evaluated_items=evaluated_items,
        )
    
    except HTTPException:
        status_code = 400
        raise
    except Exception as e:
        logger.exception("Error in /evaluate")
        status_code = 500
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        duration = (datetime.datetime.now() - start_time).total_seconds()
        API_REQUESTS_TOTAL.labels(
            endpoint=endpoint, method=method, status_code=str(status_code)
        ).inc()
        API_REQUEST_DURATION_SECONDS.labels(
            endpoint=endpoint, method=method, status_code=str(status_code)
        ).observe(duration)

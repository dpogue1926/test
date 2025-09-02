"""
Prediction & Forecast Dashboard (Streamlit)
-------------------------------------------
Single‑file Streamlit app for:
  • Time‑series forecasting (Holt‑Winters Exponential Smoothing)
  • Tabular predictions (classification & regression) with scikit‑learn pipelines

Features
  - Upload CSVs (training + optional scoring/new data)
  - Clean preprocessing (numeric + categorical)
  - Train/validation split with metrics
  - Interactive controls for model choices & hyperparameters
  - Visualizations (timeseries with forecasts; confusion matrix; feature importances)
  - Downloadable predictions and trained model (.joblib)

Run:
  streamlit run predict_forecast_dashboard_streamlit.py

Dependencies (install once):
  pip install streamlit pandas numpy scikit-learn statsmodels matplotlib joblib

Notes
  - Forecasting uses statsmodels' Holt‑Winters ExponentialSmoothing. It performs well
    for many seasonal series without heavy dependencies (no Prophet install required).
  - For classification metrics, ROC AUC is shown when problem is binary and probabilities are available.
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe for Streamlit
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# =====================================
# Utilities
# =====================================

def _fmt(x: float) -> str:
    try:
        return f"{x:,.4f}"
    except Exception:
        return str(x)

@st.cache_data(show_spinner=False)
def read_csv_cached(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

@st.cache_data(show_spinner=False)
def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def infer_seasonal_period(idx: pd.DatetimeIndex) -> int:
    """Heuristic seasonal period based on inferred frequency.
    Fallbacks if freq not set.
    """
    try:
        freq = pd.infer_freq(idx)
    except Exception:
        freq = None

    if freq is None:
        # Guess by median spacing
        if len(idx) >= 3:
            median_delta = np.median(np.diff(idx.values).astype("timedelta64[D]").astype(float))
            if 0.9 <= median_delta <= 1.1:
                return 7  # daily ~ weekly seasonality
            if 27 <= median_delta <= 31:
                return 12  # monthly ~ yearly seasonality
        return 7

    freq = freq.upper()
    if freq in {"D"}:
        return 7
    if freq in {"B"}:  # business daily
        return 5
    if freq in {"W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT"}:
        return 52
    if freq.startswith("M") or freq.startswith("MS"):
        return 12
    if freq.startswith("Q"):
        return 4
    if freq.startswith("H"):
        return 24
    return 7


def plot_timeseries_with_forecast(train: pd.Series, test: pd.Series, fc: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test")
    fc.plot(ax=ax, label="Forecast")
    ax.set_title("Observed vs Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.8, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    return fig


# =====================================
# Sidebar — App Mode
# =====================================
st.set_page_config(page_title="Prediction & Forecast Dashboard", layout="wide")
st.title("🔮 Prediction & 📈 Forecast Dashboard")
st.caption("Upload your data and build models without leaving the browser.")

with st.sidebar:
    st.header("⚙️ Controls")
    mode = st.radio("Select task:", (
        "Time Series Forecast",
        "Tabular Prediction",
    ))

    st.markdown("---")
    st.write("**Data Upload**")
    train_file = st.file_uploader("Training CSV", type=["csv"], key="train")
    scoring_file = None
    if mode == "Tabular Prediction":
        scoring_file = st.file_uploader("Optional: New/Scoring CSV", type=["csv"], key="score")

# Load
train_df: Optional[pd.DataFrame] = None
if train_file is not None:
    train_df = read_csv_cached(train_file)

# Sample generators (if no file uploaded)
SAMPLE_TS_NAME = "_sample_timeseries"
SAMPLE_TAB_NAME = "_sample_tabular"

if train_df is None:
    if mode == "Time Series Forecast":
        st.info("No file uploaded. Using a synthetic seasonal sample (daily with weekly seasonality). Upload a CSV to use your own data.")
        rng = pd.date_range("2022-01-01", periods=365, freq="D")
        y = 20 + 5*np.sin(2*np.pi*rng.dayofyear/7) + np.random.normal(0, 1.2, len(rng))
        train_df = pd.DataFrame({"ds": rng, "y": y})
        train_df[SAMPLE_TS_NAME] = True
    else:
        st.info("No file uploaded. Using a synthetic tabular sample. Upload a CSV to use your own data.")
        rs = np.random.RandomState(42)
        n = 500
        df_syn = pd.DataFrame({
            "age": rs.randint(18, 80, size=n),
            "income": rs.normal(65_000, 15_000, size=n).round(0),
            "tenure": rs.randint(1, 10, size=n),
            "region": rs.choice(["North", "South", "East", "West"], size=n),
        })
        # Binary target influenced by features
        logits = -3 + 0.03*(df_syn["age"]) + 0.00002*(df_syn["income"]) + 0.12*(df_syn["tenure"]) + df_syn["region"].map({"North":0.2,"South":0.0,"East":-0.1,"West":0.1}).values
        p = 1/(1+np.exp(-logits))
        df_syn["churn"] = (rs.rand(n) < p).astype(int)
        train_df = df_syn.copy()
        train_df[SAMPLE_TAB_NAME] = True

# =====================================
# Time Series Forecast Mode
# =====================================
if mode == "Time Series Forecast":
    st.subheader("Time Series Forecasting (Holt‑Winters)")
    st.write("Provide a date/time column and a numeric target column.")

    # Column selection
    cols = list(train_df.columns)
    # Try to auto-pick a date column
    def _auto_date_col(columns: List[str]) -> Optional[str]:
        for c in columns:
            lc = str(c).lower()
            if any(k in lc for k in ["date", "ds", "time", "timestamp"]):
                return c
        return None

    date_col = st.selectbox("Date column", options=cols, index=cols.index(_auto_date_col(cols)) if _auto_date_col(cols) in cols else 0)
    target_col = st.selectbox("Target (numeric) column", options=[c for c in cols if c != date_col])

    # Parse datetime and sort
    ts = train_df[[date_col, target_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col]).sort_values(date_col)
    ts = ts.set_index(date_col)

    # Ensure numeric
    y = pd.to_numeric(ts[target_col], errors="coerce").dropna()

    st.write(f"**Row count after parsing:** {len(y):,}")

    # Controls
    horizon = st.number_input("Forecast horizon (steps)", min_value=1, max_value=max(1, len(y)//2), value=min(30, max(1, len(y)//4)))
    test_size = st.slider("Holdout size (for validation)", min_value=5, max_value=min(365, max(5, len(y)//3)), value=min(30, max(5, len(y)//5)))

    seasonal_period_default = infer_seasonal_period(y.index)
    seasonal_periods = st.number_input("Seasonal period", min_value=1, max_value=10_000, value=int(seasonal_period_default))
    trend_opt = st.selectbox("Trend", options=["add", "mul", "none"], index=0)
    seasonal_opt = st.selectbox("Seasonal", options=["add", "mul", "none"], index=0)

    go = st.button("Train & Forecast", type="primary")

    if go:
        # Train/valid split by time
        train_y = y.iloc[:-test_size]
        test_y = y.iloc[-test_size:]

        # Build HWES model
        trend = None if trend_opt == "none" else trend_opt
        seasonal = None if seasonal_opt == "none" else seasonal_opt
        try:
            model = ExponentialSmoothing(
                train_y,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods if seasonal else None,
                initialization_method="estimated",
            ).fit(optimized=True)
        except Exception as ex:
            st.error(f"Model failed to fit: {ex}")
            st.stop()

        # Forecast for validation and future horizon
        fc_valid = model.forecast(steps=test_size)
        fc_future = model.forecast(steps=horizon)

        # Metrics on validation window
        mae = mean_absolute_error(test_y, fc_valid)
        rmse = np.sqrt(mean_squared_error(test_y, fc_valid))
        mape = np.mean(np.abs((test_y - fc_valid) / np.maximum(1e-8, np.abs(test_y)))) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", _fmt(mae))
        c2.metric("RMSE", _fmt(rmse))
        c3.metric("MAPE %", _fmt(mape))

        fig = plot_timeseries_with_forecast(train_y, test_y, fc_valid)
        st.pyplot(fig, use_container_width=True)

        st.markdown("### Forecast")
        fc_df = pd.DataFrame({
            "date": fc_future.index,
            "forecast": fc_future.values,
        })
        st.dataframe(fc_df, use_container_width=True)

        # Download
        st.download_button(
            "Download forecast CSV",
            data=to_csv_download(fc_df),
            file_name="forecast.csv",
            mime="text/csv",
        )

        # Save model
        buf = io.BytesIO()
        joblib.dump(model, buf)
        st.download_button(
            "Download trained HWES model (.joblib)",
            data=buf.getvalue(),
            file_name="hwes_model.joblib",
            mime="application/octet-stream",
        )

# =====================================
# Tabular Prediction Mode
# =====================================
if mode == "Tabular Prediction":
    st.subheader("Tabular Prediction (Classification / Regression)")
    st.write("Pick a target column and we’ll build a pipeline.")

    cols = list(train_df.columns)
    # drop sample marker if present
    cols = [c for c in cols if c not in (SAMPLE_TAB_NAME, SAMPLE_TS_NAME)]

    target_col = st.selectbox("Target column", options=cols, index=max(0, len(cols)-1))

    feature_cols = st.multiselect(
        "Feature columns (exclude target)",
        options=[c for c in cols if c != target_col],
        default=[c for c in cols if c != target_col],
    )

    if not feature_cols:
        st.warning("Select at least one feature column.")
        st.stop()

    # Determine task type
    y_series = train_df[target_col]

    def _is_regression(s: pd.Series) -> bool:
        if pd.api.types.is_float_dtype(s):
            return True
        # many unique ints -> regression
        if pd.api.types.is_integer_dtype(s) and s.nunique() > 20:
            return True
        return False

    task_guess_is_reg = _is_regression(y_series)
    problem_type = st.radio("Problem type:", ["Auto (guess)", "Regression", "Classification"], index=0, horizontal=True)
    if problem_type == "Regression":
        is_regression = True
    elif problem_type == "Classification":
        is_regression = False
    else:
        is_regression = task_guess_is_reg

    # Split controls
    test_size = st.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5) / 100.0
    random_state = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

    # Model choice & hyperparameters
    if is_regression:
        model_name = st.selectbox("Model", ["LinearRegression", "RandomForestRegressor"], index=1)
        if model_name == "RandomForestRegressor":
            n_estimators = st.slider("n_estimators", 50, 500, 200, step=25)
            max_depth = st.slider("max_depth (0 = None)", 0, 50, 0, step=1)
        else:
            n_estimators, max_depth = None, None
    else:
        model_name = st.selectbox("Model", ["LogisticRegression", "RandomForestClassifier"], index=1)
        if model_name == "RandomForestClassifier":
            n_estimators = st.slider("n_estimators", 50, 500, 200, step=25)
            max_depth = st.slider("max_depth (0 = None)", 0, 50, 0, step=1)
        else:
            n_estimators, max_depth = None, None

    go2 = st.button("Train & Evaluate", type="primary")

    if go2:
        df = train_df[feature_cols + [target_col]].copy()

        # Separate X/y
        X = df[feature_cols]
        y = df[target_col]

        # Identify dtypes
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if c not in num_cols]

        # Build encoders with back‑compat for scikit‑learn versions
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        numeric_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
        ])
        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ])

        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols),
            ],
            remainder="drop",
        )

        # Choose model
        if is_regression:
            if model_name == "LinearRegression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=None if max_depth == 0 else int(max_depth),
                    random_state=random_state,
                    n_jobs=-1,
                )
        else:
            if model_name == "LogisticRegression":
                model = LogisticRegression(max_iter=1_000)
            else:
                model = RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=None if max_depth == 0 else int(max_depth),
                    random_state=random_state,
                    n_jobs=-1,
                )

        pipe = Pipeline(steps=[("pre", pre), ("model", model)])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None if is_regression else y
        )

        # Fit & predict
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Metrics
        if is_regression:
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", _fmt(mae))
            c2.metric("RMSE", _fmt(rmse))
            c3.metric("R²", _fmt(r2))
        else:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", _fmt(acc))
            c2.metric("F1 (weighted)", _fmt(f1))

            # ROC AUC (binary only)
            try:
                if hasattr(pipe, "predict_proba") and len(np.unique(y_test)) == 2:
                    y_proba = pipe.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                    st.metric("ROC AUC", _fmt(auc))
            except Exception:
                pass

            # Confusion matrix
            try:
                labels_sorted = sorted(pd.unique(y_test))
                cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
                fig_cm = plot_confusion_matrix(cm, [str(x) for x in labels_sorted])
                st.pyplot(fig_cm, use_container_width=False)
            except Exception:
                pass

        # Feature importances if RandomForest
        if (isinstance(model, RandomForestRegressor) or isinstance(model, RandomForestClassifier)):
            try:
                # get processed feature names
                feature_names = []
                if num_cols:
                    feature_names.extend(num_cols)
                if cat_cols:
                    # Extract categories from OHE
                    ohe_step = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
                    if hasattr(ohe_step, "get_feature_names_out"):
                        ohe_names = list(ohe_step.get_feature_names_out(cat_cols))
                    else:
                        # fallback
                        ohe_names = []
                    feature_names = num_cols + ohe_names

                importances = pipe.named_steps["model"].feature_importances_
                # Build a small DataFrame
                fi_df = pd.DataFrame({"feature": feature_names[:len(importances)], "importance": importances})
                fi_df = fi_df.sort_values("importance", ascending=False).head(20)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
                ax.set_title("Feature Importance (top 20)")
                ax.set_xlabel("Importance")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
            except Exception:
                pass

        # Downloads — trained model
        buf = io.BytesIO()
        joblib.dump(pipe, buf)
        st.download_button(
            "Download trained pipeline (.joblib)",
            data=buf.getvalue(),
            file_name="trained_pipeline.joblib",
            mime="application/octet-stream",
        )

        # Scoring new data (optional)
        if scoring_file is not None:
            try:
                new_df = read_csv_cached(scoring_file)
                st.markdown("### Scoring / New Predictions")
                st.write("Uploaded scoring data shape:", new_df.shape)

                new_preds = pipe.predict(new_df[feature_cols])
                out_df = new_df.copy()
                out_df["prediction"] = new_preds

                if not is_regression and hasattr(pipe, "predict_proba"):
                    try:
                        proba = pipe.predict_proba(new_df[feature_cols])
                        # Only add positive class probability for binary; else add max probability
                        if proba.ndim == 2 and proba.shape[1] == 2:
                            out_df["prob_positive"] = proba[:, 1]
                        else:
                            out_df["prob_max"] = proba.max(axis=1)
                    except Exception:
                        pass

                st.dataframe(out_df.head(50), use_container_width=True)
                st.download_button(
                    "Download predictions CSV",
                    data=to_csv_download(out_df),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as ex:
                st.error(f"Scoring failed: {ex}")

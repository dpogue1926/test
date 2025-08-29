# forecast_dashboard_streamlit.py
# Streamlit dashboard for monthly forecasting (SARIMAX vs ETS) with safe file upload handling.
import os
import io
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import streamlit as st

warnings.filterwarnings("ignore")

# =========================
# Helpers
# =========================
def autodetect_date_col(df: pd.DataFrame) -> Optional[str]:
    """Pick the column most likely to be a date/time by successful parse count."""
    candidates = [c for c in df.columns if any(tok in str(c).lower() for tok in
                    ["date", "time", "start", "end", "timestamp", "month", "day"])]
    parsed_counts = []
    to_try = candidates if candidates else list(df.columns)
    for c in to_try:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            parsed_counts.append((parsed.notna().sum(), c))
        except Exception:
            continue
    if not parsed_counts:
        return None
    parsed_counts.sort(reverse=True)
    return parsed_counts[0][1]

def autodetect_target_col(df: pd.DataFrame) -> str:
    """Choose a numeric column to forecast, preferring known names, else highest variance."""
    for pref in ["PKG QTY", "Fill Qty"]:
        if pref in df.columns:
            return pref
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        # Try coercing numerics broadly, then re-check
        coerce_df = df.copy()
        for c in df.columns:
            try:
                coerce_df[c] = pd.to_numeric(coerce_df[c], errors="coerce")
            except Exception:
                pass
        num_cols = [c for c in coerce_df.columns if pd.api.types.is_numeric_dtype(coerce_df[c])]
        if not num_cols:
            raise ValueError("No numeric columns found for forecasting.")
        df = coerce_df
    ranked = sorted([(float(df[c].var() or 0), c) for c in num_cols], reverse=True)
    return ranked[0][1]

def monthly_series(df: pd.DataFrame, date_col: str, target_col: str, agg: str) -> pd.Series:
    """Aggregate to monthly series with robust numeric coercion and interpolation for gaps."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # coerce target to numeric even if it contains text like "1,234" or units
    df[target_col] = (df[target_col]
                      .astype(str).str.replace(r"[^\d\.\-eE]", "", regex=True)
                      .replace({"": np.nan})
                      .astype(float))
    df = df.dropna(subset=[date_col, target_col]).sort_values(date_col)
    if df.empty:
        return pd.Series(dtype=float)

    if agg == "sum":
        s = df.groupby(pd.Grouper(key=date_col, freq="MS"))[target_col].sum()
    else:
        s = df.groupby(pd.Grouper(key=date_col, freq="MS"))[target_col].mean()

    s = s.asfreq("MS")
    if s.isna().all():
        # Try alternative aggregation if initial produced all NaN
        alt = "mean" if agg == "sum" else "sum"
        s_alt = df.groupby(pd.Grouper(key=date_col, freq="MS"))[target_col].mean() if alt == "mean" \
                else df.groupby(pd.Grouper(key=date_col, freq="MS"))[target_col].sum()
        s = s_alt.asfreq("MS")

    if s.isna().any():
        s = s.interpolate(limit_direction="both")

    return s.astype(float)

def train_sarimax(y: pd.Series, seasonal_periods: int = 12):
    """Small SARIMAX grid; returns (result, aic) or None."""
    best = (None, np.inf)
    y = y.astype(float)
    seasonal_ok = y.dropna().shape[0] >= 2 * seasonal_periods

    p = d = q = range(0, 2)  # 0,1 keeps it snappy
    seasonal_grid = [(0, 0, 0)] if not seasonal_ok else [(Pi, Di, Qi) for Pi in (0,1) for Di in (0,1) for Qi in (0,1)]

    for pi in p:
        for di in d:
            for qi in q:
                for (Pi, Di, Qi) in seasonal_grid:
                    try:
                        mod = sm.tsa.statespace.SARIMAX(
                            y,
                            order=(pi, di, qi),
                            seasonal_order=(Pi, Di, Qi, seasonal_periods) if seasonal_ok else (0, 0, 0, 0),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        res = mod.fit(disp=False)
                        if res.aic < best[1]:
                            best = (res, res.aic)
                    except Exception:
                        continue
    return best if best[0] is not None else None

def train_ets(y: pd.Series, seasonal_periods: int = 12):
    """Holt-Winters ETS with graceful fallbacks."""
    y = y.astype(float)
    enough_for_season = y.dropna().shape[0] >= 2 * seasonal_periods
    if not enough_for_season:
        return ExponentialSmoothing(
            y, trend="add", seasonal=None, initialization_method="estimated"
        ).fit(optimized=True)
    try:
        return ExponentialSmoothing(
            y, trend="add", seasonal="add",
            seasonal_periods=seasonal_periods, initialization_method="estimated"
        ).fit(optimized=True)
    except Exception:
        if (y > 0).all():
            try:
                return ExponentialSmoothing(
                    y, trend="add", seasonal="mul",
                    seasonal_periods=seasonal_periods, initialization_method="estimated"
                ).fit(optimized=True)
            except Exception:
                pass
        return ExponentialSmoothing(
            y, trend="add", seasonal=None, initialization_method="estimated"
        ).fit(optimized=True)

def mape(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt != 0
    return np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0 if mask.any() else np.nan

def _clip_nonneg(series_like):
    s = pd.Series(series_like, index=getattr(series_like, "index", None), dtype=float)
    return s.clip(lower=0.0)

def _pick_first_non_meta_sheet(xls) -> str:
    for s in xls.sheet_names:
        lname = s.lower()
        if not any(tok in lname for tok in ["readme", "meta", "dictionary", "schema"]):
            return s
    return xls.sheet_names[0]

def load_table_from_upload(buffer: io.BytesIO, filename: str, sheet_name=None) -> Tuple[pd.DataFrame, Optional[str]]:
    """Load DataFrame from uploaded file (BytesIO). Returns (df, used_sheet)."""
    ext = os.path.splitext(filename)[1].lower()
    buffer.seek(0)
    if ext == ".csv":
        return pd.read_csv(buffer), None

    # Excel handling
    buffer.seek(0)
    xls = pd.ExcelFile(buffer)
    if isinstance(sheet_name, (str, int)) and sheet_name != "":
        use = sheet_name if not isinstance(sheet_name, str) or sheet_name in xls.sheet_names else _pick_first_non_meta_sheet(xls)
    else:
        use = _pick_first_non_meta_sheet(xls)
    buffer.seek(0)
    df = pd.read_excel(buffer, sheet_name=use)
    if isinstance(df, dict):
        key = list(df.keys())[0]
        return df[key], key
    return df, use

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Syringe Line Forecast Dashboard", layout="wide")
st.title("🧪 Syringe Line Forecast Dashboard")
st.caption("Upload a CSV or Excel file and configure columns. The app builds a monthly series, compares SARIMAX vs ETS via AIC, and produces a 6-month forecast with a quick backtest.")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload data (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])
    default_sheet = st.text_input("Sheet name (Excel only, optional)", value="")
    date_col_override = st.text_input("Date column (optional)", value="Fill Scheduled Start Date")
    target_col_override = st.text_input("Target column (optional)", value="PKG QTY")
    agg_func = st.selectbox("Monthly aggregation", options=["sum", "mean"], index=0)
    fc_steps = st.number_input("Forecast months ahead", min_value=1, max_value=36, value=6, step=1)
    seasonal_periods = st.number_input("Seasonal periods (12=annual)", min_value=1, max_value=24, value=12, step=1)

# Guard: require an upload before proceeding
if uploaded is None:
    st.info("👆 Upload a CSV or Excel file in the sidebar to begin.")
    st.stop()

# Read uploaded file into memory (compatible with various Streamlit versions)
if hasattr(uploaded, "getbuffer"):
    # getbuffer() returns a memoryview; convert to bytes
    file_bytes = bytes(uploaded.getbuffer())
else:
    file_bytes = uploaded.read()

buf = io.BytesIO(file_bytes)


# Load DataFrame
df, used_sheet = load_table_from_upload(buf, uploaded.name, default_sheet.strip() or None)

# Column picks (with autodetect fallback)
date_col = date_col_override if (date_col_override in df.columns) else autodetect_date_col(df)
target_col = target_col_override if (target_col_override in df.columns) else autodetect_target_col(df)

# Metrics
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.metric("Rows", f"{len(df):,}")
with mcol2:
    st.metric("Columns", f"{len(df.columns):,}")
with mcol3:
    st.metric("Date column", str(date_col))
with mcol4:
    st.metric("Target column", str(target_col))

if date_col is None or target_col is None:
    st.error("Could not detect a valid date or target column. Adjust the sidebar inputs.")
    st.stop()

# Build monthly series
y = monthly_series(df, date_col, target_col, agg_func)
if y.empty:
    st.warning("No usable monthly data from the chosen columns/aggregation. Try a different column or aggregation.")
    st.stop()

# Forecasting
near_constant = (y.nunique(dropna=True) <= 1) or (np.nanstd(y.values) < 1e-6)
if near_constant:
    chosen_name = "Naive (flat)"
    last_val = float(y.iloc[-1])
    idx_fc = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=int(fc_steps), freq="MS")
    yhat = pd.Series([last_val] * int(fc_steps), index=idx_fc, name="yhat")
    band = max(0.5 * abs(last_val), 1.0)
    conf = pd.DataFrame({"lower": yhat - band, "upper": yhat + band}, index=idx_fc)
    # Use a flat fitted series for plotting
    fitted_vals = pd.Series(data=[last_val] * len(y), index=y.index)
else:
    sarimax_best = train_sarimax(y, seasonal_periods=int(seasonal_periods))
    ets_model = train_ets(y, seasonal_periods=int(seasonal_periods))

    sarimax_aic = sarimax_best[1] if sarimax_best is not None else np.inf
    ets_aic = getattr(ets_model, "aic", np.inf)

    if sarimax_aic < ets_aic:
        chosen = sarimax_best[0]
        chosen_name = "SARIMAX"
        fitted_vals = chosen.fittedvalues
        fc_res = chosen.get_forecast(steps=int(fc_steps))
        yhat = fc_res.predicted_mean
        conf = fc_res.conf_int(alpha=0.05)
        # Normalize CI shape & names
        if conf.shape[1] >= 2:
            conf = conf.iloc[:, :2]
        conf.columns = ["lower", "upper"]
    else:
        chosen = ets_model
        chosen_name = "ETS"
        fitted_vals = chosen.fittedvalues
        yhat = chosen.forecast(int(fc_steps))
        resid = y - fitted_vals
        sigma = float(np.nanstd(resid))
        if not np.isfinite(sigma) or sigma == 0:
            med = np.nanmedian(resid)
            mad = np.nanmedian(np.abs(resid - med))
            sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else (0.10 * max(1.0, float(np.nanmean(y))))
        conf = pd.DataFrame({"lower": yhat - 1.96 * sigma, "upper": yhat + 1.96 * sigma}, index=yhat.index)

# Non-negative clip for quantities
yhat = _clip_nonneg(yhat)
conf["lower"] = _clip_nonneg(conf["lower"])
conf["upper"] = _clip_nonneg(conf["upper"])

# =========================
# Layout & Plots
# =========================
st.subheader("Time Series & Forecast")
fig_hist, ax = plt.subplots()
ax.plot(y.index, y.values, label="History")
if fitted_vals is not None and len(pd.Series(fitted_vals).dropna()) > 0:
    ax.plot(fitted_vals.index, fitted_vals.values, label="Fitted")
ax.plot(yhat.index, yhat.values, label="Forecast")
ax.fill_between(yhat.index, conf["lower"].values, conf["upper"].values, alpha=0.25, label="95% CI")
ax.set_xlabel("Month")
ax.set_ylabel(str(target_col))
ax.set_title(f"{int(fc_steps)}-Month Forecast ({chosen_name})")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
fig_hist.autofmt_xdate()
ax.legend()
st.pyplot(fig_hist)

left, right = st.columns([1,1])
with left:
    st.subheader("Monthly Series")
    s_df = y.rename("value").to_frame()
    st.dataframe(s_df.style.format({"value": "{:,.0f}"}), use_container_width=True)
with right:
    st.subheader("Forecast Table")
    forecast_tbl = pd.DataFrame({"yhat": yhat, "yhat_lower": conf["lower"], "yhat_upper": conf["upper"]})
    st.dataframe(forecast_tbl.style.format("{:,.0f}"), use_container_width=True)

# Download
csv_bytes = forecast_tbl.to_csv(index_label="month", float_format="%.6f").encode("utf-8")
st.download_button("Download forecast CSV", data=csv_bytes, file_name="forecast_6mo.csv", mime="text/csv")

# =========================
# Backtest (last ≤6 months)
# =========================
try:
    k = min(6, max(0, len(y) - 12))
    if k >= 1:
        y_bt_train = y[:-k]
        if chosen_name == "SARIMAX":
            sarimax_bt = train_sarimax(y_bt_train, seasonal_periods=int(seasonal_periods))
            pred_bt = sarimax_bt[0].get_forecast(steps=k).predicted_mean
        elif chosen_name == "ETS":
            ets_bt = train_ets(y_bt_train, seasonal_periods=int(seasonal_periods))
            pred_bt = ets_bt.forecast(k)
        else:
            pred_bt = pd.Series([float(y_bt_train.iloc[-1])] * k, index=y.index[-k:])

        y_true = y[-k:]
        mae = float(np.mean(np.abs(y_true - pred_bt)))
        mp = float(mape(y_true, pred_bt))

        st.subheader("Backtest (last ≤6 months)")
        st.write(f"**MAE:** {mae:,.2f} &nbsp;&nbsp; **MAPE:** {mp:,.2f}%")

        fig_bt, ax_bt = plt.subplots()
        ax_bt.plot(y_true.index, y_true.values, label="Actual")
        ax_bt.plot(pred_bt.index, pred_bt.values, label="Predicted")
        ax_bt.set_title("Backtest")
        ax_bt.set_xlabel("Month")
        ax_bt.set_ylabel(str(target_col))
        ax_bt.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        fig_bt.autofmt_xdate()
        ax_bt.legend()
        st.pyplot(fig_bt)
except Exception as e:
    st.info(f"(Backtest skipped: {e})")
# forecast_dashboard_streamlit.py
# Streamlit dashboard for monthly forecasting (SARIMAX vs ETS) with safe file upload handling.
import os
import io
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import streamlit as st

warnings.filterwarnings("ignore")

# =========================
# Helpers
# =========================
def autodetect_date_col(df: pd.DataFrame) -> Optional[str]:
    """Pick the column most likely to be a date/time by successful parse count."""
    candidates = [c for c in df.columns if any(tok in str(c).lower() for tok in
                    ["date", "time", "start", "end", "timestamp", "month", "day"])]
    parsed_counts = []
    to_try = candidates if candidates else list(df.columns)
    for c in to_try:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            parsed_counts.append((parsed.notna().sum(), c))
        except Exception:
            continue
    if not parsed_counts:
        return None
    parsed_counts.sort(reverse=True)
    return parsed_counts[0][1]

def autodetect_target_col(df: pd.DataFrame) -> str:
    """Choose a numeric column to forecast, preferring known names, else highest variance."""
    for pref in ["PKG QTY", "Fill Qty"]:
        if pref in df.columns:
            return pref
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        # Try coercing numerics broadly, then re-check
        coerce_df = df.copy()
        for c in df.columns:
            try:
                coerce_df[c] = pd.to_numeric(coerce_df[c], errors="coerce")
            except Exception:
                pass
        num_cols = [c for c in coerce_df.columns if pd.api.types.is_numeric_dtype(coerce_df[c])]
        if not num_cols:
            raise ValueError("No numeric columns found for forecasting.")
        df = coerce_df
    ranked = sorted([(float(df[c].var() or 0), c) for c in num_cols], reverse=True)
    return ranked[0][1]

def monthly_series(df: pd.DataFrame, date_col: str, target_col: str, agg: str) -> pd.Series:
    """Aggregate to monthly series with robust numeric coercion and interpolation for gaps."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # coerce target to numeric even if it contains text like "1,234" or units
    df[target_col] = (df[target_col]
                      .astype(str).str.replace(r"[^\d\.\-eE]", "", regex=True)
                      .replace({"": np.nan})
                      .astype(float))
    df = df.dropna(subset=[date_col, target_col]).sort_values(date_col)
    if df.empty:
        return pd.Series(dtype=float)

    if agg == "sum":
        s = df.groupby(pd.Grouper(key=date_col, freq="MS"))[target_col].sum()
    else:
        s = df.groupby(pd.Grouper(key=date_col, freq="MS"))[target_col].mean()

    s = s.asfreq("MS")
    if s.isna().all():
        # Try alternative aggregation if initial produced all NaN
        alt = "mean" if agg == "sum" else "sum"
        s_alt = df.groupby(pd.Grouper(key=date_col, freq="MS"))[target_col].mean() if alt == "mean" \
                else df.groupby(pd.Grouper(key=date_col, freq="MS"))[target_col].sum()
        s = s_alt.asfreq("MS")

    if s.isna().any():
        s = s.interpolate(limit_direction="both")

    return s.astype(float)

def train_sarimax(y: pd.Series, seasonal_periods: int = 12):
    """Small SARIMAX grid; returns (result, aic) or None."""
    best = (None, np.inf)
    y = y.astype(float)
    seasonal_ok = y.dropna().shape[0] >= 2 * seasonal_periods

    p = d = q = range(0, 2)  # 0,1 keeps it snappy
    seasonal_grid = [(0, 0, 0)] if not seasonal_ok else [(Pi, Di, Qi) for Pi in (0,1) for Di in (0,1) for Qi in (0,1)]

    for pi in p:
        for di in d:
            for qi in q:
                for (Pi, Di, Qi) in seasonal_grid:
                    try:
                        mod = sm.tsa.statespace.SARIMAX(
                            y,
                            order=(pi, di, qi),
                            seasonal_order=(Pi, Di, Qi, seasonal_periods) if seasonal_ok else (0, 0, 0, 0),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        res = mod.fit(disp=False)
                        if res.aic < best[1]:
                            best = (res, res.aic)
                    except Exception:
                        continue
    return best if best[0] is not None else None

def train_ets(y: pd.Series, seasonal_periods: int = 12):
    """Holt-Winters ETS with graceful fallbacks."""
    y = y.astype(float)
    enough_for_season = y.dropna().shape[0] >= 2 * seasonal_periods
    if not enough_for_season:
        return ExponentialSmoothing(
            y, trend="add", seasonal=None, initialization_method="estimated"
        ).fit(optimized=True)
    try:
        return ExponentialSmoothing(
            y, trend="add", seasonal="add",
            seasonal_periods=seasonal_periods, initialization_method="estimated"
        ).fit(optimized=True)
    except Exception:
        if (y > 0).all():
            try:
                return ExponentialSmoothing(
                    y, trend="add", seasonal="mul",
                    seasonal_periods=seasonal_periods, initialization_method="estimated"
                ).fit(optimized=True)
            except Exception:
                pass
        return ExponentialSmoothing(
            y, trend="add", seasonal=None, initialization_method="estimated"
        ).fit(optimized=True)

def mape(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt != 0
    return np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0 if mask.any() else np.nan

def _clip_nonneg(series_like):
    s = pd.Series(series_like, index=getattr(series_like, "index", None), dtype=float)
    return s.clip(lower=0.0)

def _pick_first_non_meta_sheet(xls) -> str:
    for s in xls.sheet_names:
        lname = s.lower()
        if not any(tok in lname for tok in ["readme", "meta", "dictionary", "schema"]):
            return s
    return xls.sheet_names[0]

def load_table_from_upload(buffer: io.BytesIO, filename: str, sheet_name=None) -> Tuple[pd.DataFrame, Optional[str]]:
    """Load DataFrame from uploaded file (BytesIO). Returns (df, used_sheet)."""
    ext = os.path.splitext(filename)[1].lower()
    buffer.seek(0)
    if ext == ".csv":
        return pd.read_csv(buffer), None

    # Excel handling
    buffer.seek(0)
    xls = pd.ExcelFile(buffer)
    if isinstance(sheet_name, (str, int)) and sheet_name != "":
        use = sheet_name if not isinstance(sheet_name, str) or sheet_name in xls.sheet_names else _pick_first_non_meta_sheet(xls)
    else:
        use = _pick_first_non_meta_sheet(xls)
    buffer.seek(0)
    df = pd.read_excel(buffer, sheet_name=use)
    if isinstance(df, dict):
        key = list(df.keys())[0]
        return df[key], key
    return df, use

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Syringe Line Forecast Dashboard", layout="wide")
st.title("🧪 Syringe Line Forecast Dashboard")
st.caption("Upload a CSV or Excel file and configure columns. The app builds a monthly series, compares SARIMAX vs ETS via AIC, and produces a 6-month forecast with a quick backtest.")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload data (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])
    default_sheet = st.text_input("Sheet name (Excel only, optional)", value="")
    date_col_override = st.text_input("Date column (optional)", value="Fill Scheduled Start Date")
    target_col_override = st.text_input("Target column (optional)", value="PKG QTY")
    agg_func = st.selectbox("Monthly aggregation", options=["sum", "mean"], index=0)
    fc_steps = st.number_input("Forecast months ahead", min_value=1, max_value=36, value=6, step=1)
    seasonal_periods = st.number_input("Seasonal periods (12=annual)", min_value=1, max_value=24, value=12, step=1)

# Guard: require an upload before proceeding
if uploaded is None:
    st.info("👆 Upload a CSV or Excel file in the sidebar to begin.")
    st.stop()

# Read uploaded file into memory
file_bytes = uploaded.getvalue()
buf = io.BytesIO(file_bytes)

# Load DataFrame
df, used_sheet = load_table_from_upload(buf, uploaded.name, default_sheet.strip() or None)

# Column picks (with autodetect fallback)
date_col = date_col_override if (date_col_override in df.columns) else autodetect_date_col(df)
target_col = target_col_override if (target_col_override in df.columns) else autodetect_target_col(df)

# Metrics
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.metric("Rows", f"{len(df):,}")
with mcol2:
    st.metric("Columns", f"{len(df.columns):,}")
with mcol3:
    st.metric("Date column", str(date_col))
with mcol4:
    st.metric("Target column", str(target_col))

if date_col is None or target_col is None:
    st.error("Could not detect a valid date or target column. Adjust the sidebar inputs.")
    st.stop()

# Build monthly series
y = monthly_series(df, date_col, target_col, agg_func)
if y.empty:
    st.warning("No usable monthly data from the chosen columns/aggregation. Try a different column or aggregation.")
    st.stop()

# Forecasting
near_constant = (y.nunique(dropna=True) <= 1) or (np.nanstd(y.values) < 1e-6)
if near_constant:
    chosen_name = "Naive (flat)"
    last_val = float(y.iloc[-1])
    idx_fc = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=int(fc_steps), freq="MS")
    yhat = pd.Series([last_val] * int(fc_steps), index=idx_fc, name="yhat")
    band = max(0.5 * abs(last_val), 1.0)
    conf = pd.DataFrame({"lower": yhat - band, "upper": yhat + band}, index=idx_fc)
    # Use a flat fitted series for plotting
    fitted_vals = pd.Series(data=[last_val] * len(y), index=y.index)
else:
    sarimax_best = train_sarimax(y, seasonal_periods=int(seasonal_periods))
    ets_model = train_ets(y, seasonal_periods=int(seasonal_periods))

    sarimax_aic = sarimax_best[1] if sarimax_best is not None else np.inf
    ets_aic = getattr(ets_model, "aic", np.inf)

    if sarimax_aic < ets_aic:
        chosen = sarimax_best[0]
        chosen_name = "SARIMAX"
        fitted_vals = chosen.fittedvalues
        fc_res = chosen.get_forecast(steps=int(fc_steps))
        yhat = fc_res.predicted_mean
        conf = fc_res.conf_int(alpha=0.05)
        # Normalize CI shape & names
        if conf.shape[1] >= 2:
            conf = conf.iloc[:, :2]
        conf.columns = ["lower", "upper"]
    else:
        chosen = ets_model
        chosen_name = "ETS"
        fitted_vals = chosen.fittedvalues
        yhat = chosen.forecast(int(fc_steps))
        resid = y - fitted_vals
        sigma = float(np.nanstd(resid))
        if not np.isfinite(sigma) or sigma == 0:
            med = np.nanmedian(resid)
            mad = np.nanmedian(np.abs(resid - med))
            sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else (0.10 * max(1.0, float(np.nanmean(y))))
        conf = pd.DataFrame({"lower": yhat - 1.96 * sigma, "upper": yhat + 1.96 * sigma}, index=yhat.index)

# Non-negative clip for quantities
yhat = _clip_nonneg(yhat)
conf["lower"] = _clip_nonneg(conf["lower"])
conf["upper"] = _clip_nonneg(conf["upper"])

# =========================
# Layout & Plots
# =========================
st.subheader("Time Series & Forecast")
fig_hist, ax = plt.subplots()
ax.plot(y.index, y.values, label="History")
if fitted_vals is not None and len(pd.Series(fitted_vals).dropna()) > 0:
    ax.plot(fitted_vals.index, fitted_vals.values, label="Fitted")
ax.plot(yhat.index, yhat.values, label="Forecast")
ax.fill_between(yhat.index, conf["lower"].values, conf["upper"].values, alpha=0.25, label="95% CI")
ax.set_xlabel("Month")
ax.set_ylabel(str(target_col))
ax.set_title(f"{int(fc_steps)}-Month Forecast ({chosen_name})")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
fig_hist.autofmt_xdate()
ax.legend()
st.pyplot(fig_hist)

left, right = st.columns([1,1])
with left:
    st.subheader("Monthly Series")
    s_df = y.rename("value").to_frame()
    st.dataframe(s_df.style.format({"value": "{:,.0f}"}), use_container_width=True)
with right:
    st.subheader("Forecast Table")
    forecast_tbl = pd.DataFrame({"yhat": yhat, "yhat_lower": conf["lower"], "yhat_upper": conf["upper"]})
    st.dataframe(forecast_tbl.style.format("{:,.0f}"), use_container_width=True)

# Download
csv_bytes = forecast_tbl.to_csv(index_label="month", float_format="%.6f").encode("utf-8")
st.download_button("Download forecast CSV", data=csv_bytes, file_name="forecast_6mo.csv", mime="text/csv")

# =========================
# Backtest (last ≤6 months)
# =========================
try:
    k = min(6, max(0, len(y) - 12))
    if k >= 1:
        y_bt_train = y[:-k]
        if chosen_name == "SARIMAX":
            sarimax_bt = train_sarimax(y_bt_train, seasonal_periods=int(seasonal_periods))
            pred_bt = sarimax_bt[0].get_forecast(steps=k).predicted_mean
        elif chosen_name == "ETS":
            ets_bt = train_ets(y_bt_train, seasonal_periods=int(seasonal_periods))
            pred_bt = ets_bt.forecast(k)
        else:
            pred_bt = pd.Series([float(y_bt_train.iloc[-1])] * k, index=y.index[-k:])

        y_true = y[-k:]
        mae = float(np.mean(np.abs(y_true - pred_bt)))
        mp = float(mape(y_true, pred_bt))

        st.subheader("Backtest (last ≤6 months)")
        st.write(f"**MAE:** {mae:,.2f} &nbsp;&nbsp; **MAPE:** {mp:,.2f}%")

        fig_bt, ax_bt = plt.subplots()
        ax_bt.plot(y_true.index, y_true.values, label="Actual")
        ax_bt.plot(pred_bt.index, pred_bt.values, label="Predicted")
        ax_bt.set_title("Backtest")
        ax_bt.set_xlabel("Month")
        ax_bt.set_ylabel(str(target_col))
        ax_bt.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        fig_bt.autofmt_xdate()
        ax_bt.legend()
        st.pyplot(fig_bt)
except Exception as e:
    st.info(f"(Backtest skipped: {e})")


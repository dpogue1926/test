

import os
import math
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# ---------- CONFIG ----------
DEFAULT_FILE   = r"C:\Users\Deshawn.Pogue\Downloads\NN Syringe Line 2.xlsx"
DEFAULT_SHEET  = "query (10)"     # or None to auto-pick first sheet
PINNED_TARGET  = "PKG QTY"        # will auto-fallback if unusable
TEST_FRACTION  = 0.20
MODEL_PATH     = "model_sl2.pkl"

PREFERRED_DATE_NAMES = [
    "Fill Scheduled Start Date", "PKG Actual Start Date", "Fill Actual Start Date",
    "Date", "Start Date", "End Date"
]
TARGET_FALLBACKS = ["PKG QTY", "Fill Qty", "Quantity", "Qty"]


# ---------- Helpers ----------
def load_table(path: str, sheet: Optional[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path), None
    df = pd.read_excel(path, sheet_name=sheet if sheet else 0)
    if isinstance(df, dict):  # defensive
        key = list(df.keys())[0]
        return df[key], key
    return df, (sheet if sheet else pd.ExcelFile(path).sheet_names[0])


def autodetect_date_col(df: pd.DataFrame) -> Optional[str]:
    # Prefer common names first
    for c in PREFERRED_DATE_NAMES:
        if c in df.columns:
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except Exception:
                pass
    # Otherwise pick the column that parses best
    candidates = [c for c in df.columns if any(tok in str(c).lower()
                   for tok in ["date", "time", "start", "end", "timestamp", "month", "day"])]
    scores = []
    for c in (candidates or list(df.columns)):
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            scores.append((parsed.notna().sum(), c))
        except Exception:
            continue
    if not scores:
        return None
    scores.sort(reverse=True)
    return scores[0][1]


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce strings like '1,234', '(123)', 'abc 45.6' to numeric."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    x = s.astype(str).str.strip()
    x = x.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (123) -> -123
    x = x.str.replace(",", "", regex=False)
    x = x.str.replace("%", "", regex=False)
    extracted = x.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def pick_target(df: pd.DataFrame, pinned: Optional[str]) -> str:
    candidates: List[str] = []
    if pinned:
        candidates.append(pinned)
    for c in TARGET_FALLBACKS:
        if c not in candidates:
            candidates.append(c)
    # add numeric by variance as last resort
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        ranked = sorted([(float(df[c].var() or 0), c) for c in num_cols], reverse=True)
        for _, c in ranked:
            if c not in candidates:
                candidates.append(c)
    for cand in candidates:
        if cand in df.columns:
            s = clean_numeric_series(df[cand])
            if s.notna().sum() >= 3:
                return cand
    return pinned or df.columns[0]


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = r2_score(y_true, y_pred) if np.unique(y_true).size > 1 else float("nan")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # no squared=; works on older sklearn
    return r2, mae, rmse


def make_ohe():
    """Construct OneHotEncoder across sklearn versions."""
    try:
        # Newer sklearn (>=1.2+ prefers sparse_output)
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Older sklearn: use 'sparse'
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe()),
    ])
    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop"
    )
    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    return Pipeline([("prep", preproc), ("model", model)])


def get_ohe_feature_names(prep, categorical_cols):
    try:
        ohe = prep.named_transformers_["cat"].named_steps["ohe"]
        if hasattr(ohe, "get_feature_names_out"):
            return ohe.get_feature_names_out(categorical_cols).tolist()
        if hasattr(ohe, "get_feature_names"):
            return ohe.get_feature_names(categorical_cols).tolist()
    except Exception:
        pass
    return []


# ---------- Train & Save ----------
def train_and_save(path=DEFAULT_FILE, sheet=DEFAULT_SHEET, pinned_target=PINNED_TARGET, out_model=MODEL_PATH):
    df, used_sheet = load_table(path, sheet)
    print(f"Loaded: {path}" + (f" (sheet: {used_sheet})" if used_sheet else ""))

    target = pick_target(df, pinned_target)
    y_series = clean_numeric_series(df[target])
    usable = y_series.notna()
    print(f"Using target: {target} | rows with numeric target: {usable.sum()} / {len(df)}")

    if usable.sum() == 0:
        print("No numeric target rows found. Saving baseline constant model (0.0).")
        joblib.dump({"type": "baseline", "value": 0.0, "target": target}, out_model)
        print(f"Saved baseline model -> {out_model}")
        return

    df2 = df.loc[usable].copy()
    df2[target] = y_series.loc[usable].astype(float)

    # Identify date column for time-ordered split
    date_col = autodetect_date_col(df2)
    if date_col:
        print(f"Detected date column for split: {date_col}")
        dt = pd.to_datetime(df2[date_col], errors="coerce")
        order = np.argsort(dt.fillna(pd.Timestamp.min).values)
        df2 = df2.iloc[order].reset_index(drop=True)

    # Features/target
    X_full = df2.drop(columns=[target])
    y_full = df2[target].astype(float)

    # Drop the date column from features if present
    if date_col and date_col in X_full.columns:
        X_full = X_full.drop(columns=[date_col])

    # Time split if date col present; else random split
    if date_col:
        split_idx = int((1 - TEST_FRACTION) * len(X_full))
        X_train, X_test = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
        y_train, y_test = y_full.iloc[:split_idx], y_full.iloc[split_idx:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=TEST_FRACTION, random_state=42
        )

    # If training empty, borrow from test
    if len(X_train) == 0 and len(X_test) > 0:
        X_train = X_test.iloc[:1].copy()
        y_train = y_test.iloc[:1].copy()
        X_test = X_test.iloc[1:].copy()
        y_test = y_test.iloc[1:].copy()

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    if len(X_train) == 0:
        median_val = float(y_full.median()) if len(y_full) else 0.0
        print(f"No trainable rows. Saving baseline median model ({median_val}).")
        joblib.dump({"type": "baseline", "value": median_val, "target": target}, out_model)
        print(f"Saved baseline model -> {out_model}")
        if len(X_test):
            preds = np.full(len(X_test), median_val)
            out = X_test.copy()
            out[target + "_actual"] = y_test.values
            out[target + "_pred"] = preds
            out.to_csv("test_predictions.csv", index=False)
            print("Saved test_predictions.csv (baseline).")
        return

    # Drop columns that are entirely missing in training (avoid imputer warnings)
    all_missing_cols = [c for c in X_train.columns if X_train[c].isna().all()]
    if all_missing_cols:
        X_train = X_train.drop(columns=all_missing_cols)
        X_test = X_test.drop(columns=all_missing_cols, errors="ignore")

    # Keep date-like names out of numeric features
    date_like_tokens = ("date", "time", "start", "end", "timestamp", "month", "day")
    date_like_cols = [c for c in X_train.columns if any(tok in str(c).lower() for tok in date_like_tokens)]

    numeric_cols = [c for c in X_train.columns
                    if pd.api.types.is_numeric_dtype(X_train[c]) and c not in date_like_cols]
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    print(f"Numeric features: {len(numeric_cols)}, Categorical features: {len(categorical_cols)}")

    pipe = build_pipeline(numeric_cols, categorical_cols)
    pipe.fit(X_train, y_train)

    # Evaluate if we have a holdout
    if len(X_test) > 0:
        y_pred = pipe.predict(X_test)
        r2, mae, rmse = evaluate(y_test, y_pred)
        print(f"R²: {r2:.3f} | MAE: {mae:.3f} | RMSE: {rmse:.3f}")

        preds_df = X_test.copy()
        preds_df[target + "_actual"] = y_test.values
        preds_df[target + "_pred"] = y_pred
        preds_df.to_csv("test_predictions.csv", index=False)
        print("Saved test_predictions.csv")
    else:
        print("No holdout rows — skipped test evaluation.")

    # Save model bundle
    joblib.dump({
        "type": "sklearn",
        "pipeline": pipe,
        "target": target,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "date_col": date_col,
        "used_sheet": used_sheet,
    }, out_model)
    print(f"Saved model -> {out_model}")

    # Feature importances (if available)
    try:
        pipe.fit(pd.concat([X_train, X_test], axis=0),
                 pd.concat([y_train, y_test], axis=0) if len(X_test) > 0 else y_train)
        model = pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            prep = pipe.named_steps["prep"]
            cat_names = get_ohe_feature_names(prep, categorical_cols)
            feat_names = list(numeric_cols) + cat_names
            importances = model.feature_importances_
            k = min(len(feat_names), len(importances))
            fsum = pd.DataFrame({"feature": feat_names[:k], "importance": importances[:k]}) \
                     .sort_values("importance", ascending=False)
            fsum.to_csv("feature_summary.csv", index=False)
            print("Saved feature_summary.csv")
    except Exception as e:
        print(f"(Skipping feature summary: {e})")


# ---------- Batch Inference ----------
def predict_new(model_path: str, data_path: str, sheet: Optional[str] = None, out_csv: str = "predictions_new_data.csv"):
    bundle = joblib.load(model_path)
    target = bundle.get("target", "target")

    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".csv":
        new_df = pd.read_csv(data_path)
    else:
        new_df = pd.read_excel(data_path, sheet_name=sheet if sheet else 0)
        if isinstance(new_df, dict):
            new_df = list(new_df.values())[0]

    if bundle["type"] == "baseline":
        preds = np.full(len(new_df), float(bundle["value"]))
        out = new_df.copy()
        out[target + "_pred"] = preds
        out.to_csv(out_csv, index=False)
        print(f"[baseline] Saved predictions -> {out_csv}")
        return

    date_col = bundle.get("date_col")
    newX = new_df.drop(columns=[date_col], errors="ignore")
    pipe = bundle["pipeline"]
    preds = pipe.predict(newX)
    out = new_df.copy()
    out[target + "_pred"] = preds
    out.to_csv(out_csv, index=False)
    print(f"[sklearn] Saved predictions -> {out_csv}")


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Robust training and prediction for NN Syringe Line 2.")
    ap.add_argument("--file", type=str, default=DEFAULT_FILE, help="Path to XLSX/CSV")
    ap.add_argument("--sheet", type=str, default=DEFAULT_SHEET, help="Excel sheet name (ignored for CSV)")
    ap.add_argument("--target", type=str, default=PINNED_TARGET, help="Pinned target (auto-fallback if unusable)")
    ap.add_argument("--model", type=str, default=MODEL_PATH, help="Output model path")
    ap.add_argument("--predict", type=str, help="Path to new data (XLSX/CSV) for batch prediction")
    ap.add_argument("--predict-sheet", type=str, help="Sheet name for new Excel data")
    args = ap.parse_args()

    if args.predict:
        if not os.path.exists(args.model):
            train_and_save(path=args.file, sheet=args.sheet, pinned_target=args.target, out_model=args.model)
        predict_new(args.model, args.predict, sheet=args.predict_sheet)
    else:
        train_and_save(path=args.file, sheet=args.sheet, pinned_target=args.target, out_model=args.model)




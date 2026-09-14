# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Collections forecasting: public baseline
#
# This is a deliberately limited, reproducible benchmark suitable for GitHub. It uses only the aggregate country-month feature table and asks one question: **does a later forecasting model improve on simply carrying forward the recent three-month average?**
#
# Scope: this notebook contains no company data, no row-level features, no final modelling pipeline, and no saved forecast values. Keep all outputs cleared before committing it.

# %% [markdown]
# ## Method
#
# - Estimation window ends 31 March 2026.
# - Apr–Jun 2026 is the visible validation quarter.
# - Jul–Sep 2026 actuals must not be present.
# - The benchmark forecast is the mean of the latest three observed country collection months.
# - MAE, WAPE and signed Bias are reported for validation.
#
# This benchmark is intentionally not the final answer. It creates a transparent floor that a cohort/vintage model must beat.

# %%
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

ESTIMATION_END = pd.Timestamp("2026-03-31")
VALIDATION_END = pd.Timestamp("2026-06-30")
FORECAST_MONTHS = pd.date_range("2026-07-31", "2026-09-30", freq="ME")


# %% tags=["public_core"]
def score_monthly(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    error = forecast - actual
    denom = np.abs(actual).sum()
    return {
        "MAE": float(np.mean(np.abs(error))),
        "WAPE": float(np.abs(error).sum() / denom) if denom else np.nan,
        "Bias": float(error.sum() / denom) if denom else np.nan,
    }


def assert_no_sealed_actuals(frame, cutoff=pd.Timestamp("2026-06-30")):
    max_month = pd.to_datetime(frame["month"], errors="coerce").max()
    if pd.notna(max_month) and max_month > cutoff:
        raise AssertionError(f"Sealed-period leakage: found {max_month.date()}")
    return True


def naive_forecast(history, origin, forecast_months, lookback=3):
    origin = pd.Timestamp(origin)
    past = history.loc[
        pd.to_datetime(history["month"]).le(origin), "source_reported_cash_usd"
    ].tail(lookback)
    if len(past) < lookback:
        raise ValueError(f"Need at least {lookback} historical months")
    return pd.DataFrame({"month": pd.to_datetime(forecast_months), "forecast": float(past.mean())})


# %%
try:
    from google.colab import files
    uploaded = files.upload()
    zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
    if not zip_names:
        raise FileNotFoundError("Upload dlight_feature_engineering_outputs_v3.zip")
    feature_zip = io.BytesIO(uploaded[zip_names[0]])
except ImportError:
    feature_zip = Path("dlight_feature_engineering_outputs_v3.zip")

with zipfile.ZipFile(feature_zip, "r") as zf:
    member = next(name for name in zf.namelist() if name.endswith("country_month_features_v3.csv"))
    with zf.open(member) as stream:
        country_month = pd.read_csv(stream)

country_month["month"] = pd.to_datetime(country_month["month"])
required = {"month", "source_reported_cash_usd"}
if required - set(country_month.columns):
    raise AssertionError(f"Missing columns: {sorted(required - set(country_month.columns))}")
assert_no_sealed_actuals(country_month)
print("Aggregate history loaded. Latest month:", country_month["month"].max().date())

# %%
validation_months = pd.date_range("2026-04-30", "2026-06-30", freq="ME")
validation_forecast = naive_forecast(country_month, ESTIMATION_END, validation_months)
validation_actual = country_month.set_index("month").reindex(validation_months)["source_reported_cash_usd"]
baseline_validation = validation_forecast.copy()
baseline_validation["actual"] = validation_actual.to_numpy()
baseline_metrics = pd.DataFrame([{"model": "recent_3_month_average", **score_monthly(baseline_validation["actual"], baseline_validation["forecast"])}])

# Final naive floor uses the latest visible three months. Q3 actuals remain absent.
github_baseline_forecast_q3_2026 = naive_forecast(country_month, VALIDATION_END, FORECAST_MONTHS)

display(baseline_validation.style.format({"forecast": "${:,.0f}", "actual": "${:,.0f}"}))
display(baseline_metrics.style.format({"MAE": "${:,.0f}", "WAPE": "{:.1%}", "Bias": "{:.1%}"}))
display(github_baseline_forecast_q3_2026.style.format({"forecast": "${:,.0f}"}))

# %%
baseline_validation.to_csv("baseline_validation.csv", index=False)
baseline_metrics.to_csv("baseline_model_metrics.csv", index=False)
github_baseline_forecast_q3_2026.to_csv("github_baseline_forecast_q3_2026.csv", index=False)

try:
    from google.colab import files
    for name in ["baseline_validation.csv", "baseline_model_metrics.csv", "github_baseline_forecast_q3_2026.csv"]:
        files.download(name)
except ImportError:
    print("CSV files written to the current directory.")

# %% [markdown]
# ## Before committing to GitHub
#
# Use **Edit → Clear all outputs**, confirm that no CSV/ZIP/data file is staged, and commit only this notebook.

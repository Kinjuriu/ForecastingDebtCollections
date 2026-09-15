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

# %% [markdown] id="m4U-DikZBKwv"
# # Q3 2026 collections forecast — working notebook
#
# This notebook starts from the frozen `dlight_feature_engineering_outputs_v3.zip`. It does not reopen cleaning or feature engineering, and it never asks for or stores AI-assistant conversations. It writes the Q3 forecast before any Jul–Sep actuals are available.
#
# Forecast story: one economically grounded cohort/vintage model, checked against a naive floor and damped Holt, with a direct-horizon Ridge model used only as a challenger.

# %% [markdown] id="XRprS3OMBKw9"
# ## Run order
#
# 1. Run the setup cell.
# 2. Upload `dlight_feature_engineering_outputs_v3.zip` when prompted.
# 3. Run all remaining cells.
# 4. Download the output ZIP only after the QA table says **PASS**.
#
# The notebook will reject any feature file containing rows after 30 June 2026. Q3 actuals are never loaded.

# %% colab={"base_uri": "https://localhost:8080/"} id="TzChOfaQBKw_" outputId="f7d1946a-a353-4386-b09a-45c6ccff63b6"
# %pip -q install statsmodels scikit-learn

import io
import json
import math
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 160)
pd.set_option("display.width", 220)

ANALYSIS_START = pd.Timestamp("2024-10-31")
ESTIMATION_END = pd.Timestamp("2026-03-31")
VALIDATION_END = pd.Timestamp("2026-06-30")
FORECAST_MONTHS = pd.date_range("2026-07-31", "2026-09-30", freq="ME")
AVG_DAYS_PER_MONTH = 365.25 / 12
Q3_SALES_PLAN = {
    pd.Timestamp("2026-07-31"): 3100,
    pd.Timestamp("2026-08-31"): 3200,
    pd.Timestamp("2026-09-30"): 3200,
}
OUTPUT_DIR = Path("/content/dlight_q3_forecast_private")
OUTPUT_DIR.mkdir(exist_ok=True)

RUN_CHALLENGER = True
SALES_PLAN_STRESS = {"low": 0.95, "base": 1.00, "high": 1.05}
print("Setup complete. Private outputs will be written to", OUTPUT_DIR)

# %% colab={"base_uri": "https://localhost:8080/", "height": 109} id="Wv63_Ow3BKxC" outputId="ead81116-f65a-47e7-e783-3d1d24f695af"
try:
    from google.colab import files
    uploaded = files.upload()
    zip_candidates = [name for name in uploaded if name.lower().endswith(".zip")]
    if not zip_candidates:
        raise FileNotFoundError("Upload dlight_feature_engineering_outputs_v3.zip")
    preferred = [name for name in zip_candidates if "feature_engineering_outputs_v3" in name.lower()]
    feature_zip_name = preferred[0] if preferred else zip_candidates[0]
    feature_zip_bytes = io.BytesIO(uploaded[feature_zip_name])
except ImportError:
    feature_zip_path = Path("dlight_feature_engineering_outputs_v3.zip")
    if not feature_zip_path.exists():
        raise FileNotFoundError(
            "Place dlight_feature_engineering_outputs_v3.zip beside the notebook."
        )
    feature_zip_bytes = feature_zip_path

EXTRACT_DIR = Path("/content/dlight_feature_outputs_v3")
EXTRACT_DIR.mkdir(exist_ok=True)
with zipfile.ZipFile(feature_zip_bytes, "r") as zf:
    zf.extractall(EXTRACT_DIR)

def find_output(name):
    matches = list(EXTRACT_DIR.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Missing required feature output: {name}")
    return matches[0]

panel = pd.read_csv(find_output("contract_month_features_v3.csv"))
country_month = pd.read_csv(find_output("country_month_features_v3.csv"))
region_month = pd.read_csv(find_output("region_month_features_v3.csv"))
reconciliation = pd.read_csv(find_output("payment_reconciliation_by_month_v3.csv"))

for frame in (panel, country_month, region_month, reconciliation):
    frame["month"] = pd.to_datetime(frame["month"])
panel["sales_month"] = pd.to_datetime(panel["sales_month"])

print("Loaded panel:", panel.shape)
print("Loaded country-month table:", country_month.shape)

# %% tags=["model_core"] id="BVJaOcE8BKxF"
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def month_difference(later, earlier):
    later = pd.to_datetime(later)
    earlier = pd.to_datetime(earlier)
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def score_monthly(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if len(actual) != len(forecast) or len(actual) == 0:
        raise ValueError("actual and forecast must be non-empty and equally sized")
    error = forecast - actual
    denom = np.abs(actual).sum()
    return {
        "MAE": float(np.mean(np.abs(error))),
        "WAPE": float(np.abs(error).sum() / denom) if denom else np.nan,
        "Bias": float(error.sum() / denom) if denom else np.nan,
        "Bias_USD": float(error.sum()),
    }


def assert_no_sealed_actuals(*frames, cutoff=pd.Timestamp("2026-06-30")):
    for frame in frames:
        if "month" not in frame.columns:
            continue
        max_month = pd.to_datetime(frame["month"], errors="coerce").max()
        if pd.notna(max_month) and max_month > cutoff:
            raise AssertionError(
                f"Sealed-period leakage: found actual month {max_month.date()} after {cutoff.date()}"
            )
    return True


def expected_due_for_mob(price, deposit, daily_amount, tenor_days, mob):
    price = float(price) if pd.notna(price) else 0.0
    deposit = float(deposit) if pd.notna(deposit) else 0.0
    daily_amount = float(daily_amount) if pd.notna(daily_amount) else 0.0
    tenor_days = float(tenor_days) if pd.notna(tenor_days) else 0.0
    mob = int(mob)
    if mob < 0:
        return 0.0
    if mob == 0:
        return min(price, max(0.0, deposit))
    elapsed_before = min(tenor_days, max(0.0, (mob - 1) * AVG_DAYS_PER_MONTH))
    elapsed_through = min(tenor_days, max(0.0, mob * AVG_DAYS_PER_MONTH))
    due = daily_amount * max(0.0, elapsed_through - elapsed_before)
    return min(max(0.0, price), max(0.0, due))


def naive_forecast(history, origin, forecast_months, lookback=3):
    origin = pd.Timestamp(origin)
    past = history.loc[
        pd.to_datetime(history["month"]).le(origin), "source_reported_cash_usd"
    ].tail(lookback)
    if len(past) < lookback:
        raise ValueError(f"Need at least {lookback} historical months for naive forecast")
    return pd.DataFrame({
        "month": pd.to_datetime(forecast_months),
        "forecast": float(past.mean()),
    })


def holt_forecast(history, origin, forecast_months):
    from statsmodels.tsa.holtwinters import Holt
    origin = pd.Timestamp(origin)
    y = history.loc[
        pd.to_datetime(history["month"]).le(origin), "source_reported_cash_usd"
    ].astype(float)
    if len(y) < 6:
        raise ValueError("Need at least six months for damped Holt")
    fitted = Holt(y.to_numpy(), damped_trend=True, initialization_method="estimated").fit(
        optimized=True
    )
    steps = max(month_difference(pd.Timestamp(m), origin) for m in forecast_months)
    all_future = np.asarray(fitted.forecast(steps), dtype=float)
    values = [all_future[month_difference(pd.Timestamp(m), origin) - 1] for m in forecast_months]
    return pd.DataFrame({"month": pd.to_datetime(forecast_months), "forecast": values})


def make_ridge_pipeline(numeric_features, categorical_features, alpha=10.0):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, list(numeric_features)),
        ("categorical", categorical_pipe, list(categorical_features)),
    ])
    return Pipeline([("prep", prep), ("model", Ridge(alpha=alpha, solver="lsqr"))])


def validate_scenario_table(table):
    required = {"month", "low", "base", "high"}
    missing = required - set(table.columns)
    if missing:
        raise AssertionError(f"Missing forecast columns: {sorted(missing)}")
    if table[list(required - {"month"})].isna().any().any():
        raise AssertionError("Scenario forecast contains missing values")
    if not ((table["low"] <= table["base"]) & (table["base"] <= table["high"])).all():
        raise AssertionError("Scenario ordering must be low <= base <= high")
    return True


# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="w_fnBs_MBKxG" outputId="8245465a-43b7-42c2-bfd9-4284176efb9e"
REQUIRED_PANEL_COLUMNS = {
    "contractid", "month", "sales_month", "region", "contract_type",
    "product", "payment_frequency", "price_usd", "deposit_usd",
    "daily_amount_usd", "tenor_length", "months_on_book",
    "expected_cash_due_this_month_usd", "actual_cumulative_cash_through_month_usd",
    "actual_cumulative_cash_before_month_usd", "target_payment_usd",
    "within_scheduled_tenor_proxy", "post_tenor_recovery_proxy",
    "payment_history_reliable",
}
missing = REQUIRED_PANEL_COLUMNS - set(panel.columns)
if missing:
    raise AssertionError(f"Feature panel is missing required columns: {sorted(missing)}")

assert_no_sealed_actuals(panel, country_month, region_month, reconciliation)
if panel.duplicated(["contractid", "month"]).any():
    raise AssertionError("Duplicate contract-month keys found")
if panel["target_payment_usd"].dropna().lt(0).any():
    raise AssertionError("Negative targets found")

outreach_tokens = {"channel", "attempts", "reached", "cost_usd", "contact_month"}
forbidden = [c for c in panel.columns if c in outreach_tokens or c.startswith("outreach_")]
if forbidden:
    raise AssertionError(f"Outreach-treatment fields entered the base panel: {forbidden}")

qa_input = pd.DataFrame([
    {"check": "sealed_q3_actuals_absent", "status": "PASS"},
    {"check": "contract_month_key_unique", "status": "PASS"},
    {"check": "negative_targets_absent", "status": "PASS"},
    {"check": "outreach_fields_absent", "status": "PASS"},
])
display(qa_input)


# %% tags=["cohort_core"] id="W5gz6rF1BKxH"
def _bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype("string").str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False)


for col in ["within_scheduled_tenor_proxy", "post_tenor_recovery_proxy", "payment_history_reliable"]:
    panel[col] = _bool_series(panel[col]).astype(bool)


def fit_cohort_parameters(panel, country_month, cutoff, shrinkage_contracts=200):
    cutoff = pd.Timestamp(cutoff)
    train = panel[(panel["month"] <= cutoff) & panel["payment_history_reliable"]].copy()
    scheduled = train[
        train["within_scheduled_tenor_proxy"]
        & train["expected_cash_due_this_month_usd"].gt(0)
        & train["target_payment_usd"].notna()
    ].copy()

    country = scheduled.groupby(["contract_type", "months_on_book"], as_index=False).agg(
        actual=("target_payment_usd", "sum"),
        expected=("expected_cash_due_this_month_usd", "sum"),
        contracts=("contractid", "nunique"),
    )
    country["eff"] = (country["actual"] / country["expected"].replace(0, np.nan)).clip(0, 1.50)
    country_map = {(r.contract_type, int(r.months_on_book)): float(r.eff) for r in country.itertuples()}

    region = scheduled.groupby(["region", "contract_type", "months_on_book"], as_index=False).agg(
        actual=("target_payment_usd", "sum"),
        expected=("expected_cash_due_this_month_usd", "sum"),
        contracts=("contractid", "nunique"),
    )
    region["country_eff"] = [country_map.get((t, int(m)), np.nan) for t, m in zip(region["contract_type"], region["months_on_book"])]
    weight = region["contracts"] / (region["contracts"] + shrinkage_contracts)
    raw = region["actual"] / region["expected"].replace(0, np.nan)
    region["eff"] = (weight * raw + (1 - weight) * region["country_eff"]).clip(0, 1.50)
    region_map = {(r.region, r.contract_type, int(r.months_on_book)): float(r.eff) for r in region.itertuples()}

    type_default = scheduled.groupby("contract_type").apply(
        lambda x: x["target_payment_usd"].sum() / x["expected_cash_due_this_month_usd"].sum()
    ).clip(0, 1.50).to_dict()

    recovery = train[
        train["post_tenor_recovery_proxy"]
        & train["target_payment_usd"].notna()
    ].copy()
    recovery["balance_before"] = (
        recovery["price_usd"] - recovery["actual_cumulative_cash_before_month_usd"]
    ).clip(lower=0)
    recovery["months_past_tenor"] = (
        recovery["months_on_book"]
        - np.ceil(recovery["tenor_length"] / AVG_DAYS_PER_MONTH).fillna(0).astype(int)
    ).clip(lower=0, upper=12).astype(int)
    rec = recovery.groupby("months_past_tenor", as_index=False).agg(
        actual=("target_payment_usd", "sum"), balance=("balance_before", "sum"), rows=("contractid", "size")
    )
    overall_rec = float(recovery["target_payment_usd"].sum() / recovery["balance_before"].sum()) if recovery["balance_before"].sum() else 0.0
    rec["rate"] = ((rec["actual"] + 500 * overall_rec) / (rec["balance"] + 500)).clip(0, 0.50)
    recovery_map = {int(r.months_past_tenor): float(r.rate) for r in rec.itertuples()}

    bridge_hist = country_month[country_month["month"] <= cutoff].tail(6)["source_minus_attributable_cash_usd"].dropna()
    bridge_level = float(bridge_hist.mean()) if len(bridge_hist) else 0.0
    return {
        "cutoff": cutoff, "country_eff": country_map, "region_eff": region_map,
        "type_default": type_default, "recovery": recovery_map,
        "recovery_default": overall_rec, "bridge_level": bridge_level,
    }


def lookup_eff(params, region, contract_type, mob):
    key = (region, contract_type, int(mob))
    if key in params["region_eff"] and np.isfinite(params["region_eff"][key]):
        return params["region_eff"][key]
    country_key = (contract_type, int(mob))
    if country_key in params["country_eff"] and np.isfinite(params["country_eff"][country_key]):
        return params["country_eff"][country_key]
    candidates = [(abs(m - int(mob)), value) for (t, m), value in params["country_eff"].items() if t == contract_type and np.isfinite(value)]
    if candidates:
        return min(candidates, key=lambda x: x[0])[1]
    return float(params["type_default"].get(contract_type, 0.0))


def lookup_recovery(params, months_past_tenor):
    bucket = int(np.clip(months_past_tenor, 0, 12))
    if bucket in params["recovery"]:
        return params["recovery"][bucket]
    return float(params["recovery_default"])


def recent_sales_mix(panel, origin, months=6, cash_share_override=None):
    origin = pd.Timestamp(origin)
    start = origin - pd.offsets.MonthEnd(months)
    cohorts = panel[(panel["months_on_book"] == 0) & panel["sales_month"].gt(start) & panel["sales_month"].le(origin)].copy()
    cohorts = cohorts.drop_duplicates("contractid")
    if cohorts.empty:
        raise ValueError("No recent sales cohorts available for mix assumptions")
    group_cols = ["region", "contract_type", "product", "payment_frequency"]
    mix = cohorts.groupby(group_cols, dropna=False, as_index=False).agg(
        contracts=("contractid", "nunique"), price=("price_usd", "mean"),
        deposit=("deposit_usd", "mean"), daily=("daily_amount_usd", "mean"),
        tenor=("tenor_length", "mean"),
    )
    mix["share"] = mix["contracts"] / mix["contracts"].sum()
    if cash_share_override is not None:
        cash_share_override = float(np.clip(cash_share_override, 0, 1))
        for ctype, desired in [("CASH", cash_share_override), ("FINANCED", 1 - cash_share_override)]:
            mask = mix["contract_type"].eq(ctype)
            current = mix.loc[mask, "share"].sum()
            if current > 0:
                mix.loc[mask, "share"] *= desired / current
    return mix


def forecast_cohort(
    panel, country_month, origin, forecast_months, sales_units,
    parameter_cutoff=None, collection_factor=1.0, recovery_factor=1.0,
    sales_factor=1.0, cash_share_override=None, bridge_override=None,
):
    origin = pd.Timestamp(origin)
    forecast_months = pd.DatetimeIndex(pd.to_datetime(forecast_months))
    cutoff = pd.Timestamp(parameter_cutoff) if parameter_cutoff is not None else origin
    params = fit_cohort_parameters(panel, country_month, cutoff)
    mix = recent_sales_mix(panel, origin, cash_share_override=cash_share_override)

    snap = panel[(panel["month"] == origin) & panel["contract_type"].eq("FINANCED")].copy()
    snap = snap.drop_duplicates("contractid")
    snap["balance"] = (snap["price_usd"] - snap["actual_cumulative_cash_through_month_usd"]).clip(lower=0)

    rows, region_rows = [], []
    for month in forecast_months:
        scheduled_by_region, recovery_by_region = {}, {}
        scheduled_total = recovery_total = 0.0
        for idx, row in snap.iterrows():
            mob = month_difference(month, row["sales_month"])
            tenor_days = 0.0 if pd.isna(row["tenor_length"]) else float(row["tenor_length"])
            elapsed_before = min(tenor_days, max(0.0, (mob - 1) * AVG_DAYS_PER_MONTH))
            within = elapsed_before < tenor_days
            region = row["region"]
            if within:
                due = expected_due_for_mob(row["price_usd"], row["deposit_usd"], row["daily_amount_usd"], row["tenor_length"], mob)
                pred = min(row["balance"], due * lookup_eff(params, region, "FINANCED", mob) * collection_factor)
                scheduled_total += pred
                scheduled_by_region[region] = scheduled_by_region.get(region, 0.0) + pred
            else:
                tenor_mob = int(math.ceil(tenor_days / AVG_DAYS_PER_MONTH))
                pred = min(row["balance"], row["balance"] * lookup_recovery(params, mob - tenor_mob) * recovery_factor)
                recovery_total += pred
                recovery_by_region[region] = recovery_by_region.get(region, 0.0) + pred
            snap.at[idx, "balance"] = max(0.0, row["balance"] - pred)

        new_financed_total = new_cash_total = 0.0
        new_by_region = {}
        for sale_month in forecast_months[forecast_months <= month]:
            units = float(sales_units.get(pd.Timestamp(sale_month), 0.0)) * sales_factor
            mob = month_difference(month, sale_month)
            for seg in mix.itertuples():
                count = units * float(seg.share)
                if seg.contract_type == "CASH":
                    pred = count * float(seg.price) * lookup_eff(params, seg.region, "CASH", 0) * collection_factor if mob == 0 else 0.0
                    new_cash_total += pred
                else:
                    due = expected_due_for_mob(seg.price, seg.deposit, seg.daily, seg.tenor, mob)
                    pred = count * due * lookup_eff(params, seg.region, "FINANCED", mob) * collection_factor
                    new_financed_total += pred
                new_by_region[seg.region] = new_by_region.get(seg.region, 0.0) + pred

        bridge = float(params["bridge_level"] if bridge_override is None else bridge_override)
        total = scheduled_total + recovery_total + new_financed_total + new_cash_total + bridge
        rows.append({
            "month": month,
            "existing_financed_scheduled_collections": scheduled_total,
            "post_tenor_recovery": recovery_total,
            "new_financed_sales": new_financed_total,
            "new_cash_sales": new_cash_total,
            "reconciliation_adjustment": bridge,
            "total": total,
        })
        all_regions = set(scheduled_by_region) | set(recovery_by_region) | set(new_by_region)
        for region in all_regions:
            region_rows.append({
                "month": month, "region": region,
                "forecast_portfolio_cash_usd": scheduled_by_region.get(region, 0.0) + recovery_by_region.get(region, 0.0) + new_by_region.get(region, 0.0),
            })
    return pd.DataFrame(rows), pd.DataFrame(region_rows)


# %% id="iMmFsXnYBKxL"
NUMERIC_FEATURES = [c for c in [
    "months_on_book", "price_usd", "perc_deposit", "daily_amount_usd", "tenor_length",
    "expected_cash_due_this_month_usd", "expected_cumulative_cash_before_month_usd",
    "actual_cumulative_cash_through_month_usd", "payment_lag_1m", "payment_lag_2m",
    "payment_lag_3m", "payment_trailing_3m_sum", "payment_trailing_6m_sum",
    "prior_paying_months", "zero_payment_months_prior_3", "zero_payment_months_prior_6",
    "months_since_last_positive_payment", "calls_cumulative_before_month",
    "tickets_cumulative_before_month",
] if c in panel.columns]
CATEGORICAL_FEATURES = [c for c in [
    "region", "product", "payment_frequency", "scheduled_status"
] if c in panel.columns]


def ridge_existing_forecast(panel, prediction_origin, forecast_months, training_target_cutoff):
    prediction_origin = pd.Timestamp(prediction_origin)
    training_target_cutoff = pd.Timestamp(training_target_cutoff)
    forecast_months = pd.DatetimeIndex(pd.to_datetime(forecast_months))
    target_lookup = panel[["contractid", "month", "target_payment_usd"]].rename(
        columns={"month": "target_month", "target_payment_usd": "future_target"}
    )
    prediction_rows = panel[
        panel["month"].eq(prediction_origin) & panel["contract_type"].eq("FINANCED")
    ].drop_duplicates("contractid").copy()
    if prediction_rows.empty:
        raise ValueError(f"No financed-book snapshot at {prediction_origin.date()}")

    output = []
    for horizon, target_month in enumerate(forecast_months, start=1):
        latest_training_origin = training_target_cutoff - pd.offsets.MonthEnd(horizon)
        eligible_origins = pd.date_range(
            max(ANALYSIS_START, panel["month"].min()), latest_training_origin, freq="ME"
        )
        train = panel[
            panel["month"].isin(eligible_origins) & panel["contract_type"].eq("FINANCED")
        ].copy()
        train["target_month"] = train["month"] + pd.offsets.MonthEnd(horizon)
        train = train.merge(target_lookup, on=["contractid", "target_month"], how="left")
        train = train[train["future_target"].notna()].copy()
        if len(train) < 1000:
            raise ValueError(f"Too few Ridge training rows for horizon +{horizon}: {len(train)}")

        model = make_ridge_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES, alpha=10.0)
        model.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["future_target"])
        pred = np.clip(model.predict(prediction_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]), 0, None)
        output.append({"month": target_month, "forecast_existing_cash_usd": float(pred.sum())})
    return pd.DataFrame(output)


def actual_sales_units(panel, months):
    first_rows = panel[panel["months_on_book"].eq(0)].drop_duplicates("contractid")
    counts = first_rows.groupby("sales_month")["contractid"].nunique().to_dict()
    return {pd.Timestamp(m): int(counts.get(pd.Timestamp(m), 0)) for m in months}


# %% [markdown] id="-VkOBzMgBKxQ"
# ## Leakage-safe historical backtests
#
# Each origin uses only information available by that origin. The final development validation is Apr–Jun 2026 from the 31 March origin. East/West residuals are shown separately because their non-randomized pilots operate during that quarter.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="guKKIcDtBKxS" outputId="bc7384f1-53d9-4d98-acb6-91a371d98865"
BACKTEST_ORIGINS = pd.to_datetime(["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"])
monthly_rows = []
region_validation_rows = []

for origin in BACKTEST_ORIGINS:
    months = pd.date_range(origin + pd.offsets.MonthEnd(1), periods=3, freq="ME")
    actual = country_month.set_index("month").reindex(months)["source_reported_cash_usd"].astype(float)
    if actual.isna().any():
        raise AssertionError(f"Missing actuals for backtest from {origin.date()}")
    sales_units = actual_sales_units(panel, months)

    candidates = {}
    candidates["naive"] = naive_forecast(country_month, origin, months).set_index("month")["forecast"]
    candidates["holt_damped"] = holt_forecast(country_month, origin, months).set_index("month")["forecast"]

    cohort_components, cohort_regions = forecast_cohort(
        panel, country_month, origin, months, sales_units, parameter_cutoff=origin
    )
    candidates["cohort"] = cohort_components.set_index("month")["total"]

    if RUN_CHALLENGER:
        ridge_existing = ridge_existing_forecast(panel, origin, months, training_target_cutoff=origin)
        non_existing = cohort_components.set_index("month")[[
            "new_financed_sales", "new_cash_sales", "reconciliation_adjustment"
        ]].sum(axis=1)
        candidates["ridge_challenger"] = ridge_existing.set_index("month")["forecast_existing_cash_usd"] + non_existing

    for model_name, forecast in candidates.items():
        for month in months:
            monthly_rows.append({
                "model": model_name, "historical_origin": origin,
                "month": month, "actual": float(actual.loc[month]),
                "forecast": float(forecast.loc[month]),
            })

    if origin == ESTIMATION_END:
        actual_region = region_month.set_index(["month", "region"])["panel_cash_usd"]
        for row in cohort_regions.itertuples():
            key = (row.month, row.region)
            if key in actual_region.index:
                region_validation_rows.append({
                    "month": row.month, "region": row.region,
                    "actual_portfolio_cash_usd": float(actual_region.loc[key]),
                    "forecast_portfolio_cash_usd": float(row.forecast_portfolio_cash_usd),
                    "residual_actual_minus_forecast_usd": float(actual_region.loc[key] - row.forecast_portfolio_cash_usd),
                    "pilot_context": "SMS pilot" if row.region == "East" else "Outbound-call pilot" if row.region == "West" else "No pilot",
                })

model_backtest_monthly = pd.DataFrame(monthly_rows)
summaries = []
for (model_name, origin), g in model_backtest_monthly.groupby(["model", "historical_origin"]):
    summaries.append({"model": model_name, "historical_origin": origin, **score_monthly(g["actual"], g["forecast"])})
model_backtest_summary = pd.DataFrame(summaries).sort_values(["historical_origin", "WAPE"])
region_validation_residuals = pd.DataFrame(region_validation_rows)

display(model_backtest_summary.style.format({"MAE": "${:,.0f}", "WAPE": "{:.1%}", "Bias": "{:.1%}", "Bias_USD": "${:,.0f}"}))
display(region_validation_residuals.head(20))

# %% [markdown] id="pmnLpRipBKxT"
# ## Freeze and write the Q3 forecast
#
# The cohort parameters are frozen at 31 March 2026 so the non-randomized Apr–Jun pilots do not become normal collection behaviour. The contract state is observed through 30 June. Holt and the naive floor are independent cross-checks. Ridge is trained only on targets ending by 31 March and predicts from the June snapshot.

# %% colab={"base_uri": "https://localhost:8080/", "height": 231} id="env6BGa9BKxU" outputId="69a514fd-06a5-4d6f-f6c4-fab47168d31a"
FINAL_ORIGIN = VALIDATION_END
PARAMETER_CUTOFF = ESTIMATION_END

base_components, _ = forecast_cohort(
    panel, country_month, FINAL_ORIGIN, FORECAST_MONTHS, Q3_SALES_PLAN,
    parameter_cutoff=PARAMETER_CUTOFF,
)
naive_q3 = naive_forecast(country_month, FINAL_ORIGIN, FORECAST_MONTHS)
holt_q3 = holt_forecast(country_month, FINAL_ORIGIN, FORECAST_MONTHS)

comparison_parts = [
    naive_q3.assign(model="naive"),
    base_components[["month", "total"]].rename(columns={"total": "forecast"}).assign(model="cohort"),
    holt_q3.assign(model="holt_damped"),
]
if RUN_CHALLENGER:
    ridge_existing_q3 = ridge_existing_forecast(
        panel, FINAL_ORIGIN, FORECAST_MONTHS, training_target_cutoff=PARAMETER_CUTOFF
    )
    non_existing_q3 = base_components.set_index("month")[[
        "new_financed_sales", "new_cash_sales", "reconciliation_adjustment"
    ]].sum(axis=1)
    ridge_q3 = ridge_existing_q3.set_index("month")["forecast_existing_cash_usd"] + non_existing_q3
    comparison_parts.append(ridge_q3.rename("forecast").reset_index().assign(model="ridge_challenger"))

model_comparison = pd.concat(comparison_parts, ignore_index=True)[["model", "month", "forecast"]]
display(model_comparison.pivot(index="month", columns="model", values="forecast").style.format("${:,.0f}"))

# %% colab={"base_uri": "https://localhost:8080/", "height": 504} id="CZwaLLSABKxU" outputId="a33fa528-9474-4cad-9f36-35c97afca5a8"
cohort_bt = model_backtest_monthly[model_backtest_monthly["model"].eq("cohort")].copy()
cohort_bt["actual_to_forecast"] = cohort_bt["actual"] / cohort_bt["forecast"].replace(0, np.nan)
efficiency_low = min(1.0, float(cohort_bt["actual_to_forecast"].quantile(0.25)))
efficiency_high = max(1.0, float(cohort_bt["actual_to_forecast"].quantile(0.75)))

train_for_ranges = panel[(panel["month"] <= PARAMETER_CUTOFF) & panel["payment_history_reliable"]].copy()
post = train_for_ranges[train_for_ranges["post_tenor_recovery_proxy"]].copy()
post["balance_before"] = (post["price_usd"] - post["actual_cumulative_cash_before_month_usd"]).clip(lower=0)
rec_month = post.groupby("month", as_index=False).agg(actual=("target_payment_usd", "sum"), balance=("balance_before", "sum"))
rec_month["rate"] = rec_month["actual"] / rec_month["balance"].replace(0, np.nan)
base_recovery = float(rec_month["rate"].median()) if len(rec_month) else 0.0
recovery_low_factor = min(1.0, float(rec_month["rate"].quantile(0.25) / base_recovery)) if base_recovery else 1.0
recovery_high_factor = max(1.0, float(rec_month["rate"].quantile(0.75) / base_recovery)) if base_recovery else 1.0

cohort_zero = panel[panel["months_on_book"].eq(0)].drop_duplicates("contractid")
monthly_mix = cohort_zero[cohort_zero["sales_month"] <= PARAMETER_CUTOFF].groupby("sales_month")["contract_type"].apply(lambda s: s.eq("CASH").mean())
base_mix = recent_sales_mix(panel, FINAL_ORIGIN)
cash_share_base = float(base_mix.loc[base_mix["contract_type"].eq("CASH"), "share"].sum())
cash_share_low = min(cash_share_base, float(monthly_mix.tail(12).quantile(0.25)))
cash_share_high = max(cash_share_base, float(monthly_mix.tail(12).quantile(0.75)))

bridge_hist = country_month[country_month["month"] <= PARAMETER_CUTOFF]["source_minus_attributable_cash_usd"].dropna().tail(12)
bridge_base = fit_cohort_parameters(panel, country_month, PARAMETER_CUTOFF)["bridge_level"]
bridge_low = min(bridge_base, float(bridge_hist.quantile(0.25)))
bridge_high = max(bridge_base, float(bridge_hist.quantile(0.75)))

assumptions = {
    "low": dict(collection_factor=efficiency_low, recovery_factor=recovery_low_factor, sales_factor=SALES_PLAN_STRESS["low"], cash_share_override=cash_share_low, bridge_override=bridge_low),
    "base": dict(collection_factor=1.0, recovery_factor=1.0, sales_factor=SALES_PLAN_STRESS["base"], cash_share_override=cash_share_base, bridge_override=bridge_base),
    "high": dict(collection_factor=efficiency_high, recovery_factor=recovery_high_factor, sales_factor=SALES_PLAN_STRESS["high"], cash_share_override=cash_share_high, bridge_override=bridge_high),
}

scenario_components = {}
for scenario, kwargs in assumptions.items():
    scenario_components[scenario], _ = forecast_cohort(
        panel, country_month, FINAL_ORIGIN, FORECAST_MONTHS, Q3_SALES_PLAN,
        parameter_cutoff=PARAMETER_CUTOFF, **kwargs
    )

forecast_q3_2026 = scenario_components["base"][["month", "total"]].rename(columns={"total": "base"})
forecast_q3_2026["low"] = scenario_components["low"]["total"].to_numpy()
forecast_q3_2026["high"] = scenario_components["high"]["total"].to_numpy()
forecast_q3_2026 = forecast_q3_2026[["month", "low", "base", "high"]]

validate_scenario_table(forecast_q3_2026)

assumption_table = pd.DataFrame([
    {"assumption": "Existing-book collection efficiency", "low": efficiency_low, "base": 1.0, "high": efficiency_high, "evidence": "25th/75th percentiles of historical cohort actual-to-forecast ratios"},
    {"assumption": "Post-tenor recovery", "low": recovery_low_factor, "base": 1.0, "high": recovery_high_factor, "evidence": "25th/75th percentiles of monthly pre-Apr-2026 recovery rates"},
    {"assumption": "Q3 sales-plan realization", "low": 0.95, "base": 1.0, "high": 1.05, "evidence": "Explicit management stress; replace if plan-attainment history becomes available"},
    {"assumption": "Cash share", "low": cash_share_low, "base": cash_share_base, "high": cash_share_high, "evidence": "Recent historical monthly mix"},
    {"assumption": "Monthly reconciliation bridge (USD)", "low": bridge_low, "base": bridge_base, "high": bridge_high, "evidence": "Historical monthly source-minus-attributable cash"},
])
display(forecast_q3_2026.style.format({"low": "${:,.0f}", "base": "${:,.0f}", "high": "${:,.0f}"}))
display(assumption_table)

# %% colab={"base_uri": "https://localhost:8080/", "height": 758} id="k3ixjUeOBKxV" outputId="138491e7-19ea-47eb-b400-728f7fcb956c"
base_kwargs = assumptions["base"].copy()
sensitivity_rows = []
lever_map = {
    "existing_book_collection_efficiency": ("collection_factor", efficiency_low, efficiency_high),
    "post_tenor_recovery": ("recovery_factor", recovery_low_factor, recovery_high_factor),
    "q3_sales_plan_realisation": ("sales_factor", 0.95, 1.05),
    "cash_financed_mix": ("cash_share_override", cash_share_low, cash_share_high),
    "reconciliation_bridge": ("bridge_override", bridge_low, bridge_high),
}
base_q3_total = float(scenario_components["base"]["total"].sum())
for lever, (key, low_value, high_value) in lever_map.items():
    for case, value in [("low", low_value), ("high", high_value)]:
        kwargs = base_kwargs.copy()
        kwargs[key] = value
        result, _ = forecast_cohort(
            panel, country_month, FINAL_ORIGIN, FORECAST_MONTHS, Q3_SALES_PLAN,
            parameter_cutoff=PARAMETER_CUTOFF, **kwargs
        )
        q3_total = float(result["total"].sum())
        sensitivity_rows.append({
            "assumption": lever, "case": case, "assumption_value": value,
            "q3_cash_usd": q3_total, "change_vs_base_usd": q3_total - base_q3_total,
            "absolute_change_usd": abs(q3_total - base_q3_total),
        })

forecast_sensitivity = pd.DataFrame(sensitivity_rows).sort_values("absolute_change_usd", ascending=False)
largest_sensitivity = forecast_sensitivity.iloc[0]
print("Largest tested sensitivity:", largest_sensitivity["assumption"], f"({largest_sensitivity['change_vs_base_usd']:+,.0f} USD)")
display(forecast_sensitivity.style.format({"q3_cash_usd": "${:,.0f}", "change_vs_base_usd": "${:+,.0f}", "absolute_change_usd": "${:,.0f}"}))

# %% colab={"base_uri": "https://localhost:8080/", "height": 527} id="ScVId_Y6BKxV" outputId="88b5dc83-e442-4c5c-cae6-a3279c463111"
import matplotlib.dates as mdates

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# -------------------------
# 1. Forecast chart
# -------------------------
axes[0].plot(
    forecast_q3_2026["month"],
    forecast_q3_2026["base"],
    marker="o",
    label="Base"
)

axes[0].fill_between(
    forecast_q3_2026["month"],
    forecast_q3_2026["low"],
    forecast_q3_2026["high"],
    alpha=0.2,
    label="Low–high"
)

axes[0].set_title("Q3 2026 country collections forecast", fontsize=13)
axes[0].set_ylabel("USD")

# Show only one clean date label per month
axes[0].xaxis.set_major_locator(mdates.MonthLocator())
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(loc="upper left")

# -------------------------
# 2. Components chart
# -------------------------
comp_plot = forecast_components.set_index("month")[required_component_cols]

comp_plot.plot(
    kind="bar",
    stacked=True,
    ax=axes[1]
)

axes[1].set_title("Base forecast components", fontsize=13)
axes[1].set_ylabel("USD")

# Clean month labels
axes[1].set_xticklabels(
    [pd.to_datetime(x).strftime("%b %Y") for x in comp_plot.index],
    rotation=0
)

# Put legend outside the plotting area
axes[1].legend(
    fontsize=8,
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0
)

plt.tight_layout()

chart_path = OUTPUT_DIR / "forecast_charts.png"

plt.savefig(
    chart_path,
    dpi=180,
    bbox_inches="tight"
)

plt.show()

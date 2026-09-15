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
# # Part 1 — Collections Forecast
#
# This notebook is designed for **Google Colab** and keeps the forecasting workflow auditable.
#
# ## Modelling decisions encoded here
#
# 1. **Final holdout is Jul–Sep 2026.** Nothing from those months is used to choose the model, segmentation depth, lookback window, or assumptions.
# 2. **Model selection uses rolling-origin backtests of the whole pipeline.** The cohort/vintage model is rebuilt as it would have existed at each historical origin.
# 3. **Headline model = bottom-up financed-book vintage model.** It models monthly cash collections as a share of the contract balance still outstanding, by months-on-book, with optional region / payment-frequency / product segmentation.
# 4. **Naive floor = trailing 3-month national collections average.** The headline model must beat or at least clearly justify itself against this.
# 5. **Top-down cross-check = damped Holt/ETS.** No annual seasonality is forced because the aggregate history is short.
# 6. **Outreach is NOT used as a core forecasting feature.** The pilots start only in Apr-2026, cover two regions, and were not randomized. A causal uplift should be handled in Part 2 and only added to Part 1 as an explicit scenario if justified.
# 7. **Service tickets are kept out of the headline forecast.** They are a diagnostic / downside lens unless the data shows a stable, material relationship.
# 8. **New-sales backtests use only information available at the origin.** Historical future sales are not leaked into the backtest. The final Jul–Sep forecast uses the stated sales plan: 3,100 / 3,200 / 3,200 units.
#
# ## What I would criticize in the proposed approach
#
# - Saying the cohort structure “rescues” the short-history problem is directionally right, but thousands of contract-month rows are **not thousands of independent time-series observations**. Contracts share calendar shocks, products, regions and cohorts. Backtesting is still the arbiter.
# - “Tune the outreach uplift at an earlier origin” is not clean causal validation. The pilots did not exist before Apr-2026 and were not randomized. Use temporal separation for prediction, but do not mistake it for causal identification.
# - MAPE is readable but should not be the only score. This notebook reports **MAE in dollars, WAPE, MAPE and bias** at the country-month level.
# - Segmentation should not be chosen by visually inspecting curves. This notebook compares a small set of pre-specified segmentation depths and applies a minimum-cell rule.
# - A bottom-up schedule based on `daily_amount_usd × days` is intuitive, but exact sale dates are unavailable and customers can pay late beyond tenor. The headline vintage model here instead estimates **cash collected as a share of remaining contract balance by months-on-book**, which handles deposits, arrears and late recovery empirically while still capping collections at the remaining balance.
# - Historical backtests do not have the historical sales plans that management had at the time. Therefore the backtest uses a trailing sales run-rate as an origin-available proxy. That means backtest error includes both collection-model error and sales-assumption error. The final forecast is stronger because Q3 sales units are explicitly given.

# %% [markdown]
# ## 1. Upload the five CSV files from your computer
#
# Run the next cell and select:
#
# - `contracts.csv`
# - `payments.csv`
# - `calls.csv`
# - `service_tickets.csv`
# - `collections_outreach.csv`
#
# Colab will show an upload chooser. At the end of the notebook, `files.download(...)` will create browser download prompts for the outputs. **A notebook cannot force your browser to save specifically to Desktop**; choose Desktop in the browser save dialog if your browser asks, or move the downloaded ZIP there afterward.

# %%
from google.colab import files
uploaded = files.upload()

print("\nUploaded:")
for name in uploaded:
    print(" -", name)

# %% [markdown]
# ## 2. Imports and configuration

# %%
import io
import re
import math
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)

# -----------------------------
# FINAL HOLDOUT — DO NOT CHANGE
# -----------------------------
FINAL_ORIGIN = pd.Timestamp("2026-06-30")
FINAL_FORECAST_MONTHS = pd.to_datetime([
    "2026-07-31",
    "2026-08-31",
    "2026-09-30"
])

FINAL_SALES_PLAN = {
    pd.Timestamp("2026-07-31"): 3100,
    pd.Timestamp("2026-08-31"): 3200,
    pd.Timestamp("2026-09-30"): 3200,
}

# Earlier 3-month origins used for model selection.
# Each origin forecasts the NEXT 3 months.
BACKTEST_ORIGINS = pd.to_datetime([
    "2025-09-30",  # forecasts Oct-Dec 2025
    "2025-12-31",  # forecasts Jan-Mar 2026
    "2026-03-31",  # forecasts Apr-Jun 2026
])

# Candidate segmentation depths.
SEGMENT_CANDIDATES = {
    "mob_only": [],
    "region_mob": ["region"],
    "region_frequency_mob": ["region", "payment_frequency"],
    "region_product_mob": ["region", "product"],
}

# All-history vs recent-history rate estimation.
LOOKBACK_CANDIDATES = {
    "all_history": None,
    "trailing_12m": 12,
}

MIN_CELL_CONTRACT_MONTHS = 100
RECENT_SALES_MIX_MONTHS = 3
BACKTEST_SALES_RUNRATE_MONTHS = 3

# Explicit planning uncertainty where historical plan accuracy is unavailable.
# Collection-rate scenario bounds will be data-driven later.
SALES_PLAN_LOW_MULTIPLIER = 0.95
SALES_PLAN_HIGH_MULTIPLIER = 1.05

# Keep the final Jul-Sep actuals sealed until AFTER forecast outputs are written.
RUN_FINAL_HOLDOUT_EVALUATION = False

OUTPUT_DIR = Path("/content/dlight_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# %% [markdown]
# ## 3. Load files robustly from the Colab upload

# %%
def _normalise_filename(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

def find_uploaded_file(keyword):
    matches = []
    key = _normalise_filename(keyword)
    for name in uploaded.keys():
        norm = _normalise_filename(name)
        if key in norm:
            matches.append(name)
    if not matches:
        raise FileNotFoundError(f"Could not find an uploaded file matching: {keyword}")
    if len(matches) > 1:
        print(f"Multiple matches for {keyword}: {matches}. Using {matches[0]}")
    return matches[0]

def read_uploaded_csv(keyword):
    name = find_uploaded_file(keyword)
    return pd.read_csv(io.BytesIO(uploaded[name]))

contracts_raw = read_uploaded_csv("contracts")
payments_raw = read_uploaded_csv("payments")
calls_raw = read_uploaded_csv("calls")
service_raw = read_uploaded_csv("service_tickets")
outreach_raw = read_uploaded_csv("collections_outreach")

print("Raw shapes")
print("contracts:", contracts_raw.shape)
print("payments:", payments_raw.shape)
print("calls:", calls_raw.shape)
print("service_tickets:", service_raw.shape)
print("collections_outreach:", outreach_raw.shape)


# %% [markdown]
# ## 4. Cleaning helpers
#
# Cleaning is deliberately conservative:
#
# - fix provable schema / formatting problems;
# - preserve original values where a correction is material;
# - keep genuine missingness;
# - flag ambiguous records instead of silently deleting them.

# %%
def clean_column_names(df):
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    out = out.dropna(axis=1, how="all")
    return out

def clean_id(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

def clean_text(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

def parse_mixed_date(series):
    raw = series.astype("string").str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")

    result = pd.to_datetime(raw.where(numeric.isna()), errors="coerce")

    numeric_mask = numeric.notna()
    if numeric_mask.any():
        result.loc[numeric_mask] = pd.to_datetime(
            numeric.loc[numeric_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
    return result

def month_end(series):
    s = pd.to_datetime(series, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp("M")


# %% [markdown]
# ## 5. Clean contracts

# %%
contracts = clean_column_names(contracts_raw)

for col in ["contractid", "sales_person_id"]:
    if col in contracts.columns:
        contracts[col] = clean_id(contracts[col])

contracts["sales_month"] = month_end(parse_mixed_date(contracts["sales_month"]))

numeric_cols = [
    "household_size",
    "price_usd",
    "perc_deposit",
    "daily_amount_usd",
    "tenor_length",
]
for col in numeric_cols:
    if col in contracts.columns:
        contracts[col] = pd.to_numeric(contracts[col], errors="coerce")

for col in ["region", "product"]:
    contracts[col] = clean_text(contracts[col])

contracts["region"] = contracts["region"].str.title()
contracts["contract_type"] = clean_text(contracts["contract_type"]).str.upper()
contracts["payment_frequency"] = clean_text(contracts["payment_frequency"]).str.upper()
contracts["customer_gender"] = clean_text(contracts["customer_gender"]).str.upper()
contracts["occupation"] = clean_text(contracts["occupation"]).str.upper()

# Preserve the source field before correcting the known deposit-scale issue.
contracts["perc_deposit_original"] = contracts["perc_deposit"]

deposit_scale_mask = (
    contracts["contract_type"].eq("FINANCED")
    & contracts["perc_deposit"].gt(1)
    & contracts["price_usd"].gt(0)
)

contracts["deposit_scale_corrected"] = deposit_scale_mask

contracts.loc[deposit_scale_mask, "perc_deposit"] = (
    contracts.loc[deposit_scale_mask, "perc_deposit"]
    / contracts.loc[deposit_scale_mask, "price_usd"]
)

print("Contracts:", contracts.shape)
print("Deposit-scale corrections:", int(deposit_scale_mask.sum()))
print("\nMissingness:")
display(contracts.isna().sum().sort_values(ascending=False).to_frame("missing"))

# %% [markdown]
# ## 6. Clean payments

# %%
payments = clean_column_names(payments_raw)
if "contract_id" in payments.columns:
    payments = payments.rename(columns={"contract_id": "contractid"})

payments["contractid"] = clean_id(payments["contractid"])
payments["pay_month"] = month_end(parse_mixed_date(payments["pay_month"]))
payments["total_paid"] = pd.to_numeric(payments["total_paid"], errors="coerce")

# Audit all repeated contract-months BEFORE changing anything.
payment_contract_month_audit = payments[
    payments.duplicated(["contractid", "pay_month"], keep=False)
].sort_values(["contractid", "pay_month"]).copy()

exact_payment_duplicates = payments.duplicated(keep=False)
payment_exact_duplicate_audit = payments[exact_payment_duplicates].copy()

# Remove exact duplicate rows, then collapse any remaining repeated contract-month
# records by summing distinct payment records.
payments = payments.drop_duplicates().copy()

payments = (
    payments.groupby(["contractid", "pay_month"], as_index=False, dropna=False)
    .agg(total_paid=("total_paid", "sum"))
)

print("Clean payments:", payments.shape)
print("Rows in repeated contract-month audit:", len(payment_contract_month_audit))
print("Rows in exact-duplicate audit:", len(payment_exact_duplicate_audit))
print("Negative payment rows:", int((payments["total_paid"] < 0).sum()))

# %% [markdown]
# ## 7. Clean calls, service tickets and outreach

# %%
# ---- Calls ----
calls = clean_column_names(calls_raw)
calls["contractid"] = clean_id(calls["contractid"])
calls["call_date"] = parse_mixed_date(calls["call_date"])
calls["call_reason"] = (
    clean_text(calls["call_reason"])
    .str.upper()
    .str.replace(" ", "_", regex=False)
)

call_duplicate_audit = calls[
    calls.duplicated(["contractid", "call_date", "call_reason"], keep=False)
].copy()

# Do NOT drop call duplicates: two same-day calls can be real events.

# ---- Service tickets ----
service = clean_column_names(service_raw)
service["contractid"] = clean_id(service["contractid"])
service["ticket_date"] = parse_mixed_date(service["ticket_date"])
service["ticket_outcome"] = clean_text(service["ticket_outcome"]).str.upper()

service["ticket_reason_original"] = service["ticket_reason"]

ticket_reason_key = (
    clean_text(service["ticket_reason"])
    .str.lower()
    .str.replace("_", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
)

ticket_reason_map = {
    "battery fault": "BATTERY_FAULT",
    "battery failure": "BATTERY_FAULT",
    "batt fault": "BATTERY_FAULT",
    "charging issue": "CHARGING_ISSUE",
    "not charging": "CHARGING_ISSUE",
    "won't charge": "CHARGING_ISSUE",
    "no charge": "CHARGING_ISSUE",
    "panel damage": "PANEL_DAMAGE",
    "panel broken": "PANEL_DAMAGE",
    "panel cracked": "PANEL_DAMAGE",
    "cable cut": "CABLE_FAULT",
    "cable fault": "CABLE_FAULT",
    "wire fault": "CABLE_FAULT",
    "lamp fault": "LIGHT_FAULT",
    "bulb not working": "LIGHT_FAULT",
    "light fault": "LIGHT_FAULT",
    "other": "OTHER",
    "misc": "OTHER",
}

service["ticket_reason"] = (
    ticket_reason_key.map(ticket_reason_map)
    .fillna(
        ticket_reason_key
        .str.upper()
        .str.replace(" ", "_", regex=False)
    )
)

# ---- Outreach ----
outreach = clean_column_names(outreach_raw)
outreach["contractid"] = clean_id(outreach["contractid"])
outreach["contact_month"] = month_end(parse_mixed_date(outreach["contact_month"]))
outreach["region"] = clean_text(outreach["region"]).str.title()
outreach["channel"] = clean_text(outreach["channel"]).str.upper()
outreach["attempts"] = pd.to_numeric(outreach["attempts"], errors="coerce")
outreach["cost_usd"] = pd.to_numeric(outreach["cost_usd"], errors="coerce")

outreach["reached"] = (
    outreach["reached"].astype("string").str.strip().str.lower()
    .map({
        "true": True, "false": False,
        "1": True, "0": False,
        "yes": True, "no": False
    })
    .astype("boolean")
)

print("Possible duplicate-looking calls (not deleted):", len(call_duplicate_audit))
print("\nClean ticket reasons:")
display(service["ticket_reason"].value_counts(dropna=False).to_frame("rows"))
print("\nOutreach region x channel:")
display(outreach.groupby(["region", "channel"]).size().to_frame("rows"))

# %% [markdown]
# ## 8. Data-quality audit before modelling

# %%
contract_dates = contracts[["contractid", "sales_month"]].copy()

def add_before_sale_flag(df, date_col):
    out = df.merge(contract_dates, on="contractid", how="left")
    event_month = pd.to_datetime(out[date_col]).dt.to_period("M")
    sale_month = pd.to_datetime(out["sales_month"]).dt.to_period("M")
    out["before_sale_month"] = event_month < sale_month
    return out

payments_check = add_before_sale_flag(payments, "pay_month")
calls_check = add_before_sale_flag(calls, "call_date")
service_check = add_before_sale_flag(service, "ticket_date")
outreach_check = add_before_sale_flag(outreach, "contact_month")

quality_rows = []

for name, df in {
    "contracts": contracts,
    "payments": payments,
    "calls": calls,
    "service_tickets": service,
    "collections_outreach": outreach,
}.items():
    quality_rows.append({
        "dataset": name,
        "rows": len(df),
        "columns": df.shape[1],
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
    })

quality_summary = pd.DataFrame(quality_rows)

extra_checks = pd.DataFrame([
    {"check": "deposit_scale_corrected_contracts", "value": int(contracts["deposit_scale_corrected"].sum())},
    {"check": "payment_contract_month_rows_flagged", "value": len(payment_contract_month_audit)},
    {"check": "possible_duplicate_call_rows_flagged", "value": len(call_duplicate_audit)},
    {"check": "payments_before_sale_month", "value": int(payments_check["before_sale_month"].sum())},
    {"check": "calls_before_sale_month", "value": int(calls_check["before_sale_month"].sum())},
    {"check": "service_tickets_before_sale_month", "value": int(service_check["before_sale_month"].sum())},
    {"check": "outreach_before_sale_month", "value": int(outreach_check["before_sale_month"].sum())},
])

display(quality_summary)
display(extra_checks)

# %% [markdown]
# ## 9. Seal the final holdout
#
# The model-selection data ends at **30-Jun-2026**.
#
# Jul–Sep 2026 payments may exist in the source file, but this notebook does not use them until the optional final-holdout evaluation cell at the very end.

# %%
payments_dev = payments[payments["pay_month"] <= FINAL_ORIGIN].copy()

assert payments_dev["pay_month"].max() <= FINAL_ORIGIN

print("Development payment data ends:", payments_dev["pay_month"].max().date())
print("Final holdout months:", [d.date() for d in FINAL_FORECAST_MONTHS])


# %% [markdown]
# ## 10. Build the financed vintage panel
#
# ### Why this version of a cohort model?
#
# Rather than reconstructing exact monthly billing from `daily_amount_usd`, this notebook estimates historical **cash collected as a share of the balance still outstanding at the start of each month**, by months-on-book.
#
# That choice is deliberate:
#
# - exact sale dates are not available, only sale month;
# - customers can pay late beyond the contractual tenor;
# - deposits, catch-up payments and arrears recovery are observed directly in cash;
# - forecast payments are capped at the remaining contractual balance.
#
# The contract schedule fields remain valuable for diagnostics and a challenger model, but they are not required for the headline vintage curve.

# %%
def month_diff(later, earlier):
    later = pd.to_datetime(later)
    earlier = pd.to_datetime(earlier)
    return (
        (later.dt.year - earlier.dt.year) * 12
        + (later.dt.month - earlier.dt.month)
    )

def build_financed_panel(contracts_df, payments_df, panel_end):
    financed = contracts_df[
        (contracts_df["contract_type"] == "FINANCED")
        & contracts_df["sales_month"].notna()
        & contracts_df["price_usd"].gt(0)
        & (contracts_df["sales_month"] <= panel_end)
    ].copy()

    keep_cols = [
        "contractid", "sales_month", "region", "product",
        "payment_frequency", "price_usd", "perc_deposit",
        "daily_amount_usd", "tenor_length"
    ]
    financed = financed[keep_cols]

    months = pd.DataFrame({
        "month": pd.date_range(
            financed["sales_month"].min(),
            panel_end,
            freq="ME"
        )
    })

    panel = financed.assign(_k=1).merge(
        months.assign(_k=1),
        on="_k",
        how="inner"
    ).drop(columns="_k")

    panel = panel[panel["month"] >= panel["sales_month"]].copy()
    panel["mob"] = month_diff(panel["month"], panel["sales_month"]).astype(int)

    pay = payments_df.rename(columns={"pay_month": "month"}).copy()
    panel = panel.merge(
        pay[["contractid", "month", "total_paid"]],
        on=["contractid", "month"],
        how="left"
    )
    panel["total_paid"] = panel["total_paid"].fillna(0.0)

    panel = panel.sort_values(["contractid", "month"]).reset_index(drop=True)

    # Remaining balance immediately BEFORE the month's observed payment.
    panel["cum_paid_before"] = (
        panel.groupby("contractid")["total_paid"]
        .cumsum()
        - panel["total_paid"]
    )

    panel["remaining_start"] = (
        panel["price_usd"] - panel["cum_paid_before"]
    ).clip(lower=0)

    # Model target is capped at contractual remaining balance.
    # Any excess is preserved as an audit flag rather than allowed to distort the curve.
    panel["payment_above_remaining"] = (
        panel["total_paid"] > panel["remaining_start"] + 1e-9
    )

    panel["paid_for_model"] = np.minimum(
        panel["total_paid"].clip(lower=0),
        panel["remaining_start"]
    )

    return panel

financed_panel = build_financed_panel(
    contracts,
    payments_dev,
    FINAL_ORIGIN
)

print("Financed contract-month panel:", financed_panel.shape)
print("Rows with payment above contractual remaining balance:",
      int(financed_panel["payment_above_remaining"].sum()))
display(financed_panel.head())


# %% [markdown]
# ## 11. Rate tables with hierarchical fallback
#
# For a given historical origin:
#
# 1. use only contract-months available at that origin;
# 2. optionally restrict to the most recent 12 calendar months to handle drift;
# 3. estimate the weighted monthly collection rate:
#
# \[
# \text{rate} = \frac{\sum \text{cash collected}}{\sum \text{remaining balance at month start}}
# \]
#
# 4. segmented cells must have at least `MIN_CELL_CONTRACT_MONTHS`;
# 5. sparse or unseen cells fall back to the overall months-on-book curve.

# %%
def build_rate_tables(panel, origin, segment_cols=None, lookback_months=None, min_cell=100):
    segment_cols = segment_cols or []

    hist = panel[
        (panel["month"] <= origin)
        & (panel["remaining_start"] > 0)
    ].copy()

    if lookback_months is not None:
        start = (origin - pd.DateOffset(months=lookback_months - 1)).to_period("M").to_timestamp("M")
        hist = hist[hist["month"] >= start].copy()

    overall = (
        hist.groupby("mob", as_index=False)
        .agg(
            paid=("paid_for_model", "sum"),
            balance=("remaining_start", "sum"),
            n=("contractid", "size")
        )
    )
    overall["rate"] = np.where(
        overall["balance"] > 0,
        overall["paid"] / overall["balance"],
        np.nan
    )
    overall = overall[["mob", "rate", "n"]].sort_values("mob")

    segmented = None
    if segment_cols:
        segmented = (
            hist.groupby(segment_cols + ["mob"], as_index=False)
            .agg(
                paid=("paid_for_model", "sum"),
                balance=("remaining_start", "sum"),
                n=("contractid", "size")
            )
        )
        segmented["rate"] = np.where(
            segmented["balance"] > 0,
            segmented["paid"] / segmented["balance"],
            np.nan
        )
        segmented = segmented[segmented["n"] >= min_cell].copy()
        segmented = segmented[segment_cols + ["mob", "rate", "n"]]

    return {
        "segment_cols": segment_cols,
        "overall": overall,
        "segmented": segmented,
    }

def add_collection_rate(rows, rate_tables, rate_multiplier=1.0):
    out = rows.copy()
    seg_cols = rate_tables["segment_cols"]
    overall = rate_tables["overall"].copy()

    max_mob = int(overall["mob"].max())
    out["_mob_fallback"] = out["mob"].clip(upper=max_mob)

    # Overall age fallback.
    overall_lookup = overall[["mob", "rate"]].rename(
        columns={"mob": "_mob_fallback", "rate": "_overall_rate"}
    )
    out = out.merge(overall_lookup, on="_mob_fallback", how="left")

    if seg_cols and rate_tables["segmented"] is not None and len(rate_tables["segmented"]):
        seg = rate_tables["segmented"].copy().rename(columns={"rate": "_segment_rate"})
        out = out.merge(
            seg[seg_cols + ["mob", "_segment_rate"]],
            on=seg_cols + ["mob"],
            how="left"
        )
        out["collection_rate"] = out["_segment_rate"].fillna(out["_overall_rate"])
    else:
        out["collection_rate"] = out["_overall_rate"]

    # Conservative fallback if an early age is somehow absent.
    fallback_scalar = overall["rate"].median()
    out["collection_rate"] = out["collection_rate"].fillna(fallback_scalar)

    out["collection_rate"] = (out["collection_rate"] * rate_multiplier).clip(lower=0)

    drop_cols = [c for c in ["_mob_fallback", "_overall_rate", "_segment_rate"] if c in out.columns]
    return out.drop(columns=drop_cols)


# %% [markdown]
# ## 12. Origin-available new-sales assumptions
#
# ### Historical backtests
# We do **not** use actual future contracts. For each backtest origin:
#
# - future monthly units = trailing 3-month average sales volume observed at the origin;
# - new-sales contract mix and average price = recent contracts available at the origin.
#
# ### Final Q3 forecast
# Use the case-study sales plan: **3,100 / 3,200 / 3,200 units**.

# %%
def recent_sales_mix(contracts_df, origin, months=3):
    start = (origin - pd.DateOffset(months=months - 1)).to_period("M").to_timestamp("M")

    recent = contracts_df[
        (contracts_df["sales_month"] >= start)
        & (contracts_df["sales_month"] <= origin)
        & contracts_df["price_usd"].gt(0)
    ].copy()

    if recent.empty:
        raise ValueError(f"No recent contracts found for sales mix at origin {origin.date()}")

    return recent

def backtest_sales_plan(contracts_df, origin, horizon_months, runrate_months=3):
    start = (origin - pd.DateOffset(months=runrate_months - 1)).to_period("M").to_timestamp("M")
    monthly_units = (
        contracts_df[
            (contracts_df["sales_month"] >= start)
            & (contracts_df["sales_month"] <= origin)
        ]
        .groupby("sales_month")
        .size()
    )

    units = float(monthly_units.mean())
    return {m: units for m in horizon_months}


# %% [markdown]
# ## 13. Bottom-up cohort forecast function

# %%
def forecast_cohort_pipeline(
    contracts_df,
    payments_df,
    financed_panel_df,
    origin,
    forecast_months,
    sales_plan,
    segment_cols=None,
    lookback_months=None,
    min_cell=100,
    sales_mix_months=3,
    rate_multiplier=1.0,
):
    segment_cols = segment_cols or []

    rate_tables = build_rate_tables(
        financed_panel_df,
        origin=origin,
        segment_cols=segment_cols,
        lookback_months=lookback_months,
        min_cell=min_cell,
    )

    # -------------------------
    # Existing financed book
    # -------------------------
    existing = contracts_df[
        (contracts_df["contract_type"] == "FINANCED")
        & (contracts_df["sales_month"] <= origin)
        & contracts_df["price_usd"].gt(0)
    ][
        ["contractid", "sales_month", "region", "product",
         "payment_frequency", "price_usd"]
    ].copy()

    paid_to_origin = (
        payments_df[payments_df["pay_month"] <= origin]
        .groupby("contractid", as_index=False)["total_paid"]
        .sum()
        .rename(columns={"total_paid": "paid_to_origin"})
    )

    existing = existing.merge(paid_to_origin, on="contractid", how="left")
    existing["paid_to_origin"] = existing["paid_to_origin"].fillna(0.0)
    existing["remaining"] = (
        existing["price_usd"] - existing["paid_to_origin"]
    ).clip(lower=0)

    existing_state = existing.copy()
    existing_monthly = []

    for m in forecast_months:
        rows = existing_state[existing_state["remaining"] > 0].copy()
        rows["month"] = m
        rows["mob"] = month_diff(rows["month"], rows["sales_month"]).astype(int)
        rows = rows[rows["mob"] >= 0]

        rows = add_collection_rate(rows, rate_tables, rate_multiplier=rate_multiplier)
        rows["predicted_payment"] = np.minimum(
            rows["remaining"],
            rows["remaining"] * rows["collection_rate"]
        )

        existing_monthly.append({
            "month": m,
            "existing_financed": rows["predicted_payment"].sum()
        })

        pay_map = rows.set_index("contractid")["predicted_payment"]
        existing_state["remaining"] = (
            existing_state["remaining"]
            - existing_state["contractid"].map(pay_map).fillna(0)
        ).clip(lower=0)

    existing_monthly = pd.DataFrame(existing_monthly)

    # -------------------------
    # New sales mix
    # -------------------------
    recent = recent_sales_mix(contracts_df, origin, months=sales_mix_months)
    total_recent = len(recent)

    # Cash: full price assumed collected in month of sale, per contract definition.
    recent_cash = recent[recent["contract_type"] == "CASH"]
    cash_share = len(recent_cash) / total_recent if total_recent else 0
    mean_cash_price = recent_cash["price_usd"].mean() if len(recent_cash) else 0.0

    # Financed templates are weighted by share of ALL recent sales.
    recent_fin = recent[recent["contract_type"] == "FINANCED"].copy()

    template_group_cols = list(dict.fromkeys(segment_cols))
    if template_group_cols:
        fin_templates = (
            recent_fin.groupby(template_group_cols, dropna=False, as_index=False)
            .agg(
                units=("contractid", "size"),
                mean_price=("price_usd", "mean"),
            )
        )
    else:
        fin_templates = pd.DataFrame({
            "units": [len(recent_fin)],
            "mean_price": [recent_fin["price_usd"].mean() if len(recent_fin) else 0.0],
        })

    fin_templates["share_all_sales"] = fin_templates["units"] / total_recent

    # -------------------------
    # Simulate new cohorts
    # -------------------------
    new_financed_states = []
    new_financed_monthly = []
    new_cash_monthly = []

    for current_month in forecast_months:
        planned_units = float(sales_plan[current_month])

        # New cash sales contribution in sale month.
        new_cash_monthly.append({
            "month": current_month,
            "new_cash": planned_units * cash_share * mean_cash_price
        })

        # Add a new financed cohort for this sale month.
        cohort = fin_templates.copy()
        cohort["sales_month"] = current_month
        cohort["units_scaled"] = planned_units * cohort["share_all_sales"]
        cohort["remaining"] = cohort["units_scaled"] * cohort["mean_price"]
        cohort["cohort_id"] = current_month.strftime("%Y-%m")
        new_financed_states.append(cohort)

        month_payment = 0.0

        for i, state in enumerate(new_financed_states):
            active = state[state["remaining"] > 0].copy()
            active["month"] = current_month
            active["mob"] = month_diff(active["month"], active["sales_month"]).astype(int)

            active = add_collection_rate(active, rate_tables, rate_multiplier=rate_multiplier)
            active["predicted_payment"] = np.minimum(
                active["remaining"],
                active["remaining"] * active["collection_rate"]
            )

            month_payment += active["predicted_payment"].sum()

            # Update this cohort's remaining balance.
            state = state.copy()
            state = state.merge(
                active[["cohort_id"] + template_group_cols + ["predicted_payment"]],
                on=["cohort_id"] + template_group_cols,
                how="left"
            ) if template_group_cols else state.assign(
                predicted_payment=active["predicted_payment"].values
            )

            state["predicted_payment"] = state["predicted_payment"].fillna(0)
            state["remaining"] = (state["remaining"] - state["predicted_payment"]).clip(lower=0)
            state = state.drop(columns="predicted_payment")
            new_financed_states[i] = state

        new_financed_monthly.append({
            "month": current_month,
            "new_financed": month_payment
        })

    new_financed_monthly = pd.DataFrame(new_financed_monthly)
    new_cash_monthly = pd.DataFrame(new_cash_monthly)

    result = (
        existing_monthly
        .merge(new_financed_monthly, on="month", how="left")
        .merge(new_cash_monthly, on="month", how="left")
    )

    result["forecast"] = (
        result["existing_financed"]
        + result["new_financed"]
        + result["new_cash"]
    )

    result["model"] = "cohort"
    return result, rate_tables


# %% [markdown]
# ## 14. Baseline and ETS cross-check
#
# The baseline is intentionally simple. The ETS model is a sanity check, not the headline model.

# %%
def monthly_country_collections(payments_df):
    return (
        payments_df.groupby("pay_month", as_index=False)["total_paid"]
        .sum()
        .rename(columns={"pay_month": "month", "total_paid": "actual"})
        .sort_values("month")
    )

country_monthly = monthly_country_collections(payments)

def forecast_naive(payments_df, origin, forecast_months, trailing_months=3):
    monthly = monthly_country_collections(
        payments_df[payments_df["pay_month"] <= origin]
    )
    tail = monthly.tail(trailing_months)
    pred = tail["actual"].mean()

    return pd.DataFrame({
        "month": forecast_months,
        "forecast": pred,
        "model": "naive_3m_avg"
    })

def forecast_ets(payments_df, origin, forecast_months):
    monthly = monthly_country_collections(
        payments_df[payments_df["pay_month"] <= origin]
    ).set_index("month")["actual"]

    monthly = monthly.asfreq("ME")

    # Damped additive trend; no forced annual seasonality with short history.
    try:
        model = ExponentialSmoothing(
            monthly,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True)

        pred = model.forecast(len(forecast_months)).values
    except Exception:
        pred = np.repeat(monthly.tail(3).mean(), len(forecast_months))

    return pd.DataFrame({
        "month": forecast_months,
        "forecast": pred,
        "model": "holt_ets"
    })


# %% [markdown]
# ## 15. Country-level scoring

# %%
def score_predictions(pred_df, actual_df):
    d = pred_df.merge(actual_df, on="month", how="inner").copy()
    d["error"] = d["forecast"] - d["actual"]
    d["abs_error"] = d["error"].abs()
    d["ape"] = np.where(
        d["actual"].abs() > 0,
        d["abs_error"] / d["actual"].abs(),
        np.nan
    )

    mae = d["abs_error"].mean()
    wape = d["abs_error"].sum() / d["actual"].abs().sum()
    mape = d["ape"].mean()
    bias_pct = d["error"].sum() / d["actual"].abs().sum()

    return {
        "MAE_USD": mae,
        "WAPE": wape,
        "MAPE": mape,
        "BIAS_PCT": bias_pct,
    }, d


# %% [markdown]
# ## 16. Rolling-origin backtests and model selection
#
# The whole pipeline is rebuilt at each origin.
#
# The segmentation candidates are deliberately few. The selected model is the **simplest candidate within 1 percentage point WAPE of the best** rather than mechanically choosing the most complex winner.

# %%
def next_three_month_ends(origin):
    return pd.date_range(
        origin + pd.offsets.MonthEnd(1),
        periods=3,
        freq="ME"
    )

actual_country = monthly_country_collections(payments)

backtest_monthly_rows = []
backtest_summary_rows = []

# Naive + ETS at each origin
for origin in BACKTEST_ORIGINS:
    horizon = next_three_month_ends(origin)
    actual_h = actual_country[actual_country["month"].isin(horizon)].copy()

    for pred in [
        forecast_naive(payments, origin, horizon),
        forecast_ets(payments, origin, horizon),
    ]:
        metrics, detail = score_predictions(pred, actual_h)
        detail["origin"] = origin
        detail["candidate"] = pred["model"].iloc[0]
        detail["lookback"] = "n/a"
        backtest_monthly_rows.append(detail)

        backtest_summary_rows.append({
            "origin": origin,
            "candidate": pred["model"].iloc[0],
            "lookback": "n/a",
            **metrics,
        })

# Cohort candidates
for origin in BACKTEST_ORIGINS:
    horizon = next_three_month_ends(origin)
    actual_h = actual_country[actual_country["month"].isin(horizon)].copy()
    sales_plan = backtest_sales_plan(
        contracts,
        origin,
        horizon,
        runrate_months=BACKTEST_SALES_RUNRATE_MONTHS
    )

    for seg_name, seg_cols in SEGMENT_CANDIDATES.items():
        for lookback_name, lookback_months in LOOKBACK_CANDIDATES.items():

            pred, _ = forecast_cohort_pipeline(
                contracts_df=contracts,
                payments_df=payments,
                financed_panel_df=financed_panel,
                origin=origin,
                forecast_months=horizon,
                sales_plan=sales_plan,
                segment_cols=seg_cols,
                lookback_months=lookback_months,
                min_cell=MIN_CELL_CONTRACT_MONTHS,
                sales_mix_months=RECENT_SALES_MIX_MONTHS,
                rate_multiplier=1.0,
            )

            pred_simple = pred[["month", "forecast"]].copy()
            pred_simple["model"] = "cohort"

            metrics, detail = score_predictions(pred_simple, actual_h)
            detail["origin"] = origin
            detail["candidate"] = seg_name
            detail["lookback"] = lookback_name
            backtest_monthly_rows.append(detail)

            backtest_summary_rows.append({
                "origin": origin,
                "candidate": seg_name,
                "lookback": lookback_name,
                **metrics,
            })

backtest_monthly = pd.concat(backtest_monthly_rows, ignore_index=True)
backtest_summary = pd.DataFrame(backtest_summary_rows)

# Aggregate performance over all validation months.
model_perf = (
    backtest_monthly
    .groupby(["candidate", "lookback"], as_index=False)
    .agg(
        MAE_USD=("abs_error", "mean"),
        TOTAL_ABS_ERROR=("abs_error", "sum"),
        TOTAL_ACTUAL=("actual", lambda x: np.abs(x).sum()),
        TOTAL_ERROR=("error", "sum"),
        MAPE=("ape", "mean"),
        N_MONTHS=("month", "size")
    )
)

model_perf["WAPE"] = model_perf["TOTAL_ABS_ERROR"] / model_perf["TOTAL_ACTUAL"]
model_perf["BIAS_PCT"] = model_perf["TOTAL_ERROR"] / model_perf["TOTAL_ACTUAL"]

# Complexity order for cohort candidates.
complexity = {
    "mob_only": 0,
    "region_mob": 1,
    "region_frequency_mob": 2,
    "region_product_mob": 2,
    "naive_3m_avg": 99,
    "holt_ets": 99,
}
model_perf["complexity"] = model_perf["candidate"].map(complexity).fillna(99)

display(
    model_perf[
        ["candidate", "lookback", "MAE_USD", "WAPE", "MAPE", "BIAS_PCT", "N_MONTHS"]
    ].sort_values("WAPE")
)

# %%
# Choose only among cohort candidates for the headline model.
cohort_perf = model_perf[
    model_perf["candidate"].isin(SEGMENT_CANDIDATES.keys())
].copy()

best_wape = cohort_perf["WAPE"].min()

# Prefer simplest model within +1 percentage point absolute WAPE of best.
eligible = cohort_perf[
    cohort_perf["WAPE"] <= best_wape + 0.01
].sort_values(["complexity", "WAPE", "MAE_USD"])

selected = eligible.iloc[0]

SELECTED_SEGMENT_NAME = selected["candidate"]
SELECTED_SEGMENT_COLS = SEGMENT_CANDIDATES[SELECTED_SEGMENT_NAME]

SELECTED_LOOKBACK_NAME = selected["lookback"]
SELECTED_LOOKBACK_MONTHS = LOOKBACK_CANDIDATES[SELECTED_LOOKBACK_NAME]

print("Selected headline model")
print("-----------------------")
print("Segmentation:", SELECTED_SEGMENT_NAME, SELECTED_SEGMENT_COLS)
print("Lookback:", SELECTED_LOOKBACK_NAME, SELECTED_LOOKBACK_MONTHS)
print("Validation WAPE:", round(float(selected["WAPE"]), 4))
print("Validation MAE USD:", round(float(selected["MAE_USD"]), 2))
print("Validation bias:", round(float(selected["BIAS_PCT"]), 4))

# %% [markdown]
# ## 17. Visual backtest comparison

# %%
plot_df = backtest_monthly.copy()

# Make a single comparable series for selected cohort + baseline + ETS.
selected_cohort_bt = plot_df[
    (plot_df["candidate"] == SELECTED_SEGMENT_NAME)
    & (plot_df["lookback"] == SELECTED_LOOKBACK_NAME)
].copy()
selected_cohort_bt["series"] = "Selected cohort"

naive_bt = plot_df[plot_df["candidate"] == "naive_3m_avg"].copy()
naive_bt["series"] = "Naive 3M average"

ets_bt = plot_df[plot_df["candidate"] == "holt_ets"].copy()
ets_bt["series"] = "Holt/ETS"

compare = pd.concat([selected_cohort_bt, naive_bt, ets_bt], ignore_index=True)

fig, ax = plt.subplots(figsize=(11, 5))
for series_name, g in compare.groupby("series"):
    ax.plot(g["month"], g["forecast"], marker="o", label=series_name)

actual_plot = (
    compare[["month", "actual"]]
    .drop_duplicates()
    .sort_values("month")
)
ax.plot(actual_plot["month"], actual_plot["actual"], marker="o", linewidth=2.5, label="Actual")

ax.set_title("Rolling-origin backtests: country collections")
ax.set_ylabel("Collections (USD)")
ax.set_xlabel("")
ax.legend()
ax.grid(alpha=0.2)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 18. Data-driven collection uncertainty for low/high scenarios
#
# The base model already contains age/segment-specific collection rates.
#
# For scenario bounds, use the **recent monthly portfolio collection yield variation** available by Jun-2026:
#
# - downside multiplier = 25th percentile / recent median;
# - upside multiplier = 75th percentile / recent median.
#
# This gives a transparent empirical range rather than an arbitrary ±10% on the whole forecast.
#
# Sales-plan uncertainty remains an explicit assumption (default ±5%) because historical plan-vs-actual accuracy is not provided.

# %%
def empirical_collection_multipliers(panel, origin, recent_months=12):
    start = (origin - pd.DateOffset(months=recent_months - 1)).to_period("M").to_timestamp("M")

    recent = panel[
        (panel["month"] >= start)
        & (panel["month"] <= origin)
        & (panel["remaining_start"] > 0)
    ].copy()

    monthly = (
        recent.groupby("month", as_index=False)
        .agg(
            paid=("paid_for_model", "sum"),
            balance=("remaining_start", "sum")
        )
    )
    monthly["yield"] = monthly["paid"] / monthly["balance"]

    median = monthly["yield"].median()
    q25 = monthly["yield"].quantile(0.25)
    q75 = monthly["yield"].quantile(0.75)

    low_mult = q25 / median if median > 0 else 1.0
    high_mult = q75 / median if median > 0 else 1.0

    return low_mult, high_mult, monthly

COLLECTION_LOW_MULT, COLLECTION_HIGH_MULT, recent_collection_yield = (
    empirical_collection_multipliers(financed_panel, FINAL_ORIGIN, recent_months=12)
)

print("Collection-rate scenario multipliers")
print("Low :", round(COLLECTION_LOW_MULT, 4))
print("Base: 1.0000")
print("High:", round(COLLECTION_HIGH_MULT, 4))

display(recent_collection_yield)


# %% [markdown]
# ## 19. Produce the FINAL Jul–Sep 2026 forecast — without looking at actual Jul–Sep cash

# %%
def multiply_sales_plan(plan, multiplier):
    return {m: float(v) * multiplier for m, v in plan.items()}

# Base
final_base_detail, final_rate_tables = forecast_cohort_pipeline(
    contracts_df=contracts,
    payments_df=payments_dev,          # explicitly ends Jun-2026
    financed_panel_df=financed_panel,  # explicitly ends Jun-2026
    origin=FINAL_ORIGIN,
    forecast_months=FINAL_FORECAST_MONTHS,
    sales_plan=FINAL_SALES_PLAN,
    segment_cols=SELECTED_SEGMENT_COLS,
    lookback_months=SELECTED_LOOKBACK_MONTHS,
    min_cell=MIN_CELL_CONTRACT_MONTHS,
    sales_mix_months=RECENT_SALES_MIX_MONTHS,
    rate_multiplier=1.0,
)

# Downside
final_low_detail, _ = forecast_cohort_pipeline(
    contracts_df=contracts,
    payments_df=payments_dev,
    financed_panel_df=financed_panel,
    origin=FINAL_ORIGIN,
    forecast_months=FINAL_FORECAST_MONTHS,
    sales_plan=multiply_sales_plan(FINAL_SALES_PLAN, SALES_PLAN_LOW_MULTIPLIER),
    segment_cols=SELECTED_SEGMENT_COLS,
    lookback_months=SELECTED_LOOKBACK_MONTHS,
    min_cell=MIN_CELL_CONTRACT_MONTHS,
    sales_mix_months=RECENT_SALES_MIX_MONTHS,
    rate_multiplier=COLLECTION_LOW_MULT,
)

# Upside
final_high_detail, _ = forecast_cohort_pipeline(
    contracts_df=contracts,
    payments_df=payments_dev,
    financed_panel_df=financed_panel,
    origin=FINAL_ORIGIN,
    forecast_months=FINAL_FORECAST_MONTHS,
    sales_plan=multiply_sales_plan(FINAL_SALES_PLAN, SALES_PLAN_HIGH_MULTIPLIER),
    segment_cols=SELECTED_SEGMENT_COLS,
    lookback_months=SELECTED_LOOKBACK_MONTHS,
    min_cell=MIN_CELL_CONTRACT_MONTHS,
    sales_mix_months=RECENT_SALES_MIX_MONTHS,
    rate_multiplier=COLLECTION_HIGH_MULT,
)

forecast_table = pd.DataFrame({
    "month": FINAL_FORECAST_MONTHS,
    "low": final_low_detail["forecast"].values,
    "base": final_base_detail["forecast"].values,
    "high": final_high_detail["forecast"].values,
})

forecast_table["month"] = forecast_table["month"].dt.strftime("%Y-%m-%d")

display(
    forecast_table.style.format({
        "low": "${:,.0f}",
        "base": "${:,.0f}",
        "high": "${:,.0f}",
    })
)

# %% [markdown]
# ## 20. Where the base forecast comes from

# %%
base_components = final_base_detail[
    ["month", "existing_financed", "new_financed", "new_cash", "forecast"]
].copy()

display(
    base_components.style.format({
        "existing_financed": "${:,.0f}",
        "new_financed": "${:,.0f}",
        "new_cash": "${:,.0f}",
        "forecast": "${:,.0f}",
    })
)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(base_components))

ax.bar(x, base_components["existing_financed"], label="Existing financed")
ax.bar(
    x,
    base_components["new_financed"],
    bottom=base_components["existing_financed"],
    label="New financed"
)
ax.bar(
    x,
    base_components["new_cash"],
    bottom=base_components["existing_financed"] + base_components["new_financed"],
    label="New cash"
)

ax.set_xticks(x)
ax.set_xticklabels(pd.to_datetime(base_components["month"]).dt.strftime("%b %Y"))
ax.set_ylabel("Collections (USD)")
ax.set_title("Base forecast composition")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 21. Sensitivity: which assumption moves the number most?
#
# Change one lever at a time around the base case.

# %%
base_q3 = final_base_detail["forecast"].sum()

# Collection only
collection_low_only, _ = forecast_cohort_pipeline(
    contracts, payments_dev, financed_panel,
    FINAL_ORIGIN, FINAL_FORECAST_MONTHS, FINAL_SALES_PLAN,
    SELECTED_SEGMENT_COLS, SELECTED_LOOKBACK_MONTHS,
    MIN_CELL_CONTRACT_MONTHS, RECENT_SALES_MIX_MONTHS,
    COLLECTION_LOW_MULT
)
collection_high_only, _ = forecast_cohort_pipeline(
    contracts, payments_dev, financed_panel,
    FINAL_ORIGIN, FINAL_FORECAST_MONTHS, FINAL_SALES_PLAN,
    SELECTED_SEGMENT_COLS, SELECTED_LOOKBACK_MONTHS,
    MIN_CELL_CONTRACT_MONTHS, RECENT_SALES_MIX_MONTHS,
    COLLECTION_HIGH_MULT
)

# Sales only
sales_low_only, _ = forecast_cohort_pipeline(
    contracts, payments_dev, financed_panel,
    FINAL_ORIGIN, FINAL_FORECAST_MONTHS,
    multiply_sales_plan(FINAL_SALES_PLAN, SALES_PLAN_LOW_MULTIPLIER),
    SELECTED_SEGMENT_COLS, SELECTED_LOOKBACK_MONTHS,
    MIN_CELL_CONTRACT_MONTHS, RECENT_SALES_MIX_MONTHS,
    1.0
)
sales_high_only, _ = forecast_cohort_pipeline(
    contracts, payments_dev, financed_panel,
    FINAL_ORIGIN, FINAL_FORECAST_MONTHS,
    multiply_sales_plan(FINAL_SALES_PLAN, SALES_PLAN_HIGH_MULTIPLIER),
    SELECTED_SEGMENT_COLS, SELECTED_LOOKBACK_MONTHS,
    MIN_CELL_CONTRACT_MONTHS, RECENT_SALES_MIX_MONTHS,
    1.0
)

sensitivity = pd.DataFrame([
    {
        "assumption": "Collection performance",
        "low_q3": collection_low_only["forecast"].sum(),
        "base_q3": base_q3,
        "high_q3": collection_high_only["forecast"].sum(),
    },
    {
        "assumption": "Q3 sales units",
        "low_q3": sales_low_only["forecast"].sum(),
        "base_q3": base_q3,
        "high_q3": sales_high_only["forecast"].sum(),
    },
])

sensitivity["downside_impact"] = sensitivity["low_q3"] - sensitivity["base_q3"]
sensitivity["upside_impact"] = sensitivity["high_q3"] - sensitivity["base_q3"]
sensitivity["max_abs_swing"] = sensitivity[
    ["downside_impact", "upside_impact"]
].abs().max(axis=1)

sensitivity = sensitivity.sort_values("max_abs_swing", ascending=False)

display(
    sensitivity.style.format({
        "low_q3": "${:,.0f}",
        "base_q3": "${:,.0f}",
        "high_q3": "${:,.0f}",
        "downside_impact": "${:,.0f}",
        "upside_impact": "${:,.0f}",
        "max_abs_swing": "${:,.0f}",
    })
)

top_driver = sensitivity.iloc[0]["assumption"]
print("Largest tested forecast driver:", top_driver)

# %% [markdown]
# ## 22. Independent final cross-checks

# %%
final_naive = forecast_naive(
    payments_dev,
    FINAL_ORIGIN,
    FINAL_FORECAST_MONTHS
)

final_ets = forecast_ets(
    payments_dev,
    FINAL_ORIGIN,
    FINAL_FORECAST_MONTHS
)

crosscheck = pd.DataFrame({
    "month": FINAL_FORECAST_MONTHS,
    "cohort_base": final_base_detail["forecast"].values,
    "naive_3m_avg": final_naive["forecast"].values,
    "holt_ets": final_ets["forecast"].values,
})

display(
    crosscheck.style.format({
        "cohort_base": "${:,.0f}",
        "naive_3m_avg": "${:,.0f}",
        "holt_ets": "${:,.0f}",
    })
)

# %% [markdown]
# ## 23. Export outputs BEFORE opening the final holdout
#
# This is the point at which the forecast is considered locked.

# %%
# Core outputs
forecast_table.to_csv(OUTPUT_DIR / "forecast_q3_2026.csv", index=False)
base_components.to_csv(OUTPUT_DIR / "forecast_base_components_q3_2026.csv", index=False)
sensitivity.to_csv(OUTPUT_DIR / "forecast_sensitivity_q3_2026.csv", index=False)
crosscheck.to_csv(OUTPUT_DIR / "forecast_crosscheck_q3_2026.csv", index=False)

backtest_monthly.to_csv(OUTPUT_DIR / "backtest_monthly_predictions.csv", index=False)
model_perf.to_csv(OUTPUT_DIR / "backtest_model_summary.csv", index=False)

quality_summary.to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)
extra_checks.to_csv(OUTPUT_DIR / "data_quality_extra_checks.csv", index=False)

# Audits
payment_contract_month_audit.to_csv(
    OUTPUT_DIR / "audit_payment_contract_month_duplicates.csv", index=False
)
payment_exact_duplicate_audit.to_csv(
    OUTPUT_DIR / "audit_payment_exact_duplicates.csv", index=False
)
call_duplicate_audit.to_csv(
    OUTPUT_DIR / "audit_possible_duplicate_calls.csv", index=False
)

payments_check[payments_check["before_sale_month"]].to_csv(
    OUTPUT_DIR / "audit_payments_before_sale.csv", index=False
)
calls_check[calls_check["before_sale_month"]].to_csv(
    OUTPUT_DIR / "audit_calls_before_sale.csv", index=False
)
service_check[service_check["before_sale_month"]].to_csv(
    OUTPUT_DIR / "audit_service_before_sale.csv", index=False
)
outreach_check[outreach_check["before_sale_month"]].to_csv(
    OUTPUT_DIR / "audit_outreach_before_sale.csv", index=False
)

# Assumptions / selected model
assumptions = pd.DataFrame([
    ["final_origin", str(FINAL_ORIGIN.date())],
    ["final_holdout", "Jul-Sep 2026"],
    ["selected_segmentation", SELECTED_SEGMENT_NAME],
    ["selected_lookback", SELECTED_LOOKBACK_NAME],
    ["minimum_segment_cell", MIN_CELL_CONTRACT_MONTHS],
    ["recent_sales_mix_months", RECENT_SALES_MIX_MONTHS],
    ["backtest_sales_runrate_months", BACKTEST_SALES_RUNRATE_MONTHS],
    ["collection_low_multiplier", COLLECTION_LOW_MULT],
    ["collection_high_multiplier", COLLECTION_HIGH_MULT],
    ["sales_low_multiplier", SALES_PLAN_LOW_MULTIPLIER],
    ["sales_high_multiplier", SALES_PLAN_HIGH_MULTIPLIER],
    ["outreach_uplift_in_base_forecast", 0.0],
], columns=["assumption", "value"])

assumptions.to_csv(OUTPUT_DIR / "forecast_assumptions.csv", index=False)

# ZIP all compact model outputs.
zip_path = Path("/content/dlight_model_outputs.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUTPUT_DIR.glob("*.csv"):
        z.write(p, arcname=p.name)

print("Locked forecast outputs written to:", OUTPUT_DIR)
print("ZIP:", zip_path)

# %% [markdown]
# ## 24. Download buttons
#
# Running this cell triggers browser downloads.
#
# If your browser is configured to ask where to save files, choose **Desktop**. Otherwise it will usually save to your normal Downloads folder.

# %%
from google.colab import files

files.download("/content/dlight_model_outputs.zip")

# %% [markdown]
# ## 25. OPTIONAL: export cleaned source datasets
#
# These files can be large, especially calls. Run only if you want cleaned copies downloaded.

# %%
EXPORT_CLEANED_DATA = False

if EXPORT_CLEANED_DATA:
    clean_dir = Path("/content/dlight_cleaned_data")
    clean_dir.mkdir(exist_ok=True)

    contracts.to_csv(clean_dir / "contracts_clean.csv", index=False)
    payments.to_csv(clean_dir / "payments_clean.csv", index=False)
    calls.to_csv(clean_dir / "calls_clean.csv", index=False)
    service.to_csv(clean_dir / "service_tickets_clean.csv", index=False)
    outreach.to_csv(clean_dir / "collections_outreach_clean.csv", index=False)

    clean_zip = Path("/content/dlight_cleaned_data.zip")
    with zipfile.ZipFile(clean_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in clean_dir.glob("*.csv"):
            z.write(p, arcname=p.name)

    files.download(str(clean_zip))
else:
    print("Set EXPORT_CLEANED_DATA = True if you want a ZIP of all cleaned CSVs.")

# %% [markdown]
# # STOP HERE until the forecast is locked.
#
# Only after you have saved the forecast and model-selection outputs should you reveal Jul–Sep actuals.
#
# This prevents accidental tuning to the answer.

# %% [markdown]
# ## 26. OPTIONAL final honest holdout evaluation — Jul–Sep 2026

# %%
if RUN_FINAL_HOLDOUT_EVALUATION:
    final_actual = actual_country[
        actual_country["month"].isin(FINAL_FORECAST_MONTHS)
    ].copy()

    final_pred = final_base_detail[["month", "forecast"]].copy()

    holdout_metrics, holdout_detail = score_predictions(
        final_pred,
        final_actual
    )

    print("FINAL HOLDOUT METRICS")
    for k, v in holdout_metrics.items():
        print(k, ":", v)

    display(holdout_detail)

    holdout_detail.to_csv(
        OUTPUT_DIR / "FINAL_HOLDOUT_EVALUATION_JUL_SEP_2026.csv",
        index=False
    )
else:
    print(
        "Final Jul-Sep actuals remain sealed. "
        "Set RUN_FINAL_HOLDOUT_EVALUATION = True only after the forecast is locked."
    )

# %% [markdown]
# ## Interpretation guide for the deck
#
# If the results support it, the headline story should be:
#
# - **Headline:** Q3 base collections, with low/high range.
# - **Mechanism:** most cash comes from the existing financed book; new planned sales add deposits / early-life collections plus cash-sale receipts.
# - **Confidence:** the bottom-up model was selected using rolling 3-month backtests and compared against a naive floor and a top-down ETS cross-check.
# - **Largest swing factor:** name the sensitivity that moves Q3 dollars most; do not assert this before running the sensitivity.
# - **Outreach:** no causal pilot uplift is baked into the base forecast. Any uplift is handled explicitly after the Part 2 evaluation.
# - **Product issues:** service tickets are a diagnostic/downside lens, not silently mixed into the headline forecast.

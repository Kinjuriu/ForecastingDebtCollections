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
# # Feature Engineering — PayGo Solar Collections Portfolio (v2 — with independent sanity checks)
#
# **Scope:** create leakage-safe analytical feature tables from the cleaned development data through **30-Jun-2026**.
#
# This notebook intentionally **does not fit or select any forecasting model**.
#
# ## Inputs
#
# Use the output from cleaning v3:
#
# - `development_through_jun_2026_v3.zip`
#
# Optionally keep the cleaning audit ZIP nearby for review, but this notebook does not need it to run.
#
# ## Outputs
#
# 1. `contract_month_features.csv`  
#    Contract-month analytical panel for forecasting / collection-curve work.
#
# 2. `region_month_features.csv`  
#    Region-month summary for later pilot-aware validation.
#
# 3. `country_month_features.csv`  
#    Country-month summary for country-level scoring and sanity checks.
#
# 4. `pilot_analysis_features.csv`  
#    Separate East/West outreach dataset for Part 2. Outreach is **not** merged into the base forecast table.
#
# 5. `feature_dictionary.csv`  
#    Definitions and leakage notes.
#
# 6. `feature_quality_checks.csv`  
#    Feature engineering QA.
#
# ## Time discipline
#
# - Estimation: Oct-2024 to Mar-2026
# - Validation: Apr-2026 to Jun-2026
# - Jul-Sep-2026 is not loaded by this notebook.
#
# Every behavioural predictor for contract-month `t` uses information from **months before t only**.
#
# The same-month payment is retained only as a clearly named target:
# `target_payment_usd`.
#
# Calls and service tickets from the same month are not used as predictors because their exact ordering relative to payment is unknown.
#
# ## v2 sanity-check philosophy
#
# Feature engineering is now treated as a reconciliation problem as well as a transformation problem.
#
# The notebook produces:
# - a compact `feature_metrics.json` for cross-notebook comparison;
# - `sanity_check_report.csv` with PASS / WARN / FAIL checks;
# - `payment_reconciliation_by_month.csv` that traces source payment dollars into the feature panel;
# - `unassigned_usable_payments.csv` for any dollars that should have been model-assignable but did not enter the panel.
#
# The notebook also avoids Colab's large interactive DataFrame renderer. Small summaries are printed as text, and large outputs are written to files.
#

# %% [markdown]
# ## 1. Upload the cleaned development ZIP
#
# Select:
#
# `development_through_jun_2026_v3.zip`
#
# from your Downloads folder.

# %%
from google.colab import files
uploaded = files.upload()

print("Uploaded:")
for name in uploaded:
    print(" -", name)

# %% [markdown]
# ## 2. Imports and configuration

# %%
import io
import re
import zipfile
import json
import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 150)
pd.set_option("display.width", 200)

ANALYSIS_START = pd.Timestamp("2024-10-31")
ESTIMATION_END = pd.Timestamp("2026-03-31")
VALIDATION_END = pd.Timestamp("2026-06-30")

OUTPUT_DIR = Path("/content/dlight_feature_engineering_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Avoid stale Colab interactive-table references.
try:
    from google.colab import data_table
    data_table.disable_dataframe_formatter()
except Exception:
    pass

def show_table(df, max_rows=30, title=None):
    """Print a stable text snapshot instead of an interactive Colab table."""
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    if df is None:
        print("None")
        return
    view = df.head(max_rows)
    print(view.to_string(index=False))
    if len(df) > max_rows:
        print(f"... {len(df) - max_rows:,} additional rows not printed")



# %% [markdown]
# ## 3. Extract the cleaned development ZIP

# %%
def find_zip():
    zip_names = [
        name for name in uploaded
        if name.lower().endswith(".zip")
    ]
    if not zip_names:
        raise FileNotFoundError(
            "Upload development_through_jun_2026_v3.zip"
        )

    preferred = [
        name for name in zip_names
        if "development_through_jun_2026" in name.lower()
    ]

    return preferred[0] if preferred else zip_names[0]

zip_name = find_zip()

extract_dir = Path("/content/dlight_dev_cleaned")
extract_dir.mkdir(exist_ok=True)

with zipfile.ZipFile(
    io.BytesIO(uploaded[zip_name]),
    "r"
) as z:
    z.extractall(extract_dir)

print("Extracted:", zip_name)
for p in sorted(extract_dir.rglob("*.csv")):
    print(" -", p.name)


# %% [markdown]
# ## 4. Load cleaned development files

# %%
def find_csv(keyword):
    matches = [
        p for p in extract_dir.rglob("*.csv")
        if keyword.lower() in p.name.lower()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No CSV matching '{keyword}'"
        )
    if len(matches) > 1:
        print(
            f"Multiple matches for {keyword}:",
            [m.name for m in matches],
            "Using:", matches[0].name
        )
    return matches[0]

contracts = pd.read_csv(
    find_csv("contracts_clean")
)
payments = pd.read_csv(
    find_csv("payments_clean")
)
calls = pd.read_csv(
    find_csv("calls_clean")
)
service = pd.read_csv(
    find_csv("service_tickets_clean")
)
outreach = pd.read_csv(
    find_csv("collections_outreach_clean")
)

date_cols = {
    "contracts": (contracts, "sales_month"),
    "payments": (payments, "pay_month"),
    "calls": (calls, "call_date"),
    "service": (service, "ticket_date"),
    "outreach": (outreach, "contact_month"),
}

for _, (df, col) in date_cols.items():
    df[col] = pd.to_datetime(
        df[col],
        errors="coerce"
    )

print("Loaded shapes:")
print("contracts:", contracts.shape)
print("payments:", payments.shape)
print("calls:", calls.shape)
print("service:", service.shape)
print("outreach:", outreach.shape)


# %% [markdown]
# ## 5. Define modelling-safe payment records
#
# The cleaning notebook deliberately kept ambiguous rows rather than deleting them.
#
# For feature engineering:
#
# - pre-Oct-2024 payment rows are excluded;
# - before-sale payments are excluded;
# - orphan rows are excluded;
# - negative payments are excluded;
# - conflicting contract-month totals are **not interpreted as zero**.
#
# A contract-month with a payment conflict receives:
# - `current_payment_conflict = 1`
# - `target_payment_usd = NaN`
#
# Rows after an earlier conflict receive:
# - `prior_payment_conflict_count > 0`
#
# That gives the later modelling notebook a clean eligibility rule instead of silently inventing a payment amount.

# %%
# Defensive boolean conversion because CSV round-trips may load booleans as strings.
def as_bool(series):
    if series.dtype == bool:
        return series
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False
        })
        .fillna(False)
        .astype(bool)
    )

for col in [
    "conflicting_contract_month",
    "zero_payment",
    "negative_payment",
    "orphan_contractid",
    "before_sale_month",
    "before_expected_data_start",
    "payment_after_apparent_payoff"
]:
    if col in payments.columns:
        payments[col] = as_bool(
            payments[col]
        )

payment_conflicts = (
    payments.loc[
        payments["conflicting_contract_month"],
        ["contractid", "pay_month"]
    ]
    .drop_duplicates()
    .assign(current_payment_conflict=True)
)

payments_usable = payments.copy()

payments_usable = payments_usable[
    (payments_usable["pay_month"] >= ANALYSIS_START)
    &
    (~payments_usable["orphan_contractid"])
    &
    (~payments_usable["before_sale_month"])
    &
    (~payments_usable["negative_payment"])
    &
    (~payments_usable["conflicting_contract_month"])
].copy()

# Defensive uniqueness check.
assert not payments_usable.duplicated(
    ["contractid", "pay_month"]
).any(), (
    "Usable payments still contain repeated contract-months."
)

print(
    "Usable payment rows:",
    len(payments_usable)
)
print(
    "Conflicting contract-months withheld:",
    len(payment_conflicts)
)

# %% [markdown]
# ## 6. Build the contract-month panel
#
# ### Panel logic
#
# **FINANCED contracts:** one row for every month from the later of:
# - the contract's sale month;
# - Oct-2024;
#
# through Jun-2026.
#
# **CASH contracts:** one row only in their sale month, because they are paid upfront.
#
# Contracts sold before Oct-2024 are retained as a **legacy book** if they are still present in the source. Their pre-Oct payment history is unavailable, so that limitation is flagged.
#
# No Jul-Sep rows are created.

# %%
contracts["sales_month"] = pd.to_datetime(
    contracts["sales_month"]
)

contracts["contract_type"] = (
    contracts["contract_type"]
    .astype("string")
    .str.upper()
)

contracts["legacy_pre_oct_2024"] = (
    contracts["sales_month"] < ANALYSIS_START
)

# Create monthly panel efficiently via month cross-join.
all_months = pd.DataFrame({
    "month": pd.date_range(
        ANALYSIS_START,
        VALIDATION_END,
        freq="ME"
    )
})

base_contract_cols = [
    "contractid",
    "sales_person_id",
    "sales_month",
    "region",
    "customer_gender",
    "household_size",
    "occupation",
    "contract_type",
    "product",
    "price_usd",
    "payment_frequency",
    "perc_deposit",
    "daily_amount_usd",
    "tenor_length",
    "deposit_scale_corrected",
    "legacy_pre_oct_2024"
]

base_contract_cols = [
    c for c in base_contract_cols
    if c in contracts.columns
]

financed = contracts[
    contracts["contract_type"].eq("FINANCED")
][base_contract_cols].copy()

cash = contracts[
    contracts["contract_type"].eq("CASH")
][base_contract_cols].copy()

fin_panel = (
    financed.assign(_key=1)
    .merge(
        all_months.assign(_key=1),
        on="_key",
        how="inner"
    )
    .drop(columns="_key")
)

fin_panel = fin_panel[
    fin_panel["month"]
    >= fin_panel["sales_month"].clip(
        lower=ANALYSIS_START
    )
].copy()

cash_panel = cash[
    cash["sales_month"].between(
        ANALYSIS_START,
        VALIDATION_END
    )
].copy()

cash_panel["month"] = cash_panel[
    "sales_month"
]

panel = pd.concat(
    [
        fin_panel,
        cash_panel
    ],
    ignore_index=True
)

panel = panel.sort_values(
    ["contractid", "month"]
).reset_index(drop=True)

print(
    "Contract-month rows:",
    len(panel)
)
print(
    "Unique contracts:",
    panel["contractid"].nunique()
)


# %% [markdown]
# ## 7. Static contract features
#
# These are known from the contract itself and do not depend on future behaviour.

# %%
def month_difference(later, earlier):
    later = pd.to_datetime(later)
    earlier = pd.to_datetime(earlier)

    return (
        (later.dt.year - earlier.dt.year) * 12
        +
        (later.dt.month - earlier.dt.month)
    )

panel["months_on_book"] = month_difference(
    panel["month"],
    panel["sales_month"]
).clip(lower=0)

panel["deposit_usd"] = np.where(
    panel["price_usd"].notna()
    &
    panel["perc_deposit"].notna(),
    panel["price_usd"]
    * panel["perc_deposit"],
    np.nan
)

panel["financed_amount_after_deposit_usd"] = np.where(
    panel["contract_type"].eq("FINANCED"),
    (
        panel["price_usd"]
        -
        panel["deposit_usd"]
    ).clip(lower=0),
    0.0
)

panel["tenor_months_proxy"] = (
    panel["tenor_length"]
    / 30.4375
)

panel["days_in_month"] = (
    panel["month"].dt.days_in_month
)

panel["scheduled_amount_proxy_usd"] = np.where(
    panel["contract_type"].eq("FINANCED"),
    panel["daily_amount_usd"]
    * panel["days_in_month"],
    0.0
)

# This is explicitly a proxy because only sales month,
# not exact sale date, is available.
panel["past_tenor_proxy"] = np.where(
    panel["contract_type"].eq("FINANCED"),
    (
        panel["months_on_book"]
        * 30.4375
        >
        panel["tenor_length"]
    ),
    False
)

panel["calendar_month_num"] = (
    panel["month"].dt.month
)

panel["calendar_quarter"] = (
    panel["month"].dt.quarter
)

panel["time_layer"] = np.select(
    [
        panel["month"] <= ESTIMATION_END,
        (
            (panel["month"] > ESTIMATION_END)
            &
            (panel["month"] <= VALIDATION_END)
        )
    ],
    [
        "estimation",
        "validation"
    ],
    default="other"
)

panel["known_by_mar_2026_origin"] = (
    panel["sales_month"] <= ESTIMATION_END
)

panel["known_by_jun_2026_origin"] = (
    panel["sales_month"] <= VALIDATION_END
)

# %% [markdown]
# ## 8. Merge the monthly payment target
#
# `target_payment_usd` is the amount actually paid in the row's calendar month.
#
# It is an **outcome**, not a predictor.
#
# Where the source had an unresolved conflicting monthly total, the target is left `NaN` rather than treated as zero.

# %%
target_payments = payments_usable[
    ["contractid", "pay_month", "total_paid"]
].rename(
    columns={
        "pay_month": "month",
        "total_paid": "target_payment_usd"
    }
)

panel = panel.merge(
    target_payments,
    on=[
        "contractid",
        "month"
    ],
    how="left"
)

panel = panel.merge(
    payment_conflicts.rename(
        columns={"pay_month": "month"}
    ),
    on=[
        "contractid",
        "month"
    ],
    how="left"
)

panel["current_payment_conflict"] = (
    panel["current_payment_conflict"]
    .fillna(False)
    .astype(bool)
)

# No record means zero observed payment,
# except for unresolved conflicting contract-months.
panel["target_payment_usd"] = (
    panel["target_payment_usd"]
    .fillna(0.0)
)

panel.loc[
    panel["current_payment_conflict"],
    "target_payment_usd"
] = np.nan

panel["target_paid_any"] = np.where(
    panel["target_payment_usd"].notna(),
    panel["target_payment_usd"] > 0,
    np.nan
)

# %% [markdown]
# ## 8a. Payment-dollar reconciliation
#
# This is the most important new sanity check.
#
# The cleaned monthly payment table is the accounting source. The contract-month panel is an analytical representation. Those two dollar totals should never silently diverge.
#
# For each month this section reports:
#
# - `source_payment_total_usd`: all cleaned payment dollars in the modelling window after exact de-duplication;
# - `conflicting_payment_rows_usd`: dollars sitting in unresolved duplicate contract-month records;
# - `before_sale_payment_usd`: non-conflicting dollars dated before the contract's sale month;
# - `model_usable_payment_usd`: dollars left after the explicit feature-engineering exclusions;
# - `panel_target_payment_usd`: dollars actually attached to contract-month rows;
# - `unassigned_usable_payment_usd`: usable dollars that still failed to enter the panel.
#
# The last quantity should be approximately zero. If it is not, the notebook classifies the unmatched rows so we know whether the issue comes from CASH contracts outside their sale month, a panel-construction error, or something else.
#
# This section also explains why a country-month total from the feature panel can legitimately differ from the raw cleaned payment total. The difference must be **named and quantified**, not hidden.

# %%
# Scope to the analytical period only.
payment_scope = payments[
    payments["pay_month"].between(
        ANALYSIS_START,
        VALIDATION_END
    )
].copy()

# Defensive flags after CSV round-trip.
for _col in [
    "conflicting_contract_month",
    "before_sale_month",
    "orphan_contractid",
    "negative_payment"
]:
    if _col in payment_scope.columns:
        payment_scope[_col] = as_bool(payment_scope[_col])

# Source accounting total after cleaning-v3 exact deduplication.
source_by_month = (
    payment_scope.groupby("pay_month", as_index=False)["total_paid"]
    .sum()
    .rename(columns={
        "pay_month": "month",
        "total_paid": "source_payment_total_usd"
    })
)

conflict_by_month = (
    payment_scope.loc[
        payment_scope["conflicting_contract_month"]
    ]
    .groupby("pay_month", as_index=False)["total_paid"]
    .sum()
    .rename(columns={
        "pay_month": "month",
        "total_paid": "conflicting_payment_rows_usd"
    })
)

before_sale_by_month = (
    payment_scope.loc[
        (~payment_scope["conflicting_contract_month"])
        & payment_scope["before_sale_month"]
    ]
    .groupby("pay_month", as_index=False)["total_paid"]
    .sum()
    .rename(columns={
        "pay_month": "month",
        "total_paid": "before_sale_payment_usd"
    })
)

usable_by_month = (
    payments_usable.groupby("pay_month", as_index=False)["total_paid"]
    .sum()
    .rename(columns={
        "pay_month": "month",
        "total_paid": "model_usable_payment_usd"
    })
)

panel_target_by_month = (
    panel.groupby("month", as_index=False)["target_payment_usd"]
    .sum(min_count=1)
    .rename(columns={
        "target_payment_usd": "panel_target_payment_usd"
    })
)

payment_reconciliation_by_month = (
    source_by_month
    .merge(conflict_by_month, on="month", how="left")
    .merge(before_sale_by_month, on="month", how="left")
    .merge(usable_by_month, on="month", how="left")
    .merge(panel_target_by_month, on="month", how="left")
    .sort_values("month")
)

for _col in [
    "conflicting_payment_rows_usd",
    "before_sale_payment_usd",
    "model_usable_payment_usd",
    "panel_target_payment_usd"
]:
    payment_reconciliation_by_month[_col] = (
        payment_reconciliation_by_month[_col]
        .fillna(0.0)
    )

payment_reconciliation_by_month[
    "excluded_or_unresolved_from_source_usd"
] = (
    payment_reconciliation_by_month["source_payment_total_usd"]
    - payment_reconciliation_by_month["model_usable_payment_usd"]
)

payment_reconciliation_by_month[
    "unassigned_usable_payment_usd"
] = (
    payment_reconciliation_by_month["model_usable_payment_usd"]
    - payment_reconciliation_by_month["panel_target_payment_usd"]
)

payment_reconciliation_by_month[
    "panel_share_of_source"
] = np.where(
    payment_reconciliation_by_month["source_payment_total_usd"].abs() > 0,
    payment_reconciliation_by_month["panel_target_payment_usd"]
    / payment_reconciliation_by_month["source_payment_total_usd"],
    np.nan
)

# Identify model-usable payments that do not have a corresponding panel key.
_panel_keys = panel[
    ["contractid", "month"]
].drop_duplicates()

_unassigned = (
    payments_usable.rename(columns={"pay_month": "month"})
    .merge(
        _panel_keys.assign(_in_panel=True),
        on=["contractid", "month"],
        how="left"
    )
)

unassigned_usable_payments = _unassigned[
    _unassigned["_in_panel"].isna()
].drop(columns="_in_panel").copy()

if len(unassigned_usable_payments):
    _contract_lookup = contracts[
        ["contractid", "contract_type", "sales_month"]
    ].drop_duplicates("contractid")

    unassigned_usable_payments = (
        unassigned_usable_payments
        .merge(
            _contract_lookup,
            on="contractid",
            how="left"
        )
    )

    unassigned_usable_payments["unassigned_reason"] = np.select(
        [
            (
                unassigned_usable_payments["contract_type"].eq("CASH")
                & (
                    unassigned_usable_payments["month"]
                    != unassigned_usable_payments["sales_month"]
                )
            ),
            unassigned_usable_payments["contract_type"].eq("CASH"),
            unassigned_usable_payments["contract_type"].eq("FINANCED"),
            unassigned_usable_payments["contract_type"].isna(),
        ],
        [
            "CASH_PAYMENT_OUTSIDE_SALE_MONTH",
            "CASH_PANEL_KEY_MISSING",
            "FINANCED_PANEL_KEY_MISSING",
            "CONTRACT_LOOKUP_MISSING",
        ],
        default="OTHER"
    )
else:
    unassigned_usable_payments["unassigned_reason"] = pd.Series(dtype="string")

show_table(
    payment_reconciliation_by_month,
    max_rows=30,
    title="Payment reconciliation by month"
)

if len(unassigned_usable_payments):
    print("\nUnassigned usable payments by reason")
    _unassigned_reason_summary = (
        unassigned_usable_payments
        .groupby("unassigned_reason", as_index=False)
        .agg(
            rows=("contractid", "size"),
            total_paid_usd=("total_paid", "sum")
        )
        .sort_values("total_paid_usd", ascending=False)
    )
    show_table(_unassigned_reason_summary, max_rows=30)
else:
    print("\nPASS: every model-usable payment dollar maps to a contract-month panel row.")

# %% [markdown]
# ## 9. Payment-history features — prior months only
#
# These are designed so the row for month `t` never uses payment from month `t`.
#
# Features include:
#
# - previous month's payment;
# - trailing 3- and 6-month payment totals / means;
# - cumulative payments before the month;
# - share of contract price already paid;
# - remaining contractual balance before the month;
# - number of prior paying months;
# - prior zero-payment counts;
# - months since last positive payment;
# - payment-history conflict flags.
#
# For legacy contracts sold before Oct-2024, cumulative history is left-censored because earlier payments are not in the supplied window. That is explicitly flagged.

# %%
panel = panel.sort_values(
    ["contractid", "month"]
).copy()

# For historical feature construction, NaN conflict targets stay NaN.
# Rolling sums use available values but reliability flags tell us
# whether an unresolved conflict occurred earlier.

g = panel.groupby(
    "contractid",
    group_keys=False
)

panel["payment_lag_1m"] = (
    g["target_payment_usd"]
    .shift(1)
)

panel["payment_lag_2m"] = (
    g["target_payment_usd"]
    .shift(2)
)

panel["payment_lag_3m"] = (
    g["target_payment_usd"]
    .shift(3)
)

panel["payment_trailing_3m_sum"] = (
    g["target_payment_usd"]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(
            3,
            min_periods=1
        )
        .sum()
    )
)

panel["payment_trailing_3m_mean"] = (
    g["target_payment_usd"]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(
            3,
            min_periods=1
        )
        .mean()
    )
)

panel["payment_trailing_6m_sum"] = (
    g["target_payment_usd"]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(
            6,
            min_periods=1
        )
        .sum()
    )
)

panel["payment_trailing_6m_mean"] = (
    g["target_payment_usd"]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(
            6,
            min_periods=1
        )
        .mean()
    )
)

# Prior cumulative cash. Conflicting values are excluded, so
# reliability is separately tracked.
panel["_payment_for_cumsum"] = (
    panel["target_payment_usd"]
    .fillna(0.0)
)

panel["cumulative_paid_before_month"] = (
    panel.groupby("contractid")[
        "_payment_for_cumsum"
    ]
    .cumsum()
    -
    panel["_payment_for_cumsum"]
)

panel["pct_contract_paid_before_month"] = (
    panel["cumulative_paid_before_month"]
    /
    panel["price_usd"].replace(0, np.nan)
)

panel["remaining_contract_balance_before_month"] = (
    panel["price_usd"]
    -
    panel["cumulative_paid_before_month"]
).clip(lower=0)

panel["_paid_positive"] = np.where(
    panel["target_payment_usd"].fillna(0) > 0,
    1,
    0
)

panel["prior_paying_months"] = (
    panel.groupby("contractid")[
        "_paid_positive"
    ]
    .cumsum()
    -
    panel["_paid_positive"]
)

panel["_zero_observed_payment"] = np.where(
    panel["target_payment_usd"].eq(0),
    1,
    0
)

panel["zero_payment_months_prior_3"] = (
    panel.groupby("contractid")[
        "_zero_observed_payment"
    ]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(
            3,
            min_periods=1
        )
        .sum()
    )
)

panel["zero_payment_months_prior_6"] = (
    panel.groupby("contractid")[
        "_zero_observed_payment"
    ]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(
            6,
            min_periods=1
        )
        .sum()
    )
)

# Prior payment-conflict history.
panel["_payment_conflict_int"] = (
    panel["current_payment_conflict"]
    .astype(int)
)

panel["prior_payment_conflict_count"] = (
    panel.groupby("contractid")[
        "_payment_conflict_int"
    ]
    .cumsum()
    -
    panel["_payment_conflict_int"]
)

# Months since last positive observed payment.
panel["_positive_payment_month_index"] = np.where(
    panel["_paid_positive"].eq(1),
    panel["months_on_book"],
    np.nan
)

panel["_last_positive_mob_before"] = (
    panel.groupby("contractid")[
        "_positive_payment_month_index"
    ]
    .transform(
        lambda s:
        s.shift(1)
        .ffill()
    )
)

panel["months_since_last_positive_payment"] = (
    panel["months_on_book"]
    -
    panel["_last_positive_mob_before"]
)

# If no prior positive payment has been observed in the available window,
# keep missing rather than fabricating a duration.
panel.loc[
    panel["_last_positive_mob_before"].isna(),
    "months_since_last_positive_payment"
] = np.nan

panel["payment_history_left_censored"] = (
    panel["legacy_pre_oct_2024"]
)

panel["payment_history_reliable"] = (
    (~panel["current_payment_conflict"])
    &
    (panel["prior_payment_conflict_count"] == 0)
    &
    (~panel["payment_history_left_censored"])
)

# %% [markdown]
# ## 10. Calls features — prior months only
#
# These are **inbound customer calls**, not collections-agent calls.
#
# Current-month calls are excluded from predictors because we do not know whether the call happened before or after that month's payment.
#
# The notebook uses the call vocabulary already present in the cleaned development file; no sealed-period categories exist in this notebook.

# %%
# Exclude impossible / orphan records from aggregation.
for col in [
    "orphan_contractid",
    "before_sale_month"
]:
    if col in calls.columns:
        calls[col] = as_bool(
            calls[col]
        )

calls_usable = calls[
    (~calls["orphan_contractid"])
    &
    (~calls["before_sale_month"])
    &
    (calls["call_date"] >= ANALYSIS_START)
    &
    (calls["call_date"] <= VALIDATION_END)
].copy()

calls_usable["month"] = (
    calls_usable["call_date"]
    .dt.to_period("M")
    .dt.to_timestamp(how="end")
    .dt.normalize()
)

call_month_total = (
    calls_usable.groupby(
        ["contractid", "month"],
        as_index=False
    )
    .size()
    .rename(
        columns={"size": "calls_this_month"}
    )
)

call_reason_month = (
    calls_usable.groupby(
        [
            "contractid",
            "month",
            "call_reason"
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
    .reset_index()
)

# Safe feature names.
call_reason_rename = {}
for col in call_reason_month.columns:
    if col in [
        "contractid",
        "month"
    ]:
        continue

    safe = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        str(col)
    ).strip("_").lower()

    call_reason_rename[col] = (
        f"calls_reason_{safe}_this_month"
    )

call_reason_month = call_reason_month.rename(
    columns=call_reason_rename
)

call_monthly = call_month_total.merge(
    call_reason_month,
    on=[
        "contractid",
        "month"
    ],
    how="left"
)

panel = panel.merge(
    call_monthly,
    on=[
        "contractid",
        "month"
    ],
    how="left"
)

call_month_cols = [
    c for c in panel.columns
    if c == "calls_this_month"
    or (
        c.startswith("calls_reason_")
        and c.endswith("_this_month")
    )
]

panel[call_month_cols] = (
    panel[call_month_cols]
    .fillna(0)
)

for col in call_month_cols:
    prefix = col.replace(
        "_this_month",
        ""
    )

    panel[
        f"{prefix}_prior_1m"
    ] = (
        panel.groupby(
            "contractid"
        )[col]
        .shift(1)
        .fillna(0)
    )

    panel[
        f"{prefix}_prior_3m"
    ] = (
        panel.groupby(
            "contractid"
        )[col]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                3,
                min_periods=1
            )
            .sum()
        )
        .fillna(0)
    )

# Cumulative inbound-call count before month.
panel["calls_cumulative_before_month"] = (
    panel.groupby(
        "contractid"
    )["calls_this_month"]
    .cumsum()
    -
    panel["calls_this_month"]
)

# %% [markdown]
# ## 11. Service-ticket features — prior months only
#
# Ticket features use only months before the current contract-month.
#
# The existing canonical distinctions are retained, including:
#
# - `CABLE_CUT`
# - `CABLE_FAULT`
# - `WIRE_FAULT`
#
# Current-month tickets are not predictors because event order within the month is unknown.

# %%
for col in [
    "orphan_contractid",
    "before_sale_month"
]:
    if col in service.columns:
        service[col] = as_bool(
            service[col]
        )

service_usable = service[
    (~service["orphan_contractid"])
    &
    (~service["before_sale_month"])
    &
    (service["ticket_date"] >= ANALYSIS_START)
    &
    (service["ticket_date"] <= VALIDATION_END)
].copy()

service_usable["month"] = (
    service_usable["ticket_date"]
    .dt.to_period("M")
    .dt.to_timestamp(how="end")
    .dt.normalize()
)

ticket_month_total = (
    service_usable.groupby(
        ["contractid", "month"],
        as_index=False
    )
    .size()
    .rename(
        columns={"size": "tickets_this_month"}
    )
)

ticket_reason_month = (
    service_usable.groupby(
        [
            "contractid",
            "month",
            "ticket_reason"
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
    .reset_index()
)

ticket_reason_rename = {}
for col in ticket_reason_month.columns:
    if col in [
        "contractid",
        "month"
    ]:
        continue

    safe = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        str(col)
    ).strip("_").lower()

    ticket_reason_rename[col] = (
        f"tickets_reason_{safe}_this_month"
    )

ticket_reason_month = (
    ticket_reason_month.rename(
        columns=ticket_reason_rename
    )
)

ticket_monthly = (
    ticket_month_total.merge(
        ticket_reason_month,
        on=[
            "contractid",
            "month"
        ],
        how="left"
    )
)

panel = panel.merge(
    ticket_monthly,
    on=[
        "contractid",
        "month"
    ],
    how="left"
)

ticket_month_cols = [
    c for c in panel.columns
    if c == "tickets_this_month"
    or (
        c.startswith("tickets_reason_")
        and c.endswith("_this_month")
    )
]

panel[ticket_month_cols] = (
    panel[ticket_month_cols]
    .fillna(0)
)

for col in ticket_month_cols:
    prefix = col.replace(
        "_this_month",
        ""
    )

    panel[
        f"{prefix}_prior_1m"
    ] = (
        panel.groupby(
            "contractid"
        )[col]
        .shift(1)
        .fillna(0)
    )

    panel[
        f"{prefix}_prior_3m"
    ] = (
        panel.groupby(
            "contractid"
        )[col]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                3,
                min_periods=1
            )
            .sum()
        )
        .fillna(0)
    )

panel["tickets_cumulative_before_month"] = (
    panel.groupby(
        "contractid"
    )["tickets_this_month"]
    .cumsum()
    -
    panel["tickets_this_month"]
)

# %% [markdown]
# ## 12. Remove current-month event counts from the base predictor set
#
# We keep current-month counts temporarily only to calculate lagged features.
#
# They are dropped from the final forecast feature table so a future model cannot accidentally use information that may have occurred after payment.

# %%
leakage_event_cols = (
    call_month_cols
    +
    ticket_month_cols
)

panel = panel.drop(
    columns=[
        c for c in leakage_event_cols
        if c in panel.columns
    ]
)

# Remove temporary calculation columns.
temporary_cols = [
    "_payment_for_cumsum",
    "_paid_positive",
    "_zero_observed_payment",
    "_payment_conflict_int",
    "_positive_payment_month_index",
    "_last_positive_mob_before"
]

panel = panel.drop(
    columns=[
        c for c in temporary_cols
        if c in panel.columns
    ]
)

# %% [markdown]
# ## 13. Model-eligibility flags
#
# These are not modelling choices; they are transparent quality flags for the next notebook.
#
# `forecast_feature_row_eligible` means:
# - target is known;
# - payment history has no unresolved conflict;
# - the row is inside Oct-2024 to Jun-2026.
#
# The future modelling notebook may impose additional restrictions.

# %%
panel["forecast_feature_row_eligible"] = (
    panel["target_payment_usd"].notna()
    &
    panel["payment_history_reliable"]
    &
    panel["month"].between(
        ANALYSIS_START,
        VALIDATION_END
    )
)

panel["is_estimation_row"] = (
    panel["month"] <= ESTIMATION_END
)

panel["is_validation_row"] = (
    (panel["month"] > ESTIMATION_END)
    &
    (panel["month"] <= VALIDATION_END)
)

# %% [markdown]
# ## 14. Region-month summary
#
# This table is especially important later because Apr–Jun contains the East/West pilots.
#
# The future modelling notebook should inspect validation performance by region rather than allowing East/West pilot-era behaviour to disappear inside a country total.

# %%
region_month = (
    panel.groupby(
        [
            "month",
            "region"
        ],
        dropna=False,
        as_index=False
    )
    .agg(
        total_collections_usd=(
            "target_payment_usd",
            "sum"
        ),
        contract_month_rows=(
            "contractid",
            "size"
        ),
        unique_contracts=(
            "contractid",
            "nunique"
        ),
        financed_contract_months=(
            "contract_type",
            lambda s:
            (s == "FINANCED").sum()
        ),
        cash_contract_months=(
            "contract_type",
            lambda s:
            (s == "CASH").sum()
        ),
        remaining_balance_before_month_usd=(
            "remaining_contract_balance_before_month",
            "sum"
        ),
        new_contracts=(
            "months_on_book",
            lambda s:
            (s == 0).sum()
        ),
        eligible_feature_rows=(
            "forecast_feature_row_eligible",
            "sum"
        )
    )
)

region_month["time_layer"] = np.where(
    region_month["month"] <= ESTIMATION_END,
    "estimation",
    "validation"
)

show_table(region_month.tail(16), max_rows=16, title="Region-month tail")

# %% [markdown]
# ## 15. Country-month summary

# %%
country_month = (
    panel.groupby(
        "month",
        as_index=False
    )
    .agg(
        total_collections_usd=(
            "target_payment_usd",
            "sum"
        ),
        contract_month_rows=(
            "contractid",
            "size"
        ),
        unique_contracts=(
            "contractid",
            "nunique"
        ),
        financed_contract_months=(
            "contract_type",
            lambda s:
            (s == "FINANCED").sum()
        ),
        cash_contract_months=(
            "contract_type",
            lambda s:
            (s == "CASH").sum()
        ),
        remaining_balance_before_month_usd=(
            "remaining_contract_balance_before_month",
            "sum"
        ),
        new_contracts=(
            "months_on_book",
            lambda s:
            (s == 0).sum()
        ),
        eligible_feature_rows=(
            "forecast_feature_row_eligible",
            "sum"
        )
    )
)

country_month["time_layer"] = np.where(
    country_month["month"] <= ESTIMATION_END,
    "estimation",
    "validation"
)

show_table(country_month, max_rows=50, title="Country-month summary")

# %% [markdown]
# ## 16. Separate pilot-analysis feature table
#
# **Outreach is not merged into the base forecast panel.**
#
# For each pilot record, this table attaches only features that would have been available at the **start of the contact month**.
#
# It also creates payment outcomes separately:
# - contact-month payment;
# - next-month payment, where that month is still within Jun-2026;
# - second-next-month payment, where available.
#
# Important causal warning:
# the contact date is only monthly, so contact-month payment may have occurred before or after outreach. `next_month_payment_usd` is temporally cleaner, but using it reduces the usable pilot sample, especially for June contacts.

# %%
# Start-of-month feature columns to carry into pilot analysis.
pilot_feature_cols = [
    "contractid",
    "month",
    "sales_month",
    "region",
    "product",
    "contract_type",
    "payment_frequency",
    "price_usd",
    "perc_deposit",
    "daily_amount_usd",
    "tenor_length",
    "months_on_book",
    "deposit_usd",
    "financed_amount_after_deposit_usd",
    "scheduled_amount_proxy_usd",
    "payment_lag_1m",
    "payment_lag_2m",
    "payment_lag_3m",
    "payment_trailing_3m_sum",
    "payment_trailing_3m_mean",
    "payment_trailing_6m_sum",
    "payment_trailing_6m_mean",
    "cumulative_paid_before_month",
    "pct_contract_paid_before_month",
    "remaining_contract_balance_before_month",
    "prior_paying_months",
    "zero_payment_months_prior_3",
    "zero_payment_months_prior_6",
    "months_since_last_positive_payment",
    "calls_cumulative_before_month",
    "tickets_cumulative_before_month",
    "payment_history_reliable",
]

# Include all lagged call/ticket reason features.
pilot_feature_cols += [
    c for c in panel.columns
    if (
        c.startswith("calls_")
        or c.startswith("tickets_")
    )
    and (
        c.endswith("_prior_1m")
        or c.endswith("_prior_3m")
    )
]

pilot_feature_cols = list(
    dict.fromkeys(
        [
            c for c in pilot_feature_cols
            if c in panel.columns
        ]
    )
)

pilot_base = panel[
    pilot_feature_cols
].copy()

outreach_for_pilot = outreach.copy()

for col in [
    "orphan_contractid",
    "before_sale_month"
]:
    if col in outreach_for_pilot.columns:
        outreach_for_pilot[col] = as_bool(
            outreach_for_pilot[col]
        )

outreach_for_pilot = outreach_for_pilot[
    (~outreach_for_pilot["orphan_contractid"])
    &
    (~outreach_for_pilot["before_sale_month"])
    &
    (outreach_for_pilot["contact_month"] >= pd.Timestamp("2026-04-30"))
    &
    (outreach_for_pilot["contact_month"] <= VALIDATION_END)
].copy()

pilot = outreach_for_pilot.merge(
    pilot_base,
    left_on=[
        "contractid",
        "contact_month"
    ],
    right_on=[
        "contractid",
        "month"
    ],
    how="left",
    suffixes=(
        "_outreach",
        "_contract"
    )
)

# Payment outcome lookup.
payment_target_lookup = panel[
    [
        "contractid",
        "month",
        "target_payment_usd"
    ]
].copy()

pilot["contact_month_outcome"] = (
    pilot["contact_month"]
)

pilot["next_month_outcome"] = (
    pilot["contact_month"]
    +
    pd.offsets.MonthEnd(1)
)

pilot["second_next_month_outcome"] = (
    pilot["contact_month"]
    +
    pd.offsets.MonthEnd(2)
)

def attach_outcome(
    df,
    outcome_month_col,
    output_col
):
    lookup = payment_target_lookup.rename(
        columns={
            "month": outcome_month_col,
            "target_payment_usd": output_col
        }
    )

    return df.merge(
        lookup,
        on=[
            "contractid",
            outcome_month_col
        ],
        how="left"
    )

pilot = attach_outcome(
    pilot,
    "contact_month_outcome",
    "payment_contact_month_usd"
)

pilot = attach_outcome(
    pilot,
    "next_month_outcome",
    "payment_next_month_usd"
)

pilot = attach_outcome(
    pilot,
    "second_next_month_outcome",
    "payment_second_next_month_usd"
)

pilot["next_month_outcome_observed"] = (
    pilot["next_month_outcome"]
    <= VALIDATION_END
)

pilot["second_next_month_outcome_observed"] = (
    pilot["second_next_month_outcome"]
    <= VALIDATION_END
)

# Never interpret an unavailable Jul/Aug outcome as zero.
pilot.loc[
    ~pilot["next_month_outcome_observed"],
    "payment_next_month_usd"
] = np.nan

pilot.loc[
    ~pilot["second_next_month_outcome_observed"],
    "payment_second_next_month_usd"
] = np.nan

print("Pilot-analysis rows:", len(pilot))
show_table(
    pilot[
        [
            "contact_month",
            "region_outreach",
            "channel",
            "attempts",
            "reached",
            "cost_usd",
            "payment_contact_month_usd",
            "payment_next_month_usd",
            "next_month_outcome_observed"
        ]
    ].head(10),
    max_rows=10,
    title="Pilot feature sample"
)

# %% [markdown]
# ## 17. Feature quality checks
#
# These checks make sure feature engineering itself did not introduce leakage or structural problems.

# %%
quality_checks = []

quality_checks.append([
    "contract_month_rows",
    len(panel)
])

quality_checks.append([
    "duplicate_contract_month_rows",
    int(
        panel.duplicated(
            ["contractid", "month"]
        ).sum()
    )
])

quality_checks.append([
    "rows_after_jun_2026",
    int(
        (panel["month"] > VALIDATION_END).sum()
    )
])

quality_checks.append([
    "rows_before_oct_2024",
    int(
        (panel["month"] < ANALYSIS_START).sum()
    )
])

quality_checks.append([
    "rows_with_current_payment_conflict",
    int(
        panel["current_payment_conflict"].sum()
    )
])

quality_checks.append([
    "rows_with_prior_payment_conflict",
    int(
        (panel["prior_payment_conflict_count"] > 0).sum()
    )
])

quality_checks.append([
    "legacy_contract_month_rows_left_censored",
    int(
        panel["payment_history_left_censored"].sum()
    )
])

quality_checks.append([
    "forecast_feature_eligible_rows",
    int(
        panel["forecast_feature_row_eligible"].sum()
    )
])

quality_checks.append([
    "pilot_rows",
    len(pilot)
])

quality_checks.append([
    "pilot_rows_with_missing_start_of_month_features",
    int(
        pilot["months_on_book"].isna().sum()
    )
])

quality_checks.append([
    "model_usable_minus_panel_payment_usd",
    float(
        payment_reconciliation_by_month[
            "unassigned_usable_payment_usd"
        ].sum()
    )
])

quality_checks.append([
    "source_minus_model_usable_payment_usd",
    float(
        payment_reconciliation_by_month[
            "source_payment_total_usd"
        ].sum()
        -
        payment_reconciliation_by_month[
            "model_usable_payment_usd"
        ].sum()
    )
])

feature_quality_checks = pd.DataFrame(
    quality_checks,
    columns=[
        "check",
        "value"
    ]
)

show_table(feature_quality_checks, max_rows=50, title="Feature quality checks")

assert feature_quality_checks.loc[
    feature_quality_checks["check"].eq(
        "duplicate_contract_month_rows"
    ),
    "value"
].iloc[0] == 0

assert feature_quality_checks.loc[
    feature_quality_checks["check"].eq(
        "rows_after_jun_2026"
    ),
    "value"
].iloc[0] == 0

# %% [markdown]
# ## 17a. Independent sanity checks and cross-notebook metrics
#
# The checks below are intentionally redundant. They recompute expected values from source-level monthly aggregates rather than trusting the feature columns simply because the code ran.
#
# ### Checks
#
# - contract-month key uniqueness;
# - no rows after Jun-2026;
# - region totals reconcile exactly to country totals;
# - every model-usable payment dollar is accounted for;
# - `payment_lag_1m` agrees with a separately shifted payment lookup;
# - prior-1-month inbound-call counts agree with independently shifted call aggregates;
# - prior-1-month ticket counts agree with independently shifted ticket aggregates;
# - no same-month call/ticket columns remain in the base table;
# - no collections-outreach treatment columns entered the base table.
#
# A `WARN` is not automatically a bug. For example, source dollars excluded because they are dated before the contract sale month are a business-data issue that must be documented.
#
# The compact JSON generated here is the preferred way to compare two notebooks.

# %%
# ---------------------------
# Sanity report helper
# ---------------------------
_sanity_rows = []

def add_check(name, status, actual, expected, layer, note=""):
    _sanity_rows.append({
        "check": name,
        "status": status,
        "actual": actual,
        "expected": expected,
        "layer": layer,
        "note": note
    })

# 1) Panel key / date checks.
_duplicate_panel_keys = int(
    panel.duplicated(["contractid", "month"]).sum()
)
add_check(
    "contract_month_key_unique",
    "PASS" if _duplicate_panel_keys == 0 else "FAIL",
    _duplicate_panel_keys,
    0,
    "engineered",
    "There must be one row per contract-month."
)

_future_rows = int(
    (panel["month"] > VALIDATION_END).sum()
)
add_check(
    "no_rows_after_jun_2026",
    "PASS" if _future_rows == 0 else "FAIL",
    _future_rows,
    0,
    "engineered",
    "Jul-Sep must remain sealed."
)

# 2) Region -> country aggregation reconciliation.
_region_rollup = (
    region_month.groupby("month", as_index=False)["total_collections_usd"]
    .sum()
    .rename(columns={"total_collections_usd": "region_rollup_usd"})
)
_country_recon = country_month[
    ["month", "total_collections_usd"]
].merge(
    _region_rollup,
    on="month",
    how="outer"
)
_country_recon["abs_diff"] = (
    _country_recon["total_collections_usd"]
    - _country_recon["region_rollup_usd"]
).abs()

_region_country_max_diff = float(
    _country_recon["abs_diff"].fillna(np.inf).max()
)
add_check(
    "region_sums_equal_country_total",
    "PASS" if _region_country_max_diff < 0.01 else "FAIL",
    round(_region_country_max_diff, 6),
    "< 0.01 USD",
    "engineered",
    "Regional aggregation should exactly reproduce country totals."
)

# 3) Model-usable dollars -> panel target.
_unassigned_total = float(
    payment_reconciliation_by_month[
        "unassigned_usable_payment_usd"
    ].sum()
)
add_check(
    "model_usable_payment_dollars_reconcile_to_panel",
    "PASS" if abs(_unassigned_total) < 0.01 else "FAIL",
    round(_unassigned_total, 2),
    "0.00 USD",
    "engineered",
    "Any non-zero amount is traced in unassigned_usable_payments.csv."
)

# 4) Source -> usable gap is expected to be explainable, not necessarily zero.
_source_total = float(
    payment_reconciliation_by_month[
        "source_payment_total_usd"
    ].sum()
)
_usable_total = float(
    payment_reconciliation_by_month[
        "model_usable_payment_usd"
    ].sum()
)
_source_gap = _source_total - _usable_total

add_check(
    "source_vs_model_usable_payment_gap",
    "WARN" if abs(_source_gap) >= 0.01 else "PASS",
    round(_source_gap, 2),
    "Explainable, not necessarily zero",
    "input",
    "Gap should be explained by conflicts, before-sale rows, or other explicit exclusions."
)

# ---------------------------
# Independent payment lag-1 check
# ---------------------------
_prev_pay_lookup = payments_usable[
    ["contractid", "pay_month", "total_paid"]
].copy()
_prev_pay_lookup["month"] = (
    _prev_pay_lookup["pay_month"]
    + pd.offsets.MonthEnd(1)
)
_prev_pay_lookup = _prev_pay_lookup[
    ["contractid", "month", "total_paid"]
].rename(columns={
    "total_paid": "_expected_payment_lag_1m"
})

_prev_conflict_lookup = payment_conflicts.copy()
_prev_conflict_lookup["month"] = (
    _prev_conflict_lookup["pay_month"]
    + pd.offsets.MonthEnd(1)
)
_prev_conflict_lookup = _prev_conflict_lookup[
    ["contractid", "month"]
].assign(_prev_month_conflict=True)

_lag_check = panel[
    ["contractid", "month", "months_on_book", "payment_lag_1m"]
].merge(
    _prev_pay_lookup,
    on=["contractid", "month"],
    how="left"
).merge(
    _prev_conflict_lookup,
    on=["contractid", "month"],
    how="left"
)

_lag_check["_prev_month_conflict"] = (
    _lag_check["_prev_month_conflict"]
    .fillna(False)
)

# Only rows that actually have a prior contract-month in the panel.
_lag_check = _lag_check[
    _lag_check["months_on_book"] > 0
].copy()

# If previous month had no payment record, expected observed payment is zero.
_lag_check["_expected_payment_lag_1m"] = (
    _lag_check["_expected_payment_lag_1m"]
    .fillna(0.0)
)

# If previous month was unresolved conflict, v1/v2 intentionally has NaN lag.
_lag_expected_nan = _lag_check["_prev_month_conflict"]

_lag_numeric_mismatch = (
    (~_lag_expected_nan)
    & (
        (
            _lag_check["payment_lag_1m"]
            - _lag_check["_expected_payment_lag_1m"]
        ).abs() > 1e-9
    )
)

_lag_nan_mismatch = (
    _lag_expected_nan
    & _lag_check["payment_lag_1m"].notna()
)

_payment_lag1_mismatches = int(
    (_lag_numeric_mismatch | _lag_nan_mismatch).sum()
)

add_check(
    "payment_lag_1m_matches_shifted_source",
    "PASS" if _payment_lag1_mismatches == 0 else "FAIL",
    _payment_lag1_mismatches,
    0,
    "engineered",
    "Independent check against prior-month cleaned payment source."
)

# ---------------------------
# Independent calls lag-1 check
# ---------------------------
_expected_calls_prev = call_month_total.copy()
_expected_calls_prev["month"] = (
    _expected_calls_prev["month"]
    + pd.offsets.MonthEnd(1)
)
_expected_calls_prev = _expected_calls_prev.rename(
    columns={"calls_this_month": "_expected_calls_prior_1m"}
)

_calls_check = panel[
    ["contractid", "month", "calls_prior_1m"]
].merge(
    _expected_calls_prev[
        ["contractid", "month", "_expected_calls_prior_1m"]
    ],
    on=["contractid", "month"],
    how="left"
)

_calls_check["_expected_calls_prior_1m"] = (
    _calls_check["_expected_calls_prior_1m"]
    .fillna(0)
)

_call_lag1_mismatches = int(
    (
        _calls_check["calls_prior_1m"]
        - _calls_check["_expected_calls_prior_1m"]
    ).abs().gt(1e-9).sum()
)

add_check(
    "calls_prior_1m_matches_shifted_source",
    "PASS" if _call_lag1_mismatches == 0 else "FAIL",
    _call_lag1_mismatches,
    0,
    "engineered",
    "Independent check against source call counts shifted one month."
)

# ---------------------------
# Independent ticket lag-1 check
# ---------------------------
_expected_tickets_prev = ticket_month_total.copy()
_expected_tickets_prev["month"] = (
    _expected_tickets_prev["month"]
    + pd.offsets.MonthEnd(1)
)
_expected_tickets_prev = _expected_tickets_prev.rename(
    columns={"tickets_this_month": "_expected_tickets_prior_1m"}
)

_tickets_check = panel[
    ["contractid", "month", "tickets_prior_1m"]
].merge(
    _expected_tickets_prev[
        ["contractid", "month", "_expected_tickets_prior_1m"]
    ],
    on=["contractid", "month"],
    how="left"
)

_tickets_check["_expected_tickets_prior_1m"] = (
    _tickets_check["_expected_tickets_prior_1m"]
    .fillna(0)
)

_ticket_lag1_mismatches = int(
    (
        _tickets_check["tickets_prior_1m"]
        - _tickets_check["_expected_tickets_prior_1m"]
    ).abs().gt(1e-9).sum()
)

add_check(
    "tickets_prior_1m_matches_shifted_source",
    "PASS" if _ticket_lag1_mismatches == 0 else "FAIL",
    _ticket_lag1_mismatches,
    0,
    "engineered",
    "Independent check against source ticket counts shifted one month."
)

# 5) Leakage-column checks.
_same_month_event_cols = [
    c for c in panel.columns
    if c.endswith("_this_month")
    and (
        c.startswith("calls")
        or c.startswith("tickets")
    )
]
add_check(
    "no_same_month_call_or_ticket_predictors",
    "PASS" if len(_same_month_event_cols) == 0 else "FAIL",
    len(_same_month_event_cols),
    0,
    "engineered",
    ", ".join(_same_month_event_cols[:10])
)

_outreach_terms = {
    "channel", "attempts", "reached", "cost_usd",
    "contact_month"
}
_outreach_cols_in_panel = [
    c for c in panel.columns
    if c in _outreach_terms
    or c.startswith("outreach_")
]
add_check(
    "no_outreach_treatment_columns_in_base_panel",
    "PASS" if len(_outreach_cols_in_panel) == 0 else "FAIL",
    len(_outreach_cols_in_panel),
    0,
    "engineered",
    ", ".join(_outreach_cols_in_panel[:10])
)

sanity_check_report = pd.DataFrame(_sanity_rows)

show_table(
    sanity_check_report,
    max_rows=100,
    title="Sanity check report"
)

# ---------------------------
# Compact cross-notebook metrics
# ---------------------------
def _round_float(x, ndigits=6):
    if pd.isna(x):
        return None
    return round(float(x), ndigits)

def _month_dict(df, month_col, value_col, ndigits=2):
    out = {}
    for _, row in df[[month_col, value_col]].iterrows():
        key = pd.Timestamp(row[month_col]).strftime("%Y-%m-%d")
        value = row[value_col]
        out[key] = None if pd.isna(value) else round(float(value), ndigits)
    return out

def _feature_sum(column_name):
    if column_name not in panel.columns:
        return None
    return _round_float(
        pd.to_numeric(panel[column_name], errors="coerce").sum()
    )

feature_metrics = {
    "input_layer": {
        "contracts_rows": int(len(contracts)),
        "contracts_unique_ids": int(contracts["contractid"].nunique()),
        "payments_rows": int(len(payments)),
        "payments_total_usd_oct_to_jun": _round_float(_source_total, 2),
        "payments_model_usable_usd_oct_to_jun": _round_float(_usable_total, 2),
        "payment_conflict_contract_months": int(len(payment_conflicts)),
        "calls_rows": int(len(calls)),
        "service_ticket_rows": int(len(service)),
        "outreach_rows": int(len(outreach)),
        "source_payment_by_month_usd": _month_dict(
            payment_reconciliation_by_month,
            "month",
            "source_payment_total_usd",
            2
        ),
        "model_usable_payment_by_month_usd": _month_dict(
            payment_reconciliation_by_month,
            "month",
            "model_usable_payment_usd",
            2
        ),
    },
    "engineered_layer": {
        "panel_rows": int(len(panel)),
        "panel_unique_contracts": int(panel["contractid"].nunique()),
        "panel_unique_contract_months": int(
            panel[["contractid", "month"]].drop_duplicates().shape[0]
        ),
        "panel_column_count": int(panel.shape[1]),
        "panel_columns_sha256": hashlib.sha256(
            "\n".join(sorted(panel.columns)).encode("utf-8")
        ).hexdigest(),
        "forecast_feature_eligible_rows": int(
            panel["forecast_feature_row_eligible"].sum()
        ),
        "current_payment_conflict_rows": int(
            panel["current_payment_conflict"].sum()
        ),
        "prior_payment_conflict_rows": int(
            (panel["prior_payment_conflict_count"] > 0).sum()
        ),
        "panel_target_total_usd": _round_float(
            panel["target_payment_usd"].sum(),
            2
        ),
        "panel_target_by_month_usd": _month_dict(
            country_month,
            "month",
            "total_collections_usd",
            2
        ),
        "payment_lag_1m_sum": _feature_sum("payment_lag_1m"),
        "payment_trailing_3m_sum_total": _feature_sum("payment_trailing_3m_sum"),
        "payment_trailing_6m_sum_total": _feature_sum("payment_trailing_6m_sum"),
        "cumulative_paid_before_month_sum": _feature_sum("cumulative_paid_before_month"),
        "remaining_balance_before_month_sum": _feature_sum("remaining_contract_balance_before_month"),
        "calls_prior_1m_sum": _feature_sum("calls_prior_1m"),
        "calls_prior_3m_sum": _feature_sum("calls_prior_3m"),
        "tickets_prior_1m_sum": _feature_sum("tickets_prior_1m"),
        "tickets_prior_3m_sum": _feature_sum("tickets_prior_3m"),
        "region_month_rows": int(len(region_month)),
        "country_month_rows": int(len(country_month)),
        "pilot_rows": int(len(pilot)),
    },
    "reconciliation": {
        "source_minus_model_usable_usd": _round_float(_source_gap, 2),
        "model_usable_minus_panel_usd": _round_float(_unassigned_total, 2),
        "region_country_max_abs_diff_usd": _round_float(_region_country_max_diff, 6),
    },
    "sanity": {
        row["check"]: row["status"]
        for row in _sanity_rows
    }
}

# ---------------------------
# Cross-notebook comparison helper
# ---------------------------
def flatten_metrics(obj, prefix=""):
    flat = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_metrics(value, next_prefix))
    else:
        flat[prefix] = obj
    return flat

def compare_metrics(mine, theirs, atol=1e-6, rtol=1e-9):
    a = flatten_metrics(mine)
    b = flatten_metrics(theirs)

    rows = []
    for key in sorted(set(a) | set(b)):
        left = a.get(key, "__MISSING__")
        right = b.get(key, "__MISSING__")

        if left == "__MISSING__" or right == "__MISSING__":
            same = False
            delta = None
        elif (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            same = math.isclose(
                float(left),
                float(right),
                abs_tol=atol,
                rel_tol=rtol
            )
            delta = float(left) - float(right)
        else:
            same = left == right
            delta = None

        if key.startswith("input_layer."):
            layer = "INPUT"
        elif key.startswith("engineered_layer."):
            layer = "ENGINEERED"
        elif key.startswith("reconciliation."):
            layer = "RECONCILIATION"
        else:
            layer = "SANITY"

        rows.append({
            "metric": key,
            "layer": layer,
            "mine": left,
            "theirs": right,
            "delta_mine_minus_theirs": delta,
            "result": "MATCH" if same else "DIFF"
        })

    return pd.DataFrame(rows)

# Demonstration:
# mine = json.load(open(OUTPUT_DIR / "feature_metrics.json"))
# theirs = json.load(open("/content/feature_metrics_OTHER.json"))
# comparison = compare_metrics(mine, theirs)
# show_table(comparison[comparison["result"] == "DIFF"], max_rows=200)


# %% [markdown]
# ## 18. Feature dictionary

# %%
feature_dictionary_rows = [
    (
        "target_payment_usd",
        "target",
        "Actual cash paid by the contract in the row month. Never use as a predictor for the same row."
    ),
    (
        "months_on_book",
        "contract",
        "Calendar months since sales_month."
    ),
    (
        "deposit_usd",
        "contract",
        "price_usd × cleaned perc_deposit."
    ),
    (
        "financed_amount_after_deposit_usd",
        "contract",
        "Price less deposit for financed contracts."
    ),
    (
        "scheduled_amount_proxy_usd",
        "contract",
        "daily_amount_usd × days in calendar month. Proxy only because exact sale date is unavailable."
    ),
    (
        "payment_lag_1m",
        "payment_history",
        "Previous calendar month's payment only."
    ),
    (
        "payment_trailing_3m_sum",
        "payment_history",
        "Sum of payments in up to the 3 months strictly before the row month."
    ),
    (
        "payment_trailing_6m_sum",
        "payment_history",
        "Sum of payments in up to the 6 months strictly before the row month."
    ),
    (
        "cumulative_paid_before_month",
        "payment_history",
        "Observed cumulative cash strictly before the row month."
    ),
    (
        "pct_contract_paid_before_month",
        "payment_history",
        "Observed cumulative cash before month divided by contract price."
    ),
    (
        "remaining_contract_balance_before_month",
        "payment_history",
        "Contract price minus observed cumulative payments before the month, floored at zero."
    ),
    (
        "zero_payment_months_prior_3",
        "payment_history",
        "Count of zero-payment months in the prior 3 contract-month rows."
    ),
    (
        "months_since_last_positive_payment",
        "payment_history",
        "Months since the last observed positive payment, using past months only."
    ),
    (
        "calls_*_prior_1m / prior_3m",
        "calls",
        "Inbound customer-call counts from prior month(s) only."
    ),
    (
        "tickets_*_prior_1m / prior_3m",
        "service",
        "Service-ticket counts from prior month(s) only."
    ),
    (
        "payment_history_reliable",
        "quality",
        "False if payment history is left-censored or has unresolved payment conflicts."
    ),
    (
        "known_by_mar_2026_origin",
        "time",
        "Whether the contract existed by the estimation/validation split origin."
    ),
    (
        "is_estimation_row",
        "time",
        "Row month is Oct-2024 through Mar-2026."
    ),
    (
        "is_validation_row",
        "time",
        "Row month is Apr-Jun-2026."
    ),
]

feature_dictionary = pd.DataFrame(
    feature_dictionary_rows,
    columns=[
        "feature",
        "group",
        "definition_and_leakage_note"
    ]
)

show_table(feature_dictionary, max_rows=50, title="Feature dictionary")

# %% [markdown]
# ## 19. Save feature-engineering outputs
#
# No forecast is produced here.

# %%
# Save panel with month-end dates as ISO text.
panel_export = panel.copy()
region_export = region_month.copy()
country_export = country_month.copy()
pilot_export = pilot.copy()

panel_export.to_csv(
    OUTPUT_DIR / "contract_month_features.csv",
    index=False
)

region_export.to_csv(
    OUTPUT_DIR / "region_month_features.csv",
    index=False
)

country_export.to_csv(
    OUTPUT_DIR / "country_month_features.csv",
    index=False
)

pilot_export.to_csv(
    OUTPUT_DIR / "pilot_analysis_features.csv",
    index=False
)

feature_dictionary.to_csv(
    OUTPUT_DIR / "feature_dictionary.csv",
    index=False
)

feature_quality_checks.to_csv(
    OUTPUT_DIR / "feature_quality_checks.csv",
    index=False
)

print("Saved feature outputs to:", OUTPUT_DIR)

payment_reconciliation_by_month.to_csv(
    OUTPUT_DIR / "payment_reconciliation_by_month.csv",
    index=False
)

unassigned_usable_payments.to_csv(
    OUTPUT_DIR / "unassigned_usable_payments.csv",
    index=False
)

sanity_check_report.to_csv(
    OUTPUT_DIR / "sanity_check_report.csv",
    index=False
)

with open(
    OUTPUT_DIR / "feature_metrics.json",
    "w"
) as f:
    json.dump(
        feature_metrics,
        f,
        indent=2,
        sort_keys=True
    )

print("Also saved:")
print(" - payment_reconciliation_by_month.csv")
print(" - unassigned_usable_payments.csv")
print(" - sanity_check_report.csv")
print(" - feature_metrics.json")


# %% [markdown]
# ## 20. Package and download

# %%
zip_path = Path(
    "/content/dlight_feature_engineering_outputs_v2.zip"
)

with zipfile.ZipFile(
    zip_path,
    "w",
    zipfile.ZIP_DEFLATED
) as z:
    for p in OUTPUT_DIR.glob("*.csv"):
        z.write(
            p,
            arcname=p.name
        )

print("Created:", zip_path)

from google.colab import files
files.download(str(zip_path))

# %% [markdown]
# # Stop here
#
# The feature engineering stage is complete.
#
# The next notebook should decide:
#
# - which feature groups belong in the **headline cohort/vintage model**;
# - which belong only in a **challenger model**;
# - how to validate the base model by region during Apr–Jun;
# - how to handle contracts with unreliable payment history;
# - how to use the given Jul/Aug/Sep sales plan for genuinely unsold contracts;
# - how to estimate pilot effectiveness separately without contaminating the base forecast.
#
# Do not open the sealed Jul–Sep cleaning ZIP for any of those choices.
#
# ## Comparing against a second notebook
#
# The first comparison artifacts to exchange are:
#
# 1. `feature_metrics.json`
# 2. `sanity_check_report.csv`
# 3. `payment_reconciliation_by_month.csv`
# 4. `feature_quality_checks.csv`
# 5. `country_month_features.csv`
#
# Do **not** exchange the full 700k-row contract-month file unless these compact checks reveal a discrepancy that needs row-level investigation.
#

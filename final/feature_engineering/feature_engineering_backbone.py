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
# # Feature Engineering Backbone
#
# This notebook is the **feature-engineering layer only**. It does not fit or select a forecasting model.
#
# It incorporates the lessons from two independent implementations:
#
# 1. **The cohort / contractual backbone must exist explicitly.**
#    We calculate expected deposit, expected repayment schedule, cumulative expected cash, cumulative actual cash, and collection efficiency by months-on-book.
#
# 2. **Late cash must not disappear just because a contract is past tenor.**
#    Financed contracts remain in the analytical panel through the development horizon, but every row is labelled as either inside the scheduled repayment window or post-tenor recovery.
#
# 3. **Accounting cash and model-attributable cash are separate concepts.**
#    A monthly reconciliation bridge explains every difference between source payment dollars and the clean contract-month target. Excluded / unresolved cash is retained as a separate component for the future forecast.
#
# 4. **Behavioural features remain useful, but they are a challenger layer.**
#    Lagged payments, inbound calls, and service tickets use prior months only.
#
# 5. **Cross-notebook comparison uses one canonical flat JSON schema.**
#    The notebook creates and downloads `feature_metrics.json` directly, plus a richer detailed JSON and CSV sanity reports.
#
# ## Time discipline
#
# - Estimation: Oct-2024 to Mar-2026
# - Validation: Apr-2026 to Jun-2026
# - Jul-Sep-2026 remains sealed and is not loaded here.
#
# ## Input
#
# Upload:
#
# `development_through_jun_2026_v3.zip`
#
# from the cleaning notebook.

# %% [markdown]
# ## 1. Upload the cleaned development ZIP

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
import json
import math
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Keep warnings visible during development.
pd.set_option("display.max_columns", 160)
pd.set_option("display.width", 220)

try:
    from google.colab import data_table
    data_table.disable_dataframe_formatter()
except Exception:
    pass

ANALYSIS_START = pd.Timestamp("2024-10-31")
ESTIMATION_END = pd.Timestamp("2026-03-31")
VALIDATION_END = pd.Timestamp("2026-06-30")

# Month-age proxy. Exact sale day is unavailable.
AVG_DAYS_PER_MONTH = 365.25 / 12

OUTPUT_DIR = Path("/content/dlight_feature_engineering_v3")
OUTPUT_DIR.mkdir(exist_ok=True)

def show_table(df, max_rows=30, title=None):
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    if df is None:
        print("None")
        return
    print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"... {len(df) - max_rows:,} additional rows not printed")


# %% [markdown]
# ## 3. Extract and load the cleaned development files

# %%
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
zip_name = preferred[0] if preferred else zip_names[0]

EXTRACT_DIR = Path("/content/dlight_dev_v3")
EXTRACT_DIR.mkdir(exist_ok=True)

with zipfile.ZipFile(
    io.BytesIO(uploaded[zip_name]),
    "r"
) as z:
    z.extractall(EXTRACT_DIR)

def find_csv(keyword):
    matches = [
        p for p in EXTRACT_DIR.rglob("*.csv")
        if keyword.lower() in p.name.lower()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No CSV matching '{keyword}'"
        )
    return matches[0]

contracts = pd.read_csv(find_csv("contracts_clean"))
payments = pd.read_csv(find_csv("payments_clean"))
calls = pd.read_csv(find_csv("calls_clean"))
service = pd.read_csv(find_csv("service_tickets_clean"))
outreach = pd.read_csv(find_csv("collections_outreach_clean"))

for df, col in [
    (contracts, "sales_month"),
    (payments, "pay_month"),
    (calls, "call_date"),
    (service, "ticket_date"),
    (outreach, "contact_month"),
]:
    df[col] = pd.to_datetime(
        df[col],
        errors="coerce"
    )

print("Loaded development data only:")
print("contracts:", contracts.shape)
print("payments:", payments.shape)
print("calls:", calls.shape)
print("service:", service.shape)
print("outreach:", outreach.shape)


# %% [markdown]
# ## 4. Boolean helper and input-period assertions

# %%
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

for df, date_col in [
    (contracts, "sales_month"),
    (payments, "pay_month"),
    (calls, "call_date"),
    (service, "ticket_date"),
    (outreach, "contact_month"),
]:
    assert (
        df[date_col].dropna().max()
        <= VALIDATION_END
    ), f"{date_col} contains data after Jun-2026"

print("PASS: no Jul-Sep records are loaded.")

# %% [markdown]
# ## 5. Payment accounting layers
#
# We keep three distinct concepts:
#
# ### A. Source-reported cleaned cash
# All payment rows remaining after the cleaning notebook's exact de-duplication.
#
# ### B. Model-attributable cash
# Rows that can be attached confidently to a contract-month:
# - inside the Oct-2024 to Jun-2026 analysis window;
# - not orphaned;
# - not before sale;
# - not negative;
# - not an unresolved conflicting contract-month.
#
# ### C. Excluded / unresolved cash
# The difference between A and B is **not thrown away**. It is quantified by reason and exported for the later forecast reconciliation.
#
# This prevents the forecast from silently becoming a forecast of only the clean subset of country collections.

# %%
payment_scope = payments[
    payments["pay_month"].between(
        ANALYSIS_START,
        VALIDATION_END
    )
].copy()

payment_conflicts = (
    payment_scope.loc[
        payment_scope["conflicting_contract_month"],
        ["contractid", "pay_month"]
    ]
    .drop_duplicates()
    .assign(current_payment_conflict=True)
)

payments_usable = payment_scope[
    (~payment_scope["orphan_contractid"])
    &
    (~payment_scope["before_sale_month"])
    &
    (~payment_scope["negative_payment"])
    &
    (~payment_scope["conflicting_contract_month"])
].copy()

assert not payments_usable.duplicated(
    ["contractid", "pay_month"]
).any()

print("Source payment rows in model period:", len(payment_scope))
print("Model-attributable payment rows:", len(payments_usable))
print("Conflicting contract-months:", len(payment_conflicts))

# %% [markdown]
# ## 6. Monthly payment reconciliation bridge

# %%
source_month = (
    payment_scope.groupby(
        "pay_month",
        as_index=False
    )["total_paid"]
    .sum()
    .rename(columns={
        "pay_month": "month",
        "total_paid": "source_reported_cash_usd"
    })
)

usable_month = (
    payments_usable.groupby(
        "pay_month",
        as_index=False
    )["total_paid"]
    .sum()
    .rename(columns={
        "pay_month": "month",
        "total_paid": "model_attributable_cash_usd"
    })
)

conflict_month = (
    payment_scope.loc[
        payment_scope["conflicting_contract_month"]
    ]
    .groupby(
        "pay_month",
        as_index=False
    )["total_paid"]
    .sum()
    .rename(columns={
        "pay_month": "month",
        "total_paid": "unresolved_conflict_rows_usd"
    })
)

before_sale_month = (
    payment_scope.loc[
        (~payment_scope["conflicting_contract_month"])
        &
        payment_scope["before_sale_month"]
    ]
    .groupby(
        "pay_month",
        as_index=False
    )["total_paid"]
    .sum()
    .rename(columns={
        "pay_month": "month",
        "total_paid": "before_sale_cash_usd"
    })
)

orphan_month = (
    payment_scope.loc[
        (~payment_scope["conflicting_contract_month"])
        &
        (~payment_scope["before_sale_month"])
        &
        payment_scope["orphan_contractid"]
    ]
    .groupby(
        "pay_month",
        as_index=False
    )["total_paid"]
    .sum()
    .rename(columns={
        "pay_month": "month",
        "total_paid": "orphan_cash_usd"
    })
)

negative_month = (
    payment_scope.loc[
        (~payment_scope["conflicting_contract_month"])
        &
        (~payment_scope["before_sale_month"])
        &
        (~payment_scope["orphan_contractid"])
        &
        payment_scope["negative_payment"]
    ]
    .groupby(
        "pay_month",
        as_index=False
    )["total_paid"]
    .sum()
    .rename(columns={
        "pay_month": "month",
        "total_paid": "negative_cash_usd"
    })
)

payment_reconciliation = (
    source_month
    .merge(usable_month, on="month", how="left")
    .merge(conflict_month, on="month", how="left")
    .merge(before_sale_month, on="month", how="left")
    .merge(orphan_month, on="month", how="left")
    .merge(negative_month, on="month", how="left")
    .sort_values("month")
)

for col in payment_reconciliation.columns:
    if col != "month":
        payment_reconciliation[col] = (
            payment_reconciliation[col]
            .fillna(0.0)
        )

payment_reconciliation[
    "source_minus_attributable_cash_usd"
] = (
    payment_reconciliation[
        "source_reported_cash_usd"
    ]
    -
    payment_reconciliation[
        "model_attributable_cash_usd"
    ]
)

payment_reconciliation[
    "attributable_share_of_source"
] = np.where(
    payment_reconciliation[
        "source_reported_cash_usd"
    ].abs() > 0,
    payment_reconciliation[
        "model_attributable_cash_usd"
    ]
    /
    payment_reconciliation[
        "source_reported_cash_usd"
    ],
    np.nan
)

show_table(
    payment_reconciliation,
    max_rows=30,
    title="Payment accounting bridge"
)

# %% [markdown]
# ## 7. Build the full financed recovery panel
#
# This is the key change from the earlier notebooks.
#
# ### FINANCED
# A contract receives one row for every month from the later of:
# - sale month;
# - Oct-2024;
#
# through Jun-2026.
#
# We **do not stop at tenor** because post-tenor recovery is still real cash.
#
# Instead, each row receives:
# - `within_scheduled_tenor_proxy`
# - `post_tenor_recovery_proxy`
#
# ### CASH
# A cash contract receives a sale-month panel row because contractually its full price is due upfront.
#
# If later cash-contract payments exist in the source, the accounting reconciliation will expose them rather than silently forcing them into the contractual schedule.

# %%
contracts["contract_type"] = (
    contracts["contract_type"]
    .astype("string")
    .str.upper()
)

contracts["legacy_pre_oct_2024"] = (
    contracts["sales_month"]
    < ANALYSIS_START
)

months = pd.DataFrame({
    "month": pd.date_range(
        ANALYSIS_START,
        VALIDATION_END,
        freq="ME"
    )
})

base_cols = [
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

base_cols = [
    c for c in base_cols
    if c in contracts.columns
]

financed = contracts.loc[
    contracts["contract_type"].eq("FINANCED"),
    base_cols
].copy()

cash = contracts.loc[
    contracts["contract_type"].eq("CASH"),
    base_cols
].copy()

fin_panel = (
    financed.assign(_k=1)
    .merge(
        months.assign(_k=1),
        on="_k",
        how="inner"
    )
    .drop(columns="_k")
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

cash_panel["month"] = (
    cash_panel["sales_month"]
)

panel = pd.concat(
    [fin_panel, cash_panel],
    ignore_index=True
)

panel = panel.sort_values(
    ["contractid", "month"]
).reset_index(drop=True)

assert not panel.duplicated(
    ["contractid", "month"]
).any()

print("Panel rows:", len(panel))
print("Unique contracts:", panel["contractid"].nunique())


# %% [markdown]
# ## 8. Contract-age and contractual schedule features

# %%
def month_difference(later, earlier):
    later = pd.to_datetime(later)
    earlier = pd.to_datetime(earlier)

    return (
        (later.dt.year - earlier.dt.year) * 12
        +
        (later.dt.month - earlier.dt.month)
    )

panel["months_on_book"] = (
    month_difference(
        panel["month"],
        panel["sales_month"]
    )
    .clip(lower=0)
    .astype(int)
)

panel["deposit_usd"] = np.where(
    panel["price_usd"].notna()
    &
    panel["perc_deposit"].notna(),
    panel["price_usd"]
    * panel["perc_deposit"],
    np.nan
)

panel["financed_principal_after_deposit_usd"] = np.where(
    panel["contract_type"].eq("FINANCED"),
    (
        panel["price_usd"]
        -
        panel["deposit_usd"]
    ).clip(lower=0),
    0.0
)

# Sanity: what contract terms imply if daily_amount × tenor is added to deposit.
panel["term_implied_total_usd"] = np.where(
    panel["contract_type"].eq("FINANCED"),
    panel["deposit_usd"]
    +
    panel["daily_amount_usd"]
    * panel["tenor_length"],
    panel["price_usd"]
)

panel["term_implied_gap_usd"] = (
    panel["term_implied_total_usd"]
    -
    panel["price_usd"]
)

panel["term_implied_gap_pct"] = np.where(
    panel["price_usd"].abs() > 0,
    panel["term_implied_gap_usd"]
    /
    panel["price_usd"],
    np.nan
)

# Approximate elapsed scheduled repayment days.
# MOB0 = deposit month only.
panel["scheduled_elapsed_days_through_month"] = np.where(
    panel["contract_type"].eq("FINANCED"),
    np.minimum(
        panel["tenor_length"],
        panel["months_on_book"]
        * AVG_DAYS_PER_MONTH
    ),
    0.0
)

panel["scheduled_elapsed_days_before_month"] = np.where(
    panel["contract_type"].eq("FINANCED"),
    np.minimum(
        panel["tenor_length"],
        np.maximum(
            0.0,
            (panel["months_on_book"] - 1)
            * AVG_DAYS_PER_MONTH
        )
    ),
    0.0
)

# Expected cumulative cash through current month.
panel["expected_cumulative_cash_through_month_usd"] = np.where(
    panel["contract_type"].eq("CASH"),
    panel["price_usd"],
    np.minimum(
        panel["price_usd"],
        panel["deposit_usd"]
        +
        panel["daily_amount_usd"]
        * panel["scheduled_elapsed_days_through_month"]
    )
)

# Expected cumulative cash before current month.
panel["expected_cumulative_cash_before_month_usd"] = np.where(
    panel["contract_type"].eq("CASH"),
    np.where(
        panel["months_on_book"] > 0,
        panel["price_usd"],
        0.0
    ),
    np.where(
        panel["months_on_book"].eq(0),
        0.0,
        np.minimum(
            panel["price_usd"],
            panel["deposit_usd"]
            +
            panel["daily_amount_usd"]
            * panel["scheduled_elapsed_days_before_month"]
        )
    )
)

panel["expected_cash_due_this_month_usd"] = (
    panel[
        "expected_cumulative_cash_through_month_usd"
    ]
    -
    panel[
        "expected_cumulative_cash_before_month_usd"
    ]
).clip(lower=0)

panel["within_scheduled_tenor_proxy"] = np.where(
    panel["contract_type"].eq("FINANCED"),
    (
        panel["scheduled_elapsed_days_before_month"]
        < panel["tenor_length"]
    ),
    panel["months_on_book"].eq(0)
)

panel["post_tenor_recovery_proxy"] = (
    panel["contract_type"].eq("FINANCED")
    &
    (~panel["within_scheduled_tenor_proxy"])
)

panel["scheduled_status"] = np.select(
    [
        panel["contract_type"].eq("CASH"),
        panel["within_scheduled_tenor_proxy"]
    ],
    [
        "CASH_UPFRONT",
        "WITHIN_SCHEDULE"
    ],
    default="POST_TENOR_RECOVERY"
)

panel["time_layer"] = np.where(
    panel["month"] <= ESTIMATION_END,
    "estimation",
    "validation"
)

# %% [markdown]
# ## 9. Contract-term sanity table
#
# This does not remove rows. It checks whether:
#
# `deposit + daily_amount × tenor`
#
# approximately reconstructs contract price.
#
# If it does, the expected schedule formula is well-supported. If not, the gap is visible rather than hidden.

# %%
contract_term_sanity = (
    panel[
        panel["months_on_book"].eq(0)
    ][
        [
            "contractid",
            "contract_type",
            "price_usd",
            "deposit_usd",
            "daily_amount_usd",
            "tenor_length",
            "term_implied_total_usd",
            "term_implied_gap_usd",
            "term_implied_gap_pct"
        ]
    ]
    .drop_duplicates("contractid")
)

term_sanity_summary = (
    contract_term_sanity[
        contract_term_sanity[
            "contract_type"
        ].eq("FINANCED")
    ][
        "term_implied_gap_pct"
    ]
    .describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
    .rename("value")
    .reset_index()
    .rename(columns={"index": "stat"})
)

show_table(
    term_sanity_summary,
    max_rows=20,
    title="Financed contract term reconstruction"
)

# %% [markdown]
# ## 10. Attach actual monthly cash target

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
    on=["contractid", "month"],
    how="left"
)

panel = panel.merge(
    payment_conflicts.rename(
        columns={"pay_month": "month"}
    ),
    on=["contractid", "month"],
    how="left"
)

panel["current_payment_conflict"] = (
    panel["current_payment_conflict"]
    .fillna(False)
    .astype(bool)
)

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
# ## 11. Reconcile attributable payment cash to panel

# %%
panel_target_month = (
    panel.groupby(
        "month",
        as_index=False
    )["target_payment_usd"]
    .sum(min_count=1)
    .rename(columns={
        "target_payment_usd":
        "panel_target_cash_usd"
    })
)

payment_reconciliation = (
    payment_reconciliation.merge(
        panel_target_month,
        on="month",
        how="left"
    )
)

payment_reconciliation[
    "panel_target_cash_usd"
] = (
    payment_reconciliation[
        "panel_target_cash_usd"
    ]
    .fillna(0.0)
)

payment_reconciliation[
    "attributable_minus_panel_usd"
] = (
    payment_reconciliation[
        "model_attributable_cash_usd"
    ]
    -
    payment_reconciliation[
        "panel_target_cash_usd"
    ]
)

# Row-level unmatched usable payment records.
panel_keys = panel[
    ["contractid", "month"]
].drop_duplicates()

unassigned_usable_payments = (
    payments_usable.rename(
        columns={"pay_month": "month"}
    )
    .merge(
        panel_keys.assign(in_panel=True),
        on=["contractid", "month"],
        how="left"
    )
)

unassigned_usable_payments = (
    unassigned_usable_payments[
        unassigned_usable_payments[
            "in_panel"
        ].isna()
    ]
    .drop(columns="in_panel")
    .copy()
)

if len(unassigned_usable_payments):
    contract_lookup = contracts[
        [
            "contractid",
            "contract_type",
            "sales_month",
            "tenor_length"
        ]
    ].drop_duplicates("contractid")

    unassigned_usable_payments = (
        unassigned_usable_payments
        .merge(
            contract_lookup,
            on="contractid",
            how="left"
        )
    )

    unassigned_usable_payments[
        "unassigned_reason"
    ] = np.select(
        [
            (
                unassigned_usable_payments[
                    "contract_type"
                ].eq("CASH")
                &
                (
                    unassigned_usable_payments[
                        "month"
                    ]
                    !=
                    unassigned_usable_payments[
                        "sales_month"
                    ]
                )
            ),
            unassigned_usable_payments[
                "contract_type"
            ].eq("FINANCED"),
            unassigned_usable_payments[
                "contract_type"
            ].isna()
        ],
        [
            "CASH_PAYMENT_OUTSIDE_SALE_MONTH",
            "FINANCED_PANEL_KEY_MISSING",
            "CONTRACT_LOOKUP_MISSING"
        ],
        default="OTHER"
    )

show_table(
    payment_reconciliation,
    max_rows=30,
    title="Payment-to-panel reconciliation"
)

print(
    "Unassigned model-attributable cash:",
    round(
        float(
            unassigned_usable_payments[
                "total_paid"
            ].sum()
        )
        if len(unassigned_usable_payments)
        else 0.0,
        2
    )
)

# %% [markdown]
# ## 12. Actual cumulative cash and collection-efficiency backbone
#
# For every contract-month:
#
# - `actual_cumulative_cash_before_month_usd`
# - `actual_cumulative_cash_through_month_usd`
# - `expected_cumulative_cash_through_month_usd`
# - `cumulative_collection_efficiency`
#
# `cumulative_collection_efficiency` is an **outcome / diagnostic**, not a same-row predictor.
#
# Legacy contracts sold before Oct-2024 are left-censored and excluded from the clean efficiency curves.

# %%
panel = panel.sort_values(
    ["contractid", "month"]
).copy()

panel["_target_for_cumsum"] = (
    panel["target_payment_usd"]
    .fillna(0.0)
)

panel[
    "actual_cumulative_cash_through_month_usd"
] = (
    panel.groupby("contractid")[
        "_target_for_cumsum"
    ]
    .cumsum()
)

panel[
    "actual_cumulative_cash_before_month_usd"
] = (
    panel[
        "actual_cumulative_cash_through_month_usd"
    ]
    -
    panel["_target_for_cumsum"]
)

panel[
    "cumulative_collection_efficiency"
] = np.where(
    panel[
        "expected_cumulative_cash_through_month_usd"
    ] > 0,
    panel[
        "actual_cumulative_cash_through_month_usd"
    ]
    /
    panel[
        "expected_cumulative_cash_through_month_usd"
    ],
    np.nan
)

# Do not cap the raw metric; retain over-collection if present.
panel[
    "cumulative_collection_efficiency_capped_1"
] = (
    panel[
        "cumulative_collection_efficiency"
    ]
    .clip(upper=1.0)
)

# %% [markdown]
# ## 13. Payment-history behavioural features — prior months only

# %%
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
        .rolling(3, min_periods=1)
        .sum()
    )
)

panel["payment_trailing_6m_sum"] = (
    g["target_payment_usd"]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(6, min_periods=1)
        .sum()
    )
)

panel["payment_trailing_3m_mean"] = (
    g["target_payment_usd"]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(3, min_periods=1)
        .mean()
    )
)

panel["payment_trailing_6m_mean"] = (
    g["target_payment_usd"]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(6, min_periods=1)
        .mean()
    )
)

panel["_positive_payment"] = np.where(
    panel["target_payment_usd"]
    .fillna(0) > 0,
    1,
    0
)

panel["prior_paying_months"] = (
    panel.groupby("contractid")[
        "_positive_payment"
    ]
    .cumsum()
    -
    panel["_positive_payment"]
)

panel["_zero_payment"] = np.where(
    panel["target_payment_usd"].eq(0),
    1,
    0
)

panel["zero_payment_months_prior_3"] = (
    panel.groupby("contractid")[
        "_zero_payment"
    ]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(3, min_periods=1)
        .sum()
    )
)

panel["zero_payment_months_prior_6"] = (
    panel.groupby("contractid")[
        "_zero_payment"
    ]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(6, min_periods=1)
        .sum()
    )
)

panel["_conflict_int"] = (
    panel[
        "current_payment_conflict"
    ]
    .astype(int)
)

panel["prior_payment_conflict_count"] = (
    panel.groupby("contractid")[
        "_conflict_int"
    ]
    .cumsum()
    -
    panel["_conflict_int"]
)

panel["_positive_payment_mob"] = np.where(
    panel["_positive_payment"].eq(1),
    panel["months_on_book"],
    np.nan
)

panel["_last_positive_mob_before"] = (
    panel.groupby("contractid")[
        "_positive_payment_mob"
    ]
    .transform(
        lambda s:
        s.shift(1).ffill()
    )
)

panel["months_since_last_positive_payment"] = (
    panel["months_on_book"]
    -
    panel["_last_positive_mob_before"]
)

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
    (
        panel["prior_payment_conflict_count"]
        == 0
    )
    &
    (
        ~panel["payment_history_left_censored"]
    )
)

# %% [markdown]
# ## 14. Prior-month inbound call features

# %%
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
    calls["call_date"].between(
        ANALYSIS_START,
        VALIDATION_END
    )
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
        ["contractid", "month", "call_reason"]
    )
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

rename_map = {}
for col in call_reason_month.columns:
    if col in ["contractid", "month"]:
        continue

    safe = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        str(col)
    ).strip("_").lower()

    rename_map[col] = (
        f"calls_reason_{safe}_this_month"
    )

call_reason_month = (
    call_reason_month.rename(
        columns=rename_map
    )
)

call_monthly = (
    call_month_total.merge(
        call_reason_month,
        on=["contractid", "month"],
        how="left"
    )
)

panel = panel.merge(
    call_monthly,
    on=["contractid", "month"],
    how="left"
)

call_current_cols = [
    c for c in panel.columns
    if (
        c == "calls_this_month"
        or
        (
            c.startswith("calls_reason_")
            and c.endswith("_this_month")
        )
    )
]

panel[call_current_cols] = (
    panel[call_current_cols]
    .fillna(0)
)

for col in call_current_cols:
    prefix = col.replace(
        "_this_month",
        ""
    )

    panel[f"{prefix}_prior_1m"] = (
        panel.groupby("contractid")[col]
        .shift(1)
        .fillna(0)
    )

    panel[f"{prefix}_prior_3m"] = (
        panel.groupby("contractid")[col]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(3, min_periods=1)
            .sum()
        )
        .fillna(0)
    )

panel["calls_cumulative_before_month"] = (
    panel.groupby("contractid")[
        "calls_this_month"
    ]
    .cumsum()
    -
    panel["calls_this_month"]
)

# %% [markdown]
# ## 15. Prior-month service-ticket features

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
    service["ticket_date"].between(
        ANALYSIS_START,
        VALIDATION_END
    )
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
        ["contractid", "month", "ticket_reason"]
    )
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

rename_map = {}
for col in ticket_reason_month.columns:
    if col in ["contractid", "month"]:
        continue

    safe = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        str(col)
    ).strip("_").lower()

    rename_map[col] = (
        f"tickets_reason_{safe}_this_month"
    )

ticket_reason_month = (
    ticket_reason_month.rename(
        columns=rename_map
    )
)

ticket_monthly = (
    ticket_month_total.merge(
        ticket_reason_month,
        on=["contractid", "month"],
        how="left"
    )
)

panel = panel.merge(
    ticket_monthly,
    on=["contractid", "month"],
    how="left"
)

ticket_current_cols = [
    c for c in panel.columns
    if (
        c == "tickets_this_month"
        or
        (
            c.startswith("tickets_reason_")
            and c.endswith("_this_month")
        )
    )
]

panel[ticket_current_cols] = (
    panel[ticket_current_cols]
    .fillna(0)
)

for col in ticket_current_cols:
    prefix = col.replace(
        "_this_month",
        ""
    )

    panel[f"{prefix}_prior_1m"] = (
        panel.groupby("contractid")[col]
        .shift(1)
        .fillna(0)
    )

    panel[f"{prefix}_prior_3m"] = (
        panel.groupby("contractid")[col]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(3, min_periods=1)
            .sum()
        )
        .fillna(0)
    )

panel["tickets_cumulative_before_month"] = (
    panel.groupby("contractid")[
        "tickets_this_month"
    ]
    .cumsum()
    -
    panel["tickets_this_month"]
)

# %% [markdown]
# ## 16. Remove same-month event leakage and temporary columns

# %%
panel = panel.drop(
    columns=[
        c for c in (
            call_current_cols
            +
            ticket_current_cols
        )
        if c in panel.columns
    ]
)

temporary_cols = [
    "_target_for_cumsum",
    "_positive_payment",
    "_zero_payment",
    "_conflict_int",
    "_positive_payment_mob",
    "_last_positive_mob_before"
]

panel = panel.drop(
    columns=[
        c for c in temporary_cols
        if c in panel.columns
    ]
)

# %% [markdown]
# ## 17. Cohort / vintage efficiency tables
#
# These are the missing headline features from the earlier notebook.
#
# The clean country curve excludes:
# - left-censored legacy contracts;
# - rows with current or prior unresolved payment conflicts.
#
# Efficiency is calculated as:
#
# `sum(actual cumulative cash through MOB) / sum(expected cumulative cash through MOB)`
#
# This is a portfolio-weighted efficiency, not the average of individual customer ratios.

# %%
curve_base = panel[
    panel["payment_history_reliable"]
    &
    panel[
        "expected_cumulative_cash_through_month_usd"
    ].gt(0)
].copy()

country_curve = (
    curve_base.groupby(
        ["contract_type", "months_on_book"],
        as_index=False
    )
    .agg(
        contracts=("contractid", "nunique"),
        actual_cum_usd=(
            "actual_cumulative_cash_through_month_usd",
            "sum"
        ),
        expected_cum_usd=(
            "expected_cumulative_cash_through_month_usd",
            "sum"
        ),
        current_month_cash_usd=(
            "target_payment_usd",
            "sum"
        )
    )
)

country_curve[
    "cumulative_collection_efficiency"
] = np.where(
    country_curve["expected_cum_usd"] > 0,
    country_curve["actual_cum_usd"]
    /
    country_curve["expected_cum_usd"],
    np.nan
)

region_curve = (
    curve_base.groupby(
        [
            "region",
            "contract_type",
            "months_on_book"
        ],
        as_index=False
    )
    .agg(
        contracts=("contractid", "nunique"),
        actual_cum_usd=(
            "actual_cumulative_cash_through_month_usd",
            "sum"
        ),
        expected_cum_usd=(
            "expected_cumulative_cash_through_month_usd",
            "sum"
        ),
        current_month_cash_usd=(
            "target_payment_usd",
            "sum"
        )
    )
)

region_curve[
    "cumulative_collection_efficiency"
] = np.where(
    region_curve["expected_cum_usd"] > 0,
    region_curve["actual_cum_usd"]
    /
    region_curve["expected_cum_usd"],
    np.nan
)

show_table(
    country_curve.head(25),
    max_rows=25,
    title="Country cohort efficiency curve sample"
)

# %% [markdown]
# ## 18. Region-month and country-month tables

# %%
region_month = (
    panel.groupby(
        ["month", "region"],
        as_index=False,
        dropna=False
    )
    .agg(
        panel_cash_usd=(
            "target_payment_usd",
            "sum"
        ),
        expected_cash_due_usd=(
            "expected_cash_due_this_month_usd",
            "sum"
        ),
        expected_cumulative_cash_usd=(
            "expected_cumulative_cash_through_month_usd",
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
        post_tenor_rows=(
            "post_tenor_recovery_proxy",
            "sum"
        )
    )
)

country_month = (
    panel.groupby(
        "month",
        as_index=False
    )
    .agg(
        panel_cash_usd=(
            "target_payment_usd",
            "sum"
        ),
        expected_cash_due_usd=(
            "expected_cash_due_this_month_usd",
            "sum"
        ),
        expected_cumulative_cash_usd=(
            "expected_cumulative_cash_through_month_usd",
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
        post_tenor_rows=(
            "post_tenor_recovery_proxy",
            "sum"
        )
    )
)

country_month = (
    country_month.merge(
        payment_reconciliation[
            [
                "month",
                "source_reported_cash_usd",
                "model_attributable_cash_usd",
                "source_minus_attributable_cash_usd",
                "attributable_share_of_source"
            ]
        ],
        on="month",
        how="left"
    )
)

show_table(
    country_month,
    max_rows=30,
    title="Country-month feature summary"
)

# %% [markdown]
# ## 19. Pilot analysis table kept separate
#
# Outreach treatment variables do not enter the base forecast panel.
#
# This table carries only pre-contact features plus separately labelled outcomes.

# %%
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
    "scheduled_status",
    "expected_cash_due_this_month_usd",
    "expected_cumulative_cash_before_month_usd",
    "actual_cumulative_cash_before_month_usd",
    "payment_lag_1m",
    "payment_lag_2m",
    "payment_lag_3m",
    "payment_trailing_3m_sum",
    "payment_trailing_6m_sum",
    "prior_paying_months",
    "zero_payment_months_prior_3",
    "zero_payment_months_prior_6",
    "months_since_last_positive_payment",
    "calls_cumulative_before_month",
    "tickets_cumulative_before_month",
    "payment_history_reliable"
]

pilot_feature_cols += [
    c for c in panel.columns
    if (
        (
            c.startswith("calls_")
            or c.startswith("tickets_")
        )
        and (
            c.endswith("_prior_1m")
            or c.endswith("_prior_3m")
        )
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

for col in [
    "orphan_contractid",
    "before_sale_month"
]:
    if col in outreach.columns:
        outreach[col] = as_bool(
            outreach[col]
        )

pilot_outreach = outreach[
    (~outreach["orphan_contractid"])
    &
    (~outreach["before_sale_month"])
    &
    outreach["contact_month"].between(
        pd.Timestamp("2026-04-30"),
        VALIDATION_END
    )
].copy()

pilot = pilot_outreach.merge(
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

target_lookup = panel[
    [
        "contractid",
        "month",
        "target_payment_usd"
    ]
].copy()

pilot["next_month"] = (
    pilot["contact_month"]
    +
    pd.offsets.MonthEnd(1)
)

next_lookup = target_lookup.rename(
    columns={
        "month": "next_month",
        "target_payment_usd":
        "payment_next_month_usd"
    }
)

pilot = pilot.merge(
    next_lookup,
    on=["contractid", "next_month"],
    how="left"
)

pilot["next_month_observed"] = (
    pilot["next_month"]
    <= VALIDATION_END
)

pilot.loc[
    ~pilot["next_month_observed"],
    "payment_next_month_usd"
] = np.nan

print("Pilot rows:", len(pilot))

# %% [markdown]
# ## 20. Independent sanity checks
#
# These checks do not trust the engineered columns merely because they were produced by the same function.
#
# They independently validate:
# - panel key uniqueness;
# - sealed-period exclusion;
# - region-to-country aggregation;
# - attributable payment cash landing in panel;
# - payment lag-1;
# - call lag-1;
# - ticket lag-1;
# - no same-month event predictors;
# - no outreach treatment variables in the base panel.

# %%
sanity_rows = []

def add_check(
    name,
    status,
    actual,
    expected,
    layer,
    note=""
):
    sanity_rows.append({
        "check": name,
        "status": status,
        "actual": actual,
        "expected": expected,
        "layer": layer,
        "note": note
    })

dup_keys = int(
    panel.duplicated(
        ["contractid", "month"]
    ).sum()
)

add_check(
    "contract_month_key_unique",
    "PASS" if dup_keys == 0 else "FAIL",
    dup_keys,
    0,
    "engineered"
)

future_rows = int(
    (panel["month"] > VALIDATION_END).sum()
)

add_check(
    "no_rows_after_jun_2026",
    "PASS" if future_rows == 0 else "FAIL",
    future_rows,
    0,
    "engineered"
)

region_rollup = (
    region_month.groupby(
        "month",
        as_index=False
    )["panel_cash_usd"]
    .sum()
    .rename(columns={
        "panel_cash_usd":
        "region_rollup_usd"
    })
)

rc = country_month[
    ["month", "panel_cash_usd"]
].merge(
    region_rollup,
    on="month",
    how="outer"
)

rc["abs_diff"] = (
    rc["panel_cash_usd"]
    -
    rc["region_rollup_usd"]
).abs()

region_country_max_diff = float(
    rc["abs_diff"].fillna(np.inf).max()
)

add_check(
    "region_sums_equal_country_total",
    "PASS"
    if region_country_max_diff < 0.01
    else "FAIL",
    round(
        region_country_max_diff,
        6
    ),
    "< 0.01 USD",
    "engineered"
)

unassigned_usable_usd = float(
    unassigned_usable_payments[
        "total_paid"
    ].sum()
) if len(
    unassigned_usable_payments
) else 0.0

add_check(
    "model_attributable_cash_lands_in_panel",
    "PASS"
    if abs(unassigned_usable_usd) < 0.01
    else "FAIL",
    round(
        unassigned_usable_usd,
        2
    ),
    "0.00 USD",
    "reconciliation"
)

# Independent payment lag-1 source check.
prev_pay = payments_usable[
    ["contractid", "pay_month", "total_paid"]
].copy()

prev_pay["month"] = (
    prev_pay["pay_month"]
    +
    pd.offsets.MonthEnd(1)
)

prev_pay = prev_pay[
    ["contractid", "month", "total_paid"]
].rename(columns={
    "total_paid":
    "_expected_payment_lag_1m"
})

lag_check = panel[
    [
        "contractid",
        "month",
        "months_on_book",
        "payment_lag_1m"
    ]
].merge(
    prev_pay,
    on=["contractid", "month"],
    how="left"
)

# A prior panel month with no payment row means zero cash.
lag_check = lag_check[
    lag_check["months_on_book"] > 0
].copy()

lag_check[
    "_expected_payment_lag_1m"
] = lag_check[
    "_expected_payment_lag_1m"
].fillna(0.0)

lag_mismatch = int(
    (
        lag_check[
            "payment_lag_1m"
        ].fillna(0.0)
        -
        lag_check[
            "_expected_payment_lag_1m"
        ]
    ).abs().gt(1e-9).sum()
)

add_check(
    "payment_lag_1m_matches_shifted_source",
    "PASS"
    if lag_mismatch == 0
    else "FAIL",
    lag_mismatch,
    0,
    "engineered"
)

# Independent calls lag-1.
expected_calls = call_month_total.copy()
expected_calls["month"] = (
    expected_calls["month"]
    +
    pd.offsets.MonthEnd(1)
)

expected_calls = expected_calls.rename(
    columns={
        "calls_this_month":
        "_expected_calls_prior_1m"
    }
)

calls_check = panel[
    ["contractid", "month", "calls_prior_1m"]
].merge(
    expected_calls[
        [
            "contractid",
            "month",
            "_expected_calls_prior_1m"
        ]
    ],
    on=["contractid", "month"],
    how="left"
)

calls_check[
    "_expected_calls_prior_1m"
] = calls_check[
    "_expected_calls_prior_1m"
].fillna(0)

calls_mismatch = int(
    (
        calls_check[
            "calls_prior_1m"
        ]
        -
        calls_check[
            "_expected_calls_prior_1m"
        ]
    ).abs().gt(1e-9).sum()
)

add_check(
    "calls_prior_1m_matches_shifted_source",
    "PASS"
    if calls_mismatch == 0
    else "FAIL",
    calls_mismatch,
    0,
    "engineered"
)

# Independent tickets lag-1.
expected_tickets = ticket_month_total.copy()
expected_tickets["month"] = (
    expected_tickets["month"]
    +
    pd.offsets.MonthEnd(1)
)

expected_tickets = expected_tickets.rename(
    columns={
        "tickets_this_month":
        "_expected_tickets_prior_1m"
    }
)

tickets_check = panel[
    [
        "contractid",
        "month",
        "tickets_prior_1m"
    ]
].merge(
    expected_tickets[
        [
            "contractid",
            "month",
            "_expected_tickets_prior_1m"
        ]
    ],
    on=["contractid", "month"],
    how="left"
)

tickets_check[
    "_expected_tickets_prior_1m"
] = tickets_check[
    "_expected_tickets_prior_1m"
].fillna(0)

tickets_mismatch = int(
    (
        tickets_check[
            "tickets_prior_1m"
        ]
        -
        tickets_check[
            "_expected_tickets_prior_1m"
        ]
    ).abs().gt(1e-9).sum()
)

add_check(
    "tickets_prior_1m_matches_shifted_source",
    "PASS"
    if tickets_mismatch == 0
    else "FAIL",
    tickets_mismatch,
    0,
    "engineered"
)

same_month_event_cols = [
    c for c in panel.columns
    if (
        c.endswith("_this_month")
        and (
            c.startswith("calls")
            or c.startswith("tickets")
        )
    )
]

add_check(
    "no_same_month_call_or_ticket_predictors",
    "PASS"
    if len(same_month_event_cols) == 0
    else "FAIL",
    len(same_month_event_cols),
    0,
    "leakage"
)

outreach_terms = {
    "channel",
    "attempts",
    "reached",
    "cost_usd",
    "contact_month"
}

outreach_cols_in_panel = [
    c for c in panel.columns
    if (
        c in outreach_terms
        or c.startswith("outreach_")
    )
]

add_check(
    "no_outreach_treatment_columns_in_base_panel",
    "PASS"
    if len(outreach_cols_in_panel) == 0
    else "FAIL",
    len(outreach_cols_in_panel),
    0,
    "leakage"
)

sanity_check_report = pd.DataFrame(
    sanity_rows
)

show_table(
    sanity_check_report,
    max_rows=100,
    title="Sanity checks"
)


# %% [markdown]
# ## 21. Canonical metrics for cross-notebook comparison
#
# This v3 notebook deliberately writes **flat metric names** so another notebook can match them without knowing this notebook's internal JSON structure.
#
# Important definitions:
#
# - `sum_total_paid_model_period` = source-reported cash Oct-2024 through Jun-2026.
# - `sum_model_attributable_cash` = clean payment cash that can be confidently attached to contract-months.
# - `payments_not_landed_in_panel` = model-attributable cash that still failed to map to a panel row. This should be zero.
# - `sum_expected_cum_final` = sum of each eligible contract's expected cumulative amount at its latest observed panel month.
# - `sum_actual_cum_final` = corresponding observed cumulative cash for the same contracts.

# %%
def weighted_efficiency(
    contract_type,
    mob
):
    sub = country_curve[
        country_curve[
            "contract_type"
        ].eq(contract_type)
        &
        country_curve[
            "months_on_book"
        ].eq(mob)
    ]

    if sub.empty:
        return None

    return round(
        float(
            sub[
                "cumulative_collection_efficiency"
            ].iloc[0]
        ),
        6
    )

latest_contract_rows = (
    panel.sort_values(
        ["contractid", "month"]
    )
    .groupby(
        "contractid",
        as_index=False
    )
    .tail(1)
)

# Exclude left-censored / conflict-affected contracts
# from final expected-vs-actual cohort metrics.
latest_reliable = latest_contract_rows[
    latest_contract_rows[
        "payment_history_reliable"
    ]
].copy()

panel_hash_source = (
    panel[
        [
            "contractid",
            "month",
            "contract_type",
            "months_on_book",
            "scheduled_status"
        ]
    ]
    .sort_values(
        ["contractid", "month"]
    )
    .astype(str)
)

panel_hash = hashlib.sha256(
    pd.util.hash_pandas_object(
        panel_hash_source,
        index=False
    ).values.tobytes()
).hexdigest()[:16]

source_total_model_period = float(
    payment_reconciliation[
        "source_reported_cash_usd"
    ].sum()
)

model_attributable_total = float(
    payment_reconciliation[
        "model_attributable_cash_usd"
    ].sum()
)

panel_total = float(
    panel["target_payment_usd"].sum()
)

flat_metrics = {
    "schema_version": "dlight_feature_metrics_v3",
    "analysis_start": str(
        ANALYSIS_START.date()
    ),
    "analysis_end": str(
        VALIDATION_END.date()
    ),

    # Inputs
    "n_contracts": int(
        contracts["contractid"].nunique()
    ),
    "n_contracts_CASH": int(
        contracts[
            "contract_type"
        ].eq("CASH").sum()
    ),
    "n_contracts_FINANCED": int(
        contracts[
            "contract_type"
        ].eq("FINANCED").sum()
    ),
    "n_payment_rows_dev": int(
        len(payments)
    ),
    "sum_total_paid_model_period": round(
        source_total_model_period,
        2
    ),

    # Panel
    "n_panel_rows": int(
        len(panel)
    ),
    "n_panel_unique_contracts": int(
        panel[
            "contractid"
        ].nunique()
    ),
    "panel_hash": panel_hash,
    "n_post_tenor_recovery_rows": int(
        panel[
            "post_tenor_recovery_proxy"
        ].sum()
    ),

    # Reconciliation
    "sum_model_attributable_cash": round(
        model_attributable_total,
        2
    ),
    "sum_panel_target_cash": round(
        panel_total,
        2
    ),
    "payments_not_landed_in_panel": round(
        unassigned_usable_usd,
        2
    ),
    "source_minus_model_attributable_cash": round(
        source_total_model_period
        -
        model_attributable_total,
        2
    ),

    # Contract economics
    "sum_price_usd": round(
        float(
            contracts[
                "price_usd"
            ].sum()
        ),
        2
    ),
    "sum_expected_deposit": round(
        float(
            panel.loc[
                panel[
                    "months_on_book"
                ].eq(0),
                "deposit_usd"
            ].sum()
        ),
        2
    ),
    "sum_expected_cum_final": round(
        float(
            latest_reliable[
                "expected_cumulative_cash_through_month_usd"
            ].sum()
        ),
        2
    ),
    "sum_actual_cum_final": round(
        float(
            latest_reliable[
                "actual_cumulative_cash_through_month_usd"
            ].sum()
        ),
        2
    ),

    # Cohort efficiency checkpoints
    "eff_CASH_mob0": weighted_efficiency(
        "CASH",
        0
    ),
    "eff_FINANCED_mob0": weighted_efficiency(
        "FINANCED",
        0
    ),
    "eff_FINANCED_mob1": weighted_efficiency(
        "FINANCED",
        1
    ),
    "eff_FINANCED_mob3": weighted_efficiency(
        "FINANCED",
        3
    ),
    "eff_FINANCED_mob6": weighted_efficiency(
        "FINANCED",
        6
    ),

    # Behavioural feature control totals
    "sum_payment_lag_1m": round(
        float(
            panel[
                "payment_lag_1m"
            ].sum()
        ),
        2
    ),
    "sum_payment_trailing_3m": round(
        float(
            panel[
                "payment_trailing_3m_sum"
            ].sum()
        ),
        2
    ),
    "sum_calls_prior_1m": round(
        float(
            panel[
                "calls_prior_1m"
            ].sum()
        ),
        2
    ),
    "sum_calls_prior_3m": round(
        float(
            panel[
                "calls_prior_3m"
            ].sum()
        ),
        2
    ),
    "sum_tickets_prior_1m": round(
        float(
            panel[
                "tickets_prior_1m"
            ].sum()
        ),
        2
    ),
    "sum_tickets_prior_3m": round(
        float(
            panel[
                "tickets_prior_3m"
            ].sum()
        ),
        2
    ),

    # Other
    "n_pilot_rows": int(
        len(pilot)
    )
}

show_table(
    pd.DataFrame(
        [
            {
                "metric": k,
                "value": v
            }
            for k, v in flat_metrics.items()
        ]
    ),
    max_rows=100,
    title="Canonical feature metrics"
)


# %% [markdown]
# ## 22. Cross-notebook comparison helper

# %%
def compare_metrics(
    mine,
    theirs,
    atol=1e-6,
    rtol=1e-9
):
    rows = []

    for key in sorted(
        set(mine) | set(theirs)
    ):
        left = mine.get(
            key,
            "__MISSING__"
        )
        right = theirs.get(
            key,
            "__MISSING__"
        )

        if (
            left == "__MISSING__"
            or right == "__MISSING__"
        ):
            match = False
            delta = None

        elif (
            isinstance(
                left,
                (int, float)
            )
            and not isinstance(
                left,
                bool
            )
            and isinstance(
                right,
                (int, float)
            )
            and not isinstance(
                right,
                bool
            )
        ):
            match = math.isclose(
                float(left),
                float(right),
                abs_tol=atol,
                rel_tol=rtol
            )
            delta = (
                float(left)
                -
                float(right)
            )

        else:
            match = (
                left == right
            )
            delta = None

        rows.append({
            "metric": key,
            "notebook_A": left,
            "notebook_B": right,
            "delta_A_minus_B": delta,
            "status":
            "MATCH"
            if match
            else "DIFF"
        })

    return pd.DataFrame(rows)

# Example:
#
# with open("/content/feature_metrics_OTHER.json") as f:
#     other = json.load(f)
#
# comparison = compare_metrics(
#     flat_metrics,
#     other
# )
#
# show_table(
#     comparison[
#         comparison["status"].eq("DIFF")
#     ],
#     max_rows=200,
#     title="Metric differences"
# )


# %% [markdown]
# ## 23. Feature quality summary

# %%
feature_quality_checks = pd.DataFrame([
    {
        "check":
        "duplicate_contract_month_rows",
        "value":
        int(
            panel.duplicated(
                ["contractid", "month"]
            ).sum()
        )
    },
    {
        "check":
        "rows_after_jun_2026",
        "value":
        int(
            (
                panel["month"]
                > VALIDATION_END
            ).sum()
        )
    },
    {
        "check":
        "unassigned_model_attributable_cash_usd",
        "value":
        round(
            unassigned_usable_usd,
            2
        )
    },
    {
        "check":
        "source_minus_model_attributable_cash_usd",
        "value":
        round(
            source_total_model_period
            -
            model_attributable_total,
            2
        )
    },
    {
        "check":
        "post_tenor_recovery_rows",
        "value":
        int(
            panel[
                "post_tenor_recovery_proxy"
            ].sum()
        )
    },
    {
        "check":
        "current_payment_conflict_rows",
        "value":
        int(
            panel[
                "current_payment_conflict"
            ].sum()
        )
    },
    {
        "check":
        "prior_payment_conflict_rows",
        "value":
        int(
            (
                panel[
                    "prior_payment_conflict_count"
                ] > 0
            ).sum()
        )
    },
    {
        "check":
        "legacy_left_censored_rows",
        "value":
        int(
            panel[
                "payment_history_left_censored"
            ].sum()
        )
    }
])

show_table(
    feature_quality_checks,
    max_rows=50,
    title="Feature quality checks"
)

# %% [markdown]
# ## 24. Feature dictionary

# %%
feature_dictionary = pd.DataFrame([
    [
        "target_payment_usd",
        "target",
        "Actual payment in the contract-month. Never use as a same-row predictor."
    ],
    [
        "expected_cash_due_this_month_usd",
        "contractual_backbone",
        "Expected contractual cash due this month using deposit + daily amount × tenor-day proxy."
    ],
    [
        "expected_cumulative_cash_through_month_usd",
        "contractual_backbone",
        "Expected cumulative cash through current months-on-book."
    ],
    [
        "actual_cumulative_cash_through_month_usd",
        "outcome_diagnostic",
        "Observed cumulative cash through current month; outcome, not predictor."
    ],
    [
        "cumulative_collection_efficiency",
        "outcome_diagnostic",
        "Actual cumulative / expected cumulative. Used to estimate historical cohort curves."
    ],
    [
        "within_scheduled_tenor_proxy",
        "contractual_backbone",
        "Whether contract remains inside proxy contractual payment window."
    ],
    [
        "post_tenor_recovery_proxy",
        "contractual_backbone",
        "Financed contract has passed scheduled tenor but remains in portfolio for possible recovery cash."
    ],
    [
        "scheduled_status",
        "contractual_backbone",
        "CASH_UPFRONT, WITHIN_SCHEDULE, or POST_TENOR_RECOVERY."
    ],
    [
        "payment_lag_1m",
        "behavioural",
        "Previous month's payment only."
    ],
    [
        "payment_trailing_3m_sum",
        "behavioural",
        "Payment sum over prior three months only."
    ],
    [
        "calls_*_prior_1m/prior_3m",
        "behavioural",
        "Inbound customer-call history from prior months only."
    ],
    [
        "tickets_*_prior_1m/prior_3m",
        "behavioural",
        "Service-ticket history from prior months only."
    ],
    [
        "payment_history_reliable",
        "quality",
        "False for left-censored legacy contracts or unresolved payment-conflict histories."
    ],
])

show_table(
    feature_dictionary,
    max_rows=50,
    title="Feature dictionary"
)

# %% [markdown]
# ## 25. Save outputs

# %%
panel.to_csv(
    OUTPUT_DIR /
    "contract_month_features_v3.csv",
    index=False
)

country_curve.to_csv(
    OUTPUT_DIR /
    "cohort_efficiency_country_v3.csv",
    index=False
)

region_curve.to_csv(
    OUTPUT_DIR /
    "cohort_efficiency_region_v3.csv",
    index=False
)

country_month.to_csv(
    OUTPUT_DIR /
    "country_month_features_v3.csv",
    index=False
)

region_month.to_csv(
    OUTPUT_DIR /
    "region_month_features_v3.csv",
    index=False
)

payment_reconciliation.to_csv(
    OUTPUT_DIR /
    "payment_reconciliation_by_month_v3.csv",
    index=False
)

unassigned_usable_payments.to_csv(
    OUTPUT_DIR /
    "unassigned_usable_payments_v3.csv",
    index=False
)

pilot.to_csv(
    OUTPUT_DIR /
    "pilot_analysis_features_v3.csv",
    index=False
)

sanity_check_report.to_csv(
    OUTPUT_DIR /
    "sanity_check_report_v3.csv",
    index=False
)

feature_quality_checks.to_csv(
    OUTPUT_DIR /
    "feature_quality_checks_v3.csv",
    index=False
)

feature_dictionary.to_csv(
    OUTPUT_DIR /
    "feature_dictionary_v3.csv",
    index=False
)

term_sanity_summary.to_csv(
    OUTPUT_DIR /
    "contract_term_sanity_v3.csv",
    index=False
)

# Canonical flat JSON for notebook-to-notebook comparison.
with open(
    OUTPUT_DIR /
    "feature_metrics.json",
    "w"
) as f:
    json.dump(
        flat_metrics,
        f,
        indent=2,
        sort_keys=True
    )

# Richer JSON with sanity outcomes and monthly reconciliation.
detailed_metrics = {
    "flat_metrics": flat_metrics,
    "sanity": {
        row["check"]:
        row["status"]
        for row in sanity_rows
    },
    "payment_reconciliation_by_month": {
        row["month"].strftime("%Y-%m-%d"): {
            "source_reported_cash_usd":
            round(
                float(
                    row[
                        "source_reported_cash_usd"
                    ]
                ),
                2
            ),
            "model_attributable_cash_usd":
            round(
                float(
                    row[
                        "model_attributable_cash_usd"
                    ]
                ),
                2
            ),
            "panel_target_cash_usd":
            round(
                float(
                    row[
                        "panel_target_cash_usd"
                    ]
                ),
                2
            )
        }
        for _, row
        in payment_reconciliation.iterrows()
    }
}

with open(
    OUTPUT_DIR /
    "feature_metrics_detailed.json",
    "w"
) as f:
    json.dump(
        detailed_metrics,
        f,
        indent=2,
        sort_keys=True
    )

print("Outputs saved to:", OUTPUT_DIR)

# %% [markdown]
# ## 26. Download the JSON files directly
#
# This cell gives explicit browser download buttons for the JSON files.
#
# Run it immediately after the save cell if you want only the metrics files.

# %%
from google.colab import files

files.download(
    str(
        OUTPUT_DIR /
        "feature_metrics.json"
    )
)

files.download(
    str(
        OUTPUT_DIR /
        "feature_metrics_detailed.json"
    )
)

# %% [markdown]
# ## 27. Package all outputs and download ZIP
#
# The JSON files are also included in this ZIP.

# %%
zip_path = Path(
    "/content/dlight_feature_engineering_outputs_v3.zip"
)

with zipfile.ZipFile(
    zip_path,
    "w",
    zipfile.ZIP_DEFLATED
) as z:
    for p in OUTPUT_DIR.glob("*"):
        if p.is_file():
            z.write(
                p,
                arcname=p.name
            )

print("Created:", zip_path)

files.download(
    str(zip_path)
)

# %% [markdown]
# # Stop here
#
# Do not fit the forecast yet.
#
# Before modelling, compare the v3 `feature_metrics.json` with the second implementation.
#
# The most important fields to reconcile are:
#
# - `n_contracts`
# - `n_payment_rows_dev`
# - `sum_total_paid_model_period`
# - `n_panel_rows`
# - `payments_not_landed_in_panel`
# - `sum_expected_deposit`
# - `sum_expected_cum_final`
# - `sum_actual_cum_final`
# - `eff_CASH_mob0`
# - `eff_FINANCED_mob0`
# - `eff_FINANCED_mob1`
# - `eff_FINANCED_mob3`
# - `eff_FINANCED_mob6`
#
# If panel row counts still differ, compare the number of `POST_TENOR_RECOVERY` rows before changing the panel definition. Do not discard late cash merely to force two implementations to match.

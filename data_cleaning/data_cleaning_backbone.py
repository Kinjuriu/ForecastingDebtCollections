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

# %% [markdown] id="ff68a1ba"
# # Data Cleaning Backbone
#
# **Scope:** clean, standardise, audit, split, and seal the five source datasets.
#
# This notebook intentionally **does not perform feature engineering or modelling**.
#
# ## Time design
#
# | Layer | Months | Future purpose |
# |---|---|---|
# | Estimation | Oct 2024–Mar 2026 | Fit collection curves / models |
# | Validation | Apr–Jun 2026 | Tune and test pre-Q3 performance |
# | Sealed test | Jul–Sep 2026 | Reveal only after the forecast is locked |
#
# ### Cleaning order: split early, clean consistently
#
# The safest workflow is a hybrid:
#
# 1. preserve immutable raw files;
# 2. perform only the minimum schema normalisation needed to identify records correctly;
# 3. parse dates and assign each row to a time layer;
# 4. apply the **same deterministic and reversible cleaning functions** to all partitions;
# 5. perform all visible diagnostics and vocabulary review on data through **30-Jun-2026 only**;
# 6. keep Jul–Sep outputs in a separate sealed ZIP;
# 7. stop before any transformation that learns from the data.
#
# Examples safe to apply everywhere:
# - `contract_id` → `contractid`;
# - ID type standardisation;
# - whitespace / casing cleanup;
# - deterministic date parsing;
# - known business-label mappings;
# - exact duplicate removal with an audit copy.
#
# Examples that must later be learned from estimation data only:
# - imputation values;
# - outlier thresholds;
# - rare-category pooling based on frequency;
# - feature selection;
# - segmentation depth;
# - model tuning.
#
# ## Future modelling warning: pilot-era validation
#
# Apr–Jun 2026 is both the validation quarter and the East/West pilot period. A future base forecast should therefore **not** tune blindly to country-total Apr–Jun cash and call that pilot-free performance.
#
# The intended later modelling approach is:
# - retain a pooled region-aware base rather than discarding East/West;
# - validate region by region;
# - use North/South as the cleanest pilot-free reference;
# - inspect East/West Apr–Jun residuals separately so pilot-era deviations do not silently redefine the base curve;
# - evaluate pilot uplift causally in Part 2, not as a cleaning step.

# %% [markdown] id="4b56000c"
# ## 1. Upload the five original CSV files
#
# Select:
#
# - `contracts.csv`
# - `payments.csv`
# - `calls.csv`
# - `service_tickets.csv`
# - `collections_outreach.csv`
#
# At the end, the notebook creates separate download ZIPs for:
# - development data through Jun-2026;
# - development-period cleaning audits;
# - sealed Jul–Sep data and sealed audits;
# - full cleaned data for reproducibility only.
#
# Colab cannot force your browser to save directly to Desktop. When the browser asks where to save, choose Desktop.

# %% colab={"base_uri": "https://localhost:8080/", "height": 322} id="582fcab6" outputId="743b9d24-ec36-4b98-dc63-b288daa77b5e"
from google.colab import files
uploaded = files.upload()

print("Uploaded files:")
for name in uploaded:
    print(" -", name)

# %% [markdown] id="3f7871a2"
# ## 2. Imports and configuration

# %% id="fe8a7e96"
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Keep warnings ON during development.
# Parsing / dtype warnings are useful data-quality signals.

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)

EXPECTED_DATA_START = pd.Timestamp("2024-10-01")
ESTIMATION_END = pd.Timestamp("2026-03-31")
VALIDATION_END = pd.Timestamp("2026-06-30")
SEALED_TEST_END = pd.Timestamp("2026-09-30")
EXPECTED_SEALED_MONTHS = pd.period_range("2026-07", "2026-09", freq="M")

OUTPUT_ROOT = Path("/content/dlight_cleaning_v3")
DEV_DIR = OUTPUT_ROOT / "development_through_jun_2026"
DEV_AUDIT_DIR = OUTPUT_ROOT / "development_audit"
SEALED_DIR = OUTPUT_ROOT / "SEALED_TEST_JUL_SEP_2026"
FULL_DIR = OUTPUT_ROOT / "cleaned_full_reproducibility_only"

for p in [DEV_DIR, DEV_AUDIT_DIR, SEALED_DIR, FULL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Leave FALSE until the final Jul-Sep forecast is locked.
UNSEAL_FINAL_TEST_DIAGNOSTICS = False


# %% [markdown] id="39a42ac3"
# ## 3. Load files

# %% colab={"base_uri": "https://localhost:8080/"} id="91264968" outputId="815b3991-f5a1-465f-cac0-e3e15cc35b06"
def normalise_filename(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

def uploaded_stem(name):
    return normalise_filename(Path(name).stem)

def find_uploaded_file(expected_stem):
    """
    Prefer the exact CSV stem. If Colab renamed one upload, allow a unique
    partial match. Ambiguous matches stop the run instead of choosing a file.
    """
    key = normalise_filename(expected_stem)
    names = list(uploaded.keys())

    exact = [name for name in names if uploaded_stem(name) == key]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise RuntimeError(
            f"More than one exact upload matches {expected_stem!r}: {exact}. "
            "Remove duplicates and upload the five source files again."
        )

    partial = [name for name in names if key in uploaded_stem(name)]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise FileNotFoundError(
            f"Could not find an uploaded CSV matching {expected_stem!r}."
        )
    raise RuntimeError(
        f"Ambiguous uploads for {expected_stem!r}: {partial}. "
        "Rename the intended source to its standard filename and rerun."
    )

def read_uploaded_csv(expected_stem):
    filename = find_uploaded_file(expected_stem)
    return filename, pd.read_csv(io.BytesIO(uploaded[filename]))

selected_input_files = {}

selected_input_files["contracts"], contracts_raw = read_uploaded_csv("contracts")
selected_input_files["payments"], payments_raw = read_uploaded_csv("payments")
selected_input_files["calls"], calls_raw = read_uploaded_csv("calls")
selected_input_files["service_tickets"], service_raw = read_uploaded_csv("service_tickets")
selected_input_files["collections_outreach"], outreach_raw = read_uploaded_csv("collections_outreach")

print("Selected input files:")
for dataset, filename in selected_input_files.items():
    print(f" - {dataset}: {filename}")

# Raw copies remain untouched.
print("Raw files loaded. Detailed diagnostics are deferred until after time-layer assignment.")


# %% [markdown] id="1a483717"
# ## 4. Cleaning helpers

# %% id="e3a2b905"
def clean_column_names(df):
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return out.dropna(axis=1, how="all")

def clean_id(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )

def clean_text(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )

def parse_mixed_date(series):
    raw = series.astype("string").str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")

    result = pd.to_datetime(
        raw.where(numeric.isna()),
        errors="coerce"
    )

    numeric_mask = numeric.notna()
    if numeric_mask.any():
        result.loc[numeric_mask] = pd.to_datetime(
            numeric.loc[numeric_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce"
        )
    return result

def to_month_end(series):
    '''
    Convert a date to the true calendar month-end at midnight.
    Example: 2026-07-03 -> 2026-07-31 00:00:00.
    '''
    dates = pd.to_datetime(series, errors="coerce")
    return (
        dates.dt.to_period("M")
        .dt.to_timestamp(how="end")
        .dt.normalize()
    )

def assign_time_layer(date_series):
    d = pd.to_datetime(date_series, errors="coerce")
    layer = pd.Series(pd.NA, index=d.index, dtype="string")

    layer.loc[d < EXPECTED_DATA_START] = "pre_period"
    layer.loc[
        (d >= EXPECTED_DATA_START) & (d <= ESTIMATION_END)
    ] = "estimation"
    layer.loc[
        (d > ESTIMATION_END) & (d <= VALIDATION_END)
    ] = "validation"
    layer.loc[
        (d > VALIDATION_END) & (d <= SEALED_TEST_END)
    ] = "sealed_test"
    layer.loc[d > SEALED_TEST_END] = "post_test"

    return layer

def collapse_repeated_phrase(text):
    '''
    Collapse a phrase repeated verbatim:
    "battery fault battery fault battery fault" -> "battery fault"
    '''
    if pd.isna(text):
        return text

    s = re.sub(r"\s+", " ", str(text).strip())
    words = s.split()
    n = len(words)

    for chunk_len in range(1, n // 2 + 1):
        if n % chunk_len != 0:
            continue
        chunk = words[:chunk_len]
        if chunk * (n // chunk_len) == words:
            return " ".join(chunk)

    return s

def dev_mask(df, date_col):
    d = pd.to_datetime(df[date_col], errors="coerce")
    return d <= VALIDATION_END


# %% [markdown] id="6630ba14"
# ## 5. Contracts: structural cleaning
#
# Missing values in `customer_gender`, `household_size`, and `occupation` are preserved. They are optional fields, not automatically errors.
#
# ### Deposit-scale hypothesis
#
# Rows where a financed contract has `perc_deposit > 1` are treated as a likely scale error: the source value may contain the deposit **amount** rather than the deposit **proportion**.
#
# The notebook:
# - preserves the original value;
# - applies the hypothesised correction `original / price_usd`;
# - flags every corrected row;
# - validates the hypothesis using **contracts sold through Jun-2026 only** by comparing corrected values with naturally valid financed deposits.
#
# The correction remains an explicit assumption, not a silent rewrite.

# %% colab={"base_uri": "https://localhost:8080/", "height": 161} id="28d9c3e6" outputId="78ae6323-a31e-4c3a-bdee-3c3fc0fd3ec7"
contracts = clean_column_names(contracts_raw)

contracts["source_row_id"] = np.arange(len(contracts))

contracts["contractid"] = clean_id(contracts["contractid"])
contracts["sales_person_id"] = clean_id(contracts["sales_person_id"])

contracts["sales_month_original"] = contracts["sales_month"]
contracts["sales_month"] = to_month_end(
    parse_mixed_date(contracts["sales_month"])
)
contracts["time_layer"] = assign_time_layer(contracts["sales_month"])

for col in [
    "household_size", "price_usd", "perc_deposit",
    "daily_amount_usd", "tenor_length"
]:
    contracts[col] = pd.to_numeric(
        contracts[col],
        errors="coerce"
    )

contracts["region"] = clean_text(contracts["region"]).str.title()
contracts["customer_gender"] = clean_text(contracts["customer_gender"]).str.upper()
contracts["occupation"] = clean_text(contracts["occupation"]).str.upper()
contracts["contract_type"] = clean_text(contracts["contract_type"]).str.upper()
contracts["payment_frequency"] = clean_text(contracts["payment_frequency"]).str.upper()
contracts["product"] = clean_text(contracts["product"])

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

# Visible diagnostics use development data only.
contracts_dev_diag = contracts[dev_mask(contracts, "sales_month")].copy()

display(
    contracts_dev_diag[
        ["customer_gender", "household_size", "occupation"]
    ].isna().sum().to_frame("missing_through_jun_2026")
)

print(
    "Duplicate contract IDs through Jun-2026:",
    int(contracts_dev_diag["contractid"].duplicated(keep=False).sum())
)

# %% [markdown] id="86f761ca"
# ### Deposit-scale sanity check — through Jun-2026 only

# %% colab={"base_uri": "https://localhost:8080/", "height": 433} id="7f095d97" outputId="e8fd6e20-0640-44c0-97f7-56c63723b2ce"
fin_dev = contracts_dev_diag[
    contracts_dev_diag["contract_type"].eq("FINANCED")
].copy()

natural_valid_deposits = fin_dev.loc[
    (~fin_dev["deposit_scale_corrected"])
    & fin_dev["perc_deposit"].between(0, 1),
    "perc_deposit"
].dropna()

corrected_deposits_dev = fin_dev.loc[
    fin_dev["deposit_scale_corrected"],
    "perc_deposit"
].dropna()

quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

deposit_sanity_quantiles = pd.DataFrame({
    "quantile": quantiles,
    "naturally_valid": natural_valid_deposits.quantile(quantiles).values,
    "corrected_hypothesis": corrected_deposits_dev.quantile(quantiles).values
})

if len(natural_valid_deposits):
    natural_p01 = natural_valid_deposits.quantile(0.01)
    natural_p99 = natural_valid_deposits.quantile(0.99)

    corrected_within_natural_p01_p99 = (
        corrected_deposits_dev.between(
            natural_p01,
            natural_p99
        ).mean()
        if len(corrected_deposits_dev)
        else np.nan
    )
else:
    corrected_within_natural_p01_p99 = np.nan

deposit_sanity_summary = pd.DataFrame({
    "metric": [
        "naturally_valid_financed_deposits_n",
        "corrected_deposits_n_through_jun_2026",
        "share_corrected_within_natural_p01_p99"
    ],
    "value": [
        len(natural_valid_deposits),
        len(corrected_deposits_dev),
        corrected_within_natural_p01_p99
    ]
})

display(deposit_sanity_quantiles)
display(deposit_sanity_summary)

print(
    "Interpretation: if corrected deposits occupy a similar range to naturally valid deposits, "
    "the scale-error hypothesis is more credible. If not, quarantine and revisit rather than trusting the correction."
)

# %% [markdown] id="cf95f23b"
# ## 6. Payments: structural cleaning and duplicate policy
#
# The data dictionary describes `total_paid` as the **total paid in that contract-month**.
#
# Therefore:
#
# - exact duplicates are audited and one copy is retained;
# - same contract-month rows with different amounts are **not summed automatically**;
# - conflicting contract-months are flagged for review;
# - zero and negative payments are flagged rather than deleted;
# - an immutable `source_row_id` prevents fragile joins on floating-point payment amounts.

# %% colab={"base_uri": "https://localhost:8080/", "height": 489} id="7b25b40a" outputId="4b365bb7-6f59-4c0a-8cdc-afeaa2476c4c"
payments = clean_column_names(payments_raw)
payments["source_row_id"] = np.arange(len(payments))

# Critical key-name repair.
if "contract_id" in payments.columns:
    payments = payments.rename(
        columns={"contract_id": "contractid"}
    )

payments["contractid"] = clean_id(payments["contractid"])

payments["pay_month_original"] = payments["pay_month"]
payments["pay_month"] = to_month_end(
    parse_mixed_date(payments["pay_month"])
)
payments["pay_month_parse_failed"] = (
    payments["pay_month_original"].notna()
    & payments["pay_month"].isna()
)
payments["time_layer"] = assign_time_layer(payments["pay_month"])

payments["total_paid_original"] = payments["total_paid"]
payments["total_paid"] = pd.to_numeric(
    payments["total_paid"],
    errors="coerce"
)
payments["total_paid_parse_failed"] = (
    payments["total_paid_original"].notna()
    & payments["total_paid"].isna()
)

full_payment_input_checks = pd.DataFrame({
    "check": [
        "raw_payment_rows",
        "missing_contractid_any_period",
        "invalid_pay_month_any_period",
        "invalid_total_paid_any_period",
    ],
    "value": [
        len(payments),
        int(payments["contractid"].isna().sum()),
        int(payments["pay_month_parse_failed"].sum()),
        int(payments["total_paid_parse_failed"].sum()),
    ]
})

display(full_payment_input_checks)

if payments["pay_month_parse_failed"].any():
    raise ValueError(
        "Payment dates failed to parse. Review pay_month_original before "
        "creating development or sealed outputs."
    )

if payments["total_paid_parse_failed"].any():
    raise ValueError(
        "Some payment amounts are not numeric. Review total_paid_original "
        "before creating outputs."
    )

payments["before_expected_data_start"] = (
    payments["pay_month"].notna()
    & (payments["pay_month"] < EXPECTED_DATA_START)
)

# Exact duplicates: identify using business fields, NOT source_row_id.
exact_dup_subset = [
    "contractid", "pay_month", "total_paid"
]

payment_exact_duplicate_mask = payments.duplicated(
    subset=exact_dup_subset,
    keep=False
)

payment_exact_duplicates_full = payments.loc[
    payment_exact_duplicate_mask
].copy()

payments["exact_duplicate_extra"] = payments.duplicated(
    subset=exact_dup_subset,
    keep="first"
)

# Retain first copy only.
payments = payments.loc[
    ~payments["exact_duplicate_extra"]
].copy()

# Same contract-month with different values: quarantine, do not aggregate.
payments["conflicting_contract_month"] = payments.duplicated(
    subset=["contractid", "pay_month"],
    keep=False
)

payment_conflicting_contract_months_full = (
    payments.loc[
        payments["conflicting_contract_month"]
    ]
    .sort_values(
        ["contractid", "pay_month", "total_paid"]
    )
    .copy()
)

payments["zero_payment"] = payments["total_paid"].eq(0)
payments["negative_payment"] = payments["total_paid"].lt(0)

# Development-only visible summary.
payments_dev_diag = payments[dev_mask(payments, "pay_month")].copy()

payment_exact_duplicates_dev = payment_exact_duplicates_full[
    dev_mask(payment_exact_duplicates_full, "pay_month")
].copy()

payment_conflicts_dev = payment_conflicting_contract_months_full[
    dev_mask(payment_conflicting_contract_months_full, "pay_month")
].copy()

payment_checks_dev = pd.DataFrame({
    "check": [
        "rows_through_jun_after_exact_dedup",
        "missing_contractid",
        "invalid_pay_month_any_period",
        "missing_total_paid",
        "rows_participating_in_exact_duplicate_groups",
        "conflicting_contract_month_rows",
        "zero_payment_rows",
        "negative_payment_rows",
        "rows_before_expected_oct_2024_start"
    ],
    "value": [
        len(payments_dev_diag),
        int(payments_dev_diag["contractid"].isna().sum()),
        int(payments["pay_month_parse_failed"].sum()),
        int(payments_dev_diag["total_paid"].isna().sum()),
        len(payment_exact_duplicates_dev),
        len(payment_conflicts_dev),
        int(payments_dev_diag["zero_payment"].sum()),
        int(payments_dev_diag["negative_payment"].sum()),
        int(payments_dev_diag["before_expected_data_start"].sum())
    ]
})

display(payment_checks_dev)

# %% [markdown] id="ffe69065"
# ## 7. Calls: deterministic reason cleanup
#
# The calls table is event-level. Duplicate-looking calls are flagged but not removed because two same-day calls can both be real.
#
# Repeated text is collapsed here too. For example:
#
# `payment issue payment issue` → `PAYMENT_ISSUE`
#
# No synonym mapping is learned from the sealed period. The value-count diagnostic shown below uses calls through Jun-2026 only.

# %% colab={"base_uri": "https://localhost:8080/", "height": 287} id="0d0f8859" outputId="57c2db90-bba9-4a83-9801-222efc999124"
calls = clean_column_names(calls_raw)
calls["source_row_id"] = np.arange(len(calls))

calls["contractid"] = clean_id(calls["contractid"])

calls["call_date_original"] = calls["call_date"]
calls["call_date"] = parse_mixed_date(calls["call_date"])
calls["time_layer"] = assign_time_layer(calls["call_date"])

calls["call_reason_original"] = calls["call_reason"]

call_reason_key = (
    clean_text(calls["call_reason"])
    .str.lower()
    .str.replace("_", " ", regex=False)
    .map(collapse_repeated_phrase)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

calls["call_reason"] = (
    call_reason_key
    .str.upper()
    .str.replace(" ", "_", regex=False)
)

calls["possible_duplicate_event"] = calls.duplicated(
    subset=[
        "contractid",
        "call_date",
        "call_reason"
    ],
    keep=False
)

calls_dev_diag = calls[dev_mask(calls, "call_date")].copy()

print(
    "Possible duplicate-looking call rows through Jun-2026:",
    int(calls_dev_diag["possible_duplicate_event"].sum())
)

display(
    calls_dev_diag["call_reason"]
    .value_counts(dropna=False)
    .to_frame("rows_through_jun_2026")
)

# %% [markdown] id="8f5216ac"
# ## 8. Service tickets: development-only vocabulary review
#
# The mapping is intentionally based on known business meanings and is reviewed using **ticket counts through Jun-2026 only**.
#
# The sealed Jul–Sep text is cleaned by the same fixed rules but is not displayed or used to extend the vocabulary.
#
# ### Cable taxonomy
#
# Keep these separate:
#
# - `CABLE_CUT` — explicit physical cut / break;
# - `CABLE_FAULT` — cable problem without an explicit cut;
# - `WIRE_FAULT` — reported as a wire fault.
#
# These are different reported failure modes and should not be merged.

# %% colab={"base_uri": "https://localhost:8080/", "height": 614} id="714bcf26" outputId="b721abd7-b97f-43af-aa83-c715c11ac155"
service = clean_column_names(service_raw)
service["source_row_id"] = np.arange(len(service))

service["contractid"] = clean_id(service["contractid"])

service["ticket_date_original"] = service["ticket_date"]
service["ticket_date"] = parse_mixed_date(
    service["ticket_date"]
)
service["time_layer"] = assign_time_layer(
    service["ticket_date"]
)

service["ticket_reason_original"] = service["ticket_reason"]

service["_ticket_reason_key"] = (
    clean_text(service["ticket_reason"])
    .str.lower()
    .str.replace("_", " ", regex=False)
    .map(collapse_repeated_phrase)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Review vocabulary using development data ONLY.
ticket_reason_dev_counts = (
    service.loc[
        dev_mask(service, "ticket_date"),
        "_ticket_reason_key"
    ]
    .value_counts(dropna=False)
    .rename_axis("normalised_reason")
    .reset_index(name="rows_through_jun_2026")
)

display(ticket_reason_dev_counts)

# %% [markdown] id="075ef259"
# ### Fixed ticket-reason mapping

# %% colab={"base_uri": "https://localhost:8080/", "height": 900} id="6e47852f" outputId="f600e285-58ab-4082-cb2a-c78712b86b37"
ticket_reason_map = {
    # Battery
    "battery fault": "BATTERY_FAULT",
    "battery failure": "BATTERY_FAULT",
    "batt fault": "BATTERY_FAULT",

    # Charging
    "charging issue": "CHARGING_ISSUE",
    "not charging": "CHARGING_ISSUE",
    "won't charge": "CHARGING_ISSUE",
    "no charge": "CHARGING_ISSUE",

    # Panel
    "panel damage": "PANEL_DAMAGE",
    "panel broken": "PANEL_DAMAGE",
    "panel cracked": "PANEL_DAMAGE",

    # Keep cable / wire failure modes separate
    "cable cut": "CABLE_CUT",
    "cable fault": "CABLE_FAULT",
    "wire fault": "WIRE_FAULT",

    # Lighting
    "lamp fault": "LIGHT_FAULT",
    "bulb not working": "LIGHT_FAULT",
    "light fault": "LIGHT_FAULT",

    # Other
    "other": "OTHER",
    "misc": "OTHER"
}

service["ticket_reason_mapped"] = (
    service["_ticket_reason_key"].isin(
        ticket_reason_map.keys()
    )
)

service["ticket_reason"] = (
    service["_ticket_reason_key"]
    .map(ticket_reason_map)
    .fillna(
        service["_ticket_reason_key"]
        .str.upper()
        .str.replace(" ", "_", regex=False)
    )
)

service["ticket_outcome_original"] = service["ticket_outcome"]
service["ticket_outcome"] = (
    clean_text(service["ticket_outcome"])
    .str.upper()
    .str.replace(r"\s+", "_", regex=True)
)

service_dev_diag = service[
    dev_mask(service, "ticket_date")
].copy()

ticket_reason_mapping_audit_dev = (
    service_dev_diag.groupby(
        [
            "ticket_reason_original",
            "_ticket_reason_key",
            "ticket_reason",
            "ticket_reason_mapped"
        ],
        dropna=False
    )
    .size()
    .reset_index(name="rows")
    .sort_values("rows", ascending=False)
)

unmapped_ticket_reasons_dev = (
    service_dev_diag.loc[
        ~service_dev_diag["ticket_reason_mapped"].fillna(False),
        "_ticket_reason_key"
    ]
    .value_counts(dropna=False)
    .rename_axis("unmapped_normalised_reason")
    .reset_index(name="rows")
)

fallback_share_dev = (
    (~service_dev_diag["ticket_reason_mapped"].fillna(False)).mean()
    if len(service_dev_diag)
    else np.nan
)

print(
    "Fallback/unmapped share through Jun-2026:",
    round(float(fallback_share_dev), 4)
    if pd.notna(fallback_share_dev)
    else np.nan
)

display(ticket_reason_mapping_audit_dev.head(100))

if len(unmapped_ticket_reasons_dev):
    print("Unmapped reasons to review BEFORE extending the mapping:")
    display(unmapped_ticket_reasons_dev)
else:
    print("No unmapped ticket reasons through Jun-2026.")

# %% [markdown] id="168a0ed6"
# ## 9. Collections outreach: structural cleaning only
#
# The table is not country-wide history. It represents regional pilots.
#
# This notebook standardises its fields and audits integrity but does **not** estimate an uplift.
# Visible summaries use data through Jun-2026 only.

# %% colab={"base_uri": "https://localhost:8080/", "height": 143} id="6d2ec35f" outputId="a5a4e2b0-3ff1-498a-cf40-f07c8a27e756"
outreach = clean_column_names(outreach_raw)
outreach["source_row_id"] = np.arange(len(outreach))

outreach["contractid"] = clean_id(
    outreach["contractid"]
)

outreach["contact_month_original"] = outreach["contact_month"]
outreach["contact_month"] = to_month_end(
    parse_mixed_date(outreach["contact_month"])
)
outreach["time_layer"] = assign_time_layer(
    outreach["contact_month"]
)

outreach["region"] = clean_text(
    outreach["region"]
).str.title()

outreach["channel"] = clean_text(
    outreach["channel"]
).str.upper()

outreach["attempts"] = pd.to_numeric(
    outreach["attempts"],
    errors="coerce"
)

outreach["cost_usd"] = pd.to_numeric(
    outreach["cost_usd"],
    errors="coerce"
)

outreach["reached_original"] = outreach["reached"]
outreach["reached"] = (
    outreach["reached"]
    .astype("string")
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
    .astype("boolean")
)

outreach["possible_duplicate_contract_month"] = outreach.duplicated(
    subset=["contractid", "contact_month"],
    keep=False
)

outreach_dev_diag = outreach[
    dev_mask(outreach, "contact_month")
].copy()

display(
    outreach_dev_diag
    .groupby(["region", "channel"])
    .size()
    .to_frame("rows_through_jun_2026")
)

# %% [markdown] id="90c6c8f7"
# ## 10. Cross-file orphan IDs
#
# Orphan records are flagged, not deleted.
#
# Visible counts are limited to records through Jun-2026.

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="6e6fd548" outputId="0b9c5a33-d4ca-43a1-8fa3-71b58f01c4a1"
contract_id_set = set(
    contracts["contractid"].dropna()
)

def add_orphan_flag(df):
    out = df.copy()
    out["orphan_contractid"] = (
        out["contractid"].notna()
        &
        ~out["contractid"].isin(
            contract_id_set
        )
    )
    return out

payments = add_orphan_flag(payments)
calls = add_orphan_flag(calls)
service = add_orphan_flag(service)
outreach = add_orphan_flag(outreach)

orphan_summary_dev = pd.DataFrame({
    "dataset": [
        "payments",
        "calls",
        "service_tickets",
        "collections_outreach"
    ],
    "orphan_rows_through_jun_2026": [
        int(payments.loc[dev_mask(payments, "pay_month"), "orphan_contractid"].sum()),
        int(calls.loc[dev_mask(calls, "call_date"), "orphan_contractid"].sum()),
        int(service.loc[dev_mask(service, "ticket_date"), "orphan_contractid"].sum()),
        int(outreach.loc[dev_mask(outreach, "contact_month"), "orphan_contractid"].sum())
    ]
})

display(orphan_summary_dev)

# %% [markdown] id="09980b63"
# ## 11. Events before sale month
#
# Same-month events are allowed because `sales_month` has only month-level precision.
#
# Only records strictly before the contract's sale **month** are flagged.
#
# Nothing is auto-deleted.

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="ade7d423" outputId="26ace3d1-9693-481c-b002-0baa1abca1a5"
contract_dates = contracts[
    ["contractid", "sales_month"]
].copy()

def flag_before_sale(df, date_col):
    out = df.merge(
        contract_dates,
        on="contractid",
        how="left",
        validate="m:1"
    )

    event_period = pd.to_datetime(
        out[date_col],
        errors="coerce"
    ).dt.to_period("M")

    sale_period = pd.to_datetime(
        out["sales_month"],
        errors="coerce"
    ).dt.to_period("M")

    out["before_sale_month"] = (
        event_period < sale_period
    )

    return out

payments = flag_before_sale(payments, "pay_month")
calls = flag_before_sale(calls, "call_date")
service = flag_before_sale(service, "ticket_date")
outreach = flag_before_sale(outreach, "contact_month")

before_sale_summary_dev = pd.DataFrame({
    "dataset": [
        "payments",
        "calls",
        "service_tickets",
        "collections_outreach"
    ],
    "rows_before_sale_month_through_jun_2026": [
        int(payments.loc[dev_mask(payments, "pay_month"), "before_sale_month"].sum()),
        int(calls.loc[dev_mask(calls, "call_date"), "before_sale_month"].sum()),
        int(service.loc[dev_mask(service, "ticket_date"), "before_sale_month"].sum()),
        int(outreach.loc[dev_mask(outreach, "contact_month"), "before_sale_month"].sum())
    ]
})

display(before_sale_summary_dev)

# %% [markdown] id="2942afac"
# ## 12. Payments after apparent payoff — row-ID-safe audit
#
# A payment is flagged when cumulative **prior positive payments** are already at or above contract price.
#
# Important:
# - negative rows would require business interpretation, so the logic uses positive cash only;
# - conflicting duplicate contract-months are excluded from this particular audit because their monthly total is unresolved;
# - the result is carried back using `source_row_id`, never by floating-point equality.

# %% colab={"base_uri": "https://localhost:8080/"} id="416f4243" outputId="8356c855-e085-4c04-84e4-8b4b05fae693"
payments["payment_after_apparent_payoff"] = False
payments["payoff_audit_eligible"] = (
    ~payments["conflicting_contract_month"]
    & payments["contractid"].notna()
    & payments["pay_month"].notna()
    & payments["total_paid"].notna()
)

payoff_work = payments.loc[
    payments["payoff_audit_eligible"]
].copy()

payoff_work = payoff_work.merge(
    contracts[["contractid", "price_usd"]],
    on="contractid",
    how="left",
    validate="m:1"
)

payoff_work = payoff_work.sort_values(
    ["contractid", "pay_month", "source_row_id"]
).copy()

payoff_work["_positive_paid"] = (
    payoff_work["total_paid"]
    .clip(lower=0)
    .fillna(0)
)

payoff_work["cum_positive_before"] = (
    payoff_work.groupby("contractid")["_positive_paid"]
    .cumsum()
    - payoff_work["_positive_paid"]
)

payoff_work["payment_after_apparent_payoff"] = (
    payoff_work["price_usd"].notna()
    &
    (
        payoff_work["cum_positive_before"]
        >= payoff_work["price_usd"]
    )
    &
    (payoff_work["_positive_paid"] > 0)
)

payoff_flag_by_row_id = payoff_work.set_index(
    "source_row_id"
)["payment_after_apparent_payoff"]

payments["payment_after_apparent_payoff"] = (
    payments["source_row_id"]
    .map(payoff_flag_by_row_id)
    .fillna(False)
    .astype(bool)
)

print(
    "Payments after apparent payoff through Jun-2026:",
    int(
        payments.loc[
            dev_mask(payments, "pay_month"),
            "payment_after_apparent_payoff"
        ].sum()
    )
)

# %% [markdown] id="c216f28c"
# ## 13. Partial / thin month diagnostic — development period only
#
# This checks **extraction completeness**, not business performance.
#
# A weak collections month can have lower cash while still containing a normal number of payment records and paying contracts. A truncated extract usually shows several volume indicators collapsing together.
#
# The diagnostic compares:
# - total cash;
# - payment-row count;
# - unique paying contracts;
# - mean cash per paying contract;
#
# to the previous 3-month median.
#
# Interpretation:
# - cash falls while rows/contracts stay near normal → likely lower payment amounts, not necessarily incomplete data;
# - cash + rows + unique contracts all collapse → possible truncation / extraction issue;
# - a month before Oct-2024 → pre-period anomaly, not part of the modelling window.
#
# No month is automatically deleted.

# %% colab={"base_uri": "https://localhost:8080/", "height": 759} id="3b92efaf" outputId="7eaa9c05-9bd5-414d-abd6-f530cfad5855"
payments_dev_for_diagnostic = payments[
    payments["pay_month"] <= VALIDATION_END
].copy()

monthly_payment_diagnostic = (
    payments_dev_for_diagnostic
    .groupby("pay_month", as_index=False)
    .agg(
        total_paid=("total_paid", "sum"),
        rows=("contractid", "size"),
        unique_contracts=("contractid", "nunique")
    )
    .sort_values("pay_month")
)

monthly_payment_diagnostic["mean_paid_per_contract"] = (
    monthly_payment_diagnostic["total_paid"]
    /
    monthly_payment_diagnostic["unique_contracts"].replace(0, np.nan)
)

monthly_payment_diagnostic["before_expected_data_start"] = (
    monthly_payment_diagnostic["pay_month"] < EXPECTED_DATA_START
)

for col in [
    "total_paid",
    "rows",
    "unique_contracts",
    "mean_paid_per_contract"
]:
    monthly_payment_diagnostic[
        f"{col}_prev3_median"
    ] = (
        monthly_payment_diagnostic[col]
        .shift(1)
        .rolling(3)
        .median()
    )

    monthly_payment_diagnostic[
        f"{col}_ratio_to_prev3"
    ] = (
        monthly_payment_diagnostic[col]
        /
        monthly_payment_diagnostic[
            f"{col}_prev3_median"
        ]
    )

display(monthly_payment_diagnostic)

# %% [markdown] id="B7HmJE9Jqsf5"
# ### Full-input guardrails
#
# Before the split, the notebook now checks the complete payments source for
# unparseable dates and amounts. After the split, it requires payment coverage in
# July, August and September 2026. These checks validate extraction completeness;
# they do not use Q3 outcomes to fit or change a model.
#
# The sealed ZIP also contains a three-row country actuals file for scoring the
# already-frozen forecasts without sharing customer-level payment data.
#

# %% [markdown] id="f23c0f02"
# ## 14. Optional sealed-period completeness diagnostic
#
# Do not run this while building the forecast.
#
# Only after the Jul–Sep forecast is locked should you set:
#
# `UNSEAL_FINAL_TEST_DIAGNOSTICS = True`

# %% colab={"base_uri": "https://localhost:8080/"} id="e0b8ebfc" outputId="2af13599-b5a4-480b-e7ed-98d70dfd8c7d"
if UNSEAL_FINAL_TEST_DIAGNOSTICS:
    sealed_payment_diagnostic = (
        payments[
            (payments["pay_month"] > VALIDATION_END)
            &
            (payments["pay_month"] <= SEALED_TEST_END)
        ]
        .groupby("pay_month", as_index=False)
        .agg(
            total_paid=("total_paid", "sum"),
            rows=("contractid", "size"),
            unique_contracts=("contractid", "nunique")
        )
        .sort_values("pay_month")
    )

    sealed_payment_diagnostic["mean_paid_per_contract"] = (
        sealed_payment_diagnostic["total_paid"]
        /
        sealed_payment_diagnostic["unique_contracts"].replace(0, np.nan)
    )

    display(sealed_payment_diagnostic)
else:
    print(
        "SEALED: Jul-Sep payment diagnostics are not displayed."
    )

# %% colab={"base_uri": "https://localhost:8080/", "height": 339} id="wsOJp5WxtSGz" outputId="dc39dbcc-eefc-4e69-aaf0-33335df9452b"
payment_dates = to_month_end(
    parse_mixed_date(payments_raw["pay_month"])
)

print("First payment month:", payment_dates.min())
print("Last payment month:", payment_dates.max())

display(
    payment_dates
    .dt.to_period("M")
    .value_counts()
    .sort_index()
    .tail(6)
)


# %% [markdown] id="73420a68"
# ## 15. Split cleaned data into development and sealed files
#
# Existing contracts sold on or before Jun-2026 remain in development because their terms were known at the forecast origin.
#
# New Jul–Sep contracts and Jul–Sep payment/event outcomes are placed in the sealed partition.

# %% colab={"base_uri": "https://localhost:8080/", "height": 358} id="53806181" outputId="8164e3ed-cc66-4131-ac01-b8caea1c5610"
def split_dev_sealed(df, date_col):
    d = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    dev = df[d <= VALIDATION_END].copy()

    sealed = df[
        (d > VALIDATION_END)
        &
        (d <= SEALED_TEST_END)
    ].copy()

    return dev, sealed

contracts_dev, contracts_sealed = split_dev_sealed(
    contracts, "sales_month"
)
payments_dev, payments_sealed = split_dev_sealed(
    payments, "pay_month"
)
calls_dev, calls_sealed = split_dev_sealed(
    calls, "call_date"
)
service_dev, service_sealed = split_dev_sealed(
    service, "ticket_date"
)
outreach_dev, outreach_sealed = split_dev_sealed(
    outreach, "contact_month"
)

def require_month_coverage(df, date_col, expected_months, dataset_name):
    observed = set(
        pd.to_datetime(df[date_col], errors="coerce")
        .dropna()
        .dt.to_period("M")
    )
    expected = set(expected_months)
    missing = sorted(expected - observed)
    if missing:
        missing_text = ", ".join(str(month) for month in missing)
        raise RuntimeError(
            f"{dataset_name} is missing required sealed month(s): {missing_text}. "
            "The sealed forecast cannot be scored from this extract. Check the "
            "selected source filename and the raw payment date range."
        )

require_month_coverage(
    payments_sealed,
    "pay_month",
    EXPECTED_SEALED_MONTHS,
    "payments"
)

# This is the country-level target used to score the frozen forecast. It uses
# the source-reported monthly cash after exact duplicate removal, matching the
# historical forecast target. No customer identifiers are included.
sealed_country_actuals = (
    payments_sealed
    .groupby("pay_month", as_index=False)["total_paid"]
    .sum()
    .rename(columns={
        "pay_month": "month",
        "total_paid": "actual_total_collections_usd"
    })
    .sort_values("month")
)

sealed_payment_completeness = (
    payments_sealed
    .groupby("pay_month", as_index=False)
    .agg(
        payment_rows=("contractid", "size"),
        unique_paying_contracts=("contractid", "nunique")
    )
    .rename(columns={"pay_month": "month"})
    .sort_values("month")
)

sealed_country_actuals.to_csv(
    SEALED_DIR / "SEALED_country_collections_actuals_jul_sep_2026.csv",
    index=False
)
sealed_payment_completeness.to_csv(
    SEALED_DIR / "SEALED_payment_completeness_jul_sep_2026.csv",
    index=False
)

if UNSEAL_FINAL_TEST_DIAGNOSTICS:
    display(sealed_country_actuals)
    display(sealed_payment_completeness)
else:
    print("PASS: sealed payments cover Jul, Aug and Sep 2026. Values remain hidden.")

contracts_dev.to_csv(
    DEV_DIR / "contracts_clean_through_jun_2026.csv",
    index=False
)
payments_dev.to_csv(
    DEV_DIR / "payments_clean_through_jun_2026.csv",
    index=False
)
calls_dev.to_csv(
    DEV_DIR / "calls_clean_through_jun_2026.csv",
    index=False
)
service_dev.to_csv(
    DEV_DIR / "service_tickets_clean_through_jun_2026.csv",
    index=False
)
outreach_dev.to_csv(
    DEV_DIR / "collections_outreach_clean_through_jun_2026.csv",
    index=False
)

contracts_sealed.to_csv(
    SEALED_DIR / "SEALED_new_contracts_jul_sep_2026.csv",
    index=False
)
payments_sealed.to_csv(
    SEALED_DIR / "SEALED_payments_jul_sep_2026.csv",
    index=False
)
calls_sealed.to_csv(
    SEALED_DIR / "SEALED_calls_jul_sep_2026.csv",
    index=False
)
service_sealed.to_csv(
    SEALED_DIR / "SEALED_service_tickets_jul_sep_2026.csv",
    index=False
)
outreach_sealed.to_csv(
    SEALED_DIR / "SEALED_collections_outreach_jul_sep_2026.csv",
    index=False
)

print("Development and sealed datasets written.")

# %% [markdown] id="b4ca27f8"
# ## 16. Development-period audit files
#
# Only through-Jun audits go into the normal audit ZIP.
#
# Any anomaly records from Jul–Sep are placed inside the **sealed** directory so you do not accidentally inspect them.

# %% id="c600e967"
# Development-only audits
payment_exact_duplicates_dev = payment_exact_duplicates_full[
    dev_mask(payment_exact_duplicates_full, "pay_month")
].copy()

payment_conflicts_dev = payment_conflicting_contract_months_full[
    dev_mask(payment_conflicting_contract_months_full, "pay_month")
].copy()

payment_exact_duplicates_dev.to_csv(
    DEV_AUDIT_DIR / "payment_exact_duplicate_group_rows_through_jun.csv",
    index=False
)
payment_conflicts_dev.to_csv(
    DEV_AUDIT_DIR / "payment_conflicting_contract_months_REVIEW.csv",
    index=False
)

calls_dev[
    calls_dev["possible_duplicate_event"]
].to_csv(
    DEV_AUDIT_DIR / "calls_possible_duplicate_events_REVIEW.csv",
    index=False
)

ticket_reason_dev_counts.to_csv(
    DEV_AUDIT_DIR / "ticket_reason_normalised_counts_through_jun.csv",
    index=False
)
ticket_reason_mapping_audit_dev.to_csv(
    DEV_AUDIT_DIR / "ticket_reason_mapping_audit_through_jun.csv",
    index=False
)
unmapped_ticket_reasons_dev.to_csv(
    DEV_AUDIT_DIR / "ticket_reason_unmapped_REVIEW.csv",
    index=False
)

deposit_sanity_quantiles.to_csv(
    DEV_AUDIT_DIR / "deposit_scale_sanity_quantiles.csv",
    index=False
)
deposit_sanity_summary.to_csv(
    DEV_AUDIT_DIR / "deposit_scale_sanity_summary.csv",
    index=False
)

payments_dev[
    payments_dev["orphan_contractid"]
].to_csv(
    DEV_AUDIT_DIR / "orphan_payments.csv",
    index=False
)
calls_dev[
    calls_dev["orphan_contractid"]
].to_csv(
    DEV_AUDIT_DIR / "orphan_calls.csv",
    index=False
)
service_dev[
    service_dev["orphan_contractid"]
].to_csv(
    DEV_AUDIT_DIR / "orphan_service_tickets.csv",
    index=False
)
outreach_dev[
    outreach_dev["orphan_contractid"]
].to_csv(
    DEV_AUDIT_DIR / "orphan_collections_outreach.csv",
    index=False
)

payments_dev[
    payments_dev["before_sale_month"]
].to_csv(
    DEV_AUDIT_DIR / "payments_before_sale_REVIEW.csv",
    index=False
)
calls_dev[
    calls_dev["before_sale_month"]
].to_csv(
    DEV_AUDIT_DIR / "calls_before_sale_REVIEW.csv",
    index=False
)
service_dev[
    service_dev["before_sale_month"]
].to_csv(
    DEV_AUDIT_DIR / "service_tickets_before_sale_REVIEW.csv",
    index=False
)
outreach_dev[
    outreach_dev["before_sale_month"]
].to_csv(
    DEV_AUDIT_DIR / "outreach_before_sale_REVIEW.csv",
    index=False
)

payments_dev[
    payments_dev["zero_payment"]
].to_csv(
    DEV_AUDIT_DIR / "zero_payment_rows_REVIEW.csv",
    index=False
)
payments_dev[
    payments_dev["negative_payment"]
].to_csv(
    DEV_AUDIT_DIR / "negative_payment_rows_REVIEW.csv",
    index=False
)
payments_dev[
    payments_dev["payment_after_apparent_payoff"]
].to_csv(
    DEV_AUDIT_DIR / "payments_after_apparent_payoff_REVIEW.csv",
    index=False
)
payments_dev[
    payments_dev["before_expected_data_start"]
].to_csv(
    DEV_AUDIT_DIR / "payments_before_expected_oct_2024_start_REVIEW.csv",
    index=False
)

contracts_dev[
    contracts_dev["deposit_scale_corrected"]
].to_csv(
    DEV_AUDIT_DIR / "contracts_deposit_scale_corrected_REVIEW.csv",
    index=False
)

monthly_payment_diagnostic.to_csv(
    DEV_AUDIT_DIR / "monthly_payment_completeness_through_jun_2026.csv",
    index=False
)

print("Development audits written.")

# %% [markdown] id="823c389a"
# ## 17. Sealed anomaly audits — written but not displayed

# %% id="8926a6cb"
# These files stay INSIDE the sealed ZIP.
# They are never displayed while the test is sealed.

payment_exact_duplicates_sealed = payment_exact_duplicates_full[
    (
        payment_exact_duplicates_full["pay_month"] > VALIDATION_END
    )
    &
    (
        payment_exact_duplicates_full["pay_month"] <= SEALED_TEST_END
    )
].copy()

payment_conflicts_sealed = payment_conflicting_contract_months_full[
    (
        payment_conflicting_contract_months_full["pay_month"] > VALIDATION_END
    )
    &
    (
        payment_conflicting_contract_months_full["pay_month"] <= SEALED_TEST_END
    )
].copy()

payment_exact_duplicates_sealed.to_csv(
    SEALED_DIR / "SEALED_AUDIT_payment_exact_duplicates.csv",
    index=False
)
payment_conflicts_sealed.to_csv(
    SEALED_DIR / "SEALED_AUDIT_payment_conflicts.csv",
    index=False
)

# Other anomaly subsets
for filename, df, flag in [
    ("SEALED_AUDIT_payments_before_sale.csv", payments_sealed, "before_sale_month"),
    ("SEALED_AUDIT_calls_before_sale.csv", calls_sealed, "before_sale_month"),
    ("SEALED_AUDIT_service_before_sale.csv", service_sealed, "before_sale_month"),
    ("SEALED_AUDIT_outreach_before_sale.csv", outreach_sealed, "before_sale_month"),
    ("SEALED_AUDIT_orphan_payments.csv", payments_sealed, "orphan_contractid"),
    ("SEALED_AUDIT_orphan_calls.csv", calls_sealed, "orphan_contractid"),
    ("SEALED_AUDIT_orphan_service.csv", service_sealed, "orphan_contractid"),
    ("SEALED_AUDIT_orphan_outreach.csv", outreach_sealed, "orphan_contractid"),
]:
    df.loc[df[flag]].to_csv(
        SEALED_DIR / filename,
        index=False
    )

print("Sealed audit files written without displaying their contents.")

# %% [markdown] id="c3abb8e2"
# ## 18. Development cleaning summary only
#
# The summary intentionally stops at Jun-2026.
#
# Duplicate note: `rows_participating_in_exact_duplicate_groups` counts **all** rows in duplicate groups. If duplicates mostly come in pairs, the number physically removed is roughly half that count because one copy is retained.

# %% id="2d438178"
cleaning_summary_dev = pd.DataFrame([
    ["contracts_rows_through_jun", len(contracts_dev)],
    ["contracts_gender_missing", int(contracts_dev["customer_gender"].isna().sum())],
    ["contracts_household_size_missing", int(contracts_dev["household_size"].isna().sum())],
    ["contracts_occupation_missing", int(contracts_dev["occupation"].isna().sum())],
    ["contracts_deposit_scale_corrected", int(contracts_dev["deposit_scale_corrected"].sum())],

    ["payments_rows_through_jun_after_exact_dedup", len(payments_dev)],
    ["payments_exact_duplicate_group_rows", len(payment_exact_duplicates_dev)],
    ["payments_conflicting_contract_month_rows", len(payment_conflicts_dev)],
    ["payments_zero_rows", int(payments_dev["zero_payment"].sum())],
    ["payments_negative_rows", int(payments_dev["negative_payment"].sum())],
    ["payments_before_expected_oct_2024_start", int(payments_dev["before_expected_data_start"].sum())],
    ["payments_before_sale_rows", int(payments_dev["before_sale_month"].sum())],
    ["payments_after_apparent_payoff_rows", int(payments_dev["payment_after_apparent_payoff"].sum())],

    ["calls_possible_duplicate_event_rows", int(calls_dev["possible_duplicate_event"].sum())],
    ["calls_before_sale_rows", int(calls_dev["before_sale_month"].sum())],

    ["service_ticket_canonical_reason_count", int(service_dev["ticket_reason"].nunique(dropna=True))],
    ["service_ticket_unmapped_share", fallback_share_dev],
    ["service_before_sale_rows", int(service_dev["before_sale_month"].sum())],

    ["outreach_before_sale_rows", int(outreach_dev["before_sale_month"].sum())],
    ["outreach_regions", ", ".join(sorted(outreach_dev["region"].dropna().astype(str).unique()))],
    ["outreach_channels", ", ".join(sorted(outreach_dev["channel"].dropna().astype(str).unique()))],

    ["orphan_payment_rows", int(payments_dev["orphan_contractid"].sum())],
    ["orphan_call_rows", int(calls_dev["orphan_contractid"].sum())],
    ["orphan_service_rows", int(service_dev["orphan_contractid"].sum())],
    ["orphan_outreach_rows", int(outreach_dev["orphan_contractid"].sum())],
], columns=["check", "value"])

cleaning_summary_dev.to_csv(
    DEV_AUDIT_DIR / "cleaning_summary_through_jun_2026.csv",
    index=False
)

display(cleaning_summary_dev)

# %% [markdown] id="1b00baae"
# ## 19. Full cleaned files — reproducibility only
#
# These are written so the cleaning is reproducible, but do **not** use or inspect the full ZIP while building the forecast.

# %% id="cba3fb8e"
# Drop temporary helper key before exporting service.
service_export = service.drop(
    columns=["_ticket_reason_key"]
)

contracts.to_csv(
    FULL_DIR / "contracts_clean_full.csv",
    index=False
)
payments.to_csv(
    FULL_DIR / "payments_clean_full.csv",
    index=False
)
calls.to_csv(
    FULL_DIR / "calls_clean_full.csv",
    index=False
)
service_export.to_csv(
    FULL_DIR / "service_tickets_clean_full.csv",
    index=False
)
outreach.to_csv(
    FULL_DIR / "collections_outreach_clean_full.csv",
    index=False
)

print("Full cleaned copies written, but should remain unopened during model development.")


# %% [markdown] id="e59564a4"
# ## 20. Package outputs

# %% id="d3174743"
def zip_directory(source_dir, zip_path):
    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:
        for path in source_dir.rglob("*"):
            if path.is_file():
                z.write(
                    path,
                    arcname=path.relative_to(source_dir)
                )

DEV_ZIP = Path("/content/development_through_jun_2026_v3.zip")
DEV_AUDIT_ZIP = Path("/content/data_quality_audit_through_jun_2026_v3.zip")
SEALED_ZIP = Path("/content/SEALED_TEST_JUL_SEP_2026_DO_NOT_OPEN_v3.zip")
FULL_ZIP = Path("/content/cleaned_full_reproducibility_only_v3.zip")

zip_directory(DEV_DIR, DEV_ZIP)
zip_directory(DEV_AUDIT_DIR, DEV_AUDIT_ZIP)
zip_directory(SEALED_DIR, SEALED_ZIP)
zip_directory(FULL_DIR, FULL_ZIP)

print("Created:")
print("Development:", DEV_ZIP)
print("Development audit:", DEV_AUDIT_ZIP)
print("SEALED:", SEALED_ZIP)
print("Full reproducibility:", FULL_ZIP)

# %% [markdown] id="b91af5bb"
# ## 21. Download buttons
#
# For normal work, download:
#
# 1. `development_through_jun_2026_v3.zip`
# 2. `data_quality_audit_through_jun_2026_v3.zip`
#
# Also save the sealed ZIP somewhere safe, but **do not open it** until the forecast is locked.
#
# The full reproducibility ZIP is optional.

# %% id="6193e96f"
from google.colab import files

files.download(str(DEV_ZIP))
files.download(str(DEV_AUDIT_ZIP))
files.download(str(SEALED_ZIP))
files.download(str(FULL_ZIP))

# %% [markdown] id="12e1c21b"
# # Stop here
#
# This notebook ends at cleaning and sealing.
#
# Do **not** create:
# - months-on-book;
# - arrears measures;
# - payment lags;
# - rolling collection rates;
# - call-count features;
# - service-ticket features;
# - outreach features;
# - cohort curves;
# - forecasting models.
#
# Those belong to the next stage after the development audit has been reviewed and the ambiguous records have been resolved or explicitly documented.

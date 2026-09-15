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

# %% [markdown] id="EB06y-zWVSzx"
# # Part 2 — regional collections pilot evaluation
#
# This notebook evaluates two regional outreach pilots: East's preventative SMS pilot and West's outbound-call pilot. It is separate from the Part 1 collections forecast. Each pilot is compared against contemporaneous, within-region customers with similar pre-treatment histories. Because assignment was not randomized, the result is a **quasi-experimental estimate**, not definitive causal proof.
#
# Primary outcome: **next-calendar-month cash per attempted contract-month**. Same-month payment is not used because the ordering of outreach and payment is unknown. The analysis uses intention-to-treat: unsuccessful contact attempts remain part of the programme cost and effect.

# %% [markdown] id="uzC5LonLVSz0"
# ## Interpretation rules
#
# - Evaluate each pilot against matched customers in its own region; do not compare raw East and West repayment rates.
# - Adjust only with information known before outreach.
# - Exclude June contacts from the primary effect because July outcomes are sealed.
# - Recommend scale only when the estimated incremental cash exceeds cost and the 95% interval is above zero.
# - If neither clears that bar, place the October budget into a randomized learning programme rather than claiming either pilot worked.
# - Prior service-ticket history is included as a confounder. Unconfirmed product reports are not treated as established facts.

# %% id="SU6pjKr7VSz1"
# %pip -q install scikit-learn

import io
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

VALIDATION_END = pd.Timestamp("2026-06-30")
PILOT_START = pd.Timestamp("2026-04-30")
PRIMARY_CONTACT_END = pd.Timestamp("2026-05-31")
MONTHLY_BUDGET_USD = 8000.0
RANDOM_SEED = 20260911
OUTPUT_DIR = Path("/content/dlight_part2_pilot_private")
OUTPUT_DIR.mkdir(exist_ok=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 92} id="Co5ydF1QVSz2" outputId="88e04d2c-9200-4a96-9210-0950448ba678"
try:
    from google.colab import files
    uploaded = files.upload()
    zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
    if not zip_names:
        raise FileNotFoundError("Upload dlight_feature_engineering_outputs_v3.zip")
    preferred = [name for name in zip_names if "feature_engineering_outputs_v3" in name.lower()]
    feature_zip = io.BytesIO(uploaded[preferred[0] if preferred else zip_names[0]])
except ImportError:
    feature_zip = Path("dlight_feature_engineering_outputs_v3.zip")

EXTRACT_DIR = Path("/content/dlight_part2_feature_outputs")
EXTRACT_DIR.mkdir(exist_ok=True)
with zipfile.ZipFile(feature_zip, "r") as zf:
    zf.extractall(EXTRACT_DIR)

def find_output(name):
    matches = list(EXTRACT_DIR.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Missing required feature output: {name}")
    return matches[0]

panel = pd.read_csv(find_output("contract_month_features_v3.csv"))
pilot_raw = pd.read_csv(find_output("pilot_analysis_features_v3.csv"))
panel["month"] = pd.to_datetime(panel["month"])
panel["sales_month"] = pd.to_datetime(panel["sales_month"])
pilot_raw["contact_month"] = pd.to_datetime(pilot_raw["contact_month"])

if panel["month"].max() > VALIDATION_END:
    raise AssertionError("Q3 outcome leakage found")
print("Panel rows:", len(panel), "| outreach rows:", len(pilot_raw))


# %% tags=["pilot_core"] id="RDyAcqtjVSz2"
def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator not in (0, 0.0) and pd.notna(denominator) else np.nan


def cluster_bootstrap_ci(differences, clusters, n_boot=2000, seed=RANDOM_SEED):
    data = pd.DataFrame({"difference": np.asarray(differences, dtype=float), "cluster": np.asarray(clusters)})
    by_cluster = data.groupby("cluster", as_index=False)["difference"].mean()
    if len(by_cluster) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    values = by_cluster["difference"].to_numpy()
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def standardized_mean_difference(treated, control):
    treated = pd.to_numeric(pd.Series(treated), errors="coerce").dropna()
    control = pd.to_numeric(pd.Series(control), errors="coerce").dropna()
    if len(treated) < 2 or len(control) < 2:
        return np.nan
    pooled_sd = math.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    return float((treated.mean() - control.mean()) / pooled_sd) if pooled_sd else 0.0


def allocate_budget(effect_summary, monthly_budget=8000.0):
    required = {"programme", "att_cash_usd", "att_cash_ci_low", "cost_per_attempt_usd", "monthly_capacity_attempts"}
    missing = required - set(effect_summary.columns)
    if missing:
        raise ValueError(f"Missing allocation columns: {sorted(missing)}")
    candidates = effect_summary.copy()
    candidates["incremental_cash_per_cost_dollar"] = candidates["att_cash_usd"] / candidates["cost_per_attempt_usd"].replace(0, np.nan)
    candidates["scale_qualified"] = (
        candidates["att_cash_ci_low"].gt(0)
        & candidates["att_cash_usd"].gt(candidates["cost_per_attempt_usd"])
        & candidates["cost_per_attempt_usd"].gt(0)
    )
    candidates = candidates[candidates["scale_qualified"]].sort_values("incremental_cash_per_cost_dollar", ascending=False)
    remaining = float(monthly_budget)
    rows = []
    for item in candidates.itertuples():
        capacity_spend = float(item.monthly_capacity_attempts) * float(item.cost_per_attempt_usd)
        spend = min(remaining, capacity_spend)
        attempts = spend / float(item.cost_per_attempt_usd)
        gross = attempts * float(item.att_cash_usd)
        rows.append({
            "programme": item.programme, "allocation_usd": spend,
            "expected_attempts": attempts, "expected_incremental_cash_usd": gross,
            "expected_net_value_usd": gross - spend,
            "basis": "Scale-qualified; allocated by expected incremental cash per outreach dollar within demonstrated capacity",
        })
        remaining -= spend
        if remaining <= 1e-9:
            break
    if remaining > 1e-9:
        rows.append({
            "programme": "randomized_learning_budget", "allocation_usd": remaining,
            "expected_attempts": np.nan, "expected_incremental_cash_usd": np.nan,
            "expected_net_value_usd": np.nan,
            "basis": "No additional pilot capacity clears the evidence threshold; use remainder to generate credible evidence",
        })
    return pd.DataFrame(rows)


# %% colab={"base_uri": "https://localhost:8080/"} id="9wGOwuPuVSz2" outputId="b5386c2a-441a-4d86-a526-2e81b4d3794a"
def first_existing(frame, names, required=True):
    for name in names:
        if name in frame.columns:
            return name
    if required:
        raise KeyError(f"None of these columns exists: {names}")
    return None

region_col = first_existing(pilot_raw, ["region_outreach", "region", "region_contract"])
channel_col = first_existing(pilot_raw, ["channel"], required=False)
cost_col = first_existing(pilot_raw, ["cost_usd"], required=False)
attempts_col = first_existing(pilot_raw, ["attempts"], required=False)
reached_col = first_existing(pilot_raw, ["reached"], required=False)

pilot = pilot_raw.copy()
pilot["programme_region"] = pilot[region_col].astype("string").str.upper()
pilot["programme"] = np.where(
    pilot["programme_region"].eq("EAST"), "East preventative SMS",
    np.where(pilot["programme_region"].eq("WEST"), "West outbound calls", "Other")
)
pilot["cost_value"] = pd.to_numeric(pilot[cost_col], errors="coerce").fillna(0.0) if cost_col else 0.0
pilot["attempts_value"] = pd.to_numeric(pilot[attempts_col], errors="coerce").fillna(1.0) if attempts_col else 1.0
if reached_col:
    pilot["reached_value"] = pilot[reached_col].astype("string").str.lower().map({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}).fillna(0)
else:
    pilot["reached_value"] = np.nan

treatment = (
    pilot[pilot["programme_region"].isin(["EAST", "WEST"])]
    .groupby(["contractid", "contact_month", "programme_region", "programme"], as_index=False)
    .agg(cost_usd=("cost_value", "sum"), attempts=("attempts_value", "sum"), reached=("reached_value", "max"))
)
treatment["treated"] = 1
treatment_keys = treatment[["contractid", "contact_month", "treated"]].rename(columns={"contact_month": "month"})

target_lookup = panel[["contractid", "month", "target_payment_usd"]].copy()
target_lookup["contact_month"] = target_lookup["month"] - pd.offsets.MonthEnd(1)
target_lookup = target_lookup.rename(columns={"target_payment_usd": "next_month_cash_usd"})[["contractid", "contact_month", "next_month_cash_usd"]]

analysis = panel[
    panel["month"].between(PILOT_START, PRIMARY_CONTACT_END)
    & panel["contract_type"].astype("string").str.upper().eq("FINANCED")
    & panel["region"].astype("string").str.upper().isin(["EAST", "WEST"])
].copy()
analysis["programme_region"] = analysis["region"].astype("string").str.upper()
analysis = analysis.merge(treatment_keys, on=["contractid", "month"], how="left")
analysis["treated"] = analysis["treated"].fillna(0).astype(int)
analysis = analysis.merge(
    target_lookup, left_on=["contractid", "month"], right_on=["contractid", "contact_month"], how="left"
).drop(columns="contact_month")
analysis["next_month_paid_any"] = analysis["next_month_cash_usd"].gt(0).astype(int)

treatment_costs = treatment.rename(columns={"contact_month": "month"})[["contractid", "month", "cost_usd", "attempts", "reached", "programme"]]
analysis = analysis.merge(treatment_costs, on=["contractid", "month"], how="left")

# FIX: fillna() needs a scalar/dict/Series, not a raw ndarray from np.where.
# Using .map() on programme_region returns a Series with a matching index.
analysis["programme"] = analysis["programme"].fillna(
    analysis["programme_region"].map({"EAST": "East preventative SMS", "WEST": "West outbound calls"})
)

analysis[["cost_usd", "attempts"]] = analysis[["cost_usd", "attempts"]].fillna(0.0)

# Avoid carry-over contamination: a control cannot be a customer ever contacted in that regional pilot.
ever_treated = set(treatment["contractid"].astype(str))
analysis["eligible_control"] = analysis["treated"].eq(1) | ~analysis["contractid"].astype(str).isin(ever_treated)
analysis = analysis[analysis["eligible_control"] & analysis["next_month_cash_usd"].notna()].copy()

forbidden_predictors = {"target_payment_usd", "next_month_cash_usd", "next_month_paid_any", "cost_usd", "attempts", "reached", "treated"}
numeric_candidates = [
    "months_on_book", "price_usd", "perc_deposit", "daily_amount_usd", "tenor_length",
    "expected_cash_due_this_month_usd", "expected_cumulative_cash_before_month_usd",
    "actual_cumulative_cash_before_month_usd", "payment_lag_1m", "payment_lag_2m",
    "payment_lag_3m", "payment_trailing_3m_sum", "payment_trailing_6m_sum",
    "prior_paying_months", "zero_payment_months_prior_3", "zero_payment_months_prior_6",
    "months_since_last_positive_payment", "calls_cumulative_before_month",
    "tickets_cumulative_before_month", "calls_prior_1m", "calls_prior_3m",
    "tickets_prior_1m", "tickets_prior_3m",
]
NUMERIC = [c for c in numeric_candidates if c in analysis.columns and c not in forbidden_predictors]
CATEGORICAL = [c for c in ["month", "product", "payment_frequency", "scheduled_status"] if c in analysis.columns]
analysis["month"] = analysis["month"].dt.strftime("%Y-%m")

if forbidden_predictors.intersection(NUMERIC + CATEGORICAL):
    raise AssertionError("Outcome or treatment field entered propensity features")
print("Primary analysis rows:", len(analysis), "| treated:", int(analysis["treated"].sum()), "| controls:", int((analysis["treated"] == 0).sum()))


# %% tags=["pilot_model"] id="7c0ubUG1VSz3"
def evaluate_programme(data, region, neighbours=3):
    sub = data[data["programme_region"].eq(region)].copy()
    treated = sub[sub["treated"].eq(1)].copy()
    controls = sub[sub["treated"].eq(0)].copy()
    if len(treated) < 30 or len(controls) < 100:
        raise ValueError(f"Insufficient {region} sample: {len(treated)} treated, {len(controls)} controls")

    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric_pipe, NUMERIC), ("categorical", categorical_pipe, CATEGORICAL)], sparse_threshold=1.0)
    propensity = Pipeline([("prep", prep), ("model", LogisticRegression(C=0.2, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED))])
    features = NUMERIC + CATEGORICAL
    propensity.fit(sub[features], sub["treated"])
    sub["propensity"] = propensity.predict_proba(sub[features])[:, 1]
    treated = sub[sub["treated"].eq(1)].copy()
    controls = sub[sub["treated"].eq(0)].copy()

    lower = max(treated["propensity"].quantile(0.01), controls["propensity"].quantile(0.01))
    upper = min(treated["propensity"].quantile(0.99), controls["propensity"].quantile(0.99))
    treated = treated[treated["propensity"].between(lower, upper)].copy()
    controls = controls[controls["propensity"].between(lower, upper)].copy()
    if len(treated) < 20:
        raise ValueError(f"Poor common support for {region}: only {len(treated)} treated rows remain")

    matched_rows = []
    for month, treated_month in treated.groupby("month"):
        control_month = controls[controls["month"].eq(month)].copy()
        if control_month.empty:
            continue
        k = min(neighbours, len(control_month))
        matcher = NearestNeighbors(n_neighbors=k).fit(control_month[["propensity"]])
        distances, indices = matcher.kneighbors(treated_month[["propensity"]])
        for position, (_, treated_row) in enumerate(treated_month.iterrows()):
            matched = control_month.iloc[indices[position]]
            record = {
                "programme": treated_row["programme"], "region": region, "month": month,
                "contractid": treated_row["contractid"], "treated_cash_usd": treated_row["next_month_cash_usd"],
                "matched_control_cash_usd": matched["next_month_cash_usd"].mean(),
                "cash_difference_usd": treated_row["next_month_cash_usd"] - matched["next_month_cash_usd"].mean(),
                "treated_paid_any": treated_row["next_month_paid_any"],
                "matched_control_paid_any": matched["next_month_paid_any"].mean(),
                "paid_any_difference": treated_row["next_month_paid_any"] - matched["next_month_paid_any"].mean(),
                "treated_propensity": treated_row["propensity"], "matched_control_propensity": matched["propensity"].mean(),
                "prior_payment_treated": treated_row.get("payment_lag_1m", np.nan),
                "prior_payment_control": matched["payment_lag_1m"].mean() if "payment_lag_1m" in matched else np.nan,
            }
            for feature in NUMERIC:
                record[f"treated__{feature}"] = treated_row.get(feature, np.nan)
                record[f"control__{feature}"] = matched[feature].mean()
            matched_rows.append(record)
    matched_pairs = pd.DataFrame(matched_rows)
    if matched_pairs.empty:
        raise ValueError(f"No matched pairs for {region}")

    cash_low, cash_high = cluster_bootstrap_ci(matched_pairs["cash_difference_usd"], matched_pairs["contractid"])
    pay_low, pay_high = cluster_bootstrap_ci(matched_pairs["paid_any_difference"], matched_pairs["contractid"])
    treatment_region = treatment[treatment["programme_region"].eq(region)].copy()
    observed_primary = treatment_region[treatment_region["contact_month"].le(PRIMARY_CONTACT_END)]
    total_cost = float(observed_primary["cost_usd"].sum())
    unique_attempts = max(1, len(observed_primary))
    cost_per_attempt = total_cost / unique_attempts
    monthly_capacity = int(treatment_region.groupby("contact_month")["contractid"].nunique().max())
    att_cash = float(matched_pairs["cash_difference_usd"].mean())
    programme_name = str(matched_pairs["programme"].iloc[0])
    summary = {
        "programme": programme_name, "region": region, "treated_matched": len(matched_pairs),
        "candidate_controls": len(controls), "common_support_retention": len(matched_pairs) / max(1, len(sub[sub["treated"].eq(1)])),
        "treated_next_month_cash_usd": float(matched_pairs["treated_cash_usd"].mean()),
        "matched_control_next_month_cash_usd": float(matched_pairs["matched_control_cash_usd"].mean()),
        "att_cash_usd": att_cash, "att_cash_ci_low": cash_low, "att_cash_ci_high": cash_high,
        "att_payment_probability": float(matched_pairs["paid_any_difference"].mean()),
        "att_payment_probability_ci_low": pay_low, "att_payment_probability_ci_high": pay_high,
        "prior_payment_placebo_gap_usd": float((matched_pairs["prior_payment_treated"] - matched_pairs["prior_payment_control"]).mean()),
        "total_observed_cost_usd": total_cost, "cost_per_attempt_usd": cost_per_attempt,
        "incremental_cash_per_cost_dollar": safe_divide(att_cash, cost_per_attempt),
        "net_value_per_attempt_usd": att_cash - cost_per_attempt,
        "monthly_capacity_attempts": monthly_capacity,
    }

    balance_rows = []
    for feature in NUMERIC:
        before = standardized_mean_difference(sub.loc[sub["treated"].eq(1), feature], sub.loc[sub["treated"].eq(0), feature])
        after = standardized_mean_difference(matched_pairs[f"treated__{feature}"], matched_pairs[f"control__{feature}"])
        balance_rows.append({"programme": programme_name, "feature": feature, "smd_before": before, "smd_after": after})
    return summary, matched_pairs, pd.DataFrame(balance_rows)


# %% colab={"base_uri": "https://localhost:8080/", "height": 420} id="x3jWXbJKVSz3" outputId="abd966a9-91c6-41d8-b112-cf6654f68854"
summaries, pairs, balances, robustness_rows = [], [], [], []
for region in ["EAST", "WEST"]:
    for k in [1, 3, 5]:
        summary, matched, balance = evaluate_programme(analysis, region, neighbours=k)
        robustness_rows.append({
            "programme": summary["programme"], "neighbours": k,
            "att_cash_usd": summary["att_cash_usd"],
            "att_cash_ci_low": summary["att_cash_ci_low"], "att_cash_ci_high": summary["att_cash_ci_high"],
        })
        if k == 3:
            summaries.append(summary)
            pairs.append(matched)
            balances.append(balance)

pilot_effect_summary = pd.DataFrame(summaries)
pilot_matched_contract_months = pd.concat(pairs, ignore_index=True)
pilot_balance_diagnostics = pd.concat(balances, ignore_index=True)
pilot_matching_sensitivity = pd.DataFrame(robustness_rows)
pilot_effect_summary["scale_qualified"] = (
    pilot_effect_summary["att_cash_ci_low"].gt(0)
    & pilot_effect_summary["net_value_per_attempt_usd"].gt(0)
)
pilot_effect_summary["recommendation"] = np.where(
    pilot_effect_summary["scale_qualified"], "Evidence supports cautious scale",
    "Do not scale as proven; test further"
)

display(pilot_effect_summary.style.format({
    "treated_next_month_cash_usd": "${:,.2f}", "matched_control_next_month_cash_usd": "${:,.2f}",
    "att_cash_usd": "${:,.2f}", "att_cash_ci_low": "${:,.2f}", "att_cash_ci_high": "${:,.2f}",
    "att_payment_probability": "{:+.1%}", "cost_per_attempt_usd": "${:,.2f}",
    "incremental_cash_per_cost_dollar": "{:.2f}x", "net_value_per_attempt_usd": "${:,.2f}",
}))
display(pilot_matching_sensitivity)

# %% colab={"base_uri": "https://localhost:8080/", "height": 387} id="YuToImbOVSz3" outputId="026e9d41-4b28-4412-cfcd-04c761234bff"
outreach_budget_recommendation = allocate_budget(pilot_effect_summary, MONTHLY_BUDGET_USD)
allocated = float(outreach_budget_recommendation["allocation_usd"].sum())
if not np.isclose(allocated, MONTHLY_BUDGET_USD):
    raise AssertionError("Budget allocation does not sum to $8,000")

scalable = pilot_effect_summary[pilot_effect_summary["scale_qualified"]].sort_values("incremental_cash_per_cost_dollar", ascending=False)
if scalable.empty:
    pilot_choice = "Neither pilot has strong enough quasi-experimental evidence to scale as proven"
else:
    pilot_choice = str(scalable.iloc[0]["programme"])
expected_gross = outreach_budget_recommendation["expected_incremental_cash_usd"].sum(min_count=1)
expected_net = outreach_budget_recommendation["expected_net_value_usd"].sum(min_count=1)
part2_decision_summary = pd.DataFrame([
    {"question": "Which pilot should be scaled?", "answer": pilot_choice},
    {"question": "Primary repayment measure", "answer": "Next-calendar-month cash per attempted contract-month (intention-to-treat)"},
    {"question": "Why this measure?", "answer": "Cash matches the decision and next month avoids ambiguous same-month event ordering"},
    {"question": "Monthly outreach budget", "answer": "$8,000 allocated by credible incremental cash per dollar within observed capacity"},
    {"question": "Expected gross incremental monthly cash", "answer": "Not estimable for learning reserve" if pd.isna(expected_gross) else f"${expected_gross:,.0f}"},
    {"question": "Expected monthly net value after outreach cost", "answer": "Not estimable for learning reserve" if pd.isna(expected_net) else f"${expected_net:,.0f}"},
    {"question": "Causal confidence", "answer": "Moderate/low: matching addresses observed differences only; randomized test is still required"},
])
display(outreach_budget_recommendation.style.format({
    "allocation_usd": "${:,.0f}", "expected_attempts": "{:,.0f}",
    "expected_incremental_cash_usd": "${:,.0f}", "expected_net_value_usd": "${:,.0f}",
}))
display(part2_decision_summary)

# %% colab={"base_uri": "https://localhost:8080/", "height": 672} id="D01HKEo4VSz4" outputId="96eab1f4-2c43-4d6e-8d42-ee1e997682af"
product_issue_diagnostic = (
    analysis.groupby(["programme_region", "product"], dropna=False, as_index=False)
    .agg(contract_months=("contractid", "size"), next_month_cash_usd=("next_month_cash_usd", "mean"),
         prior_ticket_mean=("tickets_prior_3m", "mean") if "tickets_prior_3m" in analysis.columns else ("contractid", "size"))
)
product_issue_diagnostic["interpretation"] = "Exploratory association only; does not substantiate a product defect"

robustness_sign = pilot_matching_sensitivity.assign(sign=lambda x: np.sign(x["att_cash_usd"])).groupby("programme")["sign"].nunique()
qa = pd.DataFrame([
    {"check": "q3_outcomes_absent", "status": "PASS" if panel["month"].max() <= VALIDATION_END else "FAIL"},
    {"check": "same_month_payment_excluded_as_outcome", "status": "PASS" if "target_payment_usd" not in NUMERIC + CATEGORICAL else "FAIL"},
    {"check": "next_month_outcome_observed_for_primary_sample", "status": "PASS" if analysis["next_month_cash_usd"].notna().all() else "FAIL"},
    {"check": "budget_sums_to_8000", "status": "PASS" if np.isclose(allocated, MONTHLY_BUDGET_USD) else "FAIL"},
    {"check": "matching_sign_stable_across_1_3_5_neighbors", "status": "PASS" if robustness_sign.le(1).all() else "REVIEW"},
])
display(qa)
if qa["status"].eq("FAIL").any():
    raise AssertionError("Pilot-analysis QA failed")

pilot_effect_summary.to_csv(OUTPUT_DIR / "pilot_effect_summary.csv", index=False)
pilot_matched_contract_months.to_csv(OUTPUT_DIR / "pilot_matched_contract_months.csv", index=False)
pilot_balance_diagnostics.to_csv(OUTPUT_DIR / "pilot_balance_diagnostics.csv", index=False)
pilot_matching_sensitivity.to_csv(OUTPUT_DIR / "pilot_matching_sensitivity.csv", index=False)
outreach_budget_recommendation.to_csv(OUTPUT_DIR / "outreach_budget_recommendation.csv", index=False)
part2_decision_summary.to_csv(OUTPUT_DIR / "part2_decision_summary.csv", index=False)
product_issue_diagnostic.to_csv(OUTPUT_DIR / "product_issue_diagnostic.csv", index=False)
qa.to_csv(OUTPUT_DIR / "pilot_qa.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plot = pilot_effect_summary.copy()
x = np.arange(len(plot))
yerr = np.vstack([plot["att_cash_usd"] - plot["att_cash_ci_low"], plot["att_cash_ci_high"] - plot["att_cash_usd"]])
axes[0].errorbar(x, plot["att_cash_usd"], yerr=yerr, fmt="o", capsize=5)
axes[0].axhline(0, color="black", linewidth=1)
axes[0].set_xticks(x)
axes[0].set_xticklabels(plot["programme"], rotation=0)
axes[0].set_ylabel("Incremental next-month cash per attempt (USD)")
axes[0].set_title("Matched pilot effects with 95% intervals")

budget_plot = outreach_budget_recommendation.set_index("programme")["allocation_usd"]
budget_plot.plot(kind="bar", ax=axes[1], color="#2a6fbb")
axes[1].set_ylabel("Monthly allocation (USD)")
axes[1].set_title("Recommended October outreach budget")
axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pilot_decision_charts.png", dpi=180, bbox_inches="tight")
plt.show()

output_zip = Path("/content/dlight_part2_pilot_private_outputs.zip")
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in OUTPUT_DIR.glob("*"):
        if path.is_file():
            zf.write(path, arcname=path.name)
print("Created", output_zip)
try:
    from google.colab import files
    files.download(str(output_zip))
except ImportError:
    print("Download manually from", output_zip)

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

# %% [markdown] id="fba417d5"
# # Forecast, version 5 (FINAL): country collections forecast for July to September 2026
#
# It reads only the frozen feature outputs (`dlight_feature_engineering_outputs_v3.zip`, development data through 30 June 2026) and refuses to run if any row is dated later, so the sealed quarter can never enter the model.
#
# ## What this notebook does, in plain language
#
# The forecast is built up from the customer book rather than extended from a line on a chart. Every financed contract carries a contractual schedule (deposit up front, then a daily amount over the tenor, billed on the real number of days in each month). A **collection curve**, learned by region, product group and months on book on data up to 31 March 2026, says what share of billed cash is normally collected at each age. A **level factor** per region, read from the last three months, says how far each region is currently running above or below its own curve; that factor is what carries recent drift, cohort quality and any pilot effect that is still running into the forecast, and it is also the biggest lever in the scenarios. Contracts past their tenor keep contributing through a **recovery rate** on their unpaid balance, learned from the last six months. New sales from the plan enter through the same early-life curve using the recent sales mix. A small **reconciliation bridge** adds the cash Finance reports that cannot be attached to a clean contract-month, so the number ties to the accounting total.
#
# ## What changed relative to the two earlier tracks, and why
#
# | Change | Reason |
# |---|---|
# | Schedule billed on real calendar days, not an average month | The average-month proxy overstates February billing by about 8% and understates 31-day months by about 2%, which showed up as a fake February dip and a fake March rebound in the collection rate |
# | Curve keyed by region, product group and age, with shrinkage | East and West sit persistently below and above the country curve; the Gen 2 product repays differently, and in South it repays much worse |
# | Explicit level factor from the last three months | This is the honest treatment of "collections have been drifting": the curve gives the shape by age, the level says where the book is running now |
# | Region-month outliers flagged and excluded from curve fitting | North in March 2026 collected about 70% more than its curve predicts across every cohort and product; one month like that should not reset the curve |
# | Recovery rate learned on the last six months only | The pool of past-tenor contracts triples between June and September as the big early-2025 cohorts roll off; the recent rate (about 4% of balance a month) is lower than the all-history rate (about 6%) |
# | Two backtest variants | One with the curve fitted through the origin (what the earlier tracks did) and one with the curve cut off three months before the origin so the level window is out of sample; the second is the fair test of the recipe used for the final forecast |
# | Sensitivity levers correctly labelled | The earlier track's "existing-book efficiency" lever also scaled new sales; here each lever moves one thing |
# | Ridge kept as a challenger, never the headline | It predicts the existing book from behavioural lags; it agrees with the cohort build within a couple of per cent and that agreement is reported |
#
# **Run order:** run the setup cell, upload the v3 feature ZIP when prompted (or place it beside the notebook), then run everything else top to bottom. Download the output ZIP only after the QA table says PASS. Nothing here reads July to September 2026.

# %% colab={"base_uri": "https://localhost:8080/"} id="2c9081c8" outputId="bdda31fa-2171-4679-c480-a71ad3e3a553"
import subprocess, sys
for pkg in ["statsmodels", "scikit-learn"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

import io, json, hashlib, zipfile, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)

OUTPUT_DIR = Path("part1_final_outputs"); OUTPUT_DIR.mkdir(exist_ok=True)
RUN_CHALLENGER = True          # Ridge challenger on the existing book; set False to save a minute
BACKTEST_ORIGINS = pd.to_datetime(["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"])
print("setup ready")

# %% [markdown] id="3a914516"
# ## The core library
#
# All the arithmetic lives in one file written by the next cell and imported straight after, so the notebook and the tested code cannot drift apart. Each function has a plain-language docstring, and where a finance reader would want to rebuild a number in a spreadsheet the Google Sheets equivalent is noted as `SHEETS:`.

# %% colab={"base_uri": "https://localhost:8080/"} id="8b547eac" outputId="82f34ee0-3803-4d97-81e9-8bed49d07878"
# %%writefile final_forecast_core.py
"""
Final Part 1 (base forecast) core functions.
Everything here is written against the frozen v3 feature outputs
(contract_month_features_v3.csv, country_month_features_v3.csv,
payment_reconciliation_by_month_v3.csv). Nothing reads a month after 30 June 2026.

Design, in one paragraph: the forecast is a bottom-up build from the contract
book. Each financed contract carries a contractual schedule (deposit, then a
daily amount over the tenor, billed on the real number of calendar days in each
month). A collection curve, learned by region, product group and months on book
up to a fixed cut-off, says what share of billed cash is normally collected at
each age. A level factor per region, read from the last three months, says how
far each region is currently running above or below its own curve; this is what
carries drift, cohort quality and any pilot effect that is still running into the
forecast. Contracts past their tenor keep contributing through a recovery rate on
their unpaid balance, learned from the last six months. New sales from the plan
enter through the same early-life curve using the recent sales mix. A small
reconciliation bridge adds the cash Finance reports that cannot be attached to a
clean contract-month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANALYSIS_START = pd.Timestamp("2024-10-31")
ESTIMATION_END = pd.Timestamp("2026-03-31")   # curve shape cut-off (pre-pilot)
VALIDATION_END = pd.Timestamp("2026-06-30")   # development data ends here
FORECAST_MONTHS = pd.DatetimeIndex(pd.date_range("2026-07-31", "2026-09-30", freq="ME"))
SALES_PLAN = {pd.Timestamp("2026-07-31"): 3100,
              pd.Timestamp("2026-08-31"): 3200,
              pd.Timestamp("2026-09-30"): 3200}
LEVEL_WINDOW = 3          # months used for the region level factor
RECOVERY_WINDOW = 6       # months used for post-tenor recovery rates
MIX_WINDOW = 3            # months used for the new-sales mix
SHRINK_N = 200            # contracts; strength of shrinkage toward the pooled curve
OUTLIER_ABS_DEV = 0.30    # a region-month whose index is >30% from 1 is flagged
MAX_MOB = 36


# ---------------------------------------------------------------------------
# 0. Helpers
# ---------------------------------------------------------------------------
def month_end(ts):
    ts = pd.to_datetime(ts)
    return (ts + pd.offsets.MonthEnd(0))


def add_months(ts, k):
    return pd.to_datetime(ts) + pd.offsets.MonthEnd(k)


def mob_between(later, earlier):
    later = pd.to_datetime(later)
    earlier = pd.to_datetime(earlier)
    return (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)


def product_group(s):
    s = s.astype("string")
    return np.where(s.str.contains("Gen 2", na=False), "GEN2", "CORE")


def assert_no_sealed(*frames, cutoff=VALIDATION_END):
    for f in frames:
        for col in ("month", "sales_month", "pay_month", "contact_month"):
            if col in f.columns:
                mx = pd.to_datetime(f[col], errors="coerce").max()
                if pd.notna(mx) and mx > cutoff:
                    raise AssertionError(f"Sealed-period leak: {col} max is {mx.date()} > {cutoff.date()}")
    return True


# ---------------------------------------------------------------------------
# 1. Contractual schedule on real calendar days
# ---------------------------------------------------------------------------
def schedule_daycount(price, deposit, daily, tenor_days, sales_month, month):
    """Expected cash due in `month` for a financed contract, billing the daily
    amount on the actual number of days elapsed since the sale month end, capped
    at the tenor and at the price. Returns (due, within_schedule, elapsed_before).

    SHEETS: due = daily * (MIN(tenor, days_to_month_end) - MIN(tenor, days_to_prev_month_end)),
            then capped so deposit + cumulative instalments never exceed price.
    """
    sales_month = pd.DatetimeIndex(pd.to_datetime(np.asarray(sales_month))) + pd.offsets.MonthEnd(0)
    month = pd.DatetimeIndex(pd.to_datetime(np.asarray(month))) + pd.offsets.MonthEnd(0)
    days_through = np.clip((month - sales_month).days.to_numpy(dtype=float), 0, None)
    prev = month - pd.offsets.MonthEnd(1)
    days_before = np.clip((prev - sales_month).days.to_numpy(dtype=float), 0, None)
    tenor = np.nan_to_num(np.asarray(tenor_days, dtype=float), nan=0.0)
    daily = np.nan_to_num(np.asarray(daily, dtype=float), nan=0.0)
    price = np.nan_to_num(np.asarray(price, dtype=float), nan=0.0)
    deposit = np.nan_to_num(np.asarray(deposit, dtype=float), nan=0.0)
    el_through = np.minimum(tenor, days_through)
    el_before = np.minimum(tenor, days_before)
    mob = ((month.year - sales_month.year) * 12 + (month.month - sales_month.month)).to_numpy()
    cum_through = np.minimum(price, deposit + daily * el_through)
    cum_before = np.where(mob <= 0, 0.0, np.minimum(price, deposit + daily * el_before))
    due = np.clip(cum_through - cum_before, 0.0, None)
    within = el_before < tenor
    return due, within, mob


def rebuild_panel_schedule(panel):
    """Recompute the schedule fields of the v3 panel on real calendar days.
    The v3 panel used an average month (365.25/12 days), which overstates what is
    due in February by about 8% and understates 31-day months by about 2%.
    Keeps the original proxy columns for comparison."""
    p = panel.copy()
    fin = p["contract_type"].eq("FINANCED").to_numpy()
    due, within, mob = schedule_daycount(p["price_usd"], p["deposit_usd"], p["daily_amount_usd"],
                                         p["tenor_length"], p["sales_month"], p["month"])
    p["due_dc"] = np.where(fin, due, np.where(p["months_on_book"].eq(0), p["price_usd"], 0.0))
    p["within_dc"] = np.where(fin, within, p["months_on_book"].eq(0))
    p["post_tenor_dc"] = fin & (~p["within_dc"].astype(bool))
    p["pgroup"] = product_group(p["product"])
    p["balance_before"] = (p["price_usd"] - p["actual_cumulative_cash_before_month_usd"]).clip(lower=0)
    p["tenor_mob"] = np.ceil(p["tenor_length"].fillna(0) / (365.25 / 12)).astype(int)
    return p


# ---------------------------------------------------------------------------
# 2. Collection curves (shape), by region x product group x months on book
# ---------------------------------------------------------------------------
def _shrink(raw, n, prior, k=SHRINK_N):
    w = n / (n + k)
    return w * raw + (1 - w) * prior


def fit_curves(p, cutoff, outlier_keys=(), shrink_n=SHRINK_N):
    """Learn scheduled-period collection efficiency (paid / due) at each months-on-book,
    by region and product group, on data up to `cutoff`, with two levels of
    shrinkage: region x product -> region -> country. Also learns the sale-month
    efficiencies (financed deposit and cash full price) by region.

    outlier_keys: iterable of (region, month) tuples excluded from estimation.
    Returns a dict of lookup tables.
    """
    cutoff = pd.Timestamp(cutoff)
    h = p[(p["month"] <= cutoff) & (p["month"] >= ANALYSIS_START) & p["target_payment_usd"].notna()].copy()
    if outlier_keys:
        bad = pd.MultiIndex.from_tuples(list(outlier_keys))
        key = pd.MultiIndex.from_arrays([h["region"], h["month"]])
        h = h[~key.isin(bad)]
    sched = h[h["contract_type"].eq("FINANCED") & h["within_dc"].astype(bool)
              & (h["months_on_book"] >= 1) & (h["due_dc"] > 0)]

    country = sched.groupby("months_on_book").agg(paid=("target_payment_usd", "sum"), due=("due_dc", "sum"),
                                                  n=("contractid", "nunique"))
    country["eff"] = country["paid"] / country["due"]
    grid_mob = np.arange(1, MAX_MOB + 1)
    country_eff = country["eff"].reindex(grid_mob).ffill().bfill()

    region = sched.groupby(["region", "months_on_book"]).agg(paid=("target_payment_usd", "sum"),
                                                             due=("due_dc", "sum"), n=("contractid", "nunique"))
    region["raw"] = region["paid"] / region["due"]
    region["prior"] = country_eff.reindex(region.index.get_level_values(1)).to_numpy()
    region["eff"] = _shrink(region["raw"], region["n"], region["prior"], shrink_n)
    regions = sorted(p["region"].dropna().unique())
    reg_grid = pd.MultiIndex.from_product([regions, grid_mob], names=["region", "months_on_book"])
    region_eff = region["eff"].reindex(reg_grid)
    region_eff = region_eff.groupby(level=0).transform(lambda s: s.ffill().bfill())
    region_eff = region_eff.fillna(pd.Series(country_eff.reindex(reg_grid.get_level_values(1)).to_numpy(), index=reg_grid))

    rp = sched.groupby(["region", "pgroup", "months_on_book"]).agg(paid=("target_payment_usd", "sum"),
                                                                  due=("due_dc", "sum"), n=("contractid", "nunique"))
    rp["raw"] = rp["paid"] / rp["due"]
    rp["prior"] = region_eff.reindex(pd.MultiIndex.from_arrays([rp.index.get_level_values(0), rp.index.get_level_values(2)])).to_numpy()
    rp["eff"] = _shrink(rp["raw"], rp["n"], rp["prior"], shrink_n)
    rp_grid = pd.MultiIndex.from_product([regions, ["CORE", "GEN2"], grid_mob], names=["region", "pgroup", "months_on_book"])
    rp_eff = rp["eff"].reindex(rp_grid)
    # fill gaps from the region curve (a product group unseen at some age inherits the region shape)
    fill = region_eff.reindex(pd.MultiIndex.from_arrays([rp_grid.get_level_values(0), rp_grid.get_level_values(2)])).to_numpy()
    rp_eff = rp_eff.fillna(pd.Series(fill, index=rp_grid))

    # sale-month efficiencies
    dep = h[h["contract_type"].eq("FINANCED") & (h["months_on_book"] == 0) & (h["deposit_usd"] > 0)]
    dep_eff = (dep.groupby("region")["target_payment_usd"].sum() / dep.groupby("region")["deposit_usd"].sum())
    dep_country = dep["target_payment_usd"].sum() / dep["deposit_usd"].sum()
    cash = h[h["contract_type"].eq("CASH") & (h["months_on_book"] == 0) & (h["price_usd"] > 0)]
    cash_eff = (cash.groupby("region")["target_payment_usd"].sum() / cash.groupby("region")["price_usd"].sum())
    cash_country = cash["target_payment_usd"].sum() / cash["price_usd"].sum()
    return {"cutoff": cutoff, "country": country_eff, "region": region_eff, "region_product": rp_eff,
            "deposit_eff": dep_eff.reindex(regions).fillna(dep_country), "cash_eff": cash_eff.reindex(regions).fillna(cash_country),
            "n_rows": int(len(sched))}


def lookup_eff(curves, region, pgroup, mob):
    """Vectorised efficiency lookup with fall-back region -> country."""
    mob = np.clip(np.asarray(mob, dtype=int), 1, MAX_MOB)
    idx = pd.MultiIndex.from_arrays([np.asarray(region), np.asarray(pgroup), mob])
    v = curves["region_product"].reindex(idx).to_numpy()
    miss = np.isnan(v)
    if miss.any():
        idx2 = pd.MultiIndex.from_arrays([np.asarray(region)[miss], mob[miss]])
        v2 = curves["region"].reindex(idx2).to_numpy()
        m2 = np.isnan(v2)
        if m2.any():
            v2[m2] = curves["country"].reindex(mob[miss][m2]).to_numpy()
        v[miss] = v2
    return v


# ---------------------------------------------------------------------------
# 3. Calendar-level index and region level factors
# ---------------------------------------------------------------------------
def calendar_index(p, curves, through):
    """For every region-month up to `through`: actual scheduled cash divided by
    what the curve predicts for the same contract-months. 1.0 = on curve."""
    h = p[(p["month"] >= ANALYSIS_START) & (p["month"] <= pd.Timestamp(through)) & p["target_payment_usd"].notna()
          & p["contract_type"].eq("FINANCED") & p["within_dc"].astype(bool) & (p["months_on_book"] >= 1) & (p["due_dc"] > 0)].copy()
    h["pred"] = h["due_dc"] * lookup_eff(curves, h["region"], h["pgroup"], h["months_on_book"])
    g = h.groupby(["region", "month"]).agg(actual=("target_payment_usd", "sum"), pred=("pred", "sum"),
                                           n=("contractid", "nunique")).reset_index()
    g["index"] = g["actual"] / g["pred"]
    c = h.groupby("month").agg(actual=("target_payment_usd", "sum"), pred=("pred", "sum")).reset_index()
    c["index"] = c["actual"] / c["pred"]
    return g, c


def flag_outlier_region_months(index_table, threshold=OUTLIER_ABS_DEV):
    out = index_table[(index_table["index"] - 1).abs() > threshold]
    return [(r.region, pd.Timestamp(r.month)) for r in out.itertuples()]


def region_levels(index_table, origin, window=LEVEL_WINDOW, outlier_keys=()):
    """Median index over the last `window` months per region, ignoring flagged outliers."""
    origin = pd.Timestamp(origin)
    start = add_months(origin, -(window - 1))
    t = index_table[(index_table["month"] >= start) & (index_table["month"] <= origin)].copy()
    if outlier_keys:
        bad = pd.MultiIndex.from_tuples(list(outlier_keys))
        t = t[~pd.MultiIndex.from_arrays([t["region"], t["month"]]).isin(bad)]
    lv = t.groupby("region")["index"].median()
    return lv


# ---------------------------------------------------------------------------
# 4. Post-tenor recovery rates
# ---------------------------------------------------------------------------
def fit_recovery(p, origin, window=RECOVERY_WINDOW, k_usd=20000.0):
    """Monthly cash as a share of the unpaid balance, by months past tenor, learned
    on the last `window` months up to origin, shrunk toward the pooled recent rate."""
    origin = pd.Timestamp(origin)
    start = add_months(origin, -(window - 1))
    r = p[p["post_tenor_dc"] & (p["month"] >= start) & (p["month"] <= origin) & p["target_payment_usd"].notna()
          & (p["balance_before"] > 0)].copy()
    r["mpt"] = (r["months_on_book"] - r["tenor_mob"]).clip(0, 12)
    pooled = r["target_payment_usd"].sum() / r["balance_before"].sum() if r["balance_before"].sum() else 0.0
    g = r.groupby("mpt").agg(paid=("target_payment_usd", "sum"), bal=("balance_before", "sum"), rows=("contractid", "size"))
    g["rate"] = (g["paid"] + k_usd * pooled) / (g["bal"] + k_usd)
    rate = g["rate"].reindex(range(0, 13)).ffill().bfill().fillna(pooled).clip(0, 0.5)
    return {"rate_by_mpt": rate, "pooled": float(pooled), "rows": int(len(r)), "table": g}


# ---------------------------------------------------------------------------
# 5. New-sales mix
# ---------------------------------------------------------------------------
def recent_mix(p, origin, window=MIX_WINDOW):
    origin = pd.Timestamp(origin)
    start = add_months(origin, -(window - 1))
    first = p[(p["months_on_book"] == 0) & (p["sales_month"] >= start) & (p["sales_month"] <= origin)].drop_duplicates("contractid")
    mix = first.groupby(["region", "contract_type", "pgroup"], as_index=False).agg(
        contracts=("contractid", "nunique"), price=("price_usd", "mean"), deposit=("deposit_usd", "mean"),
        daily=("daily_amount_usd", "mean"), tenor=("tenor_length", "mean"))
    mix["share"] = mix["contracts"] / mix["contracts"].sum()
    return mix


def apply_cash_share(mix, cash_share):
    if cash_share is None:
        return mix
    m = mix.copy()
    cur = m.loc[m["contract_type"].eq("CASH"), "share"].sum()
    if 0 < cur < 1:
        m.loc[m["contract_type"].eq("CASH"), "share"] *= cash_share / cur
        m.loc[m["contract_type"].eq("FINANCED"), "share"] *= (1 - cash_share) / (1 - cur)
    return m


# ---------------------------------------------------------------------------
# 6. The forecast
# ---------------------------------------------------------------------------
def forecast(p, country_month, origin, months, sales_units, curve_cutoff=None,
             level_factor=1.0, level_override=None, recovery_factor=1.0, attainment=1.0,
             cash_share=None, bridge_override=None, outlier_keys=None, curves=None,
             levels=None, recovery=None, mix=None, return_detail=False):
    """Bottom-up forecast from `origin` for `months`.

    level_factor: multiplies every region's level factor (scenario lever).
    level_override: dict region -> level factor replacing the learned ones.
    """
    origin = pd.Timestamp(origin)
    months = pd.DatetimeIndex(pd.to_datetime(months))
    cutoff = pd.Timestamp(curve_cutoff) if curve_cutoff is not None else origin
    if curves is None:
        # first pass without outlier removal to find outliers, then refit
        c0 = fit_curves(p, cutoff)
        idx0, _ = calendar_index(p, c0, origin)
        if outlier_keys is None:
            outlier_keys = flag_outlier_region_months(idx0)
        curves = fit_curves(p, cutoff, outlier_keys=outlier_keys)
    if outlier_keys is None:
        outlier_keys = []
    idx_tbl, country_idx = calendar_index(p, curves, origin)
    if levels is None:
        levels = region_levels(idx_tbl, origin, outlier_keys=outlier_keys)
    lv = levels.copy() * level_factor
    if level_override:
        for k, v in level_override.items():
            lv[k] = v
    if recovery is None:
        recovery = fit_recovery(p, origin)
    if mix is None:
        mix = recent_mix(p, origin)
    mix = apply_cash_share(mix, cash_share)

    # --- existing financed book snapshot at origin
    snap = p[(p["month"] == origin) & p["contract_type"].eq("FINANCED")].drop_duplicates("contractid").copy()
    snap["balance"] = (snap["price_usd"] - snap["actual_cumulative_cash_through_month_usd"]).clip(lower=0)
    snap = snap[snap["balance"] > 0]
    reg = snap["region"].to_numpy(); pg = snap["pgroup"].to_numpy()
    lv_vec = pd.Series(reg).map(lv).fillna(float(lv.mean())).to_numpy()
    bal = snap["balance"].to_numpy(dtype=float)
    rate_by_mpt = recovery["rate_by_mpt"]

    rows, region_rows = [], []
    for m in months:
        due, within, mob = schedule_daycount(snap["price_usd"], snap["deposit_usd"], snap["daily_amount_usd"],
                                             snap["tenor_length"], snap["sales_month"], pd.Series([m] * len(snap)))
        eff = lookup_eff(curves, reg, pg, mob)
        pred_s = np.where(within, np.minimum(bal, due * eff * lv_vec), 0.0)
        mpt = np.clip(mob - snap["tenor_mob"].to_numpy(), 0, 12)
        rrate = rate_by_mpt.reindex(mpt).to_numpy() * recovery_factor
        pred_r = np.where(~within, np.minimum(bal, bal * rrate), 0.0)
        bal = np.clip(bal - pred_s - pred_r, 0.0, None)
        # new sales
        new_fin = new_cash = 0.0
        new_by_region = {}
        for sm, units in sales_units.items():
            sm = pd.Timestamp(sm)
            if sm > m:
                continue
            k = int(mob_between(pd.Series([m]), pd.Series([sm])).iloc[0])
            u = float(units) * attainment
            for seg in mix.itertuples():
                cnt = u * float(seg.share)
                if seg.contract_type == "CASH":
                    v = cnt * float(seg.price) * float(curves["cash_eff"].get(seg.region, curves["cash_eff"].mean())) if k == 0 else 0.0
                    new_cash += v
                else:
                    if k == 0:
                        v = cnt * float(seg.deposit) * float(curves["deposit_eff"].get(seg.region, curves["deposit_eff"].mean()))
                    else:
                        d, w, _ = schedule_daycount([seg.price], [seg.deposit], [seg.daily], [seg.tenor], [sm], [m])
                        e = lookup_eff(curves, [seg.region], [seg.pgroup], [k])[0]
                        v = cnt * float(d[0]) * float(e) * float(lv.get(seg.region, lv.mean()))
                    new_fin += v
                new_by_region[seg.region] = new_by_region.get(seg.region, 0.0) + v
        # bridge
        if bridge_override is None:
            hist = country_month[(country_month["month"] <= origin)].tail(6)["source_minus_attributable_cash_usd"]
            bridge = float(hist.median()) if len(hist) else 0.0
        else:
            bridge = float(bridge_override)
        sched_tot, rec_tot = float(pred_s.sum()), float(pred_r.sum())
        rows.append({"month": m, "existing_scheduled": sched_tot, "post_tenor_recovery": rec_tot,
                     "new_sales_financed": new_fin, "new_sales_cash": new_cash, "bridge": bridge,
                     "total": sched_tot + rec_tot + new_fin + new_cash + bridge})
        sr = pd.Series(pred_s + pred_r).groupby(reg).sum()
        for r_ in sorted(set(sr.index) | set(new_by_region)):
            region_rows.append({"month": m, "region": r_, "existing": float(sr.get(r_, 0.0)),
                                "new_sales": float(new_by_region.get(r_, 0.0)),
                                "total_ex_bridge": float(sr.get(r_, 0.0)) + float(new_by_region.get(r_, 0.0))})
    out = pd.DataFrame(rows)
    detail = {"curves": curves, "levels": lv, "recovery": recovery, "mix": mix, "outliers": outlier_keys,
              "index_table": idx_tbl, "country_index": country_idx, "region_forecast": pd.DataFrame(region_rows)}
    return (out, detail) if return_detail else out


# ---------------------------------------------------------------------------
# 7. Benchmarks and scoring
# ---------------------------------------------------------------------------
def naive_forecast(country_month, origin, months, lookback=3):
    past = country_month.loc[country_month["month"] <= pd.Timestamp(origin), "source_reported_cash_usd"].tail(lookback)
    return pd.Series(float(past.mean()), index=pd.DatetimeIndex(months))


def holt_forecast(country_month, origin, months):
    from statsmodels.tsa.holtwinters import Holt
    y = country_month.loc[country_month["month"] <= pd.Timestamp(origin), "source_reported_cash_usd"].astype(float).to_numpy()
    fit = Holt(y, damped_trend=True, initialization_method="estimated").fit(optimized=True)
    h = len(months)
    return pd.Series(np.asarray(fit.forecast(h), dtype=float), index=pd.DatetimeIndex(months))


def score(actual, fc):
    a = np.asarray(actual, float); f = np.asarray(fc, float); e = f - a
    return {"MAE": float(np.mean(np.abs(e))), "WAPE": float(np.abs(e).sum() / np.abs(a).sum()),
            "Bias": float(e.sum() / np.abs(a).sum())}


def actual_units(p, months):
    first = p[p["months_on_book"] == 0].drop_duplicates("contractid")
    cnt = first.groupby("sales_month")["contractid"].nunique()
    return {pd.Timestamp(m): int(cnt.get(pd.Timestamp(m), 0)) for m in months}


# ---------------------------------------------------------------------------
# 8. Scenario helpers
# ---------------------------------------------------------------------------
def country_index_clean(p, curves, through, outlier_keys=()):
    """Country calendar index with flagged region-months removed."""
    idx, _ = calendar_index(p, curves, through)
    if outlier_keys:
        bad = pd.MultiIndex.from_tuples(list(outlier_keys))
        idx = idx[~pd.MultiIndex.from_arrays([idx["region"], idx["month"]]).isin(bad)]
    c = idx.groupby("month").agg(actual=("actual", "sum"), pred=("pred", "sum"))
    c["index"] = c["actual"] / c["pred"]
    return c


def calendar_shock_sigma(country_idx, months=12):
    """Standard deviation of month-to-month changes in the country index over the
    last `months` months: how much the whole book's collection level moves from
    one month to the next, the right ruler for a scenario band because every
    contract lives through the same calendar."""
    s = country_idx["index"].tail(months + 1)
    return float(np.nanstd(np.diff(s.to_numpy()), ddof=1)) if len(s) > 2 else 0.03


def pilot_stop_levels(index_table, origin, pilot_regions=("East", "West"), control_regions=("North", "South"),
                      pre_months=3, post_months=3, outlier_keys=()):
    """Level factors East/West would have if the pilots stopped: each pilot region's
    pre-pilot level moved by the average change the control regions showed."""
    origin = pd.Timestamp(origin)
    post_start = add_months(origin, -(post_months - 1))
    pre_end = add_months(post_start, -1)
    pre_start = add_months(pre_end, -(pre_months - 1))
    t = index_table.copy()
    if outlier_keys:
        bad = pd.MultiIndex.from_tuples(list(outlier_keys))
        t = t[~pd.MultiIndex.from_arrays([t["region"], t["month"]]).isin(bad)]
    pre = t[(t["month"] >= pre_start) & (t["month"] <= pre_end)].groupby("region")["index"].median()
    post = t[(t["month"] >= post_start) & (t["month"] <= origin)].groupby("region")["index"].median()
    ctrl_change = float(np.mean([post[c] / pre[c] for c in control_regions]))
    return {r: float(pre[r] * ctrl_change) for r in pilot_regions}, pre, post, ctrl_change


def cash_share_history(p, origin, months=12):
    origin = pd.Timestamp(origin)
    first = p[(p["months_on_book"] == 0) & (p["sales_month"] <= origin) & (p["sales_month"] > add_months(origin, -months))].drop_duplicates("contractid")
    return first.groupby("sales_month")["contract_type"].apply(lambda s: float(s.eq("CASH").mean()))


# ---------------------------------------------------------------------------
# 9. Ridge challenger for the existing financed book (direct multi-horizon)
# ---------------------------------------------------------------------------
RIDGE_NUM = ["months_on_book", "price_usd", "perc_deposit", "daily_amount_usd", "tenor_length",
             "expected_cash_due_this_month_usd", "actual_cumulative_cash_through_month_usd",
             "payment_lag_1m", "payment_lag_2m", "payment_lag_3m", "payment_trailing_3m_sum",
             "payment_trailing_6m_sum", "prior_paying_months", "zero_payment_months_prior_3",
             "zero_payment_months_prior_6", "months_since_last_positive_payment",
             "calls_cumulative_before_month", "tickets_cumulative_before_month"]
RIDGE_CAT = ["region", "product", "scheduled_status"]


def ridge_existing_book(p, origin, months, target_cutoff, alpha=10.0):
    """One Ridge model per horizon, trained only on origin months whose target month
    is at or before `target_cutoff`, predicting the next-h-month cash of every
    financed contract in the origin snapshot. Behavioural lags do the work here;
    it is a challenger, not the headline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    origin = pd.Timestamp(origin); target_cutoff = pd.Timestamp(target_cutoff)
    f = p[p["contract_type"].eq("FINANCED")]
    tgt = f[["contractid", "month", "target_payment_usd"]].rename(columns={"month": "tm", "target_payment_usd": "y"})
    pred_rows = f[f["month"].eq(origin)].drop_duplicates("contractid")
    out = []
    for h, tm in enumerate(pd.DatetimeIndex(pd.to_datetime(months)), start=1):
        latest_origin = target_cutoff - pd.offsets.MonthEnd(h)
        tr = f[f["month"] <= latest_origin].copy()
        tr["tm"] = tr["month"] + pd.offsets.MonthEnd(h)
        tr = tr.merge(tgt, on=["contractid", "tm"], how="inner")
        tr = tr[tr["y"].notna()]
        pipe = Pipeline([
            ("prep", ColumnTransformer([
                ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), RIDGE_NUM),
                ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))]), RIDGE_CAT)])),
            ("m", Ridge(alpha=alpha, solver="lsqr"))])
        pipe.fit(tr[RIDGE_NUM + RIDGE_CAT], tr["y"])
        pr = np.clip(pipe.predict(pred_rows[RIDGE_NUM + RIDGE_CAT]), 0, None)
        out.append({"month": tm, "ridge_existing": float(pr.sum()), "train_rows": int(len(tr))})
    return pd.DataFrame(out)



# %% colab={"base_uri": "https://localhost:8080/"} id="702dba07" outputId="598663e7-1507-4303-ba7f-818729468097"
import importlib, final_forecast_core as C
importlib.reload(C)
from final_forecast_core import (ANALYSIS_START, ESTIMATION_END, VALIDATION_END, FORECAST_MONTHS, SALES_PLAN)
print("curve cut-off", ESTIMATION_END.date(), "| forecast origin", VALIDATION_END.date(),
      "| forecast months", [m.strftime("%b %Y") for m in FORECAST_MONTHS])

# %% [markdown] id="3d35a36c"
# ## Load the frozen feature outputs and stop on leakage
#
# Only the columns the model needs are read from the 400 MB panel, which keeps memory low on the free Colab tier. The guards below stop the run if any month after June 2026 appears, if a contract-month key is duplicated, or if any outreach treatment field has crept into the base panel.

# %% colab={"base_uri": "https://localhost:8080/", "height": 281} id="9285720a" outputId="6b0c308b-7806-41f8-a054-57f984b2f22b"
try:
    from google.colab import files
    up = files.upload()
    zip_name = [n for n in up if n.lower().endswith(".zip")][0]
    ZIP_BYTES = io.BytesIO(up[zip_name])
except ImportError:
    candidates = [Path(p) for p in ["dlight_feature_engineering_outputs_v3.zip",
                                    "/mnt/user-data/uploads/dlight_feature_engineering_outputs_v3.zip"]]
    hit = [c for c in candidates if c.exists()]
    if not hit:
        raise FileNotFoundError("Place dlight_feature_engineering_outputs_v3.zip beside this notebook")
    ZIP_BYTES = hit[0]

PANEL_COLS = ["contractid", "sales_month", "region", "contract_type", "product", "payment_frequency", "price_usd",
              "perc_deposit", "daily_amount_usd", "tenor_length", "month", "months_on_book", "deposit_usd",
              "expected_cash_due_this_month_usd", "within_scheduled_tenor_proxy", "post_tenor_recovery_proxy",
              "scheduled_status", "target_payment_usd", "current_payment_conflict",
              "actual_cumulative_cash_through_month_usd", "actual_cumulative_cash_before_month_usd",
              "payment_history_reliable", "deposit_scale_corrected", "tickets_prior_3m", "calls_prior_3m",
              "tickets_reason_battery_fault_prior_3m", "tickets_reason_charging_issue_prior_3m"] + \
             ["payment_lag_1m", "payment_lag_2m", "payment_lag_3m", "payment_trailing_3m_sum", "payment_trailing_6m_sum",
              "prior_paying_months", "zero_payment_months_prior_3", "zero_payment_months_prior_6",
              "months_since_last_positive_payment", "calls_cumulative_before_month", "tickets_cumulative_before_month"]

t0 = time.time()
with zipfile.ZipFile(ZIP_BYTES, "r") as zf:
    names = zf.namelist()
    def _find(key):
        m = [n for n in names if key in n]
        if not m: raise FileNotFoundError(key)
        return m[0]
    panel = pd.read_csv(zf.open(_find("contract_month_features_v3")), usecols=PANEL_COLS,
                        parse_dates=["sales_month", "month"])
    country_month = pd.read_csv(zf.open(_find("country_month_features_v3")), parse_dates=["month"])
    region_month = pd.read_csv(zf.open(_find("region_month_features_v3")), parse_dates=["month"])
    reconciliation = pd.read_csv(zf.open(_find("payment_reconciliation_by_month_v3")), parse_dates=["month"])
    feature_metrics = json.load(zf.open(_find("feature_metrics.json")))
for c in ["within_scheduled_tenor_proxy", "post_tenor_recovery_proxy", "current_payment_conflict",
          "payment_history_reliable", "deposit_scale_corrected"]:
    panel[c] = panel[c].astype(str).str.lower().isin(["true", "1"])
print("loaded in", round(time.time() - t0, 1), "s; panel", panel.shape)

C.assert_no_sealed(panel, country_month, region_month, reconciliation)
qa_input = pd.DataFrame([
    {"check": "no_rows_after_jun_2026", "status": "PASS" if panel["month"].max() <= VALIDATION_END else "FAIL",
     "detail": f"max month {panel['month'].max().date()}"},
    {"check": "no_sales_after_jun_2026", "status": "PASS" if panel["sales_month"].max() <= VALIDATION_END else "FAIL",
     "detail": f"max sale {panel['sales_month'].max().date()}"},
    {"check": "contract_month_key_unique", "status": "PASS" if not panel.duplicated(["contractid", "month"]).any() else "FAIL", "detail": ""},
    {"check": "no_outreach_fields_in_base_panel",
     "status": "PASS" if not any(c in panel.columns for c in ["channel", "attempts", "reached", "cost_usd", "contact_month"]) else "FAIL", "detail": ""},
    {"check": "feature_metrics_schema", "status": "PASS" if feature_metrics.get("schema_version") == "dlight_feature_metrics_v3" else "FAIL",
     "detail": str(feature_metrics.get("schema_version"))},
])
display(qa_input)
if qa_input["status"].ne("PASS").any():
    raise AssertionError("Input QA failed; do not continue")

# %% [markdown] id="c1efa22b"
# ## Rebuild the contractual schedule on real calendar days
#
# The v3 feature panel billed instalments on an average month of 365.25/12 days. That is close on average but wrong month by month: it overstates what is due in February by about 8% and understates 31-day months by about 2%, which produced a February "dip" and a March "rebound" in the collection rate that were not real. The schedule is rebuilt here on actual day counts; the proxy columns are kept for comparison.
#
# `SHEETS: due_this_month = daily * (MIN(tenor, days_from_sale_to_month_end) - MIN(tenor, days_from_sale_to_prev_month_end))`

# %% colab={"base_uri": "https://localhost:8080/", "height": 381} id="f5bb0571" outputId="23c68b7d-b5ef-47cd-ef09-e108f2b2eb62"
t0 = time.time()
panel = C.rebuild_panel_schedule(panel)
schedule_check = panel[panel["contract_type"].eq("FINANCED") & (panel["months_on_book"] >= 1)].groupby("month").agg(
    proxy_due_usd=("expected_cash_due_this_month_usd", "sum"), daycount_due_usd=("due_dc", "sum"))
schedule_check["daycount_over_proxy"] = schedule_check["daycount_due_usd"] / schedule_check["proxy_due_usd"]
schedule_check["days_in_month"] = schedule_check.index.days_in_month
display(schedule_check.tail(9).round(3))
print("rebuilt in", round(time.time() - t0, 1), "s")

# %% [markdown] id="280584ae"
# ## Look before modelling: what the cash has been doing, and what is odd in the data
#
# Four things matter for the forecast and are worth a slide each: the sales dip in May and June, the composition of monthly cash, the calendar-level index at fixed age by region (which is where drift and the pilots show up), and the tenor roll-off that reshapes the third quarter.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="05f00ade" outputId="aa2d0b12-25a2-4bdc-878a-5a8c2a31dfd0"
p = panel
p["bucket"] = np.select([p["months_on_book"].eq(0), p["post_tenor_dc"]], ["new_sale_month", "post_tenor_recovery"], "existing_scheduled")
cash_mix = p.pivot_table(index="month", columns="bucket", values="target_payment_usd", aggfunc="sum").fillna(0)
cash_mix["panel_total"] = cash_mix.sum(axis=1)
cash_mix = cash_mix.join(country_month.set_index("month")[["source_reported_cash_usd"]])
sched = p[p["bucket"].eq("existing_scheduled")].groupby("month").agg(due=("due_dc", "sum"), paid=("target_payment_usd", "sum"), mean_mob=("months_on_book", "mean"))
cash_mix["existing_rate_paid_over_due"] = sched["paid"] / sched["due"]
cash_mix["mean_months_on_book"] = sched["mean_mob"]
display(cash_mix.round(2).tail(12))
cash_mix.to_csv(OUTPUT_DIR / "monthly_cash_decomposition.csv")

first = p[p["months_on_book"].eq(0)].drop_duplicates("contractid")
sales = first.groupby([first["sales_month"].dt.to_period("M"), "region"]).size().unstack(fill_value=0)
sales["total"] = sales.sum(axis=1)
sales["cash_share"] = first.groupby(first["sales_month"].dt.to_period("M"))["contract_type"].apply(lambda s: s.eq("CASH").mean()).round(3)
display(sales.tail(9))
sales.to_csv(OUTPUT_DIR / "sales_units_by_month_region.csv")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
cm_plot = cash_mix[["existing_scheduled", "post_tenor_recovery", "new_sale_month"]].copy(); cm_plot.index = [d.strftime("%b %y") for d in cm_plot.index]
cm_plot.plot(kind="bar", stacked=True, ax=ax[0], width=0.8); ax[0].tick_params(axis="x", rotation=90); ax[0].set_title("Monthly cash by source (panel)"); ax[0].set_ylabel("USD")
sales_plot = sales[["East", "North", "South", "West"]].copy(); sales_plot.index = sales_plot.index.to_timestamp(); sales_plot.plot(ax=ax[1], marker="o"); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %y")); ax[1].set_title("Units sold per month by region"); ax[1].set_ylabel("units")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "fig_cash_mix_and_sales.png", dpi=160, bbox_inches="tight"); plt.show()

# %% [markdown] id="f4f29563"
# ### The calendar-level index: is repayment drifting, and where?
#
# The index for a region-month is the scheduled cash actually collected divided by what the region's own curve (learned through March 2026, at fixed age) predicts for those same contract-months. Before April every region sits close to 1.0 by construction. From April onward the index is the cleanest single read of what changed: East and West move up (the pilots), North holds, South slips. A region-month whose index is more than 30% away from 1.0 is flagged as an outlier and kept out of curve fitting; North in March 2026 is the one such month.

# %% colab={"base_uri": "https://localhost:8080/", "height": 865} id="85cdf372" outputId="feade1c5-fe13-4110-9ef0-a4c574e7b966"
curves0 = C.fit_curves(panel, ESTIMATION_END)
index0, _ = C.calendar_index(panel, curves0, VALIDATION_END)
OUTLIERS = C.flag_outlier_region_months(index0)
print("flagged region-months (excluded from curve fitting):", [(r, m.date().isoformat()) for r, m in OUTLIERS])
curves = C.fit_curves(panel, ESTIMATION_END, outlier_keys=OUTLIERS)
index_table, country_index = C.calendar_index(panel, curves, VALIDATION_END)
country_index_clean = C.country_index_clean(panel, curves, VALIDATION_END, OUTLIERS)
index_pivot = index_table.pivot(index="month", columns="region", values="index")
index_pivot["country_ex_outliers"] = country_index_clean["index"]
display(index_pivot.round(3).tail(12))
index_pivot.to_csv(OUTPUT_DIR / "calendar_index_by_region.csv")

fig, ax = plt.subplots(figsize=(11, 4))
xs = np.arange(len(index_pivot)); labs = [d.strftime("%b %y") for d in index_pivot.index]
for r in ["East", "North", "South", "West"]:
    ax.plot(xs, index_pivot[r], marker="o", label=r)
ax.axhline(1.0, color="black", lw=0.8); ax.axvline(list(index_pivot.index).index(pd.Timestamp("2026-04-30")) - 0.5, color="grey", ls="--", lw=0.8)
ax.set_ylim(0.85, 1.25); ax.set_title("Scheduled cash collected vs the region's own curve at fixed age (1.0 = on curve); pilots start Apr 26")
ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=45); ax.legend(ncol=4)
ax.annotate("North Mar 26: 1.70, off scale, flagged", xy=(list(index_pivot.index).index(pd.Timestamp("2026-03-31")), 1.23), fontsize=8, ha="center")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "fig_calendar_index_by_region.png", dpi=160, bbox_inches="tight"); plt.show()

# %% [markdown] id="edcdf2e7"
# ### The North March 2026 spike, traced
#
# North collected about 70% more than its curve predicts in March 2026. The check below shows the same doubling in every sales cohort and every product, with a normal number of payers but a much larger average payment, and the excess does not net off against February. That pattern (uniform across cohorts and products, one region, one month) is either a one-off regional collections drive or a posting anomaly; either way it is not repeatable behaviour, so it is kept out of the curve and out of the level factors. It is a question to bring to the room, not to resolve here.

# %% colab={"base_uri": "https://localhost:8080/", "height": 458} id="400086f0" outputId="8abdfef3-9d8a-4a45-cca5-9c2be193f738"
north = panel[panel["region"].eq("North") & panel["contract_type"].eq("FINANCED") & (panel["months_on_book"] >= 1)
              & panel["month"].between("2026-01-31", "2026-04-30")]
north_check = north.groupby("month").agg(rows=("contractid", "size"), payers=("target_payment_usd", lambda s: int((s > 0).sum())),
                                          paid_usd=("target_payment_usd", "sum"), due_usd=("due_dc", "sum"))
north_check["mean_per_payer"] = north_check["paid_usd"] / north_check["payers"]
north_check["paid_over_due"] = north_check["paid_usd"] / north_check["due_usd"]
display(north_check.round(2))
by_cohort = north.assign(cohort=north["sales_month"].dt.to_period("Q").astype(str)).pivot_table(
    index="cohort", columns="month", values="target_payment_usd", aggfunc="sum").fillna(0)
by_cohort.columns = [c.strftime("%b %y") for c in by_cohort.columns]
by_cohort["Mar over avg(Feb,Apr)"] = (by_cohort["Mar 26"] / ((by_cohort["Feb 26"] + by_cohort["Apr 26"]) / 2)).round(2)
display(by_cohort.round(0))
north_check.to_csv(OUTPUT_DIR / "finding_north_march_2026.csv")

# %% [markdown] id="b8d21735"
# ### The product reports, checked
#
# Leadership mentioned unconfirmed product issues. The data can speak to that. Large Solar Gen 2 (launched November 2025, now about a quarter of units sold) repays close to the curve in East, North and West but about 25% below it in South, and its South customers log roughly four times the battery-fault and five times the charging-issue tickets of the same product elsewhere. That is specific enough to act on: it points at a South-specific Gen 2 problem (a batch, an installer, or a supplier), not at the product line as a whole.

# %% colab={"base_uri": "https://localhost:8080/", "height": 582} id="ee8a8a5e" outputId="f4441c5f-7014-42be-b155-d70d8025336b"
ex = panel[panel["contract_type"].eq("FINANCED") & panel["within_dc"].astype(bool) & (panel["months_on_book"] >= 1)
           & (panel["due_dc"] > 0) & panel["target_payment_usd"].notna() & (panel["month"] >= "2026-01-31")].copy()
ex = ex[~pd.MultiIndex.from_arrays([ex["region"], ex["month"]]).isin(pd.MultiIndex.from_tuples(OUTLIERS))] if OUTLIERS else ex
core_curve = C.fit_curves(panel[panel["pgroup"].eq("CORE")], ESTIMATION_END, outlier_keys=OUTLIERS)
ex["pred_core"] = ex["due_dc"] * C.lookup_eff(core_curve, ex["region"], np.full(len(ex), "CORE"), ex["months_on_book"])
prod = ex.groupby(["product", "region"]).agg(
    contract_months=("contractid", "size"),
    repayment_vs_core_curve=("target_payment_usd", "sum"),
    battery_tickets_prior_3m=("tickets_reason_battery_fault_prior_3m", "mean"),
    charging_tickets_prior_3m=("tickets_reason_charging_issue_prior_3m", "mean"),
    all_tickets_prior_3m=("tickets_prior_3m", "mean"))
prod["repayment_vs_core_curve"] = prod["repayment_vs_core_curve"] / ex.groupby(["product", "region"])["pred_core"].sum()
display(prod.round(3))
prod.to_csv(OUTPUT_DIR / "finding_gen2_south_product_issue.csv")

# %% [markdown] id="5bcd14dc"
# ### The tenor roll-off that reshapes Q3
#
# Contracts sold in early 2025 (the largest cohorts in the book) reach the end of their 540-day tenor in July and August 2026 with roughly half of the price still unpaid; at eighteen months the average cohort has paid only 57% of price. The number of past-tenor contracts more than doubles between June and September and their unpaid balance rises from about $0.26M to about $1.0M. Scheduled billing therefore falls through the quarter while recovery cash rises. Cash per contract does not fall off a cliff at tenor end (customers keep paying at about the same dollar pace, about 4% of the balance a month), but any model that drops contracts at tenor end will understate September by tens of thousands of dollars.

# %% colab={"base_uri": "https://localhost:8080/", "height": 458} id="2ce1e43f" outputId="80fecd44-217e-4936-897e-0501ba6448c1"
jun = panel[(panel["month"] == VALIDATION_END) & panel["contract_type"].eq("FINANCED")].drop_duplicates("contractid").copy()
jun["balance"] = (jun["price_usd"] - jun["actual_cumulative_cash_through_month_usd"]).clip(lower=0)
roll = []
for k, m in enumerate(FORECAST_MONTHS, start=1):
    _, within, _ = C.schedule_daycount(jun["price_usd"], jun["deposit_usd"], jun["daily_amount_usd"], jun["tenor_length"], jun["sales_month"], pd.Series([m] * len(jun)))
    roll.append({"month": m.strftime("%b %Y"), "contracts_in_schedule": int(within.sum()), "balance_in_schedule_usd": float(jun.loc[within, "balance"].sum()),
                 "contracts_past_tenor": int((~within).sum()), "balance_past_tenor_usd": float(jun.loc[~within, "balance"].sum())})
roll = pd.DataFrame([{"month": "Jun 2026 (actual)", "contracts_in_schedule": int(jun["within_dc"].sum()), "balance_in_schedule_usd": float(jun.loc[jun["within_dc"].astype(bool), "balance"].sum()),
                      "contracts_past_tenor": int(jun["post_tenor_dc"].sum()), "balance_past_tenor_usd": float(jun.loc[jun["post_tenor_dc"], "balance"].sum())}] + roll)
display(roll.round(0))
cohort_paid = panel[panel["contract_type"].eq("FINANCED")].assign(cohort=lambda d: d["sales_month"].dt.to_period("Q").astype(str),
    paid_share=lambda d: d["actual_cumulative_cash_through_month_usd"] / d["price_usd"]).pivot_table(index="cohort", columns="months_on_book", values="paid_share", aggfunc="mean")
display(cohort_paid[[0, 3, 6, 9, 12, 15, 17]].round(3))
roll.to_csv(OUTPUT_DIR / "finding_tenor_rolloff_q3.csv", index=False)

# %% [markdown] id="cbdf0a4c"
# ## The collection curve, the level factors, the recovery rate and the sales mix
#
# These four objects are the whole model. The curve is fitted on data through March 2026 (pre-pilot), by region, product group and months on book, with shrinkage toward the region curve and then the country curve where cells are thin. The level factors are the median calendar index over April to June per region. The recovery rate is cash as a share of unpaid balance by months past tenor, learned on the last six months. The sales mix is the last three months of sales by region, contract type and product group.
#
# `SHEETS: efficiency[region, product, age] = SUMIFS(paid, region, r, product, p, age, a) / SUMIFS(due, region, r, product, p, age, a)`

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="e25b8b75" outputId="e779561c-a678-43a2-a848-ee95d48a92f6"
LEVELS = C.region_levels(index_table, VALIDATION_END, outlier_keys=OUTLIERS)
RECOVERY = C.fit_recovery(panel, VALIDATION_END)
MIX = C.recent_mix(panel, VALIDATION_END)
curve_table = curves["region"].unstack(level=0).iloc[:24]
display(curve_table.round(3).T)
print("deposit efficiency by region:", curves["deposit_eff"].round(3).to_dict())
print("cash sale-month efficiency by region:", curves["cash_eff"].round(3).to_dict())
print("level factors (Apr-Jun median index):", LEVELS.round(3).to_dict())
print("recovery rate by months past tenor:", RECOVERY["rate_by_mpt"].round(3).to_dict(), "| pooled", round(RECOVERY["pooled"], 4))
display(MIX.round(3))
curves["region_product"].rename("efficiency").reset_index().to_csv(OUTPUT_DIR / "collection_curve_region_product_mob.csv", index=False)
RECOVERY["table"].to_csv(OUTPUT_DIR / "recovery_rates_by_months_past_tenor.csv")
MIX.to_csv(OUTPUT_DIR / "new_sales_mix.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 4))
for r in curve_table.columns:
    ax.plot(curve_table.index, curve_table[r], marker=".", label=r)
ax.set_xlabel("months on book"); ax.set_ylabel("paid / due"); ax.set_title("Scheduled collection efficiency by age and region (fitted through Mar 2026)")
ax.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "fig_collection_curves.png", dpi=160, bbox_inches="tight"); plt.show()

# %% [markdown] id="ceb5da14"
# ## Leakage-safe backtests: how well does this recipe predict months we already know?
#
# Two variants at four rolling origins, always forward in time, never a random split. Variant A fits the curve through the origin (what both earlier tracks did). Variant B fits the curve three months before the origin and reads the level from the last three months, which is exactly the recipe used for the final forecast, so it is the fair test. Both use the units actually sold in the target months, so these errors exclude sales-plan risk; the real Q3 error will be larger to the extent the plan is missed. Benchmarks: the last-three-month average, damped Holt on the accounting total, and the Ridge challenger on the existing book.
#
# Scoring is WAPE (absolute error over the quarter divided by actual) and bias (signed error divided by actual, positive means the forecast ran high).

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="f25cd1c4" outputId="627503c0-0c7a-4941-9673-cf1c74bf8c93"
bt_rows, bt_monthly = [], []
for origin in BACKTEST_ORIGINS:
    months = pd.date_range(origin + pd.offsets.MonthEnd(1), periods=3, freq="ME")
    actual = country_month.set_index("month").reindex(months)["source_reported_cash_usd"].astype(float)
    units = C.actual_units(panel, months)
    fcA, detA = C.forecast(panel, country_month, origin, months, units, curve_cutoff=origin, return_detail=True)
    cutB = origin - pd.offsets.MonthEnd(3)
    fcB, detB = C.forecast(panel, country_month, origin, months, units, curve_cutoff=cutB, return_detail=True)
    fcB_nolevel = C.forecast(panel, country_month, origin, months, units, curve_cutoff=cutB,
                             level_override={r: 1.0 for r in detB["levels"].index}, curves=detB["curves"], outlier_keys=detB["outliers"])
    cands = {"cohort_A_curve_through_origin": fcA.set_index("month")["total"],
             "cohort_B_curve_cut_3m_before_origin": fcB.set_index("month")["total"],
             "cohort_B_without_level_factor": fcB_nolevel.set_index("month")["total"],
             "naive_last_3m_average": C.naive_forecast(country_month, origin, months),
             "holt_damped": C.holt_forecast(country_month, origin, months)}
    if RUN_CHALLENGER:
        rg = C.ridge_existing_book(panel, origin, months, target_cutoff=origin).set_index("month")["ridge_existing"]
        non_existing = fcA.set_index("month")[["new_sales_financed", "new_sales_cash", "bridge"]].sum(axis=1)
        cands["ridge_challenger_existing_plus_cohort_new_sales"] = rg + non_existing
    for name, f in cands.items():
        f = f.reindex(months)
        s = C.score(actual.to_numpy(), f.to_numpy())
        bt_rows.append({"origin": origin.date(), "model": name, **s})
        for m in months:
            bt_monthly.append({"origin": origin.date(), "model": name, "month": m, "actual": float(actual[m]), "forecast": float(f[m])})
    if origin == ESTIMATION_END:
        region_resid = detA["region_forecast"].merge(region_month[["month", "region", "panel_cash_usd"]], on=["month", "region"])
        region_resid["residual_pct_actual_vs_forecast"] = region_resid["panel_cash_usd"] / region_resid["total_ex_bridge"] - 1
        act_exist = panel[panel["month"].isin(months) & (panel["sales_month"] <= origin)].groupby(["month", "region"])["target_payment_usd"].sum().rename("actual_existing_usd").reset_index()
        region_resid = region_resid.merge(act_exist, on=["month", "region"], how="left")
        region_resid["existing_residual_pct"] = region_resid["actual_existing_usd"] / region_resid["existing"] - 1
        region_resid["pilot_context"] = region_resid["region"].map({"East": "SMS pilot", "West": "outbound-call pilot"}).fillna("no pilot")
backtest_summary = pd.DataFrame(bt_rows)
backtest_monthly = pd.DataFrame(bt_monthly)
pivot_w = backtest_summary.pivot(index="model", columns="origin", values="WAPE")
pivot_w["mean_WAPE"] = pivot_w.mean(axis=1)
pivot_b = backtest_summary.pivot(index="model", columns="origin", values="Bias")
pivot_w["mean_bias"] = pivot_b.mean(axis=1)
display(pivot_w.sort_values("mean_WAPE").style.format("{:.1%}"))
backtest_summary.to_csv(OUTPUT_DIR / "backtest_summary.csv", index=False)
backtest_monthly.to_csv(OUTPUT_DIR / "backtest_monthly.csv", index=False)
print("Apr-Jun 2026 region read from the 31 Mar origin (variant A). The total residual mixes existing customers and new sales;")
print("existing_residual_pct isolates the existing book, which is where a pilot would show: East runs about 10% above, West about 5% above, North and South at or below.")
display(region_resid.round(3))
region_resid.to_csv(OUTPUT_DIR / "region_validation_residuals_apr_jun_2026.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 4))
hist = country_month[["month", "source_reported_cash_usd"]]
ax.plot(hist["month"], hist["source_reported_cash_usd"], color="black", lw=2, label="actual (accounting total)")
for origin, g in backtest_monthly[backtest_monthly["model"].eq("cohort_B_curve_cut_3m_before_origin")].groupby("origin"):
    ax.plot(g["month"], g["forecast"], marker="o", ls="--", label=f"cohort from {origin}")
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=-1, interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y")); ax.set_ylabel("USD"); ax.set_title("Rolling-origin backtests, cohort recipe B vs actual (accounting total)")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "fig_backtests.png", dpi=160, bbox_inches="tight"); plt.show()

# %% [markdown] id="3ab0aedc"
# ## The base forecast for July to September 2026
#
# Origin 30 June 2026; curve fitted through 31 March 2026; level factors from April to June; recovery from January to June; mix from April to June; plan units 3,100 / 3,200 / 3,200. Outreach is not modelled explicitly: whatever the East and West pilots were adding in April to June is inside those two regions' level factors, and the scenario below shows what the quarter loses if the pilots stop in July.

# %% colab={"base_uri": "https://localhost:8080/", "height": 665} id="9ee4fe7b" outputId="43ef7fe1-3b6b-4d5a-9157-253cd4f0bff6"
base, DET = C.forecast(panel, country_month, VALIDATION_END, FORECAST_MONTHS, SALES_PLAN, curve_cutoff=ESTIMATION_END,
                       curves=curves, outlier_keys=OUTLIERS, levels=LEVELS, recovery=RECOVERY, mix=MIX, return_detail=True)
base_show = base.copy(); base_show["month"] = base_show["month"].dt.strftime("%b %Y")
display(base_show.set_index("month").round(0).T)
print("Q3 base total:", f"${base['total'].sum():,.0f}")
region_fc = DET["region_forecast"].pivot(index="month", columns="region", values="total_ex_bridge")
display(region_fc.round(0))
base.to_csv(OUTPUT_DIR / "forecast_components_base.csv", index=False)
DET["region_forecast"].to_csv(OUTPUT_DIR / "forecast_by_region_base.csv", index=False)

# cross-checks a finance reader can do by hand
flat_june = float(country_month.loc[country_month["month"].eq(VALIDATION_END), "source_reported_cash_usd"].iloc[0])
naive3 = C.naive_forecast(country_month, VALIDATION_END, FORECAST_MONTHS)
holt = C.holt_forecast(country_month, VALIDATION_END, FORECAST_MONTHS)
cross = pd.DataFrame({"month": FORECAST_MONTHS, "cohort_base": base["total"].to_numpy(),
                      "june_held_flat": flat_june, "last_3m_average": naive3.to_numpy(), "holt_damped": holt.to_numpy()})
if RUN_CHALLENGER:
    rg = C.ridge_existing_book(panel, VALIDATION_END, FORECAST_MONTHS, target_cutoff=ESTIMATION_END)
    cross["ridge_existing_book"] = rg["ridge_existing"].to_numpy()
    cross["cohort_existing_book"] = (base["existing_scheduled"] + base["post_tenor_recovery"]).to_numpy()
    cross["ridge_challenger_total"] = cross["ridge_existing_book"] + base[["new_sales_financed", "new_sales_cash", "bridge"]].sum(axis=1).to_numpy()
cross_show = cross.copy(); cross_show["month"] = cross_show["month"].dt.strftime("%b %Y")
display(cross_show.set_index("month").round(0).T)
cross.to_csv(OUTPUT_DIR / "cross_checks.csv", index=False)

# %% [markdown] id="b156e66c"
# ## Scenarios and the tornado: what moves the number
#
# Each lever moves one thing. The low case stacks every adverse setting and the high case every favourable one, so the band is a stress range rather than a probability interval; the tornado shows each lever on its own, ranked. The collection-level lever is sized from the calendar shock, the month-to-month wobble in the whole book's collection index over the last year, two shocks down and one up, because every contract lives through the same calendar and the risk sits below the base.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="f54bcf6f" outputId="072e3dc5-dab9-4e67-d17d-16308bca46c7"
SIGMA = C.calendar_shock_sigma(country_index_clean)
STOP_LEVELS, pre_lv, post_lv, ctrl_change = C.pilot_stop_levels(index_table, VALIDATION_END, outlier_keys=OUTLIERS)
cs_hist = C.cash_share_history(panel, VALIDATION_END)
cs_base = float(MIX.loc[MIX["contract_type"].eq("CASH"), "share"].sum())
bridge_hist = country_month[country_month["month"] <= VALIDATION_END]["source_minus_attributable_cash_usd"].tail(12)
bridge_base = float(country_month[country_month["month"] <= VALIDATION_END].tail(6)["source_minus_attributable_cash_usd"].median())
june_units = int(sales.loc[pd.Period("2026-06", "M"), "total"])
attain_low = round(june_units / 3100, 2)

LEVERS = {
    "existing_book_collection_level":  {"key": "level_factor", "low": 1 - 2 * SIGMA, "base": 1.0, "high": 1 + SIGMA,
                                        "evidence": f"calendar shock sigma {SIGMA:.3f} (std of month-to-month change in the book's collection index, last 12 months); two sigma down, one up"},
    "pilots_stop_in_july":             {"key": "level_override", "low": STOP_LEVELS, "base": None, "high": None,
                                        "evidence": f"East/West levels revert to pre-pilot level moved by the control regions' change ({ctrl_change:.3f}): East {STOP_LEVELS['East']:.3f} vs {LEVELS['East']:.3f}, West {STOP_LEVELS['West']:.3f} vs {LEVELS['West']:.3f}"},
    "sales_plan_attainment":           {"key": "attainment", "low": attain_low, "base": 1.0, "high": 1.05,
                                        "evidence": f"low = June run-rate {june_units} units / 3,100 plan; high = 5% over plan"},
    "cash_share_of_new_sales":         {"key": "cash_share", "low": float(min(cs_base, cs_hist.quantile(0.25))), "base": cs_base, "high": float(max(cs_base, cs_hist.quantile(0.75))),
                                        "evidence": "25th/75th percentile of monthly cash share over the last 12 months"},
    "post_tenor_recovery_rate":        {"key": "recovery_factor", "low": 0.75, "base": 1.0, "high": 1.25,
                                        "evidence": "+/-25% on the recent rate; the Q3 pool is three times larger than any pool observed so far"},
    "reconciliation_bridge_usd":       {"key": "bridge_override", "low": float(min(bridge_base, bridge_hist.quantile(0.25))), "base": bridge_base, "high": float(max(bridge_base, bridge_hist.quantile(0.75))),
                                        "evidence": "median of last 6 months; 25th/75th of last 12"},
}
COMMON = dict(curve_cutoff=ESTIMATION_END, curves=curves, outlier_keys=OUTLIERS, levels=LEVELS, recovery=RECOVERY, mix=MIX)

def run(**kw):
    return C.forecast(panel, country_month, VALIDATION_END, FORECAST_MONTHS, SALES_PLAN, **COMMON, **kw)

low_kw = {v["key"]: v["low"] for v in LEVERS.values() if v["low"] is not None}
high_kw = {v["key"]: v["high"] for v in LEVERS.values() if v["high"] is not None}
low, high = run(**low_kw), run(**high_kw)
forecast_q3_2026 = pd.DataFrame({"month": FORECAST_MONTHS, "low": low["total"].to_numpy(), "base": base["total"].to_numpy(), "high": high["total"].to_numpy()})
assert ((forecast_q3_2026["low"] <= forecast_q3_2026["base"]) & (forecast_q3_2026["base"] <= forecast_q3_2026["high"])).all()
tbl = forecast_q3_2026.copy(); tbl["month"] = tbl["month"].dt.strftime("%b %Y")
tbl = pd.concat([tbl, pd.DataFrame([{"month": "Q3 total", "low": tbl["low"].sum(), "base": tbl["base"].sum(), "high": tbl["high"].sum()}])], ignore_index=True)
display(tbl.set_index("month").style.format("${:,.0f}"))

base_total = float(base["total"].sum())
torn = []
for name, v in LEVERS.items():
    for case in ["low", "high"]:
        if v[case] is None:
            continue
        r = run(**{v["key"]: v[case]})
        torn.append({"assumption": name, "case": case, "setting": v[case] if not isinstance(v[case], dict) else "East/West revert",
                     "q3_total_usd": float(r["total"].sum()), "change_vs_base_usd": float(r["total"].sum()) - base_total})
tornado = pd.DataFrame(torn)
tornado["abs_change"] = tornado["change_vs_base_usd"].abs()
tornado = tornado.sort_values(["abs_change"], ascending=False)
display(tornado.drop(columns="abs_change").style.format({"q3_total_usd": "${:,.0f}", "change_vs_base_usd": "${:+,.0f}"}))
largest = tornado.iloc[0]
print("Largest single sensitivity:", largest["assumption"], f"({largest['change_vs_base_usd']:+,.0f} USD on the quarter)")
forecast_q3_2026.to_csv(OUTPUT_DIR / "forecast_q3_2026_final.csv", index=False)
tornado.to_csv(OUTPUT_DIR / "forecast_sensitivity_tornado.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
x = np.arange(3); lab = [m.strftime("%b %Y") for m in FORECAST_MONTHS]
ax[0].plot(x, forecast_q3_2026["base"], marker="o", label="base"); ax[0].fill_between(x, forecast_q3_2026["low"], forecast_q3_2026["high"], alpha=0.2, label="low to high")
ax[0].axhline(flat_june, color="grey", ls=":", label="June held flat"); ax[0].set_xticks(x); ax[0].set_xticklabels(lab); ax[0].set_ylabel("USD"); ax[0].set_title("Q3 2026 collections forecast"); ax[0].legend()
comp = base.set_index("month")[["existing_scheduled", "post_tenor_recovery", "new_sales_financed", "new_sales_cash", "bridge"]]
comp.index = lab; comp.plot(kind="bar", stacked=True, ax=ax[1]); ax[1].set_title("Where the base cash comes from"); ax[1].tick_params(axis="x", rotation=0); ax[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "fig_forecast_band_and_components.png", dpi=160, bbox_inches="tight"); plt.show()
fig, ax = plt.subplots(figsize=(8, 3.6))
tp = tornado.pivot(index="assumption", columns="case", values="change_vs_base_usd").reindex(tornado["assumption"].unique()[::-1])
ax.barh(tp.index, tp.get("low", 0).fillna(0), color="#c0504d", label="adverse"); ax.barh(tp.index, tp.get("high", 0).fillna(0), color="#4f81bd", label="favourable")
ax.axvline(0, color="black", lw=0.8); ax.set_xlabel("change in Q3 total, USD"); ax.set_title("Which assumption moves the quarter most"); ax.legend()
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "fig_tornado.png", dpi=160, bbox_inches="tight"); plt.show()

# %% [markdown] id="cc8e485a"
# ## Assumptions register (for the assumptions slide)

# %% id="a1c6a7f1"
assumptions = pd.DataFrame([
    {"assumption": "Collection curve shape", "setting": "by region x product group x months on book, fitted Oct 2024 to Mar 2026, North Mar 2026 excluded", "evidence": f"{curves['n_rows']:,} scheduled contract-months; shrinkage toward region then country"},
    {"assumption": "Existing-book collection level", "setting": "region level factors " + ", ".join(f"{k} {v:.3f}" for k, v in LEVELS.items()), "evidence": "median Apr-Jun 2026 index vs own curve; includes whatever the pilots were adding"},
    {"assumption": "Post-tenor recovery", "setting": f"{RECOVERY['pooled']:.1%} of unpaid balance per month, by months past tenor", "evidence": f"{RECOVERY['rows']:,} past-tenor contract-months, Jan to Jun 2026"},
    {"assumption": "Sales plan", "setting": "3,100 / 3,200 / 3,200 units at 100% attainment", "evidence": f"May and June sold {int(sales.loc[pd.Period('2026-05','M'),'total'])} and {june_units}; West units have fallen by more than half since Q4 2025"},
    {"assumption": "New-sales mix and terms", "setting": f"Apr-Jun 2026 mix; cash share {cs_base:.1%}; Gen 2 share of financed {MIX.loc[MIX['contract_type'].eq('FINANCED') & MIX['pgroup'].eq('GEN2'),'share'].sum()/MIX.loc[MIX['contract_type'].eq('FINANCED'),'share'].sum():.1%}", "evidence": "recent_mix table above"},
    {"assumption": "Sale-month cash", "setting": "cash sales pay " + ", ".join(f"{k} {v:.1%}" for k, v in curves['cash_eff'].items()) + " of price; deposits " + ", ".join(f"{k} {v:.1%}" for k, v in curves['deposit_eff'].items()), "evidence": "sale-month efficiencies through Mar 2026"},
    {"assumption": "Reconciliation bridge", "setting": f"${bridge_base:,.0f} per month", "evidence": "median of last 6 months of accounting cash not attachable to a clean contract-month"},
    {"assumption": "Outreach in Q3", "setting": "continues at Q2 intensity (embedded in East/West levels); stop case in the low scenario", "evidence": "pilot-stop lever above"},
    {"assumption": "Schedule day count", "setting": "real calendar days; Jul 31, Aug 31, Sep 30", "evidence": "day-count rebuild table above"},
])
display(assumptions)
assumptions.to_csv(OUTPUT_DIR / "assumptions_register.csv", index=False)

# %% [markdown] id="eacfd0e4"
# ## Data-quality findings and parked questions (for the findings slide)

# %% id="3f812bcd"
findings = pd.DataFrame([
    {"finding": "North collected ~70% above its curve in March 2026, uniformly across cohorts and products", "evidence": "finding_north_march_2026.csv", "treatment": "excluded from curve and level fitting", "question_for_the_room": "Was there a March collections drive in North, or a posting error?"},
    {"finding": "Large Solar Gen 2 in South repays ~25% below curve with 4-5x the battery/charging tickets of the same product elsewhere", "evidence": "finding_gen2_south_product_issue.csv", "treatment": "own curve cell for Gen 2 by region; flagged to Operations", "question_for_the_room": "Which batch / installer / supplier served South Gen 2 units?"},
    {"finding": "Units sold fell to 2,830 in May and 2,392 in June; West has more than halved since Q4 2025", "evidence": "sales_units_by_month_region.csv", "treatment": "plan attainment lever; low case at June run-rate", "question_for_the_room": "What is behind the West sales decline, and is the Q3 plan backed by pipeline?"},
    {"finding": "Early-2025 cohorts reach tenor end in Jul-Aug with ~50% of price unpaid; past-tenor balance triples by September", "evidence": "finding_tenor_rolloff_q3.csv", "treatment": "recovery component on the balance", "question_for_the_room": "What is the collections and write-off policy past tenor?"},
    {"finding": "The average-month billing proxy created a fake February dip (-8%) and March rebound", "evidence": "schedule rebuild table", "treatment": "schedule rebuilt on real days", "question_for_the_room": "Confirm billing runs on calendar days"},
    {"finding": f"About ${feature_metrics['source_minus_model_attributable_cash']:,.0f} of accounting cash over 21 months cannot be attached to a clean contract-month (orphans, pre-sale, conflicts, negatives)", "evidence": "payment_reconciliation_by_month_v3.csv", "treatment": "bridge line in the forecast", "question_for_the_room": "Who owns the orphan contract IDs and the same-month conflicting rows?"},
    {"finding": f"{int(feature_metrics.get('n_contracts', 0)):,} contracts; 1,031 had deposit recorded as an amount not a share (corrected in cleaning)", "evidence": "cleaning audit", "treatment": "corrected, flagged", "question_for_the_room": "Confirm the deposit field definition with the sales system owner"},
])
display(findings)
findings.to_csv(OUTPUT_DIR / "data_findings_and_parked_questions.csv", index=False)

# %% [markdown] id="9d08205f"
# ## QA gate, freeze the forecast, package the outputs
#
# The forecast CSV is written with a SHA-256 fingerprint so the frozen artefact can be shown to predate any reveal of the sealed quarter. After this cell nothing in the model should change; if the sealed actuals are ever compared, that comparison lives in a separate scoring notebook that only reads the frozen CSV and the actuals.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="654620f4" outputId="7f23b8a2-aa03-4c72-b95f-4e9b48391cab"
comp_check = (base[["existing_scheduled", "post_tenor_recovery", "new_sales_financed", "new_sales_cash", "bridge"]].sum(axis=1) - base["total"]).abs().max()
qa_output = pd.DataFrame([
    {"check": "sealed_q3_actuals_never_loaded", "status": "PASS" if country_month["month"].max() <= VALIDATION_END and panel["month"].max() <= VALIDATION_END else "FAIL"},
    {"check": "forecast_months_are_jul_aug_sep_2026", "status": "PASS" if list(forecast_q3_2026["month"]) == list(FORECAST_MONTHS) else "FAIL"},
    {"check": "low_le_base_le_high", "status": "PASS" if ((forecast_q3_2026["low"] <= forecast_q3_2026["base"]) & (forecast_q3_2026["base"] <= forecast_q3_2026["high"])).all() else "FAIL"},
    {"check": "components_sum_to_total", "status": "PASS" if comp_check < 0.01 else "FAIL"},
    {"check": "region_forecasts_sum_to_country_ex_bridge", "status": "PASS" if np.allclose(region_fc.sum(axis=1).to_numpy(), (base["total"] - base["bridge"]).to_numpy(), atol=1.0) else "FAIL"},
    {"check": "cohort_beats_naive_on_mean_backtest_WAPE", "status": "PASS" if pivot_w.loc["cohort_B_curve_cut_3m_before_origin", "mean_WAPE"] < pivot_w.loc["naive_last_3m_average", "mean_WAPE"] else "FAIL"},
    {"check": "no_outreach_fields_used", "status": "PASS"},
])
display(qa_output)
if qa_output["status"].ne("PASS").any():
    raise AssertionError("Output QA failed; do not ship this forecast")

frozen = forecast_q3_2026.copy(); frozen["month"] = frozen["month"].dt.strftime("%Y-%m-%d")
frozen = frozen[["month", "low", "base", "high"]].round(2)
frozen.to_csv(OUTPUT_DIR / "forecast_q3_2026_final.csv", index=False)
digest = hashlib.sha256((OUTPUT_DIR / "forecast_q3_2026_final.csv").read_bytes()).hexdigest()
(OUTPUT_DIR / "forecast_q3_2026_final.sha256").write_text(digest + "  forecast_q3_2026_final.csv\n")
print("frozen forecast:"); print(frozen.to_string(index=False)); print("sha256:", digest)

slide_numbers = {
    "q3_base_usd": round(base_total), "q3_low_usd": round(float(low["total"].sum())), "q3_high_usd": round(float(high["total"].sum())),
    "by_month": {m.strftime("%b %Y"): {"low": round(float(l)), "base": round(float(b)), "high": round(float(h))} for m, l, b, h in zip(FORECAST_MONTHS, forecast_q3_2026["low"], forecast_q3_2026["base"], forecast_q3_2026["high"])},
    "share_existing_book": round(float((base["existing_scheduled"] + base["post_tenor_recovery"]).sum() / base_total), 3),
    "share_new_sales": round(float((base["new_sales_financed"] + base["new_sales_cash"]).sum() / base_total), 3),
    "share_recovery": round(float(base["post_tenor_recovery"].sum() / base_total), 3),
    "largest_sensitivity": largest["assumption"], "largest_sensitivity_usd": round(float(largest["change_vs_base_usd"])),
    "backtest_mean_WAPE_recipe_B": round(float(pivot_w.loc["cohort_B_curve_cut_3m_before_origin", "mean_WAPE"]), 4),
    "backtest_mean_WAPE_naive": round(float(pivot_w.loc["naive_last_3m_average", "mean_WAPE"]), 4),
    "backtest_apr_jun_WAPE_recipe_B": round(float(backtest_summary[(backtest_summary["model"].eq("cohort_B_curve_cut_3m_before_origin")) & (backtest_summary["origin"].astype(str).eq("2026-03-31"))]["WAPE"].iloc[0]), 4),
    "levels": {k: round(float(v), 3) for k, v in LEVELS.items()}, "calendar_shock_sigma": round(SIGMA, 4), "june_units": june_units,
    "forecast_sha256": digest,
}
json.dump(slide_numbers, open(OUTPUT_DIR / "slide_numbers.json", "w"), indent=2)
print(json.dumps(slide_numbers, indent=2))

zip_path = Path("part1_final_outputs.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            z.write(f, arcname=f.name)
print("wrote", zip_path, sorted(f.name for f in OUTPUT_DIR.iterdir()))
try:
    from google.colab import files as _dl
    _dl.download(str(zip_path))
except ImportError:
    pass

# %% [markdown] id="f1c09048"
# ## With direct access to the stakeholders, what would have been done differently
#
# 1. Ask Finance for the true cash-posting calendar and for the March North anomaly before fitting anything; one conversation would settle whether that month is real.
# 2. Ask Sales for the Q3 pipeline by region, because the plan assumes West sells almost double its June volume.
# 3. Ask Operations for the Gen 2 batch and installer records in South; the ticket pattern is specific enough to trace.
# 4. Ask Credit whether the East and West outreach continues in July to September at the pilot intensity, which decides which scenario is the base.
# 5. Ask for the write-off and repossession policy past tenor, since a third of the book crosses tenor end this quarter.

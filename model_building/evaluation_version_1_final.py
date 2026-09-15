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

# %% [markdown] id="1b9a072c"
# # Evaluation, version 1 (FINAL): which collections pilot is worth scaling, and how to spend $8,000 a month from October
#
# It reads only the frozen v3 feature outputs (development data through 30 June 2026) and stops if anything later appears.
#
# ## What this notebook does, in plain language
#
# Two regional pilots ran in April to June 2026: East sent preventative SMS reminders, West made outbound calls to customers already in arrears. Neither was randomised, so raw East-versus-West repayment says nothing. Two lenses are used instead, and the recommendation only leans on a result where the lens is credible for that pilot.
#
# **Lens 1, region-level difference-in-differences at fixed age.** For every scheduled contract-month the outcome is cash paid minus what the region's own pre-pilot collection curve predicts for a contract of that age and product (the same curve as Part 1). Before April every region sits near zero. After April the pilot region's residual minus the control regions' residual is the effect of running the programme in that region, in dollars per active contract-month, with a contract-clustered bootstrap interval, an event-study plot to show the pre-trend, and a placebo test inside the pre-period. This does not need look-alike untreated customers inside the pilot region, which is decisive for East, where 95% of active customers were messaged.
#
# **Lens 2, within-region matching.** Each contacted customer is matched to never-contacted customers in the same region and month with a similar pre-contact history. This estimates the effect on those actually contacted. It is credible for West (about 40% of customers were never called) and it is not credible for East (the 5% never messaged turn out to be accounts that pay nothing at all), which is shown rather than assumed.
#
# **Repayment measure.** Incremental cash, in dollars, above what the customer's age and product would predict. Same-month cash is the primary outcome for an SMS reminder, because a reminder is meant to act before the due date; next-month cash is also reported and is the primary outcome for calls to customers in arrears, matching the earlier track's reasoning. Every contact counts, reached or not, because unreached attempts cost money.
#
# **What changed relative to the two earlier tracks.** The earlier matching track reported a 24x return for East from a 2% matched slice; that slice is shown here to be uninformative. The earlier DiD track had an age confound and an interval bug; the age adjustment here is done through the curve, and the intervals come from a cluster bootstrap that runs. South's Gen 2 customers are removed from the control group because Part 1 found a product-specific repayment problem there. Pilot spend is compared with the $8,000 budget explicitly.
#
# **Run order:** setup, upload the v3 ZIP when prompted (or place it beside the notebook), run everything, download the ZIP after QA passes.

# %% colab={"base_uri": "https://localhost:8080/"} id="805ceefc" outputId="abd638c1-97ac-4232-823c-33f88fef2412"
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"], check=False)
import io, json, zipfile, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython.display import display
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)
OUTPUT_DIR = Path("part2_final_outputs"); OUTPUT_DIR.mkdir(exist_ok=True)
N_BOOT = 200
print("setup ready")

# %% colab={"base_uri": "https://localhost:8080/"} id="f0e39945" outputId="2f068a99-b445-487b-b439-e2daf2d9c38d"
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



# %% colab={"base_uri": "https://localhost:8080/"} id="87b79e48" outputId="3506f4da-e2ff-40ed-838a-7eb086420d8f"
# %%writefile final_pilot_core.py
"""
Final Part 2 (pilot evaluation) core functions.

Two lenses on two non-randomised pilots:

Lens 1, region-level difference-in-differences at fixed age. The outcome for
every scheduled contract-month is the dollar residual, cash paid minus what the
region's own pre-pilot collection curve predicts for a contract of that age and
product. Before April the pilot and control regions should both sit near zero;
after April the pilot regions' residual minus the control regions' residual is
the intention-to-treat effect of running the programme in that region, in dollars
per active contract-month. This lens does not need to find look-alike untreated
customers inside the pilot region, which matters for East where 95% of customers
were messaged.

Lens 2, within-region matching. Each contacted customer is matched to never-
contacted customers in the same region and month with a similar pre-contact
history (propensity score, nearest neighbours, common support). This is the
average effect on the treated and is only credible where there is a real
untreated comparison group; West (about 40% never called) qualifies, East (about
5% never messaged) does not, and its matched number is shown only as a caveat.

Nothing here reads a month after 30 June 2026.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VALIDATION_END = pd.Timestamp("2026-06-30")
PILOT_START = pd.Timestamp("2026-04-30")
PRE_START = pd.Timestamp("2025-10-31")
PRE_END = pd.Timestamp("2026-03-31")
CONTACT_MONTHS_NEXT = [pd.Timestamp("2026-04-30"), pd.Timestamp("2026-05-31")]      # next-month outcome observed
CONTACT_MONTHS_SAME = [pd.Timestamp("2026-04-30"), pd.Timestamp("2026-05-31"), pd.Timestamp("2026-06-30")]
MONTHLY_BUDGET = 8000.0
SEED = 20260912


# ---------------------------------------------------------------------------
# 1. Coverage and cost of the pilots
# ---------------------------------------------------------------------------
def pilot_coverage(panel, pilot):
    rows = []
    for (reg, m), g in pilot.groupby(["region_outreach", "contact_month"]):
        active = panel[(panel["region"].eq(reg)) & panel["contract_type"].eq("FINANCED") & (panel["month"].eq(m)) & (panel["months_on_book"] >= 1)]
        contacted = g["contractid"].nunique()
        rows.append({"region": reg, "channel": g["channel"].iloc[0], "month": m, "active_financed_contracts": int(len(active)),
                     "contacted": int(contacted), "share_contacted": contacted / max(1, len(active)),
                     "attempts": float(g["attempts"].sum()), "reached_share": float(g["reached"].astype(float).mean()),
                     "cost_usd": float(g["cost_usd"].sum()), "cost_per_contact_usd": float(g["cost_usd"].sum() / max(1, contacted)),
                     "share_with_zero_payment_prior_month": float((g["payment_lag_1m"].fillna(0) == 0).mean()),
                     "share_zero_last_3_months": float((g["zero_payment_months_prior_3"] == 3).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Residual outcome at fixed age (needs Part 1 curves)
# ---------------------------------------------------------------------------
def residual_frame(panel, curves, lookup_eff, start=PRE_START, end=VALIDATION_END, outlier_keys=()):
    h = panel[(panel["month"] >= start) & (panel["month"] <= end) & panel["contract_type"].eq("FINANCED")
              & panel["within_dc"].astype(bool) & (panel["months_on_book"] >= 1) & (panel["due_dc"] > 0)
              & panel["target_payment_usd"].notna()].copy()
    if outlier_keys:
        bad = pd.MultiIndex.from_tuples(list(outlier_keys))
        h = h[~pd.MultiIndex.from_arrays([h["region"], h["month"]]).isin(bad)]
    h["pred"] = h["due_dc"] * lookup_eff(curves, h["region"], h["pgroup"], h["months_on_book"])
    h["resid"] = h["target_payment_usd"] - h["pred"]
    h["post"] = (h["month"] >= PILOT_START).astype(int)
    return h


def did_regression(h, treated_region, control_regions, n_boot=300, seed=SEED, exclude_gen2_south=True):
    """Contract-month OLS of the dollar residual on region and month fixed effects
    plus treated x post; contract-clustered bootstrap for the interval."""
    d = h[h["region"].isin([treated_region] + list(control_regions))].copy()
    if exclude_gen2_south:
        d = d[~(d["region"].eq("South") & d["pgroup"].eq("GEN2"))]
    d = d.reset_index(drop=True)
    treated = d["region"].eq(treated_region).astype(float).to_numpy()
    post = d["post"].to_numpy(dtype=float)
    month_d = pd.get_dummies(d["month"].dt.strftime("%Y-%m"), drop_first=True).to_numpy(dtype=float)
    region_d = pd.get_dummies(d["region"], drop_first=True).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(d)), treated * post, month_d, region_d])
    y = d["resid"].to_numpy(dtype=float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    did = float(beta[1])
    # cluster bootstrap by contract
    groups = d.groupby("contractid").indices
    keys = np.array(list(groups.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    boots = []
    XtX_cache = None
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([groups[keys[j]] for j in pick])
        b = np.linalg.lstsq(X[rows], y[rows], rcond=None)[0][1]
        boots.append(b)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    mean_pred_post = float(d.loc[d["region"].eq(treated_region) & d["post"].eq(1), "pred"].mean())
    n_post_cm = int((d["region"].eq(treated_region) & d["post"].eq(1)).sum())
    return {"treated_region": treated_region, "controls": "+".join(control_regions), "did_usd_per_contract_month": did,
            "ci_low": float(lo), "ci_high": float(hi), "uplift_pct_of_predicted": did / mean_pred_post,
            "treated_post_contract_months": n_post_cm, "rows": int(len(d)), "contracts": int(len(keys))}


def event_study(h, treated_region, control_regions, exclude_gen2_south=True):
    d = h[h["region"].isin([treated_region] + list(control_regions))].copy()
    if exclude_gen2_south:
        d = d[~(d["region"].eq("South") & d["pgroup"].eq("GEN2"))]
    t = d[d["region"].eq(treated_region)].groupby("month")["resid"].agg(["mean", "std", "size"])
    c = d[~d["region"].eq(treated_region)].groupby("month")["resid"].agg(["mean", "std", "size"])
    out = pd.DataFrame({"treated_mean_resid": t["mean"], "control_mean_resid": c["mean"]})
    out["gap"] = out["treated_mean_resid"] - out["control_mean_resid"]
    out["gap_se"] = np.sqrt(t["std"] ** 2 / t["size"] + c["std"] ** 2 / c["size"])
    return out


def pretrend_slope(es, pre_end=PRE_END):
    pre = es[es.index <= pre_end]["gap"].dropna()
    if len(pre) < 3:
        return np.nan
    x = np.arange(len(pre))
    return float(np.polyfit(x, pre.to_numpy(), 1)[0])


def placebo_did(h, treated_region, control_regions, fake_post_start=pd.Timestamp("2026-01-31"), **kw):
    """Same regression inside the pre period with a fake April in January."""
    hp = h[h["month"] <= PRE_END].copy()
    hp["post"] = (hp["month"] >= fake_post_start).astype(int)
    return did_regression(hp, treated_region, control_regions, n_boot=kw.get("n_boot", 200), seed=kw.get("seed", SEED))


# ---------------------------------------------------------------------------
# 3. Within-region matching (average effect on the treated)
# ---------------------------------------------------------------------------
MATCH_NUM = ["months_on_book", "price_usd", "perc_deposit", "daily_amount_usd", "tenor_length", "due_dc",
             "expected_cumulative_cash_before_month_usd", "actual_cumulative_cash_before_month_usd", "balance_before",
             "payment_lag_1m", "payment_lag_2m", "payment_lag_3m", "payment_trailing_3m_sum", "payment_trailing_6m_sum",
             "prior_paying_months", "zero_payment_months_prior_3", "zero_payment_months_prior_6",
             "months_since_last_positive_payment", "calls_prior_3m", "tickets_prior_3m",
             "calls_cumulative_before_month", "tickets_cumulative_before_month"]
MATCH_CAT = ["product", "payment_frequency", "scheduled_status"]


def build_matching_frame(panel, pilot, region, contact_months):
    """Financed contract-months in `region` for the contact months, with treatment,
    same-month residual, and next-month cash. Controls are contracts never contacted
    in that pilot; contracts contacted in some other month are excluded from controls."""
    ever = set(pilot.loc[pilot["region_outreach"].eq(region), "contractid"].astype(str))
    tr = pilot[pilot["region_outreach"].eq(region)].groupby(["contractid", "contact_month"], as_index=False).agg(
        cost_usd=("cost_usd", "sum"), attempts=("attempts", "sum"), reached=("reached", "max"))
    tr["treated"] = 1
    d = panel[panel["region"].eq(region) & panel["contract_type"].eq("FINANCED") & panel["month"].isin(contact_months)
              & (panel["months_on_book"] >= 1) & panel["target_payment_usd"].notna()].copy()
    d = d.merge(tr.rename(columns={"contact_month": "month"}), on=["contractid", "month"], how="left")
    d["treated"] = d["treated"].fillna(0).astype(int)
    d = d[d["treated"].eq(1) | ~d["contractid"].astype(str).isin(ever)].copy()
    nxt = panel[["contractid", "month", "target_payment_usd"]].copy()
    nxt["month"] = nxt["month"] - pd.offsets.MonthEnd(1)
    nxt = nxt.rename(columns={"target_payment_usd": "next_month_cash"})
    d = d.merge(nxt, on=["contractid", "month"], how="left")
    d["same_month_cash"] = d["target_payment_usd"]
    return d


def smd(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna(); b = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s else 0.0


def match_region(d, outcome_cols, k=3, caliper=0.05, seed=SEED, n_boot=500):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num = [c for c in MATCH_NUM if c in d.columns]; cat = [c for c in MATCH_CAT if c in d.columns]
    d = d.copy(); d["mkey"] = d["month"].dt.strftime("%Y-%m")
    prep = ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
                              ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))]), cat + ["mkey"])])
    ps = Pipeline([("p", prep), ("m", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))])
    ps.fit(d[num + cat + ["mkey"]], d["treated"])
    d["pscore"] = ps.predict_proba(d[num + cat + ["mkey"]])[:, 1]
    d["logit"] = np.log(d["pscore"] / (1 - d["pscore"]).clip(1e-9))
    tr, co = d[d["treated"].eq(1)], d[d["treated"].eq(0)]
    lo, hi = max(tr["pscore"].min(), co["pscore"].min()), min(tr["pscore"].max(), co["pscore"].max())
    n_treated_all = len(tr)
    pairs = []
    cal = caliper * d["logit"].std()
    for mk, tm in tr[tr["pscore"].between(lo, hi)].groupby("mkey"):
        cm = co[co["mkey"].eq(mk) & co["pscore"].between(lo, hi)]
        if len(cm) < k:
            continue
        nn = NearestNeighbors(n_neighbors=k).fit(cm[["logit"]].to_numpy())
        dist, idx = nn.kneighbors(tm[["logit"]].to_numpy())
        for i in range(len(tm)):
            ok = dist[i] <= cal
            if not ok.any():
                continue
            m = cm.iloc[idx[i][ok]]
            rec = {"contractid": tm.iloc[i]["contractid"], "month": mk, "pscore": tm.iloc[i]["pscore"], "n_matched": int(ok.sum())}
            for oc in outcome_cols:
                rec[f"t_{oc}"] = float(tm.iloc[i][oc]) if pd.notna(tm.iloc[i][oc]) else np.nan
                rec[f"c_{oc}"] = float(m[oc].mean())
            for f in num:
                rec[f"t_{f}"] = tm.iloc[i][f]; rec[f"c_{f}"] = m[f].mean()
            pairs.append(rec)
    pairs = pd.DataFrame(pairs)
    res = {"n_treated_all": n_treated_all, "n_treated_matched": int(len(pairs)), "retention": len(pairs) / max(1, n_treated_all),
           "n_controls_available": int(len(co)), "common_support": (float(lo), float(hi))}
    rng = np.random.default_rng(seed)
    if len(pairs):
        for oc in outcome_cols:
            diff = (pairs[f"t_{oc}"] - pairs[f"c_{oc}"])
            valid = diff.notna()
            sub = pd.DataFrame({"d": diff[valid].to_numpy(), "c": pairs.loc[valid, "contractid"].to_numpy()})
            att = float(sub["d"].mean())
            groups = sub.groupby("c").indices; keys = np.array(list(groups.keys()), dtype=object); dvals = sub["d"].to_numpy()
            draws = []
            for _ in range(n_boot):
                pick = rng.choice(len(keys), size=len(keys), replace=True)
                rows = np.concatenate([groups[keys[j]] for j in pick])
                draws.append(dvals[rows].mean())
            res[f"att_{oc}"] = att; res[f"att_{oc}_ci_low"] = float(np.quantile(draws, 0.025)); res[f"att_{oc}_ci_high"] = float(np.quantile(draws, 0.975))
            res[f"treated_mean_{oc}"] = float(pairs.loc[valid, f"t_{oc}"].mean()); res[f"control_mean_{oc}"] = float(pairs.loc[valid, f"c_{oc}"].mean())
        bal = pd.DataFrame([{"feature": f, "smd_before": smd(tr[f], co[f]), "smd_after": smd(pairs[f"t_{f}"], pairs[f"c_{f}"])} for f in num])
    else:
        bal = pd.DataFrame(columns=["feature", "smd_before", "smd_after"])
    return res, pairs, bal, d


# ---------------------------------------------------------------------------
# 4. Budget allocation
# ---------------------------------------------------------------------------
def allocate_budget(candidates, monthly_budget=MONTHLY_BUDGET, holdout_share=0.0):
    """candidates: DataFrame with programme, incremental_cash_per_contact_usd, ci_low_per_contact_usd,
    cost_per_contact_usd, monthly_capacity_contacts. Programmes that clear the bar
    (interval above zero and cash above cost) are funded cheapest-cash-per-dollar first
    to capacity; the remainder goes to a randomised learning line."""
    c = candidates.copy()
    c["cash_per_dollar"] = c["incremental_cash_per_contact_usd"] / c["cost_per_contact_usd"]
    c["qualified"] = (c["ci_low_per_contact_usd"] > 0) & (c["incremental_cash_per_contact_usd"] > c["cost_per_contact_usd"])
    rows = []; remaining = float(monthly_budget)
    for r in c[c["qualified"]].sort_values("cash_per_dollar", ascending=False).itertuples():
        cap = float(r.monthly_capacity_contacts) * float(r.cost_per_contact_usd) * (1 - holdout_share)
        spend = min(remaining, cap)
        contacts = spend / float(r.cost_per_contact_usd)
        gross = contacts * float(r.incremental_cash_per_contact_usd)
        rows.append({"line": r.programme, "allocation_usd": spend, "contacts_per_month": contacts,
                     "expected_incremental_cash_usd": gross, "expected_net_usd": gross - spend,
                     "basis": "clears evidence bar; funded by cash per dollar to capacity" + (f" (holding out {holdout_share:.0%})" if holdout_share else "")})
        remaining -= spend
    if remaining > 1e-9:
        rows.append({"line": "randomised learning reserve", "allocation_usd": remaining, "contacts_per_month": np.nan,
                     "expected_incremental_cash_usd": np.nan, "expected_net_usd": np.nan,
                     "basis": "no further qualified capacity; spend on a randomised test to earn the evidence"})
    return pd.DataFrame(rows)



# %% colab={"base_uri": "https://localhost:8080/"} id="757e659e" outputId="3c0c5ca7-dafa-4978-89a5-15daf708a489"
import importlib, final_forecast_core as C, final_pilot_core as P
importlib.reload(C); importlib.reload(P)
from final_forecast_core import ESTIMATION_END, VALIDATION_END
print("pilot window", P.PILOT_START.date(), "to", VALIDATION_END.date(), "| pre window", P.PRE_START.date(), "to", P.PRE_END.date())

# %% [markdown] id="a6468e5c"
# ## Load the frozen feature outputs and the outreach log, and stop on leakage

# %% colab={"base_uri": "https://localhost:8080/", "height": 218} id="a8679b8c" outputId="d0a59419-1f52-46ce-8534-ee0bc037d730"
try:
    from google.colab import files
    up = files.upload()
    zip_name = [n for n in up if n.lower().endswith(".zip")][0]
    ZIP_BYTES = io.BytesIO(up[zip_name])
except ImportError:
    hit = [Path(p) for p in ["dlight_feature_engineering_outputs_v3.zip", "/mnt/user-data/uploads/dlight_feature_engineering_outputs_v3.zip"] if Path(p).exists()]
    if not hit:
        raise FileNotFoundError("Place dlight_feature_engineering_outputs_v3.zip beside this notebook")
    ZIP_BYTES = hit[0]

PANEL_COLS = ["contractid", "sales_month", "region", "contract_type", "product", "payment_frequency", "price_usd", "perc_deposit",
              "daily_amount_usd", "tenor_length", "month", "months_on_book", "deposit_usd", "expected_cash_due_this_month_usd",
              "expected_cumulative_cash_before_month_usd", "within_scheduled_tenor_proxy", "post_tenor_recovery_proxy", "scheduled_status",
              "target_payment_usd", "actual_cumulative_cash_through_month_usd", "actual_cumulative_cash_before_month_usd",
              "payment_lag_1m", "payment_lag_2m", "payment_lag_3m", "payment_trailing_3m_sum", "payment_trailing_6m_sum",
              "prior_paying_months", "zero_payment_months_prior_3", "zero_payment_months_prior_6", "months_since_last_positive_payment",
              "calls_prior_3m", "tickets_prior_3m", "calls_cumulative_before_month", "tickets_cumulative_before_month"]
PILOT_COLS = ["contractid", "contact_month", "region_outreach", "channel", "attempts", "reached", "cost_usd", "payment_lag_1m", "zero_payment_months_prior_3"]
t0 = time.time()
with zipfile.ZipFile(ZIP_BYTES, "r") as zf:
    names = zf.namelist()
    _find = lambda key: [n for n in names if key in n][0]
    panel = pd.read_csv(zf.open(_find("contract_month_features_v3")), usecols=PANEL_COLS, parse_dates=["sales_month", "month"])
    pilot = pd.read_csv(zf.open(_find("pilot_analysis_features_v3")), usecols=PILOT_COLS, parse_dates=["contact_month"])
    region_month = pd.read_csv(zf.open(_find("region_month_features_v3")), parse_dates=["month"])
for c in ["within_scheduled_tenor_proxy", "post_tenor_recovery_proxy"]:
    panel[c] = panel[c].astype(str).str.lower().isin(["true", "1"])
pilot["reached"] = pilot["reached"].astype(str).str.lower().isin(["true", "1"])
C.assert_no_sealed(panel, pilot, region_month)
panel = C.rebuild_panel_schedule(panel)
print("loaded and rebuilt in", round(time.time() - t0, 1), "s; panel", panel.shape, "| outreach rows", len(pilot))
qa_in = pd.DataFrame([
    {"check": "no_rows_after_jun_2026", "status": "PASS" if panel["month"].max() <= VALIDATION_END and pilot["contact_month"].max() <= VALIDATION_END else "FAIL"},
    {"check": "outreach_regions_are_east_and_west_only", "status": "PASS" if set(pilot["region_outreach"].unique()) == {"East", "West"} else "FAIL"},
    {"check": "outreach_months_are_apr_may_jun_2026", "status": "PASS" if set(pilot["contact_month"].unique()) == set(P.CONTACT_MONTHS_SAME) else "FAIL"},
])
display(qa_in)
if qa_in["status"].ne("PASS").any():
    raise AssertionError("Input QA failed")

# %% [markdown] id="701e579c"
# ## What the pilots actually did, and what they cost
#
# Coverage decides which lens can be trusted. East messaged about 95% of its active financed customers every month at $0.08 per customer; West called about 60% of its customers at $2.22 per customer, reaching 62% of them. Between them the pilots were spending about $19,700 a month; the October budget of $8,000 is a cut of about 60%.

# %% colab={"base_uri": "https://localhost:8080/", "height": 519} id="9f7f785a" outputId="1661daa5-8c3b-4e0c-9f33-cbe8a6e64316"
coverage = P.pilot_coverage(panel, pilot)
display(coverage.round(3))
spend = coverage.groupby("region").agg(avg_monthly_cost_usd=("cost_usd", "mean"), avg_contacts=("contacted", "mean"), cost_per_contact_usd=("cost_per_contact_usd", "mean"), share_contacted=("share_contacted", "mean"))
spend.loc["both pilots"] = [spend["avg_monthly_cost_usd"].sum(), spend["avg_contacts"].sum(), np.nan, np.nan]
spend["vs_8000_budget"] = spend["avg_monthly_cost_usd"] / 8000
display(spend.round(3))
coverage.to_csv(OUTPUT_DIR / "pilot_coverage_and_cost.csv", index=False)

# %% [markdown] id="cefa16ec"
# ## Lens 1: difference-in-differences at fixed age
#
# The curve is the Part 1 curve: fitted through March 2026 by region, product group and months on book, with North's March 2026 anomaly excluded. The residual for a contract-month is paid minus predicted. The event-study plot shows the monthly gap between each pilot region and the controls; it should sit near zero before April and move after. North is the primary control; North plus South (without South's Gen 2 customers) is the robustness control.

# %% colab={"base_uri": "https://localhost:8080/", "height": 959} id="1602e349" outputId="1ffbc9ee-6749-4805-c178-c93a33bdc86f"
curves0 = C.fit_curves(panel, ESTIMATION_END)
idx0, _ = C.calendar_index(panel, curves0, VALIDATION_END)
OUTLIERS = C.flag_outlier_region_months(idx0)
curves = C.fit_curves(panel, ESTIMATION_END, outlier_keys=OUTLIERS)
H = P.residual_frame(panel, curves, C.lookup_eff, outlier_keys=OUTLIERS)
print("scheduled contract-months in the DiD frame:", f"{len(H):,}", "| outliers excluded:", [(r, m.date().isoformat()) for r, m in OUTLIERS])

did_rows, es_tables = [], {}
t0 = time.time()
for reg, prog in [("East", "East preventative SMS"), ("West", "West outbound calls")]:
    for ctrl in [["North"], ["North", "South"]]:
        r = P.did_regression(H, reg, ctrl, n_boot=N_BOOT)
        r["programme"] = prog
        did_rows.append(r)
    es = P.event_study(H, reg, ["North", "South"])
    es_tables[reg] = es
    pl = P.placebo_did(H, reg, ["North", "South"], n_boot=100)
    did_rows.append({"programme": prog, "treated_region": reg, "controls": "placebo: fake April in January, pre-period only",
                     "did_usd_per_contract_month": pl["did_usd_per_contract_month"], "ci_low": pl["ci_low"], "ci_high": pl["ci_high"],
                     "uplift_pct_of_predicted": np.nan, "treated_post_contract_months": np.nan, "rows": pl["rows"], "contracts": pl["contracts"]})
did_summary = pd.DataFrame(did_rows)
did_summary["pretrend_slope_usd_per_month"] = did_summary["treated_region"].map({r: P.pretrend_slope(es_tables[r]) for r in es_tables})
display(did_summary[["programme", "controls", "did_usd_per_contract_month", "ci_low", "ci_high", "uplift_pct_of_predicted", "treated_post_contract_months", "pretrend_slope_usd_per_month"]].round(3))
did_summary.to_csv(OUTPUT_DIR / "did_summary.csv", index=False)
print("done in", round(time.time() - t0, 1), "s")

fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for i, reg in enumerate(["East", "West"]):
    es = es_tables[reg]
    xs = np.arange(len(es)); labs = [d.strftime("%b %y") for d in es.index]
    ax[i].errorbar(xs, es["gap"], yerr=1.96 * es["gap_se"], marker="o", capsize=3)
    ax[i].axhline(0, color="black", lw=0.8); ax[i].axvline(xs[list(es.index).index(pd.Timestamp("2026-04-30"))] - 0.5, color="grey", ls="--", lw=0.8)
    ax[i].set_xticks(xs); ax[i].set_xticklabels(labs); ax[i].set_title(f"{reg} minus controls: residual cash per contract-month (USD), pilots start Apr 26")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "fig_event_study.png", dpi=160, bbox_inches="tight"); plt.show()
pd.concat({k: v for k, v in es_tables.items()}, names=["region", "month"]).to_csv(OUTPUT_DIR / "event_study_gaps.csv")

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="rnKzTVV9pXYG" outputId="807b59c3-ef95-4ca5-825b-53064f671cdb"
# Creates:
# 1. slide_08_did_evidence.png / .svg
# 2. model_validation_progress.png / .svg
# 3. slide_visuals.zip

from pathlib import Path
import zipfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUTPUT_DIR = Path("/content/slide_visuals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Shared styling
# ------------------------------------------------------------------

NAVY = "#172033"
BLUE = "#1F77B4"
GREEN = "#169C8C"
PURPLE = "#7C4DBE"
ORANGE = "#F28E2B"
GREY = "#7B8494"
LIGHT_GREY = "#DFE5EC"
RED = "#D62728"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 24,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "axes.labelcolor": NAVY,
    "xtick.color": "#596579",
    "ytick.color": "#596579",
    "text.color": NAVY,
})


def save_figure(fig, filename):
    png_path = OUTPUT_DIR / f"{filename}.png"
    svg_path = OUTPUT_DIR / f"{filename}.svg"

    fig.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white"
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white"
    )

    return png_path, svg_path


# ==================================================================
# FIGURE 1: DIFFERENCE-IN-DIFFERENCES
# ==================================================================

months = np.arange(6)
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

# Indexed values used to explain the method.
comparison_observed = np.array([60, 62, 64, 66, 68, 70])
sms_counterfactual = np.array([68, 70, 72, 74, 76, 78])

# The post-intervention values illustrate the estimated 11% uplift.
sms_observed = np.array([
    68,
    70,
    72,
    74 * 1.11,
    76 * 1.11,
    78 * 1.11
])

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white")

# Pre-intervention and post-intervention backgrounds
ax.axvspan(-0.35, 2.5, color="#F4F7FA", zorder=0)
ax.axvspan(2.5, 5.35, color="#FFF5F4", zorder=0)

# Intervention line
ax.axvline(
    2.5,
    color=RED,
    linewidth=3,
    linestyle=(0, (6, 5)),
    zorder=2
)

ax.text(
    2.5,
    96.5,
    "INTERVENTION STARTS",
    ha="center",
    va="center",
    color="white",
    fontsize=12,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.45",
        facecolor=RED,
        edgecolor=RED
    )
)

# Observed comparison group
ax.plot(
    months,
    comparison_observed,
    color=GREY,
    linewidth=3.5,
    marker="o",
    markersize=8,
    label="Observed: comparison regions",
    zorder=3
)

# Observed SMS region
ax.plot(
    months,
    sms_observed,
    color=BLUE,
    linewidth=4,
    marker="o",
    markersize=9,
    label="Observed: SMS region",
    zorder=4
)

# Counterfactual begins from the final pre-intervention point
ax.plot(
    months[2:],
    sms_counterfactual[2:],
    color=BLUE,
    linewidth=3,
    linestyle=(0, (7, 5)),
    marker="o",
    markerfacecolor="white",
    markeredgewidth=2,
    label="Unobserved counterfactual",
    zorder=3
)

# Pre-intervention difference
ax.annotate(
    "",
    xy=(1.15, sms_counterfactual[1]),
    xytext=(1.15, comparison_observed[1]),
    arrowprops=dict(
        arrowstyle="<->",
        color=NAVY,
        linewidth=1.8
    )
)

ax.text(
    1.27,
    np.mean([sms_counterfactual[1], comparison_observed[1]]),
    "Constant difference\nin outcome",
    va="center",
    fontsize=12,
    color=NAVY,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor=LIGHT_GREY
    )
)

# Intervention effect
final_x = months[-1]
final_observed = sms_observed[-1]
final_counterfactual = sms_counterfactual[-1]

ax.annotate(
    "",
    xy=(final_x + 0.10, final_observed),
    xytext=(final_x + 0.10, final_counterfactual),
    arrowprops=dict(
        arrowstyle="<->",
        color=RED,
        linewidth=2.7
    )
)

ax.text(
    final_x - 0.08,
    np.mean([final_observed, final_counterfactual]),
    "Intervention effect\n+11%",
    ha="right",
    va="center",
    color=RED,
    fontsize=13,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="#FFF1F0",
        edgecolor=RED
    )
)

# Direct endpoint labels
ax.text(
    5.18,
    sms_observed[-1] + 0.6,
    "Observed outcome",
    color=BLUE,
    fontsize=12,
    fontweight="bold",
    va="center"
)

ax.text(
    5.18,
    sms_counterfactual[-1] - 0.6,
    "Unobserved counterfactual",
    color=BLUE,
    fontsize=12,
    va="center"
)

# Section labels
ax.text(
    1.0,
    94,
    "PRE-INTERVENTION",
    ha="center",
    fontsize=14,
    fontweight="bold"
)

ax.text(
    4.0,
    94,
    "POST-INTERVENTION",
    ha="center",
    fontsize=14,
    fontweight="bold"
)

ax.set_title(
    "Why we believe the SMS result",
    loc="left",
    pad=28
)

ax.text(
    0,
    1.025,
    "Difference-in-differences: repayment before and after the intervention",
    transform=ax.transAxes,
    fontsize=15,
    color="#566174"
)

ax.set_ylabel("Repayment outcome (indexed)")
ax.set_xlabel("Month")
ax.set_xticks(months)
ax.set_xticklabels(month_labels)
ax.set_xlim(-0.35, 5.55)
ax.set_ylim(56, 98)

ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.9)
ax.grid(axis="x", visible=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#263247")
ax.spines["bottom"].set_color("#263247")

handles, labels = ax.get_legend_handles_labels()
order = [1, 0, 2]

ax.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=3,
    frameon=False,
    fontsize=12
)

fig.text(
    0.99,
    0.015,
    "Figure 1. Illustrative indexed path. The +11% label is the estimated regional effect.",
    ha="right",
    fontsize=10,
    color="#687386"
)

fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.94])

did_files = save_figure(fig, "slide_08_did_evidence")
plt.show()


# ==================================================================
# FIGURE 2: MODEL VALIDATION PROGRESS
# ==================================================================

origins = [
    "2025-06-30",
    "2025-09-30",
    "2025-12-31",
    "2026-03-31"
]

model_results = {
    "cohort": {
        "wape": [1.5, 3.5, 6.6, 2.8],
        "color": BLUE,
        "marker": "o"
    },
    "ridge_challenger": {
        "wape": [1.5, 4.2, 6.7, 1.2],
        "color": GREEN,
        "marker": "^"
    },
    "holt_damped": {
        "wape": [6.3, 9.7, 12.1, 15.6],
        "color": PURPLE,
        "marker": "s"
    },
    "naive": {
        "wape": [12.4, 11.5, 8.4, 5.6],
        "color": ORANGE,
        "marker": "D"
    }
}

latest_results = [
    ["ridge_challenger", "$5,106", "1.2%", "+0.7%"],
    ["cohort", "$12,438", "2.8%", "+2.8%"],
    ["naive", "$24,687", "5.6%", "-1.3%"],
    ["holt_damped", "$68,777", "15.6%", "+15.6%"]
]

average_wape = {
    name: np.mean(values["wape"])
    for name, values in model_results.items()
}

fig = plt.figure(figsize=(16, 9), facecolor="white")
grid = fig.add_gridspec(
    1,
    2,
    width_ratios=[3.4, 1.45],
    wspace=0.12
)

ax = fig.add_subplot(grid[0, 0])
table_ax = fig.add_subplot(grid[0, 1])
table_ax.axis("off")

x = np.arange(len(origins))

label_offsets = {
    "cohort": [(0, 13), (0, 13), (-14, 14), (0, 13)],
    "ridge_challenger": [(0, -22), (0, -23), (15, -22), (0, -22)],
    "holt_damped": [(0, 13), (0, 13), (0, 13), (0, 13)],
    "naive": [(0, 13), (0, 13), (0, 13), (0, 13)]
}

for model_name, model in model_results.items():
    values = model["wape"]

    ax.plot(
        x,
        values,
        color=model["color"],
        marker=model["marker"],
        markersize=9,
        linewidth=3.2,
        label=model_name
    )

    for i, value in enumerate(values):
        x_offset, y_offset = label_offsets[model_name][i]

        ax.annotate(
            f"{value:.1f}%",
            (x[i], value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=model["color"]
        )

ax.set_title(
    "Rolling forecast performance by model",
    loc="left",
    pad=26
)

ax.text(
    0,
    1.025,
    "WAPE at each historical forecast origin. Lower is better.",
    transform=ax.transAxes,
    fontsize=15,
    color="#566174"
)

ax.set_ylabel("WAPE")
ax.set_xlabel("Historical forecast origin")
ax.set_xticks(x)
ax.set_xticklabels(origins)
ax.set_ylim(0, 17.5)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.9)
ax.grid(axis="x", visible=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#263247")
ax.spines["bottom"].set_color("#263247")

ax.legend(
    title="Exact model names",
    loc="upper left",
    frameon=False,
    ncol=2,
    fontsize=11,
    title_fontsize=11
)

# Latest-origin table
table_ax.text(
    0.0,
    0.93,
    "Latest origin: 2026-03-31",
    fontsize=17,
    fontweight="bold",
    transform=table_ax.transAxes
)

table_ax.text(
    0.0,
    0.885,
    "MAE, WAPE and forecast bias",
    fontsize=12,
    color="#566174",
    transform=table_ax.transAxes
)

table = table_ax.table(
    cellText=latest_results,
    colLabels=["Model", "MAE", "WAPE", "Bias"],
    cellLoc="left",
    colLoc="left",
    bbox=[0.0, 0.43, 1.0, 0.39],
    colWidths=[0.46, 0.21, 0.17, 0.16]
)

table.auto_set_font_size(False)
table.set_fontsize(10.5)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#D6DDE7")
    cell.set_linewidth(0.8)

    if row == 0:
        cell.set_facecolor(NAVY)
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    else:
        cell.set_facecolor("#F7F9FC" if row % 2 else "white")

# Average WAPE summary
table_ax.text(
    0.0,
    0.34,
    "Average WAPE across four tests",
    fontsize=15,
    fontweight="bold",
    transform=table_ax.transAxes
)

average_lines = [
    f"ridge_challenger   {average_wape['ridge_challenger']:.1f}%",
    f"cohort                    {average_wape['cohort']:.1f}%",
    f"naive                     {average_wape['naive']:.1f}%",
    f"holt_damped          {average_wape['holt_damped']:.1f}%"
]

for i, line in enumerate(average_lines):
    table_ax.text(
        0.02,
        0.285 - (i * 0.052),
        line,
        fontsize=11.5,
        family="monospace",
        transform=table_ax.transAxes
    )

table_ax.text(
    0.0,
    0.035,
    "Positive bias means the forecast was above actual collections.",
    fontsize=9.5,
    color="#687386",
    wrap=True,
    transform=table_ax.transAxes
)

fig.text(
    0.99,
    0.015,
    "Figure 2. Four forward historical tests. Values on the lines are WAPE.",
    ha="right",
    fontsize=10,
    color="#687386"
)

fig.subplots_adjust(
    top=0.86,
    bottom=0.14,
    left=0.07,
    right=0.98
)

model_files = save_figure(fig, "model_validation_progress")
plt.show()


# ==================================================================
# ZIP AND DOWNLOAD
# ==================================================================

all_files = [*did_files, *model_files]
zip_path = OUTPUT_DIR / "slide_visuals.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for file_path in all_files:
        archive.write(file_path, arcname=file_path.name)

print("Created:")
for file_path in all_files:
    print(" -", file_path)

print(" -", zip_path)

try:
    from google.colab import files
    files.download(str(zip_path))
except ImportError:
    print(f"\nDownload the files manually from: {OUTPUT_DIR}")

# %% [markdown] id="35715fd3"
# ## Lens 2: within-region matching
#
# For each pilot region and contact month, contacted customers are matched (1 to 3 nearest neighbours on the propensity score, within a caliper, on common support) to customers in the same region who were never contacted by that pilot. Three outcomes are reported: same-month cash, next-month cash (April and May contacts only, since July is sealed), and the same-month residual above the curve. A placebo on the previous month's payment (which the outreach cannot have caused) shows how well the match balanced payment history. The retention line is the share of contacted customers for whom a comparable control existed; where it is tiny the estimate describes an odd corner of the programme, not the programme.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="8aaa8895" outputId="7fb5d444-5da0-482e-c907-d98bf03cc09a"
match_rows, balances, pair_tables = [], {}, {}
t0 = time.time()
for reg, prog in [("West", "West outbound calls"), ("East", "East preventative SMS")]:
    d = P.build_matching_frame(panel, pilot, reg, P.CONTACT_MONTHS_SAME)
    d["resid_same"] = d["target_payment_usd"] - d["due_dc"] * C.lookup_eff(curves, d["region"], d["pgroup"], d["months_on_book"])
    d.loc[d["month"].eq(pd.Timestamp("2026-06-30")), "next_month_cash"] = np.nan
    res, pairs, bal, dd = P.match_region(d, ["same_month_cash", "next_month_cash", "resid_same", "payment_lag_1m"], k=3, n_boot=N_BOOT)
    res["programme"] = prog; res["region"] = reg
    res["never_contacted_mean_same_month_cash"] = float(d.loc[d["treated"].eq(0), "same_month_cash"].mean())
    res["contacted_mean_same_month_cash"] = float(d.loc[d["treated"].eq(1), "same_month_cash"].mean())
    match_rows.append(res); balances[reg] = bal; pair_tables[reg] = pairs
    print(reg, "done in", round(time.time() - t0, 1), "s")
matching = pd.DataFrame(match_rows)
show_cols = ["programme", "n_treated_all", "n_treated_matched", "retention", "n_controls_available", "never_contacted_mean_same_month_cash",
             "att_same_month_cash", "att_same_month_cash_ci_low", "att_same_month_cash_ci_high",
             "att_next_month_cash", "att_next_month_cash_ci_low", "att_next_month_cash_ci_high",
             "att_resid_same", "att_payment_lag_1m", "att_payment_lag_1m_ci_low", "att_payment_lag_1m_ci_high"]
display(matching[show_cols].round(3).T)
matching.to_csv(OUTPUT_DIR / "matching_summary.csv", index=False)
for reg in balances:
    balances[reg].to_csv(OUTPUT_DIR / f"matching_balance_{reg.lower()}.csv", index=False)
    pair_tables[reg].to_csv(OUTPUT_DIR / f"matched_pairs_{reg.lower()}.csv", index=False)
print("West balance after matching (standardised mean differences; under 0.1 is good, under 0.25 acceptable):")
display(balances["West"].round(3))

# %% [markdown] id="1c2e6a85"
# ### Why East cannot be matched
#
# The 5% of East customers who were never messaged are not a comparison group: on average they paid nothing in the pilot months. A matched estimate for East therefore compares messaged customers with accounts that were already dead, and it is not used anywhere in the recommendation. The 24x return for East in the earlier draft came from exactly this comparison.

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="d836b2a4" outputId="b7d46e6a-3c14-4f04-fc94-d0ecf372d0a8"
east_ctrl = P.build_matching_frame(panel, pilot, "East", P.CONTACT_MONTHS_SAME)
east_ctrl = east_ctrl[east_ctrl["treated"].eq(0)]
east_profile = pd.DataFrame({
    "never_messaged_east_customers": [east_ctrl["contractid"].nunique()],
    "share_paying_zero_in_month": [float((east_ctrl["same_month_cash"] == 0).mean())],
    "mean_months_since_last_payment": [float(east_ctrl["months_since_last_positive_payment"].mean())],
    "mean_zero_payment_months_prior_6": [float(east_ctrl["zero_payment_months_prior_6"].mean())],
    "mean_months_on_book": [float(east_ctrl["months_on_book"].mean())],
})
display(east_profile.round(3).T)
east_profile.to_csv(OUTPUT_DIR / "east_never_messaged_profile.csv", index=False)

# %% [markdown] id="d3841df7"
# ## Putting the two lenses side by side, and turning them into money
#
# The DiD gives incremental dollars per active contract-month in the region; multiplied by the region's active scheduled contract-months and divided by the contacts made, it becomes incremental cash per contact, which sits directly beside the cost per contact. The matched ATT for West is the effect on the customers who were actually called and is the natural upper bound on what better targeting could achieve.

# %% colab={"base_uri": "https://localhost:8080/", "height": 646} id="24e2638b" outputId="541e74ac-76b7-4a0d-952e-202870dac550"
econ = []
for reg, prog in [("East", "East preventative SMS"), ("West", "West outbound calls")]:
    dd_row = did_summary[(did_summary["treated_region"].eq(reg)) & did_summary["controls"].eq("North")].iloc[0]
    cov = coverage[coverage["region"].eq(reg)]
    contacts_pm = float(cov["contacted"].mean()); cost_pm = float(cov["cost_usd"].mean()); cpc = cost_pm / contacts_pm
    cm_pm = float(dd_row["treated_post_contract_months"]) / 3.0
    inc_pm = float(dd_row["did_usd_per_contract_month"]) * cm_pm
    inc_lo = float(dd_row["ci_low"]) * cm_pm; inc_hi = float(dd_row["ci_high"]) * cm_pm
    m = matching[matching["region"].eq(reg)].iloc[0]
    econ.append({"programme": prog, "monthly_cost_usd": cost_pm, "contacts_per_month": contacts_pm, "cost_per_contact_usd": cpc,
                 "did_uplift_pct_scheduled_cash": float(dd_row["uplift_pct_of_predicted"]),
                 "did_incremental_cash_per_month_usd": inc_pm, "did_incremental_ci_low": inc_lo, "did_incremental_ci_high": inc_hi,
                 "did_incremental_per_contact_usd": inc_pm / contacts_pm, "did_per_contact_ci_low": inc_lo / contacts_pm,
                 "did_cash_per_dollar": inc_pm / cost_pm, "did_net_per_month_usd": inc_pm - cost_pm,
                 "matched_att_same_month_per_contact_usd": float(m["att_same_month_cash"]),
                 "matched_att_next_month_per_contact_usd": float(m["att_next_month_cash"]),
                 "matched_retention": float(m["retention"]),
                 "matching_credible": bool(m["retention"] > 0.5 and m["never_contacted_mean_same_month_cash"] > 0.5)})
economics = pd.DataFrame(econ)
display(economics.round(3).T)
economics.to_csv(OUTPUT_DIR / "pilot_economics.csv", index=False)

verdict = []
for r in economics.itertuples():
    did_pos = r.did_per_contact_ci_low > 0
    pays_back = r.did_incremental_per_contact_usd > r.cost_per_contact_usd
    lenses_agree = (not r.matching_credible) or (np.sign(r.matched_att_same_month_per_contact_usd) == np.sign(r.did_incremental_per_contact_usd))
    if did_pos and pays_back and lenses_agree:
        v = "Effect is real and pays for itself: scale, with a randomised holdout as it expands"
    elif did_pos and not pays_back:
        v = "Effect is real but does not cover its cost as run: keep only a targeted, tested slice"
    else:
        v = "Not proven: test before spending"
    verdict.append({"programme": r.programme, "did_interval_above_zero": did_pos, "pays_back_its_cost": pays_back,
                    "matching_lens_credible": r.matching_credible, "lenses_agree_in_sign": lenses_agree, "verdict": v})
verdict = pd.DataFrame(verdict)
display(verdict)
verdict.to_csv(OUTPUT_DIR / "pilot_verdict.csv", index=False)

# %% [markdown] id="36fa6eec"
# ## The October budget: $8,000 a month
#
# The rule: fund what has cleared the evidence bar, cheapest cash per dollar first, up to observed capacity; do not fund a programme that loses money as run; spend the rest on buying the evidence that would unlock the next dollar, which here means extending SMS to the other regions as a randomised rollout and testing a targeted version of the calls with a held-out control. Every line carries an expected value with a low and high, and the lines that rest on untested transfer are labelled as such.

# %% colab={"base_uri": "https://localhost:8080/", "height": 700} id="8d5d3f36" outputId="4f6550e3-22a1-488c-f055-865bc2174d25"
east = economics[economics["programme"].eq("East preventative SMS")].iloc[0]
west = economics[economics["programme"].eq("West outbound calls")].iloc[0]
active = panel[(panel["month"] == VALIDATION_END) & panel["contract_type"].eq("FINANCED") & (panel["months_on_book"] >= 1)]
active_by_region = active.groupby("region")["contractid"].nunique()
sms_cpc = float(east["cost_per_contact_usd"]); call_cpc = float(west["cost_per_contact_usd"])
other_regions = ["North", "South", "West"]
n_other = int(active_by_region[other_regions].sum())

# Line 1: continue East SMS at full coverage (qualified)
east_alloc = float(active_by_region["East"]) * sms_cpc
east_ev = float(east["did_incremental_per_contact_usd"]) * float(active_by_region["East"])
east_lo = float(east["did_per_contact_ci_low"]) * float(active_by_region["East"])
# Line 2: SMS in North, South and West at 80% coverage with a 20% customer-level random holdout (learning line with value)
SMS_COVER = 0.80
sms_roll_alloc = SMS_COVER * n_other * sms_cpc
transfer = {"low": 0.0, "base": 0.5, "high": 1.0}   # share of the East per-contact effect that transfers to other regions
sms_roll_ev = {k: v * float(east["did_incremental_per_contact_usd"]) * SMS_COVER * n_other for k, v in transfer.items()}
# Line 3: a small targeted test of West calls (highest unpaid balance among recently lapsed customers) with a 20% holdout
calls_budget = 2000.0
calls_contacts = calls_budget / call_cpc
call_ev = {"low": 0.0, "base": float(west["matched_att_same_month_per_contact_usd"]) * calls_contacts,
           "high": 2.0 * float(west["matched_att_same_month_per_contact_usd"]) * calls_contacts}
reserve = 8000.0 - east_alloc - sms_roll_alloc - calls_budget
budget = pd.DataFrame([
    {"line": "East preventative SMS, all active financed customers", "allocation_usd": east_alloc, "contacts_per_month": float(active_by_region["East"]),
     "expected_incremental_low_usd": east_lo, "expected_incremental_base_usd": east_ev, "expected_incremental_high_usd": float(east["did_incremental_ci_high"]) / float(east["contacts_per_month"]) * float(active_by_region["East"]),
     "evidence": "DiD interval above zero on both controls; pays back six to seven times over"},
    {"line": "SMS in North, South, West at 80% coverage, 20% random holdout for one quarter", "allocation_usd": sms_roll_alloc, "contacts_per_month": SMS_COVER * n_other,
     "expected_incremental_low_usd": sms_roll_ev["low"], "expected_incremental_base_usd": sms_roll_ev["base"], "expected_incremental_high_usd": sms_roll_ev["high"],
     "evidence": "transfer of the East effect is untested; the random holdout answers it within a quarter"},
    {"line": "West calls, small targeted test (high balance, recently lapsed) with 20% holdout", "allocation_usd": calls_budget, "contacts_per_month": calls_contacts,
     "expected_incremental_low_usd": call_ev["low"], "expected_incremental_base_usd": call_ev["base"], "expected_incremental_high_usd": call_ev["high"],
     "evidence": "calls lose money as run ($0.50 back per $2.22); only a targeted slice that beats its cost earns more budget"},
    {"line": "unallocated: hold until the quarter's read, then release to whichever line proved out", "allocation_usd": reserve, "contacts_per_month": np.nan,
     "expected_incremental_low_usd": 0.0, "expected_incremental_base_usd": 0.0, "expected_incremental_high_usd": 0.0, "evidence": "not spending is a valid use of a budget the evidence does not yet justify"},
])
budget["spent_usd"] = np.where(budget["line"].str.startswith("unallocated"), 0.0, budget["allocation_usd"])
for k in ["low", "base", "high"]:
    budget[f"expected_net_{k}_usd"] = budget[f"expected_incremental_{k}_usd"] - budget["spent_usd"]
budget.loc[len(budget)] = {"line": "TOTAL", **{c: budget[c].sum() for c in budget.columns if c not in ["line", "evidence", "contacts_per_month"]}, "contacts_per_month": np.nan, "evidence": ""}
assert abs(budget.loc[budget["line"].eq("TOTAL"), "allocation_usd"].iloc[0] - 8000.0) < 1e-6
display(budget.round(0))
budget.to_csv(OUTPUT_DIR / "october_budget_recommendation.csv", index=False)

# what the two pilots would cost and return if simply continued as run
as_run = pd.DataFrame([
    {"option": "continue both pilots as run", "monthly_cost_usd": float(economics["monthly_cost_usd"].sum()), "did_incremental_usd": float(economics["did_incremental_cash_per_month_usd"].sum())},
    {"option": "East SMS only, as run", "monthly_cost_usd": float(east["monthly_cost_usd"]), "did_incremental_usd": float(east["did_incremental_cash_per_month_usd"])},
    {"option": "West calls only, as run", "monthly_cost_usd": float(west["monthly_cost_usd"]), "did_incremental_usd": float(west["did_incremental_cash_per_month_usd"])},
])
as_run["net_usd"] = as_run["did_incremental_usd"] - as_run["monthly_cost_usd"]; as_run["fits_8000_budget"] = as_run["monthly_cost_usd"] <= 8000
display(as_run.round(0))
as_run.to_csv(OUTPUT_DIR / "pilots_as_run_economics.csv", index=False)

# %% [markdown] id="cc6baf8b"
# ## The proper test, for the discussion at the presentation
#
# Randomise at the customer level inside each region, not at the region level, so the comparison is between like customers in the same market in the same month. Stratify by arrears status (paid last month or not) and months on book, since both drive cash and the calls programme selects on arrears. Hold out 20% of eligible customers from calls and 50% from the first quarter of the SMS rollout. Read the same-month and next-month residual cash above the curve, exactly as here, so the test answers the same question the pilots asked. The minimum detectable effect below is what one quarter of that design can see with the observed noise in monthly residual cash per customer.

# %% colab={"base_uri": "https://localhost:8080/", "height": 459} id="4307f564" outputId="7d843e74-dd0e-4f08-cf82-dd037bfa644d"
sd_resid = float(H.loc[H["post"].eq(1), "resid"].std())
rows = []
for name, n_treated, n_control in [("SMS rollout, one region, 80/20 for one quarter", 0.8 * float(active_by_region["North"]) * 3, 0.2 * float(active_by_region["North"]) * 3),
                                   ("SMS rollout, all three regions pooled, 80/20", 0.8 * n_other * 3, 0.2 * n_other * 3),
                                   ("Targeted calls test with 20% holdout, one quarter", 0.8 * calls_contacts * 3, 0.2 * calls_contacts * 3)]:
    se = sd_resid * np.sqrt(1 / n_treated + 1 / n_control)
    rows.append({"design": name, "treated_contract_months": n_treated, "control_contract_months": n_control, "sd_of_monthly_residual_usd": sd_resid,
                 "minimum_detectable_effect_usd_per_contract_month": 2.8 * se,
                 "compare_with_east_did_usd": float(did_summary.loc[did_summary["treated_region"].eq("East") & did_summary["controls"].eq("North"), "did_usd_per_contract_month"].iloc[0])})
power = pd.DataFrame(rows)
display(power.round(3))
power.to_csv(OUTPUT_DIR / "test_design_power.csv", index=False)

# %% [markdown] id="901e293e"
# ## Decision summary, QA and outputs

# %% colab={"base_uri": "https://localhost:8080/", "height": 887} id="0e99122f" outputId="152b52ef-8197-4160-d0b2-ab2eb8628d17"
summary = pd.DataFrame([
    {"question": "Which pilot should be scaled?", "answer": "East preventative SMS: about +11% scheduled cash in East, interval above zero on both controls, at $0.08 per customer. Extend it to the other regions as a randomised rollout."},
    {"question": "And West calls?", "answer": f"Real but small: about +{float(west['did_uplift_pct_scheduled_cash']):.0%} of scheduled cash at region level, roughly ${float(west['did_incremental_per_contact_usd']):.2f} per contact against a $2.22 cost. Keep a targeted, tested slice only."},
    {"question": "Repayment measure", "answer": "Incremental cash above what the customer's age and product predict; same-month for SMS, next-month for calls; every contact counts, reached or not"},
    {"question": "Why not the matched East number?", "answer": "The 5% of East never messaged paid nothing at all; they are not a comparison group"},
    {"question": "October budget", "answer": f"${east_alloc:,.0f} East SMS; ${sms_roll_alloc:,.0f} SMS in the other three regions with a 20% random holdout; ${calls_budget:,.0f} small targeted calls test with holdout; ${reserve:,.0f} held unallocated until the quarter's read"},
    {"question": "Expected monthly value of the budget", "answer": f"about ${float(budget.loc[budget['line'].eq('TOTAL'), 'spent_usd'].iloc[0]):,.0f} spent for net ${float(budget.loc[budget['line'].eq('TOTAL'), 'expected_net_low_usd'].iloc[0]):,.0f} (low) / ${float(budget.loc[budget['line'].eq('TOTAL'), 'expected_net_base_usd'].iloc[0]):,.0f} (base) / ${float(budget.loc[budget['line'].eq('TOTAL'), 'expected_net_high_usd'].iloc[0]):,.0f} (high) a month; the base assumes half the East effect transfers to the other regions"},
    {"question": "Causal confidence", "answer": "Moderate: two regions per arm, clean pre-trends and placebos, age and product adjusted; a randomised rollout settles it"},
])
display(summary)
summary.to_csv(OUTPUT_DIR / "part2_decision_summary.csv", index=False)

qa = pd.DataFrame([
    {"check": "q3_outcomes_never_loaded", "status": "PASS" if panel["month"].max() <= VALIDATION_END else "FAIL"},
    {"check": "outcome_and_treatment_fields_absent_from_propensity", "status": "PASS" if not set(["target_payment_usd", "next_month_cash", "same_month_cash", "treated", "cost_usd"]).intersection(P.MATCH_NUM + P.MATCH_CAT) else "FAIL"},
    {"check": "budget_sums_to_8000", "status": "PASS"},
    {"check": "east_did_interval_above_zero_both_controls", "status": "PASS" if (did_summary[did_summary["treated_region"].eq("East") & did_summary["controls"].isin(["North", "North+South"])]["ci_low"] > 0).all() else "REVIEW"},
    {"check": "placebo_intervals_include_zero", "status": "PASS" if ((did_summary[did_summary["controls"].str.startswith("placebo")]["ci_low"] <= 0) & (did_summary[did_summary["controls"].str.startswith("placebo")]["ci_high"] >= 0)).all() else "REVIEW"},
])
display(qa)
if qa["status"].eq("FAIL").any():
    raise AssertionError("Part 2 QA failed")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
e = economics.set_index("programme")
x = np.arange(len(e))
ax[0].bar(x - 0.2, e["did_incremental_per_contact_usd"], 0.4, label="incremental cash per contact (DiD)", yerr=[e["did_incremental_per_contact_usd"] - e["did_per_contact_ci_low"], e["did_incremental_per_contact_usd"] * 0 + (e["did_incremental_ci_high"] / e["contacts_per_month"] - e["did_incremental_per_contact_usd"])], capsize=4)
ax[0].bar(x + 0.2, e["cost_per_contact_usd"], 0.4, label="cost per contact", color="#c0504d")
ax[0].set_xticks(x); ax[0].set_xticklabels(e.index); ax[0].set_ylabel("USD per contact"); ax[0].set_title("Does a contact pay for itself?"); ax[0].legend()
b = budget[~budget["line"].eq("TOTAL")].set_index("line")["allocation_usd"]
ax[1].barh([l[:45] for l in b.index], b.values, color="#4f81bd"); ax[1].set_xlabel("USD per month"); ax[1].set_title("October budget, $8,000")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "fig_pilot_economics_and_budget.png", dpi=160, bbox_inches="tight"); plt.show()

zip_path = Path("part2_final_outputs.zip")
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

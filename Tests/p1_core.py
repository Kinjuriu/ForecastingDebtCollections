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
    _dep_den = dep.groupby("region")["deposit_usd"].sum()
    dep_eff = (dep.groupby("region")["target_payment_usd"].sum() / _dep_den.replace(0, np.nan))
    dep_country = (dep["target_payment_usd"].sum() / dep["deposit_usd"].sum()) if dep["deposit_usd"].sum() else 1.0
    cash = h[h["contract_type"].eq("CASH") & (h["months_on_book"] == 0) & (h["price_usd"] > 0)]
    _cash_den = cash.groupby("region")["price_usd"].sum()
    cash_eff = (cash.groupby("region")["target_payment_usd"].sum() / _cash_den.replace(0, np.nan))
    cash_country = (cash["target_payment_usd"].sum() / cash["price_usd"].sum()) if cash["price_usd"].sum() else 1.0
    return {"cutoff": cutoff, "country": country_eff, "region": region_eff, "region_product": rp_eff,
            "deposit_eff": dep_eff.reindex(regions).fillna(dep_country), "cash_eff": cash_eff.reindex(regions).fillna(cash_country),
            "n_rows": int(len(sched))}


def lookup_eff(curves, region, pgroup, mob):
    """Vectorised efficiency lookup with fall-back region -> country."""
    mob = np.clip(np.asarray(mob, dtype=int), 1, MAX_MOB)
    idx = pd.MultiIndex.from_arrays([np.asarray(region), np.asarray(pgroup), mob])
    v = np.array(curves["region_product"].reindex(idx).to_numpy(), dtype=float)
    miss = np.isnan(v)
    if miss.any():
        idx2 = pd.MultiIndex.from_arrays([np.asarray(region)[miss], mob[miss]])
        v2 = np.array(curves["region"].reindex(idx2).to_numpy(), dtype=float)
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
    default_level = float(lv.mean()) if len(lv) and np.isfinite(lv.mean()) else 1.0
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
    lv_vec = pd.Series(reg).map(lv).fillna(default_level).to_numpy()
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
                        v = cnt * float(d[0]) * float(e) * float(lv.get(seg.region, default_level))
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

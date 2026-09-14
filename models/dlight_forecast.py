"""
dlight_forecast.py
==================

Core, tested functions for the collections forecast (Part 1, base
model). This module is the single source of truth: the Colab notebook writes
this exact file with a %%writefile cell and then imports it, and the test file
imports it too, so there is one place where the logic lives and nothing can
quietly drift between the notebook and the tests.

Design rules baked in here, matching the locked feature-engineering conventions:

* Nothing dated after 30 June 2026 is ever accepted; the loader raises if it
  sees a later month, so the sealed July to September test can never leak in,
  and Q3 actuals are never read.
* Outreach is never used in the base forecast; the base holds outreach uplift
  at zero for causal reasons, so this module deliberately has no outreach input.
* The base collection curve is learned from the non-pilot regions only, North
  and South, so the April onward pilot effect in East and West cannot leak into
  the base rates.
* Scenario width is sized from calendar-level shocks, meaning the month-to-month
  wobble in the whole country's collection rate, not from cohort-cell sampling
  error, because contracts sold in the same month share one economy and are not
  independent draws.

Where a non-technical reader would want to follow along, the Google Sheets
equivalent of a calculation is written next to the Python in a comment, using
the form  SHEETS:  so the same number can be rebuilt in a spreadsheet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fixed facts of the case. These are the only hard-coded business numbers.
# ---------------------------------------------------------------------------
DATA_END = pd.Timestamp("2026-06-30")          # development data ends here
ESTIMATION_END = pd.Timestamp("2026-03-31")    # curves are learned up to here
FORECAST_MONTHS = [                            # the three sealed months we predict
    pd.Period("2026-07", "M"),
    pd.Period("2026-08", "M"),
    pd.Period("2026-09", "M"),
]
SALES_PLAN = {                                 # agreed Q3 unit plan
    pd.Period("2026-07", "M"): 3100,
    pd.Period("2026-08", "M"): 3200,
    pd.Period("2026-09", "M"): 3200,
}
PILOT_REGIONS = ["East", "West"]               # ran SMS / call pilots from April
NON_PILOT_REGIONS = ["North", "South"]         # clean regions the base learns from


# ---------------------------------------------------------------------------
# 0. Leakage guard
# ---------------------------------------------------------------------------
def assert_no_leakage(panel: pd.DataFrame,
                      contract_features: pd.DataFrame,
                      data_end: pd.Timestamp = DATA_END) -> bool:
    """Refuse to run if any feature row is dated after the development cut-off.

    This is the single hard stop that protects the sealed Jul to Sep test. If a
    payment month or a sale month lands after 30 June 2026, the outputs ZIP was
    built from the wrong (sealed) data and we must not model on it.
    """
    pm = pd.to_datetime(panel["pay_month"])
    if pm.max() > data_end:
        raise ValueError(
            f"STOP: vintage_panel has pay_month {pm.max().date()} after "
            f"{data_end.date()}. This looks like sealed Q3 data; not loading."
        )
    sm = pd.to_datetime(contract_features["sales_month"])
    if sm.max() > data_end:
        raise ValueError(
            f"STOP: contract_features has sales_month {sm.max().date()} after "
            f"{data_end.date()}. This looks like sealed Q3 data; not loading."
        )
    return True


# ---------------------------------------------------------------------------
# 1. A calendar of month-end day counts, used to bill instalments forward
# ---------------------------------------------------------------------------
def build_days_index(first="2024-01", last="2026-12") -> pd.Series:
    """Cumulative calendar days by month, so billing days between two months is
    a simple subtraction rather than a loop.

    days from month A+1 through month B inclusive = C[B] - C[A]
    SHEETS: put months down a column with their day counts, take a running total
    with =SUM($B$2:B2), then the days in a span are just endTotal - startTotal.
    """
    months = pd.period_range(first, last, freq="M")
    days = pd.Series([p.days_in_month for p in months], index=months, dtype=float)
    return days.cumsum()


def _cum_days_to(period, cum_days: pd.Series) -> float:
    """Cumulative days up to and including a month period; 0 if before the index."""
    if period in cum_days.index:
        return float(cum_days.loc[period])
    if period < cum_days.index[0]:
        return 0.0
    # period after the index: extend flat (should not happen inside our window)
    return float(cum_days.iloc[-1])


# ---------------------------------------------------------------------------
# 2. Learn the marginal monthly collection curve (region-aware, leakage-safe)
# ---------------------------------------------------------------------------
def learn_marginal_curve(panel: pd.DataFrame,
                         estimation_end: pd.Timestamp = ESTIMATION_END,
                         regions=None,
                         keys=("contract_type",)) -> pd.DataFrame:
    """Monthly collection rate by contract type and months on book.

    For each (contract_type, mob) we take, over the estimation window and the
    chosen regions, the money actually paid divided by the money that was due
    that month. That ratio is what we apply to future billing.

        marginal_rate[type, mob] = sum(paid) / sum(expected_this_month)
        SHEETS: =SUMIFS(paid, type, "FINANCED", mob, 3) /
                =SUMIFS(expected_this_month, type, "FINANCED", mob, 3)

    We learn it from the non-pilot regions by default so the April onward pilot
    effect never leaks into the base rate.
    """
    keys = list(keys)
    est = panel[pd.to_datetime(panel["pay_month"]) <= estimation_end].copy()
    if regions is not None:
        est = est[est["region"].isin(regions)]
    est = est[est["expected_this_month"] > 0]
    g = (est.groupby(keys + ["mob"], as_index=False)
            .agg(paid=("paid", "sum"),
                 expected_this_month=("expected_this_month", "sum"),
                 n=("contractid", "nunique")))
    g["marginal_rate"] = g["paid"] / g["expected_this_month"]
    return g


def rate_lookup(curve: pd.DataFrame) -> dict:
    """Turn the curve frame into a {(contract_type, mob): rate} dict with a
    sensible tail: for a months-on-book value beyond what we observed, we reuse
    the last observed rate for that contract type rather than inventing one."""
    lut = {}
    tail = {}
    for ct, sub in curve.sort_values("mob").groupby("contract_type"):
        for _, r in sub.iterrows():
            lut[(ct, int(r["mob"]))] = float(r["marginal_rate"])
        tail[ct] = float(sub.iloc[-1]["marginal_rate"])
        tail[(ct, "max_mob")] = int(sub.iloc[-1]["mob"])
    lut["_tail"] = tail
    return lut


def get_rate(lut: dict, contract_type: str, mob: int) -> float:
    if (contract_type, mob) in lut:
        return lut[(contract_type, mob)]
    tail = lut.get("_tail", {})
    return float(tail.get(contract_type, 0.0))


# ---------------------------------------------------------------------------
# 3. Existing book: project billing forward and apply the curve
# ---------------------------------------------------------------------------
def project_existing_billing(contract_features: pd.DataFrame,
                             cum_days: pd.Series,
                             forecast_months=FORECAST_MONTHS) -> pd.DataFrame:
    """Expected billing for each existing contract in each forecast month.

    Cash contracts were billed in full in their sale month (months on book 0),
    which is already in the past for every existing contract, so they bill zero
    new dollars in July to September; this is why the existing cash book adds
    almost nothing. Financed contracts bill a daily amount times the days in the
    month, with the running instalment total capped at what they still owe, so a
    contract past its tenor bills nothing new.

        instalment_this_month = daily * days_in_month, capped at what remains
        SHEETS: =MIN(daily*daysInMonth, installment_total - billed_so_far)
    """
    c = contract_features.copy()
    c["sales_period"] = pd.PeriodIndex(pd.to_datetime(c["sales_month"]), freq="M")
    rows = []
    prev_period = {p: (p - 1) for p in forecast_months}
    for _, r in c.iterrows():
        ct = r["contract_type"]
        S = r["sales_period"]
        d = float(r.get("daily_amount_usd", 0.0) or 0.0)
        it = float(r.get("installment_total", 0.0) or 0.0)
        for fm in forecast_months:
            mob = (fm - S).n
            if mob < 0:
                continue
            if ct == "CASH":
                exp = 0.0
            else:
                cum_now = d * (_cum_days_to(fm, cum_days) - _cum_days_to(S, cum_days))
                cum_prev = d * (_cum_days_to(prev_period[fm], cum_days) - _cum_days_to(S, cum_days))
                cum_now = min(max(cum_now, 0.0), it)
                cum_prev = min(max(cum_prev, 0.0), it)
                exp = max(cum_now - cum_prev, 0.0)
            rows.append((r["contractid"], ct, r.get("region", "NA"),
                         fm, int(mob), exp))
    out = pd.DataFrame(rows, columns=["contractid", "contract_type", "region",
                                      "forecast_month", "mob", "expected_this_month"])
    return out


def forecast_existing_book(billing: pd.DataFrame, rate_lut: dict) -> pd.DataFrame:
    """Apply the marginal curve to projected billing to get collections.

        collection = expected_this_month * marginal_rate[type, mob]
        SHEETS: =expected_this_month * VLOOKUP(mob, rateTable, 2, FALSE)
    """
    b = billing.copy()
    b["marginal_rate"] = [get_rate(rate_lut, ct, m)
                          for ct, m in zip(b["contract_type"], b["mob"])]
    b["collection"] = b["expected_this_month"] * b["marginal_rate"]
    return b


# ---------------------------------------------------------------------------
# 4. New sales from the plan, run through the early-life curve
# ---------------------------------------------------------------------------
def recent_sales_profile(contract_features: pd.DataFrame,
                         months=("2026-04", "2026-05", "2026-06")) -> dict:
    """Cash/financed mix and average economics from the most recent cohorts, so
    the new-sales forecast reflects what the country has actually been selling."""
    c = contract_features.copy()
    c["sales_period"] = pd.PeriodIndex(pd.to_datetime(c["sales_month"]), freq="M")
    recent = c[c["sales_period"].isin([pd.Period(m, "M") for m in months])]
    if len(recent) == 0:                      # fall back to the whole book
        recent = c
    n = len(recent)
    cash = recent[recent["contract_type"] == "CASH"]
    fin = recent[recent["contract_type"] == "FINANCED"]
    prof = {
        "p_cash": len(cash) / n if n else 0.0,
        "p_fin": len(fin) / n if n else 0.0,
        "cash_price": float(cash["price_usd"].mean()) if len(cash) else 0.0,
        "fin_price": float(fin["price_usd"].mean()) if len(fin) else 0.0,
        "fin_deposit": float(fin["expected_deposit"].mean()) if len(fin) else 0.0,
        "fin_daily": float(fin["daily_amount_usd"].mean()) if len(fin) else 0.0,
    }
    return prof


def forecast_new_sales(profile: dict,
                       rate_lut: dict,
                       cum_days: pd.Series,
                       plan=SALES_PLAN,
                       forecast_months=FORECAST_MONTHS,
                       sales_multiplier: float = 1.0,
                       p_cash_shift: float = 0.0,
                       price_multiplier: float = 1.0) -> pd.DataFrame:
    """Collections in Jul to Sep from units sold in Jul to Sep under the plan.

    Cash units collect their full price in the sale month; financed units collect
    their deposit in the sale month and then early instalments in the following
    months, each run through the marginal curve at the right months on book. The
    optional levers let the tornado and the scenarios move volume, mix and price.
    """
    p_cash = min(max(profile["p_cash"] + p_cash_shift, 0.0), 1.0)
    cash_r0 = get_rate(rate_lut, "CASH", 0)
    fin_r = {m: get_rate(rate_lut, "FINANCED", m) for m in range(0, 6)}
    monthly = {fm: 0.0 for fm in forecast_months}

    for born, vol in plan.items():
        if born not in monthly:
            continue
        vol = vol * sales_multiplier
        n_cash = vol * p_cash
        n_fin = vol * (1.0 - p_cash)
        cash_price = profile["cash_price"] * price_multiplier
        fin_price = profile["fin_price"] * price_multiplier
        fin_deposit = profile["fin_deposit"] * price_multiplier
        fin_daily = profile["fin_daily"]

        # cash: full price in the sale month at the month-0 rate
        monthly[born] += n_cash * cash_price * cash_r0

        # financed: deposit in the sale month, then instalments in later months
        monthly[born] += n_fin * fin_deposit * fin_r[0]
        for k in range(1, 4):
            fm = born + k
            if fm in monthly:
                days = fm.days_in_month
                inst = fin_daily * days
                monthly[fm] += n_fin * inst * fin_r.get(k, fin_r[0])

    return (pd.DataFrame({"forecast_month": list(monthly.keys()),
                          "new_sales_collection": list(monthly.values())})
            .sort_values("forecast_month").reset_index(drop=True))


# ---------------------------------------------------------------------------
# 5. Post-tenor recovery (late and post-payoff cash that still lands)
# ---------------------------------------------------------------------------
def forecast_post_tenor(panel: pd.DataFrame,
                        trailing=("2026-04", "2026-05", "2026-06")) -> float:
    """A flat monthly estimate of cash that arrives on contract-months where no
    billing was due, meaning late catch-up and post-payoff recovery. We take the
    recent three-month average and carry it forward flat.

        recovery = AVERAGE of monthly (paid where nothing was billed)
        SHEETS: =AVERAGEIFS(paid, expected_this_month, 0, month, ">=Apr26")
    """
    p = panel.copy()
    p["pay_period"] = pd.PeriodIndex(pd.to_datetime(p["pay_month"]), freq="M")
    rec = p[(p["expected_this_month"].abs() < 1e-9) & (p["paid"] > 0)]
    by_month = rec.groupby("pay_period")["paid"].sum()
    want = [pd.Period(m, "M") for m in trailing]
    vals = [by_month.get(m, 0.0) for m in want]
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# 6. Naive floors and an ETS cross-check
# ---------------------------------------------------------------------------
def country_monthly_collections(panel: pd.DataFrame) -> pd.Series:
    p = panel.copy()
    p["pay_period"] = pd.PeriodIndex(pd.to_datetime(p["pay_month"]), freq="M")
    return p.groupby("pay_period")["paid"].sum().sort_index()


def naive_baselines(panel: pd.DataFrame,
                    forecast_months=FORECAST_MONTHS) -> dict:
    """Two floors a reader can sanity-check by hand: the recent three-month average,
    and the same three months a year earlier.

    SHEETS 3-month average: =AVERAGE(Apr,May,Jun 2026)
    SHEETS same month last year: just read Jul,Aug,Sep 2025 from the actuals.
    """
    s = country_monthly_collections(panel)
    recent3 = [s.get(m, np.nan) for m in [pd.Period("2026-04", "M"),
                                          pd.Period("2026-05", "M"),
                                          pd.Period("2026-06", "M")]]
    avg3 = float(np.nanmean(recent3))
    smly = {fm: float(s.get(fm - 12, np.nan)) for fm in forecast_months}
    return {"recent_3m_avg": avg3, "same_month_last_year": smly,
            "history": s}


def ets_crosscheck(panel: pd.DataFrame,
                   forecast_months=FORECAST_MONTHS) -> dict:
    """A top-down cross-check on financed collections using a trend model with no
    twelve-month seasonal term, because there is under two years of history. If
    statsmodels is missing we fall back to a simple last-six-month linear trend.
    """
    p = panel[panel["contract_type"] == "FINANCED"].copy()
    p["pay_period"] = pd.PeriodIndex(pd.to_datetime(p["pay_month"]), freq="M")
    s = p.groupby("pay_period")["paid"].sum().sort_index()
    s = s[s.index <= pd.Period(DATA_END, "M")]
    y = s.values.astype(float)
    h = len(forecast_months)
    if len(y) < 4:
        return {"method": "insufficient_history", "values": [float("nan")] * h}
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(y, trend="add", seasonal=None,
                                     initialization_method="estimated").fit()
        fc = model.forecast(h)
        return {"method": "ETS_add_trend_no_seasonal",
                "values": [float(v) for v in fc]}
    except Exception:
        n = min(6, len(y))
        xs = np.arange(n)
        b, a = np.polyfit(xs, y[-n:], 1)
        fc = [float(a + b * (n - 1 + k)) for k in range(1, h + 1)]
        return {"method": "linear_trend_fallback", "values": fc}


# ---------------------------------------------------------------------------
# 7. Calendar-shock size for scenario widths
# ---------------------------------------------------------------------------
def calendar_shock_sigma(panel: pd.DataFrame,
                         window=("2025-07", "2026-06")) -> float:
    """How much the whole country's monthly collection rate wobbles month to
    month. This is the right ruler for the scenario band, because every contract
    in a month lives through the same economy, so the uncertainty is a calendar
    shock, not the tiny sampling error you would get by treating contracts as
    independent.

        monthly_rate = country paid / country billed, each month
        sigma = standard deviation of the month-to-month change in that rate
        SHEETS: =STDEV of the differences between consecutive monthly rates
    """
    p = panel.copy()
    p["pay_period"] = pd.PeriodIndex(pd.to_datetime(p["pay_month"]), freq="M")
    billed = p.groupby("pay_period")["expected_this_month"].sum()
    paid = p.groupby("pay_period")["paid"].sum()
    rate = (paid / billed).replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = pd.Period(window[0], "M"), pd.Period(window[1], "M")
    rate = rate[(rate.index >= lo) & (rate.index <= hi)]
    if len(rate) < 3:
        return 0.03
    return float(np.nanstd(np.diff(rate.values), ddof=1))


# ---------------------------------------------------------------------------
# 8. Assemble the base case and the low / high scenarios
# ---------------------------------------------------------------------------
def _scaled_curve(lut: dict, factor: float) -> dict:
    """Return a copy of the rate lookup with every rate scaled, for scenarios."""
    out = {}
    for k, v in lut.items():
        if k == "_tail":
            t = {}
            for tk, tv in v.items():
                t[tk] = tv * factor if isinstance(tv, float) and tk != "max_mob" \
                    and not (isinstance(tk, tuple)) else tv
            out[k] = t
        else:
            out[k] = v * factor
    return out


def run_forecast(billing: pd.DataFrame,
                 rate_lut: dict,
                 cum_days: pd.Series,
                 profile: dict,
                 post_tenor_monthly: float,
                 forecast_months=FORECAST_MONTHS,
                 rate_factor: float = 1.0,
                 sales_multiplier: float = 1.0,
                 p_cash_shift: float = 0.0,
                 price_multiplier: float = 1.0,
                 post_tenor_factor: float = 1.0) -> pd.DataFrame:
    """One full bottom-up pass: existing book plus post-tenor recovery plus new
    sales, month by month for the country. The keyword levers are what the
    scenarios and the tornado turn. Billing is passed in precomputed, because it
    does not depend on the levers, so the scenarios and the tornado reuse it and
    stay fast on the free Colab tier."""
    lut = _scaled_curve(rate_lut, rate_factor) if rate_factor != 1.0 else rate_lut
    existing = forecast_existing_book(billing, lut)
    exist_by_month = (existing.groupby("forecast_month")["collection"].sum()
                      .reindex(forecast_months).fillna(0.0))
    new = forecast_new_sales(profile, lut, cum_days, plan=SALES_PLAN,
                             forecast_months=forecast_months,
                             sales_multiplier=sales_multiplier,
                             p_cash_shift=p_cash_shift,
                             price_multiplier=price_multiplier)
    new_by_month = new.set_index("forecast_month")["new_sales_collection"]
    out = pd.DataFrame({"forecast_month": forecast_months})
    out["existing_book"] = out["forecast_month"].map(exist_by_month).values
    out["post_tenor"] = post_tenor_monthly * post_tenor_factor
    out["new_sales"] = out["forecast_month"].map(new_by_month).values
    out["total"] = out[["existing_book", "post_tenor", "new_sales"]].sum(axis=1)
    return out


def build_scenarios(billing, rate_lut, cum_days, profile,
                    post_tenor_monthly, sigma,
                    low_sigma_mult=2.0, high_sigma_mult=1.0,
                    low_sales_cut=0.05, high_sales_keep=1.0,
                    forecast_months=FORECAST_MONTHS) -> pd.DataFrame:
    """Base, low and high, one row per month for the country.

    The base continues recent rates and the plan and holds outreach at zero. The
    low softens the collection rate by about two calendar-shock sigmas, assumes
    no pilot scaling, and trims sales a little. The high lifts the rate by about
    one sigma and keeps the plan, which is deliberately less generous on the
    upside than on the downside because the drift and the unconfirmed product
    reports point the risk downward.

    SHEETS band: low = base * (1 - 2*sigma - trim), high = base * (1 + 1*sigma).
    """
    base = run_forecast(billing, rate_lut, cum_days, profile,
                        post_tenor_monthly, forecast_months)
    low = run_forecast(billing, rate_lut, cum_days, profile,
                       post_tenor_monthly, forecast_months,
                       rate_factor=max(0.0, 1.0 - low_sigma_mult * sigma),
                       sales_multiplier=1.0 - low_sales_cut)
    high = run_forecast(billing, rate_lut, cum_days, profile,
                        post_tenor_monthly, forecast_months,
                        rate_factor=1.0 + high_sigma_mult * sigma,
                        sales_multiplier=high_sales_keep)
    tbl = pd.DataFrame({
        "month": [str(m) for m in forecast_months],
        "base": base["total"].round(2).values,
        "low": low["total"].round(2).values,
        "high": high["total"].round(2).values,
    })
    return tbl


# ---------------------------------------------------------------------------
# 9. Tornado: which assumption moves the number most
# ---------------------------------------------------------------------------
def tornado(billing, rate_lut, cum_days, profile,
            post_tenor_monthly, sigma,
            forecast_months=FORECAST_MONTHS) -> pd.DataFrame:
    """Vary one assumption at a time across a plausible range and record the swing
    in the total Q3 forecast, then rank. We expect the existing-book collection
    rate to move it most on the real book, where existing collections dominate,
    and new-sales volume and mix next; the ranking is read straight from the data,
    so whatever comes out top is itself the finding to report."""
    def total(**kw):
        f = run_forecast(billing, rate_lut, cum_days, profile,
                         post_tenor_monthly, forecast_months, **kw)
        return float(f["total"].sum())

    base_total = total()
    levers = [
        ("Existing-book collection rate",
         dict(rate_factor=1.0 - 2 * sigma), dict(rate_factor=1.0 + sigma)),
        ("New-sales volume (+/-10%)",
         dict(sales_multiplier=0.90), dict(sales_multiplier=1.10)),
        ("New-sales cash/financed mix (+/-5pp cash)",
         dict(p_cash_shift=-0.05), dict(p_cash_shift=0.05)),
        ("New-sales average price (+/-5%)",
         dict(price_multiplier=0.95), dict(price_multiplier=1.05)),
        ("Post-tenor recovery (+/-50%)",
         dict(post_tenor_factor=0.5), dict(post_tenor_factor=1.5)),
    ]
    rows = []
    for name, lo_kw, hi_kw in levers:
        lo, hi = total(**lo_kw), total(**hi_kw)
        rows.append((name, round(lo, 2), round(hi, 2),
                     round(hi - lo, 2), round(abs(hi - lo), 2)))
    t = pd.DataFrame(rows, columns=["assumption", "low_total", "high_total",
                                    "swing", "abs_swing"])
    t = t.sort_values("abs_swing", ascending=False).reset_index(drop=True)
    t.attrs["base_total"] = round(base_total, 2)
    return t


# ---------------------------------------------------------------------------
# 10. Rolling-origin backtest (never a random split)
# ---------------------------------------------------------------------------
def rolling_origin_backtest(panel: pd.DataFrame,
                            origin: pd.Timestamp,
                            horizon_months: int,
                            base_regions=NON_PILOT_REGIONS) -> pd.DataFrame:
    """Stand at an origin, learn the curve on data up to it from the non-pilot
    regions, then predict the next few months of country collections by applying
    that curve to the billing that actually came due, and compare to the actual
    cash. This isolates how well the curve predicts, scored against the naive
    three-month average, always forward in time and never on a shuffled split.
    """
    origin_p = pd.Period(origin, "M")
    curve = learn_marginal_curve(panel, estimation_end=origin,
                                 regions=base_regions, keys=("contract_type",))
    lut = rate_lookup(curve)

    fut = panel.copy()
    fut["pay_period"] = pd.PeriodIndex(pd.to_datetime(fut["pay_month"]), freq="M")
    target_months = [origin_p + k for k in range(1, horizon_months + 1)]
    fut = fut[fut["pay_period"].isin(target_months)].copy()
    fut["rate"] = [get_rate(lut, ct, int(m))
                   for ct, m in zip(fut["contract_type"], fut["mob"])]
    fut["pred"] = fut["expected_this_month"] * fut["rate"]

    pred = fut.groupby("pay_period")["pred"].sum()
    actual = fut.groupby("pay_period")["paid"].sum()

    hist = country_monthly_collections(panel)
    naive_level = float(np.nanmean([hist.get(origin_p - k, np.nan)
                                    for k in range(0, 3)]))

    rows = []
    for m in target_months:
        a = float(actual.get(m, np.nan))
        pmodel = float(pred.get(m, np.nan))
        rows.append((str(m), round(a, 2), round(pmodel, 2), round(naive_level, 2),
                     round(abs(pmodel - a), 2), round(abs(naive_level - a), 2)))
    df = pd.DataFrame(rows, columns=["month", "actual", "model_pred",
                                     "naive_pred", "model_abs_err", "naive_abs_err"])
    with np.errstate(divide="ignore", invalid="ignore"):
        df["model_ape"] = (df["model_abs_err"] / df["actual"]).round(4)
        df["naive_ape"] = (df["naive_abs_err"] / df["actual"]).round(4)
    df.attrs["model_MAE"] = round(float(df["model_abs_err"].mean()), 2)
    df.attrs["naive_MAE"] = round(float(df["naive_abs_err"].mean()), 2)
    df.attrs["model_MAPE"] = round(float(df["model_ape"].mean()), 4)
    df.attrs["naive_MAPE"] = round(float(df["naive_ape"].mean()), 4)
    return df


# ---------------------------------------------------------------------------
# 11. QA gate: everything must pass before an output is allowed out
# ---------------------------------------------------------------------------
def qa_table(panel, contract_features, forecast_table, billing,
             feature_metrics: dict,
             forecast_months=FORECAST_MONTHS) -> pd.DataFrame:
    """A block of named, human-readable checks. The download is gated on all of
    these passing, so a broken run cannot quietly produce a shippable number."""
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, "PASS" if ok else "FAIL", detail))

    pm_max = pd.to_datetime(panel["pay_month"]).max()
    add("no_panel_rows_after_jun_2026", pm_max <= DATA_END, f"max={pm_max.date()}")

    sm_max = pd.to_datetime(contract_features["sales_month"]).max()
    add("no_sales_after_jun_2026", sm_max <= DATA_END, f"max={sm_max.date()}")

    fm_ok = [str(m) for m in forecast_table["month"]] == [str(m) for m in forecast_months]
    add("forecast_months_are_jul_aug_sep_2026", fm_ok)

    used_cols = set(billing.columns)
    add("no_outreach_columns_used",
        not any("outreach" in c.lower() for c in used_cols))

    reg = billing.groupby(["forecast_month", "region"])["expected_this_month"].sum()
    ctry = billing.groupby("forecast_month")["expected_this_month"].sum()
    region_ok = True
    for fm in forecast_months:
        s = float(reg.loc[fm].sum()) if fm in reg.index.get_level_values(0) else 0.0
        c = float(ctry.get(fm, 0.0))
        if abs(s - c) > 1.0:
            region_ok = False
    add("region_sums_equal_country", region_ok)

    plan_ok = sum(SALES_PLAN.values()) == 3100 + 3200 + 3200
    add("sales_plan_totals_match", plan_ok, "3100/3200/3200")

    band_ok = bool(((forecast_table["low"] <= forecast_table["base"]) &
                    (forecast_table["base"] <= forecast_table["high"])).all())
    add("base_within_low_high_each_month", band_ok)

    sv = str(feature_metrics.get("schema_version", "")).strip()
    add("feature_metrics_loaded",
        isinstance(feature_metrics, dict),
        f"schema_version={sv}" if sv else "schema_version unset (provenance only, not gated)")

    return pd.DataFrame(checks, columns=["check", "status", "detail"])


def qa_passed(qa: pd.DataFrame) -> bool:
    return bool((qa["status"] == "PASS").all())

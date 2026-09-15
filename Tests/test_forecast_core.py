"""
Unit tests for the Part 1 forecasting core (final_forecast_core / p1_core).

These test behaviour on small, hand-built fixtures where the correct answer can be
worked out with a calculator, plus the invariants the forecast must never break
(no future leakage, low <= base <= high, components sum to total, region sums to
country). They do not touch the sealed quarter and do not need the real 400 MB panel.

Run:  pytest test_forecast_core.py -v
"""
import numpy as np
import pandas as pd
import pytest

import p1_core as C


# ---------------------------------------------------------------------------
# Fixtures: a tiny two-contract panel we can reason about by hand
# ---------------------------------------------------------------------------
def _panel_rows():
    """One financed contract (price 100, deposit 20, daily 0.20, tenor 400 days,
    sold Oct 2024, North, core product) observed Oct 2024..Jun 2026, plus one cash
    contract (price 50, sold Jan 2025, South). Payments are set so cumulative
    actuals are easy to check."""
    rows = []
    sales = pd.Timestamp("2024-10-31")
    months = pd.date_range("2024-10-31", "2026-06-30", freq="ME")
    cum = 0.0
    for m in months:
        mob = (m.year - sales.year) * 12 + (m.month - sales.month)
        # pay exactly what is due each month so cumulative efficiency is ~1.0
        due, within, _ = C.schedule_daycount([100], [20], [0.20], [400], [sales], [m])
        pay = float(due[0])
        cum += pay
        rows.append(dict(contractid="F1", sales_month=sales, region="North", contract_type="FINANCED",
                         product="Basic", payment_frequency="MONTHLY", price_usd=100.0, perc_deposit=0.20,
                         daily_amount_usd=0.20, tenor_length=400.0, month=m, months_on_book=mob, deposit_usd=20.0,
                         expected_cash_due_this_month_usd=pay, within_scheduled_tenor_proxy=bool(within[0]),
                         post_tenor_recovery_proxy=False, scheduled_status="WITHIN_SCHEDULE",
                         target_payment_usd=pay, current_payment_conflict=False,
                         actual_cumulative_cash_through_month_usd=cum,
                         actual_cumulative_cash_before_month_usd=cum - pay, payment_history_reliable=True,
                         deposit_scale_corrected=False, tickets_prior_3m=0, calls_prior_3m=0,
                         tickets_reason_battery_fault_prior_3m=0, tickets_reason_charging_issue_prior_3m=0,
                         payment_lag_1m=0.0, payment_lag_2m=0.0, payment_lag_3m=0.0, payment_trailing_3m_sum=0.0,
                         payment_trailing_6m_sum=0.0, prior_paying_months=0, zero_payment_months_prior_3=0,
                         zero_payment_months_prior_6=0, months_since_last_positive_payment=1,
                         calls_cumulative_before_month=0, tickets_cumulative_before_month=0))
    csale = pd.Timestamp("2025-01-31")
    rows.append(dict(contractid="C1", sales_month=csale, region="South", contract_type="CASH",
                     product="Basic", payment_frequency="UPFRONT", price_usd=50.0, perc_deposit=np.nan,
                     daily_amount_usd=np.nan, tenor_length=np.nan, month=csale, months_on_book=0, deposit_usd=np.nan,
                     expected_cash_due_this_month_usd=50.0, within_scheduled_tenor_proxy=True,
                     post_tenor_recovery_proxy=False, scheduled_status="CASH_UPFRONT", target_payment_usd=50.0,
                     current_payment_conflict=False, actual_cumulative_cash_through_month_usd=50.0,
                     actual_cumulative_cash_before_month_usd=0.0, payment_history_reliable=True,
                     deposit_scale_corrected=False, tickets_prior_3m=0, calls_prior_3m=0,
                     tickets_reason_battery_fault_prior_3m=0, tickets_reason_charging_issue_prior_3m=0,
                     payment_lag_1m=0.0, payment_lag_2m=0.0, payment_lag_3m=0.0, payment_trailing_3m_sum=0.0,
                     payment_trailing_6m_sum=0.0, prior_paying_months=0, zero_payment_months_prior_3=0,
                     zero_payment_months_prior_6=0, months_since_last_positive_payment=0,
                     calls_cumulative_before_month=0, tickets_cumulative_before_month=0))
    # a handful of recent-cohort sales so the new-sales mix window is populated
    for i, (sm, reg, ct) in enumerate([(pd.Timestamp("2026-05-31"), "North", "FINANCED"),
                                       (pd.Timestamp("2026-06-30"), "South", "FINANCED"),
                                       (pd.Timestamp("2026-06-30"), "North", "CASH")]):
        rows.append(dict(contractid=f"R{i}", sales_month=sm, region=reg, contract_type=ct,
                         product="Basic", payment_frequency="MONTHLY", price_usd=100.0,
                         perc_deposit=0.20 if ct == "FINANCED" else np.nan,
                         daily_amount_usd=0.20 if ct == "FINANCED" else np.nan,
                         tenor_length=400.0 if ct == "FINANCED" else np.nan, month=sm, months_on_book=0,
                         deposit_usd=20.0 if ct == "FINANCED" else np.nan,
                         expected_cash_due_this_month_usd=20.0 if ct == "FINANCED" else 100.0,
                         within_scheduled_tenor_proxy=True, post_tenor_recovery_proxy=False,
                         scheduled_status="WITHIN_SCHEDULE" if ct == "FINANCED" else "CASH_UPFRONT",
                         target_payment_usd=20.0 if ct == "FINANCED" else 100.0, current_payment_conflict=False,
                         actual_cumulative_cash_through_month_usd=20.0 if ct == "FINANCED" else 100.0,
                         actual_cumulative_cash_before_month_usd=0.0, payment_history_reliable=True,
                         deposit_scale_corrected=False, tickets_prior_3m=0, calls_prior_3m=0,
                         tickets_reason_battery_fault_prior_3m=0, tickets_reason_charging_issue_prior_3m=0,
                         payment_lag_1m=0.0, payment_lag_2m=0.0, payment_lag_3m=0.0, payment_trailing_3m_sum=0.0,
                         payment_trailing_6m_sum=0.0, prior_paying_months=0, zero_payment_months_prior_3=0,
                         zero_payment_months_prior_6=0, months_since_last_positive_payment=0,
                         calls_cumulative_before_month=0, tickets_cumulative_before_month=0))
    return pd.DataFrame(rows)


@pytest.fixture
def panel():
    return C.rebuild_panel_schedule(_panel_rows())


@pytest.fixture
def country_month(panel):
    g = panel.groupby("month", as_index=False)["target_payment_usd"].sum().rename(columns={"target_payment_usd": "source_reported_cash_usd"})
    g["source_minus_attributable_cash_usd"] = 0.0
    return g


# ---------------------------------------------------------------------------
# schedule_daycount: the contractual arithmetic
# ---------------------------------------------------------------------------
class TestScheduleDaycount:
    def test_deposit_billed_in_sale_month(self):
        due, within, mob = C.schedule_daycount([100], [20], [0.20], [400], ["2025-01-31"], ["2025-01-31"])
        assert due[0] == pytest.approx(20.0)
        assert bool(within[0]) is True
        assert int(mob[0]) == 0

    def test_february_bills_28_days(self):
        due, _, _ = C.schedule_daycount([100], [20], [0.20], [400], ["2025-01-31"], ["2025-02-28"])
        assert due[0] == pytest.approx(0.20 * 28, abs=1e-9)

    def test_thirty_one_day_month_bills_31_days(self):
        due, _, _ = C.schedule_daycount([100], [20], [0.20], [400], ["2025-01-31"], ["2025-03-31"])
        assert due[0] == pytest.approx(0.20 * 31, abs=1e-9)

    def test_never_bills_beyond_price(self):
        # daily*tenor huge; cumulative due must cap at price minus nothing, i.e. <= price
        due_total = 0.0
        for m in pd.date_range("2025-01-31", "2027-01-31", freq="ME"):
            d, _, _ = C.schedule_daycount([100], [20], [5.0], [400], ["2025-01-31"], [m])
            due_total += d[0]
        assert due_total <= 100.0 + 1e-6

    def test_past_tenor_marked_not_within(self):
        _, within, _ = C.schedule_daycount([100], [20], [0.20], [400], ["2025-01-31"], ["2026-06-30"])
        assert bool(within[0]) is False

    def test_cash_style_zero_daily_gives_zero_after_deposit(self):
        due, _, _ = C.schedule_daycount([50], [0], [0.0], [0], ["2025-01-31"], ["2025-02-28"])
        assert due[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_product_group_flags_gen2(self):
        g = C.product_group(pd.Series(["Large Solar - Gen 2", "Basic Lamp", "Home 60 Gen 2"]))
        assert list(g) == ["GEN2", "CORE", "GEN2"]

    def test_month_end_normalises(self):
        assert C.month_end("2025-03-15") == pd.Timestamp("2025-03-31")

    def test_add_months(self):
        assert C.add_months("2025-01-31", 2) == pd.Timestamp("2025-03-31")
        assert C.add_months("2025-03-31", -3) == pd.Timestamp("2024-12-31")

    def test_score_bias_sign(self):
        s = C.score([100, 100], [110, 95])
        assert s["Bias"] > 0                       # forecast above actual -> positive bias
        assert s["WAPE"] == pytest.approx(15 / 200)
        assert s["MAE"] == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# rebuild_panel_schedule
# ---------------------------------------------------------------------------
class TestRebuild:
    def test_adds_expected_columns(self, panel):
        for col in ["due_dc", "within_dc", "post_tenor_dc", "pgroup", "balance_before", "tenor_mob"]:
            assert col in panel.columns

    def test_balance_before_never_negative(self, panel):
        assert (panel["balance_before"] >= 0).all()

    def test_cash_row_due_is_price_in_sale_month(self, panel):
        c = panel[panel["contractid"].eq("C1")].iloc[0]
        assert c["due_dc"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# fit_curves and lookup
# ---------------------------------------------------------------------------
class TestCurves:
    def test_efficiency_near_one_when_paid_in_full(self, panel):
        # our fixture pays exactly what's due, so scheduled efficiency should be ~1
        curves = C.fit_curves(panel, C.ESTIMATION_END)
        e = C.lookup_eff(curves, ["North"], ["CORE"], [3])
        assert e[0] == pytest.approx(1.0, abs=0.05)

    def test_lookup_falls_back_to_country_for_unseen_region(self, panel):
        curves = C.fit_curves(panel, C.ESTIMATION_END)
        # a region not in the fixture should still return a finite number via fallback
        e = C.lookup_eff(curves, ["East"], ["CORE"], [3])
        assert np.isfinite(e[0])

    def test_outlier_flagging_returns_region_month_tuples(self, panel):
        curves = C.fit_curves(panel, C.ESTIMATION_END)
        idx, _ = C.calendar_index(panel, curves, C.VALIDATION_END)
        outliers = C.flag_outlier_region_months(idx, threshold=0.3)
        assert isinstance(outliers, list)
        for o in outliers:
            assert len(o) == 2


# ---------------------------------------------------------------------------
# forecast: the invariants that must always hold
# ---------------------------------------------------------------------------
class TestForecastInvariants:
    def test_components_sum_to_total(self, panel, country_month):
        fc = C.forecast(panel, country_month, C.VALIDATION_END, C.FORECAST_MONTHS, C.SALES_PLAN, curve_cutoff=C.ESTIMATION_END)
        parts = fc[["existing_scheduled", "post_tenor_recovery", "new_sales_financed", "new_sales_cash", "bridge"]].sum(axis=1)
        assert np.allclose(parts.to_numpy(), fc["total"].to_numpy(), atol=1e-6)

    def test_three_forecast_months(self, panel, country_month):
        fc = C.forecast(panel, country_month, C.VALIDATION_END, C.FORECAST_MONTHS, C.SALES_PLAN, curve_cutoff=C.ESTIMATION_END)
        assert list(fc["month"]) == list(C.FORECAST_MONTHS)

    def test_no_negative_components(self, panel, country_month):
        fc = C.forecast(panel, country_month, C.VALIDATION_END, C.FORECAST_MONTHS, C.SALES_PLAN, curve_cutoff=C.ESTIMATION_END)
        for c in ["existing_scheduled", "post_tenor_recovery", "new_sales_financed", "new_sales_cash"]:
            assert (fc[c] >= -1e-9).all()

    def test_region_forecast_sums_to_country_ex_bridge(self, panel, country_month):
        fc, det = C.forecast(panel, country_month, C.VALIDATION_END, C.FORECAST_MONTHS, C.SALES_PLAN,
                             curve_cutoff=C.ESTIMATION_END, return_detail=True)
        rf = det["region_forecast"].groupby("month")["total_ex_bridge"].sum()
        ex_bridge = (fc.set_index("month")["total"] - fc.set_index("month")["bridge"])
        assert np.allclose(rf.reindex(ex_bridge.index).to_numpy(), ex_bridge.to_numpy(), atol=1.0)

    def test_higher_level_factor_gives_more_cash(self):
        # needs an in-schedule financed book at the origin for the level to bite,
        # so build a short-horizon contract that is still paying in Q3
        rows = []
        sales = pd.Timestamp("2026-03-31")
        for m in pd.date_range("2026-03-31", "2026-06-30", freq="ME"):
            mob = (m.year - sales.year) * 12 + (m.month - sales.month)
            due, within, _ = C.schedule_daycount([300], [30], [1.0], [365], [sales], [m])
            rows.append(dict(contractid="S1", sales_month=sales, region="North", contract_type="FINANCED",
                             product="Basic", payment_frequency="MONTHLY", price_usd=300.0, perc_deposit=0.10,
                             daily_amount_usd=1.0, tenor_length=365.0, month=m, months_on_book=mob, deposit_usd=30.0,
                             expected_cash_due_this_month_usd=float(due[0]), within_scheduled_tenor_proxy=bool(within[0]),
                             post_tenor_recovery_proxy=False, scheduled_status="WITHIN_SCHEDULE",
                             target_payment_usd=float(due[0]), current_payment_conflict=False,
                             actual_cumulative_cash_through_month_usd=30.0 + mob * 10,
                             actual_cumulative_cash_before_month_usd=30.0 + max(0, mob - 1) * 10,
                             payment_history_reliable=True, deposit_scale_corrected=False, tickets_prior_3m=0,
                             calls_prior_3m=0, tickets_reason_battery_fault_prior_3m=0,
                             tickets_reason_charging_issue_prior_3m=0, payment_lag_1m=0.0, payment_lag_2m=0.0,
                             payment_lag_3m=0.0, payment_trailing_3m_sum=0.0, payment_trailing_6m_sum=0.0,
                             prior_paying_months=0, zero_payment_months_prior_3=0, zero_payment_months_prior_6=0,
                             months_since_last_positive_payment=1, calls_cumulative_before_month=0,
                             tickets_cumulative_before_month=0))
        pnl = C.rebuild_panel_schedule(pd.DataFrame(rows))
        cm = pnl.groupby("month", as_index=False)["target_payment_usd"].sum().rename(columns={"target_payment_usd": "source_reported_cash_usd"})
        cm["source_minus_attributable_cash_usd"] = 0.0
        curves = C.fit_curves(pnl, pd.Timestamp("2026-06-30"))
        idx, _ = C.calendar_index(pnl, curves, C.VALIDATION_END)
        levels = C.region_levels(idx, C.VALIDATION_END)
        assert len(levels) > 0, "fixture must produce a non-empty level table"
        low = C.forecast(pnl, cm, C.VALIDATION_END, C.FORECAST_MONTHS, {}, curve_cutoff=pd.Timestamp("2026-06-30"), levels=levels, level_factor=0.8)
        high = C.forecast(pnl, cm, C.VALIDATION_END, C.FORECAST_MONTHS, {}, curve_cutoff=pd.Timestamp("2026-06-30"), levels=levels, level_factor=1.2)
        assert high["existing_scheduled"].sum() > low["existing_scheduled"].sum()

    def test_higher_attainment_gives_more_new_sales(self, panel, country_month):
        lo = C.forecast(panel, country_month, C.VALIDATION_END, C.FORECAST_MONTHS, C.SALES_PLAN, curve_cutoff=C.ESTIMATION_END, attainment=0.9)
        hi = C.forecast(panel, country_month, C.VALIDATION_END, C.FORECAST_MONTHS, C.SALES_PLAN, curve_cutoff=C.ESTIMATION_END, attainment=1.1)
        assert (hi["new_sales_financed"] + hi["new_sales_cash"]).sum() > (lo["new_sales_financed"] + lo["new_sales_cash"]).sum()

    def test_level_factor_does_not_touch_bridge(self, panel, country_month):
        a = C.forecast(panel, country_month, C.VALIDATION_END, C.FORECAST_MONTHS, C.SALES_PLAN, curve_cutoff=C.ESTIMATION_END, level_factor=0.7)
        b = C.forecast(panel, country_month, C.VALIDATION_END, C.FORECAST_MONTHS, C.SALES_PLAN, curve_cutoff=C.ESTIMATION_END, level_factor=1.3)
        assert np.allclose(a["bridge"].to_numpy(), b["bridge"].to_numpy())


# ---------------------------------------------------------------------------
# leakage guard
# ---------------------------------------------------------------------------
class TestLeakageGuard:
    def test_assert_no_sealed_passes_on_clean(self, panel, country_month):
        assert C.assert_no_sealed(panel, country_month) is True

    def test_assert_no_sealed_raises_on_future_row(self, panel):
        bad = panel.copy()
        bad.loc[bad.index[0], "month"] = pd.Timestamp("2026-07-31")
        with pytest.raises(AssertionError):
            C.assert_no_sealed(bad)

    def test_forecast_reads_no_future_month(self, panel, country_month):
        # forecast must not consult any month strictly after the origin when building state
        fc, det = C.forecast(panel, country_month, C.VALIDATION_END, C.FORECAST_MONTHS, C.SALES_PLAN,
                             curve_cutoff=C.ESTIMATION_END, return_detail=True)
        assert det["index_table"]["month"].max() <= C.VALIDATION_END


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------
class TestBenchmarks:
    def test_naive_is_mean_of_last_three(self, country_month):
        nv = C.naive_forecast(country_month, C.VALIDATION_END, C.FORECAST_MONTHS, lookback=3)
        last3 = country_month.loc[country_month["month"] <= C.VALIDATION_END, "source_reported_cash_usd"].tail(3).mean()
        assert nv.iloc[0] == pytest.approx(last3)
        assert (nv == nv.iloc[0]).all()          # flat

    def test_actual_units_counts_sale_month(self, panel):
        u = C.actual_units(panel, [pd.Timestamp("2025-01-31")])
        assert u[pd.Timestamp("2025-01-31")] == 1     # the cash contract C1 sold that month

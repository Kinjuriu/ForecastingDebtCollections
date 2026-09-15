"""
Unit tests for the Part 2 pilot-evaluation core (final_pilot_core / p2_core).

The hardest thing to get right in a quasi-experimental evaluation is that the
method returns *no* effect when there is none and a *known* effect when we inject
one. So the key tests build a synthetic world with a treatment effect we choose,
and check the difference-in-differences recovers it; plus a null world where the
effect must not appear. Others check the mechanical invariants: budget sums to the
cap, standardised differences behave, the placebo is centred on zero.

Run:  pytest test_pilot_core.py -v
"""
import numpy as np
import pandas as pd
import pytest

import p1_core as C
import p2_core as P


# ---------------------------------------------------------------------------
# Synthetic world: four regions, a pre period and a post period, with a chosen
# treatment effect added to one region after the pilot start.
# ---------------------------------------------------------------------------
def make_world(effect_east=0.0, effect_west=0.0, seed=0, n_per_region=400):
    """Every contract is financed, in-schedule, due $10/month. Baseline pay equals
    due plus mean-zero noise. After PILOT_START, East gets +effect_east and West
    +effect_west added to their paid amount. North and South never change."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2025-10-31", "2026-06-30", freq="ME")
    regions = ["East", "West", "North", "South"]
    rows = []
    for reg in regions:
        for i in range(n_per_region):
            cid = f"{reg}_{i}"
            sales = pd.Timestamp("2024-10-31")
            for m in months:
                mob = (m.year - sales.year) * 12 + (m.month - sales.month)
                due = 10.0
                base = due + rng.normal(0, 1.0)
                eff = 0.0
                if m >= P.PILOT_START:
                    eff = effect_east if reg == "East" else effect_west if reg == "West" else 0.0
                paid = max(0.0, base + eff)
                rows.append(dict(contractid=cid, sales_month=sales, region=reg, contract_type="FINANCED",
                                 product="Basic", payment_frequency="MONTHLY", price_usd=1000.0, perc_deposit=0.02,
                                 daily_amount_usd=0.33, tenor_length=2000.0, month=m, months_on_book=mob, deposit_usd=20.0,
                                 expected_cash_due_this_month_usd=due, within_scheduled_tenor_proxy=True,
                                 post_tenor_recovery_proxy=False, scheduled_status="WITHIN_SCHEDULE",
                                 target_payment_usd=paid, current_payment_conflict=False,
                                 actual_cumulative_cash_through_month_usd=paid * (mob + 1),
                                 actual_cumulative_cash_before_month_usd=paid * mob, payment_history_reliable=True,
                                 deposit_scale_corrected=False,
                                 payment_lag_1m=paid, payment_lag_2m=paid, payment_lag_3m=paid,
                                 payment_trailing_3m_sum=paid * 3, payment_trailing_6m_sum=paid * 6,
                                 prior_paying_months=mob, zero_payment_months_prior_3=0, zero_payment_months_prior_6=0,
                                 months_since_last_positive_payment=1, calls_prior_3m=0, tickets_prior_3m=0,
                                 calls_cumulative_before_month=0, tickets_cumulative_before_month=0,
                                 expected_cumulative_cash_before_month_usd=due * mob, balance_before=1000.0))
    panel = pd.DataFrame(rows)
    panel["due_dc"] = panel["expected_cash_due_this_month_usd"]
    panel["within_dc"] = True
    panel["post_tenor_dc"] = False
    panel["pgroup"] = "CORE"
    panel["tenor_mob"] = 66
    return panel


class FlatCurves:
    """A stand-in for the fitted curve object: every contract-month predicts its
    due amount times 1.0, so the residual is exactly paid minus due."""
    def __getitem__(self, k):
        raise KeyError(k)


def flat_lookup(curves, region, pgroup, mob):
    return np.ones(len(np.asarray(region)))


def make_pilot(panel, east_contacted=0.95, west_contacted=0.60, seed=1):
    """An outreach log covering a share of each pilot region for Apr and May."""
    rng = np.random.default_rng(seed)
    rows = []
    for reg, share, chan, cost in [("East", east_contacted, "SMS", 0.08), ("West", west_contacted, "CALL", 2.20)]:
        cids = panel.loc[panel["region"].eq(reg), "contractid"].unique()
        pick = rng.choice(cids, size=int(len(cids) * share), replace=False)
        for m in [pd.Timestamp("2026-04-30"), pd.Timestamp("2026-05-31")]:
            for cid in pick:
                rows.append(dict(contractid=cid, contact_month=m, region_outreach=reg, channel=chan,
                                 attempts=1.0, reached=True, cost_usd=cost,
                                 payment_lag_1m=0.0, zero_payment_months_prior_3=0))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DiD recovers a known effect, and finds nothing in a null world
# ---------------------------------------------------------------------------
class TestDiDRecovery:
    def test_recovers_injected_east_effect(self):
        panel = make_world(effect_east=2.0, effect_west=0.0, seed=42)
        h = P.residual_frame(panel, FlatCurves(), flat_lookup)
        r = P.did_regression(h, "East", ["North", "South"], n_boot=100, exclude_gen2_south=False)
        assert r["did_usd_per_contract_month"] == pytest.approx(2.0, abs=0.25)
        assert r["ci_low"] > 0                                   # detects the effect

    def test_null_world_interval_includes_zero(self):
        panel = make_world(effect_east=0.0, effect_west=0.0, seed=7)
        h = P.residual_frame(panel, FlatCurves(), flat_lookup)
        r = P.did_regression(h, "East", ["North", "South"], n_boot=200, exclude_gen2_south=False)
        assert r["ci_low"] <= 0 <= r["ci_high"]                 # no false positive

    def test_west_smaller_than_east_when_injected_that_way(self):
        panel = make_world(effect_east=2.0, effect_west=0.5, seed=11)
        h = P.residual_frame(panel, FlatCurves(), flat_lookup)
        re = P.did_regression(h, "East", ["North"], n_boot=100, exclude_gen2_south=False)
        rw = P.did_regression(h, "West", ["North"], n_boot=100, exclude_gen2_south=False)
        assert re["did_usd_per_contract_month"] > rw["did_usd_per_contract_month"]


# ---------------------------------------------------------------------------
# Event study and placebo
# ---------------------------------------------------------------------------
class TestEventStudyPlacebo:
    def test_pretrend_flat_before_pilot(self):
        panel = make_world(effect_east=2.0, seed=3)
        h = P.residual_frame(panel, FlatCurves(), flat_lookup)
        es = P.event_study(h, "East", ["North", "South"], exclude_gen2_south=False)
        slope = P.pretrend_slope(es)
        assert abs(slope) < 0.15                                # pre-period gap is flat

    def test_event_study_gap_jumps_after_pilot(self):
        panel = make_world(effect_east=2.0, seed=3)
        h = P.residual_frame(panel, FlatCurves(), flat_lookup)
        es = P.event_study(h, "East", ["North", "South"], exclude_gen2_south=False)
        pre = es[es.index <= P.PRE_END]["gap"].mean()
        post = es[es.index >= P.PILOT_START]["gap"].mean()
        assert post - pre == pytest.approx(2.0, abs=0.4)

    def test_placebo_finds_nothing_in_real_effect_world(self):
        # a fake April in January, run inside the pre period, must not detect the (later) effect
        panel = make_world(effect_east=2.0, seed=5)
        h = P.residual_frame(panel, FlatCurves(), flat_lookup)
        pl = P.placebo_did(h, "East", ["North", "South"], n_boot=100)
        assert pl["ci_low"] <= 0 <= pl["ci_high"]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
class TestCoverage:
    def test_coverage_share_and_cost(self):
        panel = make_world(seed=2, n_per_region=200)
        pilot = make_pilot(panel, east_contacted=0.9, west_contacted=0.5)
        cov = P.pilot_coverage(panel, pilot)
        east = cov[cov["region"].eq("East")]
        assert (east["share_contacted"] > 0.8).all()
        assert east["cost_per_contact_usd"].iloc[0] == pytest.approx(0.08, abs=1e-6)


# ---------------------------------------------------------------------------
# Matching frame and standardised differences
# ---------------------------------------------------------------------------
class TestMatching:
    def test_treated_flag_matches_outreach(self):
        panel = make_world(seed=4, n_per_region=200)
        pilot = make_pilot(panel, west_contacted=0.5)
        d = P.build_matching_frame(panel, pilot, "West", P.CONTACT_MONTHS_SAME)
        # every treated row must correspond to a contacted contract-month
        contacted = set(zip(pilot.loc[pilot["region_outreach"].eq("West"), "contractid"],
                            pilot.loc[pilot["region_outreach"].eq("West"), "contact_month"]))
        tr = d[d["treated"].eq(1)]
        for cid, m in zip(tr["contractid"], tr["month"]):
            assert (cid, m) in contacted

    def test_controls_never_contacted_in_that_pilot(self):
        panel = make_world(seed=4, n_per_region=200)
        pilot = make_pilot(panel, west_contacted=0.5)
        d = P.build_matching_frame(panel, pilot, "West", P.CONTACT_MONTHS_SAME)
        ever = set(pilot.loc[pilot["region_outreach"].eq("West"), "contractid"].astype(str))
        ctrl = d[d["treated"].eq(0)]
        assert not set(ctrl["contractid"].astype(str)).intersection(ever)

    def test_smd_zero_for_identical(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert P.smd(x, x) == pytest.approx(0.0)

    def test_smd_positive_when_treated_higher(self):
        assert P.smd([5, 6, 7], [1, 2, 3]) > 0


# ---------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------
def _candidates():
    return pd.DataFrame([
        {"programme": "East SMS", "incremental_cash_per_contact_usd": 0.53, "ci_low_per_contact_usd": 0.45,
         "cost_per_contact_usd": 0.08, "monthly_capacity_contacts": 13000},
        {"programme": "West calls", "incremental_cash_per_contact_usd": 0.50, "ci_low_per_contact_usd": 0.24,
         "cost_per_contact_usd": 2.22, "monthly_capacity_contacts": 8000},
    ])


class TestBudget:
    def test_allocation_sums_to_budget(self):
        alloc = P.allocate_budget(_candidates(), monthly_budget=8000.0)
        assert alloc["allocation_usd"].sum() == pytest.approx(8000.0, abs=1e-6)

    def test_qualified_cheap_programme_funded_first(self):
        alloc = P.allocate_budget(_candidates(), monthly_budget=8000.0)
        # East SMS clears the bar (cash>cost, ci>0) and is cheapest cash-per-dollar leader
        assert alloc.iloc[0]["line"] == "East SMS"

    def test_calls_that_lose_money_go_to_learning_reserve(self):
        # a programme whose cash is below its cost must not be funded as scale
        cands = _candidates()
        cands.loc[cands["programme"].eq("West calls"), "incremental_cash_per_contact_usd"] = 0.50  # < 2.22 cost
        alloc = P.allocate_budget(cands, monthly_budget=8000.0)
        assert "West calls" not in set(alloc["line"])
        assert "randomised learning reserve" in set(alloc["line"])

    def test_interval_below_zero_disqualifies(self):
        cands = _candidates()
        cands["ci_low_per_contact_usd"] = -0.1                    # nothing clears the bar
        alloc = P.allocate_budget(cands, monthly_budget=8000.0)
        assert alloc.iloc[0]["line"] == "randomised learning reserve"
        assert alloc["allocation_usd"].sum() == pytest.approx(8000.0)


# ---------------------------------------------------------------------------
# residual_frame respects the leakage boundary and outlier exclusion
# ---------------------------------------------------------------------------
class TestResidualFrame:
    def test_no_rows_after_validation_end(self):
        panel = make_world(seed=1, n_per_region=100)
        h = P.residual_frame(panel, FlatCurves(), flat_lookup)
        assert h["month"].max() <= P.VALIDATION_END

    def test_outlier_region_month_excluded(self):
        panel = make_world(seed=1, n_per_region=100)
        key = ("North", pd.Timestamp("2026-02-28"))
        h_all = P.residual_frame(panel, FlatCurves(), flat_lookup)
        h_ex = P.residual_frame(panel, FlatCurves(), flat_lookup, outlier_keys=[key])
        removed = len(h_all[(h_all["region"].eq("North")) & (h_all["month"].eq(key[1]))])
        assert removed > 0
        assert len(h_ex) == len(h_all) - removed

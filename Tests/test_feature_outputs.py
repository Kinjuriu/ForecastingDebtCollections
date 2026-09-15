"""
Contract tests for the frozen feature-engineering outputs (v3).

These do not rebuild anything; they assert the promises the feature stage makes to
everything downstream, run against the real artifacts in the v3 output ZIP. If a
future re-clean or re-feature run breaks one of these, the forecast and pilot
notebooks are no longer safe to run on that output, so this is the gate.

Point the suite at the extracted feature folder with FEATURES_DIR (default
/home/claude/feat). Skips cleanly if the folder is absent.

Run:  FEATURES_DIR=/path/to/feat pytest test_feature_outputs.py -v
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FEAT = Path(os.environ.get("FEATURES_DIR", "/home/claude/feat"))
VALIDATION_END = pd.Timestamp("2026-06-30")
ANALYSIS_START = pd.Timestamp("2024-10-31")

pytestmark = pytest.mark.skipif(not FEAT.exists(), reason=f"feature outputs not found at {FEAT}")


@pytest.fixture(scope="module")
def metrics():
    return json.load(open(FEAT / "feature_metrics.json"))


@pytest.fixture(scope="module")
def country_month():
    return pd.read_csv(FEAT / "country_month_features_v3.csv", parse_dates=["month"])


@pytest.fixture(scope="module")
def region_month():
    return pd.read_csv(FEAT / "region_month_features_v3.csv", parse_dates=["month"])


@pytest.fixture(scope="module")
def reconciliation():
    return pd.read_csv(FEAT / "payment_reconciliation_by_month_v3.csv", parse_dates=["month"])


@pytest.fixture(scope="module")
def sanity():
    return pd.read_csv(FEAT / "sanity_check_report_v3.csv")


@pytest.fixture(scope="module")
def pilot():
    return pd.read_csv(FEAT / "pilot_analysis_features_v3.csv", parse_dates=["contact_month"])


# ---------------------------------------------------------------------------
# Schema and leakage
# ---------------------------------------------------------------------------
class TestSchema:
    def test_metrics_schema_version(self, metrics):
        assert metrics["schema_version"] == "dlight_feature_metrics_v3"

    def test_country_month_has_required_columns(self, country_month):
        need = {"month", "panel_cash_usd", "source_reported_cash_usd", "model_attributable_cash_usd",
                "source_minus_attributable_cash_usd"}
        assert need.issubset(country_month.columns)

    def test_no_month_after_jun_2026(self, country_month, region_month, reconciliation, pilot):
        assert country_month["month"].max() <= VALIDATION_END
        assert region_month["month"].max() <= VALIDATION_END
        assert reconciliation["month"].max() <= VALIDATION_END
        assert pilot["contact_month"].max() <= VALIDATION_END

    def test_analysis_window_starts_oct_2024(self, country_month):
        assert country_month["month"].min() >= ANALYSIS_START


# ---------------------------------------------------------------------------
# The feature stage's own sanity checks must all be green
# ---------------------------------------------------------------------------
class TestSanityReport:
    def test_all_sanity_checks_pass(self, sanity):
        failed = sanity[sanity["status"].ne("PASS")]
        assert failed.empty, f"feature sanity checks not all PASS: {failed[['check','status']].to_dict('records')}"

    def test_leakage_checks_present(self, sanity):
        checks = set(sanity["check"])
        assert "no_same_month_call_or_ticket_predictors" in checks
        assert "no_outreach_treatment_columns_in_base_panel" in checks


# ---------------------------------------------------------------------------
# Reconciliation: every dollar accounted for
# ---------------------------------------------------------------------------
class TestReconciliation:
    def test_attributable_cash_all_lands_in_panel(self, metrics):
        assert abs(metrics["payments_not_landed_in_panel"]) < 0.01

    def test_source_splits_into_attributable_plus_excluded(self, reconciliation):
        r = reconciliation
        recomputed = r["source_reported_cash_usd"] - r["model_attributable_cash_usd"]
        assert np.allclose(recomputed.to_numpy(), r["source_minus_attributable_cash_usd"].to_numpy(), atol=0.01)

    def test_attributable_never_exceeds_source(self, reconciliation):
        r = reconciliation
        # attributable share is between 0 and slightly over 1 only if negatives exist; require <= 1.001
        share = r["model_attributable_cash_usd"] / r["source_reported_cash_usd"].replace(0, np.nan)
        assert (share.dropna() <= 1.001).all()

    def test_panel_target_matches_attributable(self, reconciliation):
        r = reconciliation
        if "panel_target_cash_usd" in r.columns:
            diff = (r["model_attributable_cash_usd"] - r["panel_target_cash_usd"]).abs()
            assert diff.max() < 1.0


# ---------------------------------------------------------------------------
# Region sums to country
# ---------------------------------------------------------------------------
class TestAggregation:
    def test_region_cash_sums_to_country(self, region_month, country_month):
        rollup = region_month.groupby("month")["panel_cash_usd"].sum()
        country = country_month.set_index("month")["panel_cash_usd"]
        joined = pd.concat([rollup.rename("region"), country.rename("country")], axis=1).dropna()
        assert np.allclose(joined["region"].to_numpy(), joined["country"].to_numpy(), atol=0.01)

    def test_contract_counts_consistent(self, metrics):
        assert metrics["n_contracts_CASH"] + metrics["n_contracts_FINANCED"] == metrics["n_contracts"]


# ---------------------------------------------------------------------------
# Efficiency curve sanity
# ---------------------------------------------------------------------------
class TestEfficiency:
    def test_financed_efficiency_between_zero_and_reasonable(self, metrics):
        for k in ["eff_FINANCED_mob0", "eff_FINANCED_mob1", "eff_FINANCED_mob3", "eff_FINANCED_mob6"]:
            v = metrics[k]
            assert v is None or (0.0 <= v <= 1.5), f"{k} = {v} out of range"

    def test_cash_efficiency_present(self, metrics):
        assert "eff_CASH_mob0" in metrics


# ---------------------------------------------------------------------------
# Pilot table
# ---------------------------------------------------------------------------
class TestPilotTable:
    def test_pilot_regions_are_east_and_west(self, pilot):
        assert set(pilot["region_outreach"].dropna().unique()) <= {"East", "West"}

    def test_pilot_months_within_pilot_window(self, pilot):
        assert pilot["contact_month"].min() >= pd.Timestamp("2026-04-30")
        assert pilot["contact_month"].max() <= VALIDATION_END

    def test_pilot_has_cost_and_attempts(self, pilot):
        assert {"cost_usd", "attempts", "reached", "channel"}.issubset(pilot.columns)

    def test_east_cheaper_per_contact_than_west(self, pilot):
        cpc = (pilot.groupby("region_outreach").apply(lambda d: d["cost_usd"].sum() / max(1, d["attempts"].sum())))
        assert cpc["East"] < cpc["West"]


# ---------------------------------------------------------------------------
# The frozen final forecast, if present, obeys its own contract
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def forecast():
    for cand in [FEAT / "forecast_q3_2026_final.csv",
                 FEAT.parent / "deliver" / "part1_final_outputs" / "forecast_q3_2026_final.csv"]:
        if cand.exists():
            return pd.read_csv(cand, parse_dates=["month"])
    pytest.skip("frozen forecast CSV not found")


class TestFrozenForecast:

    def test_three_months(self, forecast):
        assert len(forecast) == 3
        assert list(forecast["month"].dt.strftime("%Y-%m")) == ["2026-07", "2026-08", "2026-09"]

    def test_low_base_high_ordered(self, forecast):
        assert (forecast["low"] <= forecast["base"]).all()
        assert (forecast["base"] <= forecast["high"]).all()

    def test_values_positive_and_plausible(self, forecast):
        # sanity band: monthly country collections in the low hundreds of thousands
        assert (forecast["base"] > 100_000).all()
        assert (forecast["base"] < 2_000_000).all()

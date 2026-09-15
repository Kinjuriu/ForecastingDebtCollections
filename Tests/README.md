# Pipeline test suite

Run the whole suite from this folder:

```bash
pip install pytest pandas numpy scikit-learn statsmodels
FEATURES_DIR=/path/to/extracted/dlight_feature_engineering_outputs_v3 pytest -v
```

`FEATURES_DIR` should point at the folder where you unzipped
`dlight_feature_engineering_outputs_v3.zip` (the one holding
`contract_month_features_v3.csv`, `country_month_features_v3.csv`,
`feature_metrics.json`, etc.). If it is not set, the feature-output tests skip
and the two core suites still run. 66 tests, about five seconds.

## What is tested, and why it matters

The suite has three layers, matching the three places the work could silently go
wrong.

### 1. `test_forecast_core.py` — the forecast maths (28 tests)

Runs the forecast engine on tiny, hand-built contract books where the right
answer can be checked with a calculator.

- **The contractual schedule** bills the deposit in the sale month, then the daily
  amount on the *real* number of days in each month: February bills 28 days, a
  31-day month bills 31, and cumulative billing never exceeds the price. This is
  the fix that removed the fake February dip; the tests lock it in.
- **The forecast invariants** that must always hold: the five cash components add
  up to the total, the region forecasts add up to the country, `low <= base <=
  high` is enforced, a higher collection level produces more existing-book cash,
  higher sales attainment produces more new-sales cash, and the collection lever
  never moves the accounting bridge.
- **The leakage guard** refuses any row dated after 30 June 2026 and never reads a
  month after the forecast origin.

### 2. `test_pilot_core.py` — the pilot evaluation method (17 tests)

The important tests build a synthetic world with a treatment effect we choose,
then check the method finds it.

- **Recovery of a known effect:** inject a $2.00 per-contract-month lift in East
  and the difference-in-differences returns $2.00 (within 25 cents) with an
  interval clear of zero.
- **No false positives:** in a world with no effect, the interval includes zero.
- **Event study and placebo:** the pre-pilot gap is flat, the gap jumps by the
  injected amount after the pilot starts, and a placebo "pilot" placed in the pre
  period finds nothing.
- **Budget rules:** the allocation always sums to $8,000; a programme whose cash is
  below its cost, or whose interval touches zero, is refused scale money and the
  remainder goes to the learning reserve.
- **Matching hygiene:** treated rows really are contacted customers, controls were
  never contacted in that pilot, and the balance statistic behaves.

### 3. `test_feature_outputs.py` — the frozen feature artifacts (21 tests)

Does not rebuild anything; asserts the promises the feature stage makes to
everything downstream, against the real output files.

- Schema and the no-data-after-June boundary.
- Every one of the feature notebook's own sanity checks is green.
- **Reconciliation:** all model-attributable cash lands in the panel, and source
  cash splits exactly into attributable plus the excluded reasons.
- Region cash sums to country; contract counts add up; efficiency values are in a
  sane range; the pilot table covers only East and West within the pilot window;
  and the frozen forecast CSV has three ordered months of plausible values.

## Two bugs these tests caught

Writing the tests surfaced two real defects in the forecast core, both now fixed:

1. `lookup_eff` crashed on a read-only array when a region *and* its country
   fallback were both missing an age cell.
2. `forecast` produced NaN totals when the region level table was empty (a book
   with no in-schedule contracts at the origin); it now falls back to a neutral
   level of 1.0.

Neither could fire on the real panel, so the frozen forecast CSV is
byte-identical before and after the fix (same SHA-256). They are hardening for
re-runs on different data, which is exactly what a test suite is for.

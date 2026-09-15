# Decision log

How this project moved from a first pass at cleaning through to the final forecast and pilot evaluation, and why each stage exists.

## Data cleaning

Four iterations. The first three establish the chronological split (an estimation period for fitting, a validation period for tuning, and a sealed period held back until the forecast is frozen) and the cleaning rules: standardising fields, auditing exact duplicates, quarantining conflicting records, and flagging anything that falls outside the expected date range. The fourth iteration, `data_cleaning_backbone.ipynb`, is the version that ships: it carries the same rules with a couple of additional checks added after reviewing the first three passes.

## Feature engineering

Two independent builds ran in parallel at this stage, exploring different ways to structure the analytical panel. Both converged on the same core requirements: an explicit cohort/contractual backbone (expected deposit, expected repayment schedule, cumulative expected versus actual cash), a clear separation between in-scheduled-term and post-tenor recovery cash, and a reconciliation bridge between the row-level panel and the total reported cash. `feature_engineering_backbone.ipynb` is the merged result.

## Model building

**Public backbone v0** (`public_backbone_v0.ipynb`) is a deliberately simple benchmark: a trailing three-month average, with no segmentation and no modelling. Every later model is compared against it; if a more complex model can't beat this, the complexity isn't earning its place.

**The forecast (Part 1)** went through several rounds:

- `part1_forecast_v1.ipynb` — an early version, along with a written self-critique that shaped what came after (score on more than one metric, don't infer causality from timing alone, validate the actual pipeline rather than just the final numbers).
- `part1_forecast_v2_core.ipynb` and `part1_forecast_v2_working.ipynb` — the same underlying cohort/vintage model, split into a clean shareable version and a working notebook with more diagnostics, per-region checks, and a bottom-up-versus-top-down reconciliation.
- `part1_forecast_v3_challenger.ipynb` — adds a regression-based challenger and a damped-trend cross-check alongside the cohort model.
- `part1_forecast_final.ipynb` — the version that shipped. It corrects a systematic bias from an average-month billing assumption (real calendar days are used instead), fits the collection curve by region, product and age with shrinkage for thin segments, separates a short-window "current level" factor from the longer-history curve shape, flags and excludes anomalous region-months from curve fitting, and reports two backtest variants — one matching the earlier approach and one where the level window is kept genuinely out of sample, which is the fair test of the method actually used.

**The pilot evaluation (Part 2)** compares two outreach programmes that weren't randomly assigned, so it leans on two complementary methods rather than a raw comparison: a difference-in-differences design against non-pilot regions, and within-region matching for customers where a fair uncontacted comparison group actually exists. `part2_pilot_v1.ipynb` is the earlier version; `part2_pilot_final.ipynb` corrects an unrepresentative matched sample in the earlier version, adds a cluster bootstrap for the confidence intervals, and reports which method is credible for which pilot rather than defaulting to one method for both.

## What carried through every stage

Nothing dated after the validation cutoff is ever loaded into model development; every notebook checks for this and refuses to run if it finds a violation. Outputs are cleared before anything is committed. Each round of a notebook exists because the previous round had a specific, named shortcoming — an approximation that biased a result, a metric that hid a failure mode, a comparison that wasn't actually fair — and the next round exists to fix that one thing.

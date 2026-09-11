# Forecasting Debt Collections

Cleaning and analysing PayGo (pay-as-you-go) solar financing data, contracts,
payments, customer calls, service tickets, and collections outreach, to build a
short-term collections forecast and to evaluate whether a regional customer
outreach pilot is worth scaling. The source data is not committed here; it stays
local and is only read by the notebooks (see Data below).

## Workflow

1. **`notebooks/01_data_cleaning.ipynb`** takes the five raw source files, cleans
   and standardises each one, runs cross-file integrity and timing audits, and
   splits the result into a development set and a held-out test set so a forecast
   can be built without peeking at the answer. It does not do feature engineering
   or modelling.
2. **A forecasting / pilot-evaluation notebook** will read the cleaned
   development data, build the collections forecast, and evaluate the outreach
   pilot. It has not been built yet, see Status below.
3. Both stages currently run as Colab notebooks: upload the source CSVs at the
   start of a session, download the outputs at the end. Converting them to plain
   `.py` modules that read and write local paths directly is planned but not
   done yet.

## Status

- [x] Data cleaning notebook: done (this is the third revision; see the commit
      history on `notebooks/01_data_cleaning.ipynb` for how it evolved)
- [ ] Forecasting / modelling notebook: not started
- [ ] Pilot evaluation: not started
- [ ] Convert notebooks to `.py`

## Where things are

- `notebooks/` — the versioned notebook(s). Only the current, working version of
  each stage lives here; earlier drafts stay local.
- `data_dictionary.txt` — column-level definitions for the five source files.
- `AI_WORKFLOW.md` — how this repo's code gets written and edited with AI
  assistance, without burning tokens on things that don't need it.
- `tools/strip_notebook_outputs.py` — clears cell outputs before a notebook is
  committed, since outputs can contain real data and this repo is public.

## Data

Raw source files and any generated outputs (cleaned CSVs, audit files, ZIPs) are
intentionally excluded from version control; see `.gitignore`. Only cleaned code
and documentation are tracked.

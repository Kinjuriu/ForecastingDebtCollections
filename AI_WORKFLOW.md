# Working with AI on this repo

Practical habits for using Claude, ChatGPT or similar tools while writing and
cleaning code here, so we don't burn tokens moving text around that didn't need
moving. This isn't about agents or automated pipelines; it's about how we ask for
help day to day. Loosely inspired by how Spotify's engineering team routes coding
work to cheaper models for bulk, mechanical steps and saves the expensive model
for the parts that actually need judgment; the same idea applies to how we spend
our own attention and tokens when working with AI by hand.

## Give it the slice, not the whole notebook

The cleaning notebook is organised into numbered sections (1. Upload, 2. Imports,
3. Load files, and so on). When asking for a change, name the section and paste
only that section's cells, plus the one or two helper functions it calls, rather
than the whole notebook. The AI doesn't need section 19's packaging logic to help
fix section 6's payment-cleaning rule.

## Don't paste data, describe it

`data_dictionary.txt` already says what every column means. Point at it, or quote
the relevant three lines, instead of pasting sample rows or full column dumps.
If a specific value or shape matters, paste a `.head()` of a few rows, not a
`.info()` plus a `.describe()` plus a `.head(20)`.

## Ask for a diff, not a rewrite

For a one-function fix, ask for that function back, or a small patch, not the
"corrected full notebook." Rewriting 20 code cells to change one is how a small
edit turns into a five-minute read-and-diff exercise.

## Scope generation the same way

When the forecasting notebook gets built, build and review it section by section
(cleaning helpers, then the panel construction, then the backtest loop, then the
final forecast), the same way the cleaning notebook was built in numbered stages.
Asking for the entire modelling notebook in one shot means a much bigger result to
check line by line, and any mistake is more expensive to find.

## Keep a running note of decisions, not re-explanations

Cleaning rules and their reasoning already live in the notebook's own markdown
cells (see the "Payment duplicate policy" and "Events before sale month" sections,
for example). When asking AI to work on a later section, it's cheaper to say
"apply the same duplicate policy as section 6" than to re-explain the rule.

## Before committing a notebook

Run `python3 tools/strip_notebook_outputs.py notebooks/<file>.ipynb` first. This
repo is public, and a stray output cell is the easiest way for real numbers from
the source data to end up in git history. A pre-commit hook could automate this
later; for now it's one command, run by hand.

## Skills and hooks: not yet, but here's where they'd go

We haven't set up any Claude Code skills or hooks for this repo, and we're not
adding them speculatively. If patterns repeat enough to be worth automating
(the output-stripping step above is the most likely first candidate, followed by
a "check this cleaning notebook against the data dictionary" review skill once
the modelling notebook exists), they'd live under `.claude/`, which is already
reserved for this. Until then, this file is the reference.

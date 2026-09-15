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

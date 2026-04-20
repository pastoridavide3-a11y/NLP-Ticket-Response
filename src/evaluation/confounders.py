"""Check whether subgroup gaps survive controlling for length / richness.

A fair subgroup gap story needs to rule out trivial confounders. We regress
the target metric on ticket length, gold-response length, and a richness
proxy (type-token ratio), adding the subgroup indicator. If the subgroup
coefficient remains non-zero and the overall fit is meaningful, the gap is
not fully explained by the covariates.
"""
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _ttr(text: str) -> float:
    toks = (text or "").split()
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def build_covariates(per_ex: pd.DataFrame, ticket_col: str, gold_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=per_ex.index)
    out["ticket_len"] = per_ex[ticket_col].str.len().fillna(0).astype(float)
    out["gold_len"] = per_ex[gold_col].str.len().fillna(0).astype(float)
    out["ticket_ttr"] = per_ex[ticket_col].apply(_ttr)
    return out


def control_for_covariates(
    per_ex: pd.DataFrame, metric: str, group_col: str, ticket_col: str, gold_col: str
) -> pd.DataFrame:
    """Fit metric = f(covariates) on covariates only, then report per-group
    residual means as the confounder-adjusted signal. Including group dummies
    in the design matrix would force per-group residual means to zero by OLS
    first-order conditions, making the adjustment a no-op.
    """
    cov = build_covariates(per_ex, ticket_col, gold_col)
    X = cov.values
    y = per_ex[metric].to_numpy()

    model = LinearRegression()
    model.fit(X, y)
    y_hat = model.predict(X)
    residuals = y - y_hat

    # Adjusted mean per group = grand mean + group mean of residuals
    grand = float(y.mean())
    rows = []
    for g in per_ex[group_col].unique():
        mask = per_ex[group_col] == g
        rows.append(
            {
                "group": g,
                "n": int(mask.sum()),
                "raw_mean": float(y[mask].mean()),
                "adjusted_mean": float(grand + residuals[mask].mean()),
            }
        )
    rows.append(
        {"group": "__overall__", "n": int(len(y)), "raw_mean": grand, "adjusted_mean": grand}
    )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)

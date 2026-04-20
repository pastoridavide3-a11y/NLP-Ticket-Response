"""Subgroup aggregation + bootstrap confidence intervals."""
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd

from src import config


def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42
) -> Tuple[float, float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = rng.integers(0, n, size=(n_boot, n))
    means = values[draws].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (float(values.mean()), lo, hi)


def subgroup_table(
    per_ex: pd.DataFrame, group_col: str, metrics: Sequence[str], n_boot: int = 1000
) -> pd.DataFrame:
    rows = []
    order = None
    if group_col == "priority":
        order = config.PRIORITIES
    elif group_col == "queue":
        order = config.DEPARTMENTS
    groups = order if order else list(per_ex[group_col].unique())
    for g in groups:
        sub = per_ex[per_ex[group_col] == g]
        row = {group_col: g, "n": int(len(sub))}
        for m in metrics:
            mean, lo, hi = bootstrap_mean_ci(sub[m].to_numpy(), n_boot=n_boot)
            row[f"{m}_mean"] = mean
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
        rows.append(row)
    overall = {group_col: "__overall__", "n": int(len(per_ex))}
    for m in metrics:
        mean, lo, hi = bootstrap_mean_ci(per_ex[m].to_numpy(), n_boot=n_boot)
        overall[f"{m}_mean"] = mean
        overall[f"{m}_lo"] = lo
        overall[f"{m}_hi"] = hi
    rows.append(overall)
    return pd.DataFrame(rows)


def subgroup_gaps(table: pd.DataFrame, group_col: str, metrics: Sequence[str]) -> pd.DataFrame:
    body = table[table[group_col] != "__overall__"]
    rows = []
    for m in metrics:
        col = f"{m}_mean"
        rows.append(
            {
                "metric": m,
                "max": float(body[col].max()),
                "min": float(body[col].min()),
                "gap": float(body[col].max() - body[col].min()),
                "argmax_group": body.loc[body[col].idxmax(), group_col],
                "argmin_group": body.loc[body[col].idxmin(), group_col],
            }
        )
    return pd.DataFrame(rows)

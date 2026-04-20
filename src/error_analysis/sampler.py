"""Stratified worst-K sampler for manual review."""
import pandas as pd


def worst_k_per_group(
    per_ex: pd.DataFrame,
    metric: str,
    group_col: str,
    k: int = 10,
) -> pd.DataFrame:
    """Return the k lowest-metric rows within each group."""
    return (
        per_ex.sort_values(metric)
        .groupby(group_col, group_keys=False)
        .head(k)
        .reset_index(drop=True)
    )

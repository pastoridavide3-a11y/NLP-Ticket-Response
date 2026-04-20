"""Phase 5 + 6: evaluate all generation models, overall + per subgroup.

Consumes generations in results/generations/*.parquet, writes:
  - per-example metrics
  - overall + per-group aggregates with bootstrap 95% CIs
  - subgroup gap tables
  - confounder-adjusted per-group means
  - subgroup metric plots
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.utils import io as uio
from src.utils.plotting import save_fig
from src.utils.seeds import set_seed
from src.evaluation.lexical import score_batch
from src.evaluation.semantic import all_semantic
from src.evaluation.subgroup import subgroup_table, subgroup_gaps
from src.evaluation.confounders import control_for_covariates


METRICS_LEX = ("rouge1", "rouge2", "rougeL", "bleu", "chrf")
METRICS_SEM = ("tfidf_cos", "len_ratio")
PRIMARY_METRIC = "rougeL"  # main metric for worst-K sampling + plots


GEN_MODELS = ("constant", "retrieval", "group_template")


def _score(model: str, split: str) -> pd.DataFrame:
    path = config.GENERATIONS / f"{model}_{split}.parquet"
    df = uio.load_df(path)
    lex = score_batch(df["pred"].tolist(), df["gold"].tolist())
    sem = all_semantic(df["pred"].tolist(), df["gold"].tolist())
    for k, v in lex.items():
        df[k] = v
    for k, v in sem.items():
        df[k] = v
    return df


def _plot_group_bars(table: pd.DataFrame, group_col: str, metric: str, out_path: Path) -> None:
    body = table[table[group_col] != "__overall__"].copy()
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(body)), 4))
    x = np.arange(len(body))
    mean = body[f"{metric}_mean"].to_numpy()
    lo = mean - body[f"{metric}_lo"].to_numpy()
    hi = body[f"{metric}_hi"].to_numpy() - mean
    ax.bar(x, mean, yerr=[lo, hi], capsize=4, color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels(body[group_col].tolist(), rotation=30, ha="right")
    ax.set_ylabel(metric); ax.set_title(f"{metric} by {group_col}")
    save_fig(fig, out_path)


def main() -> None:
    set_seed(config.SEED)
    eval_tables = config.TABLES / "evaluation"
    eval_figs = config.FIGURES / "evaluation"
    eval_tables.mkdir(parents=True, exist_ok=True)
    eval_figs.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    for model in GEN_MODELS:
        for split in ("val", "test"):
            print(f"[eval] scoring {model}/{split}")
            df = _score(model, split)
            uio.save_df(df, config.GENERATIONS / f"{model}_{split}_scored.parquet")

            # overall
            row = {"model": model, "split": split, "n": int(len(df))}
            for m in METRICS_LEX + METRICS_SEM:
                row[m] = float(df[m].mean())
            overall_rows.append(row)

            # subgroup aggregates (bootstrap CIs) — only reported on test split
            if split != "test":
                continue

            for group_col in ("priority", "queue"):
                table = subgroup_table(df, group_col, METRICS_LEX + METRICS_SEM)
                uio.save_df(table, eval_tables / f"{model}_by_{group_col}.csv")

                gaps = subgroup_gaps(table, group_col, METRICS_LEX + METRICS_SEM)
                gaps.insert(0, "model", model)
                gaps.insert(1, "group_col", group_col)
                uio.save_df(gaps, eval_tables / f"{model}_gaps_{group_col}.csv")

                for m in (PRIMARY_METRIC, "tfidf_cos"):
                    _plot_group_bars(
                        table, group_col, m,
                        eval_figs / f"{model}_{group_col}_{m}.png",
                    )

            # confounder-adjusted means for primary metric
            for group_col in ("priority", "queue"):
                adj = control_for_covariates(
                    df, metric=PRIMARY_METRIC, group_col=group_col,
                    ticket_col="ticket", gold_col="gold",
                )
                adj.insert(0, "model", model)
                adj.insert(1, "group_col", group_col)
                adj.insert(2, "metric", PRIMARY_METRIC)
                uio.save_df(adj, eval_tables / f"{model}_adjusted_{group_col}.csv")

    overall = pd.DataFrame(overall_rows)
    uio.save_df(overall, eval_tables / "overall_metrics.csv")
    print(overall)

    # cross-model subgroup comparison (rougeL on test, by priority)
    frames = []
    for model in GEN_MODELS:
        t = pd.read_csv(eval_tables / f"{model}_by_priority.csv")
        t = t[t["priority"] != "__overall__"].copy()
        t["model"] = model
        frames.append(t[["model", "priority", f"{PRIMARY_METRIC}_mean", f"{PRIMARY_METRIC}_lo", f"{PRIMARY_METRIC}_hi", "n"]])
    combined = pd.concat(frames, ignore_index=True)
    uio.save_df(combined, eval_tables / f"cross_model_priority_{PRIMARY_METRIC}.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=combined, x="priority", y=f"{PRIMARY_METRIC}_mean",
        hue="model", order=config.PRIORITIES, ax=ax,
    )
    ax.set_ylabel(PRIMARY_METRIC); ax.set_title(f"{PRIMARY_METRIC} by priority across models (test)")
    save_fig(fig, eval_figs / f"cross_model_priority_{PRIMARY_METRIC}.png")


if __name__ == "__main__":
    main()

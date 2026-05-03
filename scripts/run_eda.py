"""Phase 1: exploratory data analysis. Saves tables + figures to results/."""
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


EDA_TABLES = config.TABLES / "eda"
EDA_FIGS = config.FIGURES / "eda"
EDA_TABLES.mkdir(parents=True, exist_ok=True)
EDA_FIGS.mkdir(parents=True, exist_ok=True)


def _count_tokens(series: pd.Series) -> pd.Series:
    return series.fillna("").map(lambda s: len(s.split()))


def main() -> None:
    df = uio.load_df(config.DATA_PROCESSED / "full.csv")
    print(f"[eda] rows={len(df)}")

    # --- 1. class distributions ------------------------------------------------
    pri = df["priority"].value_counts().reindex(config.PRIORITIES).fillna(0).astype(int)
    dep = df["queue"].value_counts().reindex(config.DEPARTMENTS).fillna(0).astype(int)
    uio.save_df(pri.rename_axis("priority").reset_index(name="count"), EDA_TABLES / "priority_counts.csv")
    uio.save_df(dep.rename_axis("department").reset_index(name="count"), EDA_TABLES / "department_counts.csv")

    joint = (
        df.groupby(["priority", "queue"]).size().rename("count").reset_index()
    )
    uio.save_df(joint, EDA_TABLES / "priority_department_joint.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=pri.index, y=pri.values, ax=ax, color="#4C72B0")
    ax.set_ylabel("count"); ax.set_xlabel("priority")
    save_fig(fig, EDA_FIGS / "priority_counts.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=dep.values, y=dep.index, ax=ax, color="#55A868")
    ax.set_xlabel("count"); ax.set_ylabel("department")
    save_fig(fig, EDA_FIGS / "department_counts.png")

    pivot = joint.pivot(index="queue", columns="priority", values="count").fillna(0).astype(int)
    pivot = pivot.reindex(index=config.DEPARTMENTS, columns=config.PRIORITIES).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("priority × department counts")
    save_fig(fig, EDA_FIGS / "priority_department_heatmap.png")

    # --- 2. length distributions ----------------------------------------------
    df["ticket_tokens"] = _count_tokens(df["ticket"])
    df["answer_tokens"] = _count_tokens(df["answer"])
    df["ticket_chars"] = df["ticket"].str.len()
    df["answer_chars"] = df["answer"].str.len()

    length_summary = (
        df.groupby("priority")[["ticket_tokens", "answer_tokens", "ticket_chars", "answer_chars"]]
        .agg(["mean", "median", "std"])
        .round(2)
    )
    length_summary.to_csv(EDA_TABLES / "length_by_priority.csv")

    length_summary_dep = (
        df.groupby("queue")[["ticket_tokens", "answer_tokens"]]
        .agg(["mean", "median", "std"])
        .round(2)
    )
    length_summary_dep.to_csv(EDA_TABLES / "length_by_department.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=df, x="priority", y="ticket_tokens", order=config.PRIORITIES, ax=axes[0])
    axes[0].set_title("ticket length (tokens) by priority")
    sns.boxplot(data=df, x="priority", y="answer_tokens", order=config.PRIORITIES, ax=axes[1])
    axes[1].set_title("answer length (tokens) by priority")
    save_fig(fig, EDA_FIGS / "length_by_priority.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, y="queue", x="ticket_tokens", order=config.DEPARTMENTS, ax=ax)
    ax.set_title("ticket length by department")
    save_fig(fig, EDA_FIGS / "ticket_length_by_department.png")

    # --- 3. top n-grams per priority ------------------------------------------
    from sklearn.feature_extraction.text import CountVectorizer

    def top_ngrams(texts: pd.Series, n: int = 15, ngram=(1, 2)) -> pd.DataFrame:
        cv = CountVectorizer(ngram_range=ngram, max_features=10_000, stop_words="english")
        X = cv.fit_transform(texts.fillna(""))
        sums = np.asarray(X.sum(axis=0)).ravel()
        vocab = np.array(cv.get_feature_names_out())
        order = np.argsort(-sums)[:n]
        return pd.DataFrame({"ngram": vocab[order], "count": sums[order]})

    rows = []
    for p in config.PRIORITIES:
        for ng in top_ngrams(df.loc[df["priority"] == p, "ticket"], n=15).itertuples(index=False):
            rows.append({"priority": p, "ngram": ng.ngram, "count": int(ng.count)})
    uio.save_df(pd.DataFrame(rows), EDA_TABLES / "top_ngrams_by_priority.csv")

    rows = []
    for d in config.DEPARTMENTS:
        for ng in top_ngrams(df.loc[df["queue"] == d, "ticket"], n=15).itertuples(index=False):
            rows.append({"department": d, "ngram": ng.ngram, "count": int(ng.count)})
    uio.save_df(pd.DataFrame(rows), EDA_TABLES / "top_ngrams_by_department.csv")

    # --- 4. semantic separability via TF-IDF + TruncatedSVD (2D) --------------
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    SAMPLE = min(5000, len(df))
    sub = df.sample(SAMPLE, random_state=config.SEED)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=30_000, sublinear_tf=True)
    X = vec.fit_transform(sub["ticket"])
    svd = TruncatedSVD(n_components=2, random_state=config.SEED)
    XY = svd.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in ((axes[0], "priority", "priority"), (axes[1], "queue", "department")):
        sns.scatterplot(
            x=XY[:, 0], y=XY[:, 1], hue=sub[col].values,
            palette="tab10", s=8, alpha=0.6, ax=ax, linewidth=0,
        )
        ax.set_title(f"TF-IDF → SVD(2) colored by {title}")
        ax.legend(loc="best", fontsize=7, markerscale=1.5, ncol=1)
    save_fig(fig, EDA_FIGS / "tfidf_svd_scatter.png")

    # --- 5. summary report ----------------------------------------------------
    summary = {
        "rows": int(len(df)),
        "priority_counts": pri.to_dict(),
        "department_counts": dep.to_dict(),
        "mean_ticket_tokens": float(df["ticket_tokens"].mean()),
        "mean_answer_tokens": float(df["answer_tokens"].mean()),
        "median_ticket_tokens": float(df["ticket_tokens"].median()),
        "median_answer_tokens": float(df["answer_tokens"].median()),
    }
    uio.save_json(summary, EDA_TABLES / "summary.json")
    print(f"[eda] wrote tables to {EDA_TABLES} and figures to {EDA_FIGS}")


if __name__ == "__main__":
    main()

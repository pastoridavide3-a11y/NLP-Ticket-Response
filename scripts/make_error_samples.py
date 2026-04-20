"""Phase 7: sample worst-K predictions per subgroup, attach heuristic tags."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src import config
from src.utils import io as uio
from src.utils.seeds import set_seed
from src.error_analysis.taxonomy import HeuristicTagger
from src.error_analysis.sampler import worst_k_per_group


PRIMARY = "rougeL"
MODELS = ("retrieval", "group_template")
K_PER_GROUP = 10


def main() -> None:
    set_seed(config.SEED)
    out_dir = config.SAMPLES
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = config.TABLES / "error_analysis"
    tables_dir.mkdir(parents=True, exist_ok=True)
    tagger = HeuristicTagger()

    all_rows = []
    for model in MODELS:
        df = uio.load_df(config.GENERATIONS / f"{model}_test_scored.parquet")
        tags = tagger.tag_batch(df["pred"].tolist(), df["gold"].tolist())
        tag_df = pd.DataFrame(tags)
        for c in tag_df.columns:
            df[f"tag_{c}"] = tag_df[c].to_numpy()

        for group_col in ("priority", "queue"):
            worst = worst_k_per_group(df, metric=PRIMARY, group_col=group_col, k=K_PER_GROUP)
            cols = [
                "id", "priority", "queue", "ticket", "gold", "pred",
                PRIMARY, "tfidf_cos", "tag_too_short", "tag_too_long",
                "tag_generic", "tag_low_overlap",
            ]
            cols = [c for c in cols if c in worst.columns]
            worst = worst[cols].copy()
            worst["model"] = model
            worst.to_csv(out_dir / f"worst_{model}_by_{group_col}.csv", index=False)

        # tag frequencies by subgroup
        tag_cols = [c for c in df.columns if c.startswith("tag_")]
        for group_col in ("priority", "queue"):
            freq = df.groupby(group_col)[tag_cols].mean().round(3)
            freq["n"] = df.groupby(group_col).size()
            freq = freq.reset_index()
            freq.insert(0, "model", model)
            freq.to_csv(tables_dir / f"tag_frequencies_{model}_by_{group_col}.csv", index=False)
            all_rows.append(freq)

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(tables_dir / "tag_frequencies_summary.csv", index=False)
    print(f"[errors] wrote worst-K samples to {out_dir}")
    print(f"[errors] wrote tag frequencies to {tables_dir}")


if __name__ == "__main__":
    main()

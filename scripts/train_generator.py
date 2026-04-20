"""Phase 4: fit all generation models, save predictions on val + test.

Models (first full version, sensible baselines only):
  - constant : one fixed polite reply (lower bound)
  - retrieval: TF-IDF nearest-neighbor answer (main baseline)
  - group_template: subgroup-centroid template (conditioned baseline)
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src import config
from src.utils import io as uio
from src.utils.seeds import set_seed
from src.generation.constant import ConstantResponder
from src.generation.retrieval import TfidfRetriever
from src.generation.group_template import GroupTemplateResponder


MODELS = ("constant", "retrieval", "group_template")


def _save(df: pd.DataFrame, preds, model_name: str, split: str) -> pd.DataFrame:
    out = df[["id", "ticket", "answer", "priority", "queue"]].copy()
    out = out.rename(columns={"answer": "gold"})
    out["pred"] = preds
    out["model"] = model_name
    out["split"] = split
    path = config.GENERATIONS / f"{model_name}_{split}.parquet"
    uio.save_df(out, path)
    print(f"[gen] wrote {path} rows={len(out)}")
    return out


def main() -> None:
    set_seed(config.SEED)
    train = uio.load_df(config.DATA_PROCESSED / "train.parquet")
    val = uio.load_df(config.DATA_PROCESSED / "val.parquet")
    test = uio.load_df(config.DATA_PROCESSED / "test.parquet")

    # --- constant --------------------------------------------------------------
    const = ConstantResponder().fit(train["ticket"], train["answer"])
    for name, df in (("val", val), ("test", test)):
        _save(df, const.predict(df["ticket"]), "constant", name)

    # --- retrieval -------------------------------------------------------------
    t0 = time.time()
    retr = TfidfRetriever().fit(train["ticket"].tolist(), train["answer"].tolist())
    print(f"[gen] retrieval fit in {time.time()-t0:.1f}s")
    joblib.dump(retr, config.MODELS / "retriever.joblib")
    for name, df in (("val", val), ("test", test)):
        t0 = time.time()
        preds = retr.predict(df["ticket"].tolist())
        print(f"[gen] retrieval predict {name} n={len(preds)} in {time.time()-t0:.1f}s")
        _save(df, preds, "retrieval", name)

    # --- group template --------------------------------------------------------
    gt = GroupTemplateResponder().fit(
        train["answer"].tolist(), train["priority"].tolist(), train["queue"].tolist()
    )
    joblib.dump(gt, config.MODELS / "group_template.joblib")
    for name, df in (("val", val), ("test", test)):
        preds = gt.predict(df["priority"].tolist(), df["queue"].tolist())
        _save(df, preds, "group_template", name)


if __name__ == "__main__":
    main()

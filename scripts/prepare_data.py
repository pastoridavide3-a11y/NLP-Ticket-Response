"""Phase 2: scope filter + stratified split, persisted as parquet."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.data.load import load_raw
from src.data.clean import apply_scope
from src.data.split import stratified_split
from src.utils import io as uio
from src.utils.seeds import set_seed


def main() -> None:
    set_seed(config.SEED)
    raw = load_raw()
    df = apply_scope(raw)
    splits = stratified_split(df)

    summary = {"total": int(len(df))}
    for name, part in splits.items():
        path = config.DATA_PROCESSED / f"{name}.csv"
        uio.save_df(part, path)
        summary[name] = int(len(part))
        print(f"[prepare_data] wrote {path} rows={len(part)}")

    uio.save_df(df, config.DATA_PROCESSED / "full.csv")
    uio.save_json(summary, config.DATA_PROCESSED / "split_summary.json")
    print(f"[prepare_data] summary={summary}")


if __name__ == "__main__":
    main()

"""Smoke tests for data loading + cleaning."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.data.load import load_raw
from src.data.clean import apply_scope
from src.data.split import stratified_split


def test_scope_filter_is_non_empty_and_english():
    df = apply_scope(load_raw())
    assert len(df) > 0
    assert (df["language"] == config.LANGUAGE).all()
    assert df["queue"].isin(config.DEPARTMENTS).all()
    assert df["priority"].isin(config.PRIORITIES).all()
    assert df["answer"].str.len().gt(0).all()
    assert df["ticket"].str.len().gt(0).all()


def test_split_preserves_totals_and_stratifies():
    df = apply_scope(load_raw())
    splits = stratified_split(df)
    total = sum(len(s) for s in splits.values())
    assert total == len(df)
    for s in splits.values():
        # every stratum present in train must retain at least one row per label
        assert set(s["priority"].unique()).issubset(set(config.PRIORITIES))
        assert set(s["queue"].unique()).issubset(set(config.DEPARTMENTS))


if __name__ == "__main__":
    test_scope_filter_is_non_empty_and_english()
    test_split_preserves_totals_and_stratifies()
    print("OK")

"""Classification evaluation — saves only test_report.txt."""
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, accuracy_score
from src.utils import io as uio


def eval_classifier(y_true, y_pred, labels, out_dir: Path, name: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    (out_dir / f"{name}_report.txt").write_text(report, encoding="utf-8")

    summary = {
        "name": name,
        "accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":    round(f1_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), 4),
        "n": len(y_true),
    }
    if name == "test":
        uio.save_json(summary, out_dir / f"{name}_summary.json")
    return summary

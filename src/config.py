"""Central config: paths, constants, label lists."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "customer_support_tickets.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
GENERATIONS = RESULTS / "generations"
SAMPLES = RESULTS / "samples"
MODELS = RESULTS / "models"

for _p in (DATA_PROCESSED, TABLES, FIGURES, GENERATIONS, SAMPLES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# Scope (justified in LIMITATIONS.md): English-only subset; top-10 real support queues.
LANGUAGE = "en"
DEPARTMENTS = [
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Other",
]

# departments merged into "Other" (count < 1500)
_SMALL_DEPARTMENTS = {
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
}
# English subset only has {low, medium, high}. Kept ordered for display.
PRIORITIES = ["low", "medium", "high"]

SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test

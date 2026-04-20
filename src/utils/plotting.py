"""Plot helpers — thin wrappers, fixed style."""
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")


def save_fig(fig, path, dpi: int = 150):
    path = str(path)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

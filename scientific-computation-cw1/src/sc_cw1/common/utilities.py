"""Small CSV/plot helpers; numerical updates stay in the question modules."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def output_dirs(output):
    """Create the figures and tables directories."""
    root = Path(output)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    return root


def write_csv(path, rows):
    """Save a list of row dictionaries, using LF line endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def save_history(path, result):
    """Save the initial point and all completed updates."""
    rows = []
    for k, x in enumerate(result["iterates"]):
        row = dict(k=k, x1=x[0], x2=x[1], value=result["values"][k],
                   gradient_norm=result["gradient_norms"][k], alpha=result["alphas"][k],
                   step_kind=result["step_kinds"][k], status=result["status"])
        if "residual_norms" in result:
            row["residual_norm"] = result["residual_norms"][k]
        rows.append(row)
    write_csv(path, rows)


def summary(result, method):
    """One compact row for a convergence comparison."""
    row = dict(method=method, status=result["status"], iterations=result["iterations"],
               x1=result["x"][0], x2=result["x"][1], value=result["values"][-1],
               gradient_norm=result["gradient_norms"][-1])
    if "residual_norms" in result:
        row["residual_norm"] = result["residual_norms"][-1]
    return row


def save_figure(fig, path):
    """Save a readable PNG for the report, then close it."""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def module_cli(run):
    """Allow each question module to select its output directory."""
    parser = argparse.ArgumentParser(description=run.__doc__)
    parser.add_argument("--output", default="results")
    run(parser.parse_args().output)

"""Plot interpolation errors from the laboratory CSV (requires matplotlib)."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("convergence.png"))
    args = parser.parse_args()
    with args.csv_file.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"degree", "h", "l2"}
        if not required.issubset(reader.fieldnames or []):
            parser.error("CSV must contain degree,h,l2 columns")
        norms = ["l2"] + (["h1"] if "h1" in reader.fieldnames else [])
        rows = list(reader)
    if not rows:
        parser.error("CSV contains no data; complete the starter TODOs first")
    groups = {}
    try:
        for row in rows:
            degree = int(row["degree"])
            values = {key: float(row[key]) for key in ["h"] + norms}
            if degree < 1 or any(not math.isfinite(v) or v <= 0
                                 for v in values.values()):
                raise ValueError("degrees, mesh sizes and errors must be positive")
            groups.setdefault(degree, []).append(values)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(f"Invalid CSV data: {error}")

    fig, axes = plt.subplots(1, len(norms), figsize=(6 * len(norms), 4.5),
                             squeeze=False)
    for ax, norm in zip(axes[0], norms):
        for degree, values in sorted(groups.items()):
            values.sort(key=lambda row: row["h"], reverse=True)
            hs = [row["h"] for row in values]
            errors = [row[norm] for row in values]
            if len(set(hs)) != len(hs):
                parser.error(f"Repeated mesh size for degree {degree}")
            order = degree + 1 if norm == "l2" else degree
            line, = ax.loglog(hs, errors, "o-", label=f"p={degree}")
            # Anchor the predicted slope at the finest computed point.
            reference = [errors[-1] * (h / hs[-1]) ** order for h in hs]
            ax.loglog(hs, reference, "--", color=line.get_color(), alpha=0.6,
                      label=f"slope {order}")
            if len(hs) > 1:
                rate = math.log(errors[-2] / errors[-1]) / math.log(hs[-2] / hs[-1])
                print(f"{norm}, p={degree}: last observed rate = {rate:.3f}")
        ax.set(xlabel="Cell side length h", ylabel=f"{norm.upper()} interpolation error")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"Written {args.output}")


if __name__ == "__main__":
    main()

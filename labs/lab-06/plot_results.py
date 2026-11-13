"""Render Poisson error tables and computed nodal fields; requires NumPy/Matplotlib."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from figure_policy import needs_generation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("csv_file", nargs="?", type=Path,
                    help="Student errors.csv; omit to render test data")
parser.add_argument("--output", type=Path, default=Path("convergence.png"))
parser.add_argument("--data", type=Path,
                    default=Path(__file__).resolve().parents[2] / "notes/assets/generated/lab-06")
args = parser.parse_args()
if not args.csv_file and all(not needs_generation(args.data / f"{case}-{kind}.png")
                           for case in ("exponential", "peaks")
                           for kind in ("solution", "convergence")):
    print("Laboratory 6 plots already exist; reusing them.")
    raise SystemExit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def convergence(csv_file, output):
    rows = np.atleast_1d(np.genfromtxt(csv_file, delimiter=",", names=True))
    if rows.size < 2 or not {"h", "l2", "h1"}.issubset(rows.dtype.names or ()):
        raise ValueError("Complete TODO 3 and run at least two refinements: CSV needs h,l2,h1")
    rows.sort(order="h")
    if any(np.any(~np.isfinite(rows[key])) or np.any(rows[key] <= 0)
           for key in ("h", "l2", "h1")) or len(np.unique(rows['h'])) != len(rows):
        raise ValueError("Mesh sizes and errors must be finite, positive, and mesh sizes distinct")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, norm, order in zip(axes, ("l2", "h1"), (2, 1)):
        h, error = rows['h'], rows[norm]
        ax.loglog(h, error, "o-", label="Computed error")
        ax.loglog(h, error[0]*(h/h[0])**order, "--", label=f"Slope {order}")
        rate = np.log(error[1]/error[0])/np.log(h[1]/h[0])
        print(f"{csv_file.stem} {norm}: finest rate {rate:.4f}")
        ax.set(xlabel="Cell side length h", ylabel=f"{norm.upper()} error",
               title="Full H1 norm" if norm == "h1" else "L2 norm")
        ax.grid(True, which="both", alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


if args.csv_file:
    convergence(args.csv_file, args.output)
else:
    for name in ("exponential", "peaks"):
        output = args.data / f"{name}-convergence.png"
        if needs_generation(output):
            convergence(args.data / f"{name}-errors.csv", output)
        output = args.data / f"{name}-solution.png"
        if not needs_generation(output):
            continue
        rows = np.genfromtxt(args.data / f"{name}-nodes.csv", delimiter=",", names=True)
        xs, ys = np.unique(rows['x']), np.unique(rows['y'])
        field = np.empty((len(ys), len(xs)))
        field[np.searchsorted(ys, rows['y']), np.searchsorted(xs, rows['x'])] = rows['solution']
        x, y = np.meshgrid(xs, ys)
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(x, y, field, cmap='viridis', rcount=len(ys), ccount=len(xs))
        ax.set(xlabel='x', ylabel='y', zlabel='u_h', title='Computed Q1 solution')
        ax = fig.add_subplot(122)
        picture = ax.pcolormesh(x, y, field, shading='gouraud', cmap='viridis')
        ax.contour(x, y, field, levels=10, colors='white', linewidths=.5)
        ax.set(xlabel='x', ylabel='y', aspect='equal', title='Nodal field and contours')
        fig.colorbar(picture, ax=ax, label='u_h')
        fig.tight_layout()
        fig.savefig(output, dpi=180)
        plt.close(fig)

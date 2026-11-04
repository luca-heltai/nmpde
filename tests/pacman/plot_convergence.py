"""Generate the Lecture 6 figures and tables from the local CSV files."""
import argparse
import csv
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_policy import needs_generation

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nmpde-matplotlib")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=Path("notes/assets/generated/lecture-06"))
parser.add_argument("--force", action="store_true", help="Regenerate existing outputs too")
args = parser.parse_args()
if args.force:
    os.environ["NMPDE_REGENERATE_FIGURES"] = "1"
root = args.data
from figure_policy import PLOT_NAMES
if not any(needs_generation(root / name) for name in PLOT_NAMES):
    print("All pacman plots and tables already exist; reusing them.")
    raise SystemExit(0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

labels = [r"2\pi", r"7\pi/4", r"3\pi/2", r"5\pi/4", r"\pi"]
colors = ["#315a9b", "#bd4b32", "#31805e", "#8b55a0"]
series = [np.genfromtxt(root / f"pacman-{i}/convergence.csv", delimiter=",", names=True)
          for i in range(5)]
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
# The sector diagram distinguishes the domain angle from the mesh geometry.
from matplotlib.patches import Wedge, Arc
fig, ax = plt.subplots(figsize=(5, 4.5), layout="constrained")
ax.add_patch(Wedge((0, 0), 1, 90, 360, facecolor="#e4edf8", edgecolor="#315a9b", lw=2))
ax.add_patch(Arc((0, 0), .65, .65, theta1=90, theta2=360, color="#bd4b32", lw=1.8))
ax.plot([0, -.65], [0, -.65], color="#315a9b")
ax.text(-.56, -.37, r"$r$", fontsize=16)
ax.text(-.65, .25, r"$\omega=3\pi/2$", fontsize=16, color="#bd4b32")
ax.text(.36, .44, "removed sector", ha="center", fontsize=11)
ax.text(.05, 1.03, r"$\theta=\theta_0$", fontsize=13)
ax.text(.75, -.10, r"$\theta=2\pi$", fontsize=13)
ax.plot(0, 0, "o", color="#bd4b32", ms=5)
ax.text(.04, -.17, "corner", fontsize=11)
ax.set(xlim=(-1.15, 1.15), ylim=(-1.12, 1.18), aspect="equal")
ax.axis("off")
if needs_generation(root / "pacman-geometry.svg"):
    fig.savefig(root / "pacman-geometry.svg")
plt.close(fig)
summary = []
for i, data in enumerate(series[:4]):
    alpha = 1 / (2 - i / 4)
    rates = [np.polyfit(np.log(data["h"][-3:]), np.log(data[key][-3:]), 1)[0]
             for key in ("H1_seminorm", "L2")]
    summary.append((i, alpha, rates[0], rates[1]))

for key, ylabel, filename in [
    ("H1_seminorm", r"$|u-u_h|_{H^1(\Omega)}$", "convergence-h1.svg"),
    ("L2", r"$\|u-u_h\|_{L^2(\Omega)}$", "convergence-l2.svg"),
]:
    if not any(needs_generation(root / name) for name in [filename, filename.replace(".svg", ".png")]):
        continue
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    fig.subplots_adjust(left=.08, right=.99, bottom=.17, top=.87, wspace=.26)
    for i, data in enumerate(series[:4]):
        exponent = (1 if key == "H1_seminorm" else 2) / (2 - i / 4)
        for axis, xkey in zip(axes, ("h", "dofs")):
            axis.loglog(data[xkey], data[key], "o-", color=colors[i],
                        label=rf"$\omega={labels[i]}$", markersize=4)
            x = data[xkey][-3:]
            power = exponent if xkey == "h" else -exponent / 2
            axis.loglog(x, data[key][-1] * (x / x[-1])**power,
                        "--", color=colors[i], alpha=.65)
    for axis in axes:
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=.2)
    axes[0].set_xlabel(r"Maximum cell diameter $h$")
    axes[0].invert_xaxis()
    axes[0].set_title("Refinement and corner regularity")
    axes[1].set_xlabel(r"Number of degrees of freedom $N$")
    axes[1].set_title(r"In two dimensions: $N\asymp h^{-2}$")
    axes[0].legend(fontsize=10)
    if needs_generation(root / filename):
        fig.savefig(root / filename, bbox_inches="tight")
    if needs_generation(root / filename.replace(".svg", ".png")):
        fig.savefig(root / filename.replace(".svg", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

if needs_generation(root / "tables.md"):
    with (root / "tables.md").open("w") as out:
        out.write("The following tables are generated from the CSV files by "
                  "`tests/pacman/plot_convergence.py`. Rates use successive maximum "
                  "cell diameters. The first rate is undefined.\n\n")
        for i, data in enumerate(series):
            out.write(rf"### Aperture $\omega={labels[i]}$" + "\n\n")
            if i == 4:
                out.write("Here $u=-y$ is affine. Errors measure roundoff and the linear "
                          "solver tolerance; convergence rates are omitted.\n\n")
            out.write("| Cycle | Cells | DoFs | $h$ | $L^2$ error | $H^1$ seminorm error | $p_{L^2}$ | $p_{H^1}$ |\n")
            out.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in data:
                rates = ["—", "—"] if row["cycle"] == 0 or i == 4 else [
                    f'{row["rate_L2"]:.3f}', f'{row["rate_H1"]:.3f}']
                out.write(f'| {int(row["cycle"])} | {int(row["cells"])} | {int(row["dofs"])} '
                          f'| {row["h"]:.3e} | {row["L2"]:.6e} | {row["H1_seminorm"]:.6e} '
                          f'| {rates[0]} | {rates[1]} |\n')
            out.write("\n")
        out.write("### Slopes fitted to the three finest meshes\n\n")
        out.write("| $\\omega$ | $\\alpha=\\pi/\\omega$ | Fitted $p_{H^1}$ | Fitted $p_{L^2}$ | Reference $2\\alpha$ |\n")
        out.write("|---:|---:|---:|---:|---:|\n")
        for i, alpha, h1, l2 in summary:
            out.write(rf"| ${labels[i]}$ | {alpha:.4f} | {h1:.4f} | {l2:.4f} | {2*alpha:.4f} |" + "\n")
if needs_generation(root / "fitted-rates.csv"):
    with (root / "fitted-rates.csv").open("w") as out:
        writer = csv.writer(out)
        writer.writerow(["case", "alpha", "fitted_H1", "fitted_L2"])
        writer.writerows(summary)
print(f"Generated requested missing/forced pacman plots and tables in {root}")

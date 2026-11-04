"""Run with pvpython to reproduce the five-panel Lecture 6 images."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_policy import needs_generation


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=Path("notes/assets/generated/lecture-06"))
parser.add_argument("--output", type=Path, default=Path("notes/assets/lecture-06"))
parser.add_argument("--cycle", type=int, default=1)
parser.add_argument("--force", action="store_true", help="Regenerate existing images too")
args = parser.parse_args()
if args.force:
    os.environ["NMPDE_REGENERATE_FIGURES"] = "1"
from figure_policy import PARAVIEW_NAMES
if not any(needs_generation(args.output / name) for name in PARAVIEW_NAMES):
    print("Both ParaView images already exist; reusing them.")
    raise SystemExit(0)
from paraview.simple import (CreateLayout, CreateView, XMLUnstructuredGridReader,
    Show, ColorBy, GetColorTransferFunction, GetOpacityTransferFunction,
    SaveScreenshot, Text, ResetSession, RenderAllViews)
args.output.mkdir(parents=True, exist_ok=True)
labels = ["2 pi (slit)", "7 pi / 4", "3 pi / 2", "5 pi / 4", "pi (affine control)"]
for field, association, filename in [
    ("solution", "POINTS", "pacman-solutions.png"),
    ("cell_H1_error", "POINTS", "pacman-cell-errors.png"),
]:
    if not needs_generation(args.output / filename):
        continue
    ResetSession()
    layout = CreateLayout(name=field)
    # Build five leaves by repeatedly splitting the rightmost leaf.
    leaf = 0
    leaves = []
    for remaining in (5, 4, 3, 2):
        layout.SplitHorizontal(leaf, 1.0 / remaining)
        leaves.append(2 * leaf + 1)
        leaf = 2 * leaf + 2
    leaves.append(leaf)
    for i, leaf in enumerate(leaves):
        view = CreateView("RenderView")
        layout.AssignView(leaf, view)
        view.ViewSize = [500, 600]
        view.UseColorPaletteForBackground = 0
        view.Background = [1, 1, 1]
        view.OrientationAxesVisibility = 0
        reader = XMLUnstructuredGridReader(FileName=[str(
            (args.data / f"pacman-{i}/solution-{args.cycle}.vtu").resolve())])
        reader.UpdatePipeline()
        display = Show(reader, view)
        display.Representation = "Surface With Edges"
        display.EdgeColor = [0.12, 0.18, 0.30]
        display.LineWidth = .5
        ColorBy(display, (association, field))
        lut = GetColorTransferFunction(field)
        # Use one scale across all apertures. Cell error is an integral norm
        # on a cell, not a pointwise gradient error or an estimator.
        maximum = 1.0 if field == "solution" else .12
        lut.ColorSpace = "RGB"
        lut.RGBPoints = [0, .267, .005, .329,
                         maximum*.25, .230, .322, .546,
                         maximum*.5, .128, .567, .551,
                         maximum*.75, .369, .789, .383,
                         maximum, .993, .906, .144]
        lut.AutomaticRescaleRangeMode = "Never"
        lut.RescaleTransferFunction(0, maximum)
        GetOpacityTransferFunction(field).RescaleTransferFunction(0, maximum)
        title = Text(registrationName=field + str(i))
        title.Text = "omega = " + labels[i]
        text_display = Show(title, view)
        text_display.Color = [0.08, 0.12, 0.20]
        text_display.FontSize = 16
        text_display.WindowLocation = "Upper Center"
        view.CameraParallelProjection = 1
        view.CameraPosition = [0, 0, 5]
        view.CameraFocalPoint = [0, 0, 0]
        view.CameraViewUp = [0, 1, 0]
        view.CameraParallelScale = 1.35
        if i == 4:
            display.SetScalarBarVisibility(view, True)
            from paraview.simple import GetScalarBar
            bar = GetScalarBar(lut, view)
            bar.Title = "u_h" if field == "solution" else "cell H1 error"
            bar.ComponentTitle = ""
            bar.TitleColor = [0.08, 0.12, 0.20]
            bar.LabelColor = [0.08, 0.12, 0.20]
            bar.TitleFontSize = 12
            bar.LabelFontSize = 10
            bar.Orientation = "Horizontal"
            bar.WindowLocation = "Lower Center"
    RenderAllViews()
    SaveScreenshot(str((args.output / filename).resolve()), layout,
                   ImageResolution=[2500, 600], TransparentBackground=0)
    print("Wrote", args.output / filename)

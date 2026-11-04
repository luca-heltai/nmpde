"""Regression checks for explicit force requests and preservation of outputs."""
import os
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from figure_policy import main, needs_generation, requested_by_event, PLOT_NAMES, PARAVIEW_NAMES

ROOT = Path(__file__).resolve().parents[1]


class FigurePolicy(unittest.TestCase):
    def test_commit_and_pr_keyword(self):
        self.assertFalse(requested_by_event({"head_commit": {"message": "Fix prose"}}))
        self.assertTrue(requested_by_event({"head_commit": {"message": "[regenerate figures]"}}))
        self.assertTrue(requested_by_event({"commits": [{"message": "[Regenerate Figures]"}]}))
        self.assertTrue(requested_by_event({"pull_request": {"title": "[regenerate figures]"}}))
        self.assertTrue(requested_by_event({"pull_request": {"body": "Update\n[regenerate figures]"}}))
        self.assertTrue(requested_by_event({}, ["Fix plots [regenerate figures]"]))
        self.assertFalse(requested_by_event({"pull_request": {"title": "Prose", "body": None}}))
        self.assertFalse(requested_by_event({"head_commit": {"message": "regenerate figures"}}))
        self.assertTrue(requested_by_event({}, associated_prs=[
            {"body": "[regenerate figures]", "title": "Merged PR"}]))

    def test_main_push_honors_a_marker_in_the_merged_pr_body(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            event, output = folder / "event.json", folder / "output"
            event.write_text(json.dumps({"head_commit": {"message": "Merge PR"}}))
            env = {"GITHUB_EVENT_NAME": "push", "GITHUB_EVENT_PATH": str(event),
                   "GITHUB_REPOSITORY": "example/course", "GITHUB_SHA": "a"*40,
                   "GITHUB_TOKEN": "test-only-token", "NMPDE_REGENERATE_FIGURES": "0"}
            response = io.StringIO(json.dumps([{"title": "Lecture", "body": "[regenerate figures]"}]))
            with patch.dict(os.environ, env), patch("figure_policy.urlopen", return_value=response), \
                 patch.object(sys, "argv", ["figure_policy.py", "--github-output", str(output)]):
                main()
            result = dict(line.split("=", 1) for line in output.read_text().splitlines())
            self.assertEqual(result["regenerate"], "1")
            self.assertEqual(result["plots"], "1")
            self.assertEqual(result["paraview"], "1")

    def test_missing_empty_present_and_forced(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"NMPDE_REGENERATE_FIGURES": "0"}):
            path = Path(folder) / "figure.svg"
            self.assertTrue(needs_generation(path))
            path.touch()
            self.assertTrue(needs_generation(path))
            path.write_text("existing figure")
            self.assertFalse(needs_generation(path))
            with patch.dict(os.environ, {"NMPDE_REGENERATE_FIGURES": "1"}):
                self.assertTrue(needs_generation(path))

    def test_cached_scripts_preserve_outputs_without_rendering_dependencies(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"NMPDE_REGENERATE_FIGURES": "0"}):
            folder = Path(folder)
            for name in [*PLOT_NAMES, *PARAVIEW_NAMES]:
                (folder / name).write_text("existing output")
            before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in folder.iterdir()}
            # The standard Python interpreter need not provide Matplotlib or
            # ParaView when all outputs already exist.
            for script, option in [("plot_convergence.py", "--data"),
                                   ("render_paraview.py", "--output")]:
                subprocess.run([sys.executable, str(ROOT / "tests/pacman" / script),
                                option, str(folder)], check=True, capture_output=True)
            after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in folder.iterdir()}
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

"""Shared incremental figure policy and GitHub commit/PR keyword detection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen

KEYWORD = "[regenerate figures]"
ASSETS = Path(__file__).resolve().parents[1] / "notes/assets"
PLOT_NAMES = ("pacman-geometry.svg", "convergence-h1.svg", "convergence-h1.png",
              "convergence-l2.svg", "convergence-l2.png", "tables.md", "fitted-rates.csv")
LAB06_PLOT_NAMES = tuple(f"{case}-{kind}.png" for case in ("exponential", "peaks")
                         for kind in ("solution", "convergence"))
PARAVIEW_NAMES = ("pacman-solutions.png", "pacman-cell-errors.png")


def needs_generation(path):
    path = Path(path)
    return (os.environ.get("NMPDE_REGENERATE_FIGURES") == "1"
            or not path.is_file() or path.stat().st_size == 0)


def required_files(assets=ASSETS):
    generated = assets / "generated"
    files = [generated / name for name in (
        "lecture-01-poisson-mesh.svg", "lab-02-two-quads.svg",
        "lab-02-flat-shell.svg", "lab-02-curved-shell.svg")]
    files += [generated / "lecture-06" / name for name in PLOT_NAMES]
    files += [assets / "lecture-06" / name for name in PARAVIEW_NAMES]
    for case in range(5):
        folder = generated / "lecture-06" / f"pacman-{case}"
        files += [folder / name for name in
                  ("convergence.csv", "metadata.txt", "solution.pvd")]
        files += [folder / f"solution-{cycle}.vtu" for cycle in range(6)]
    files += [generated / "lab-06" / name for name in LAB06_PLOT_NAMES]
    for case in ("exponential", "peaks"):
        files += [generated / "lab-06" / f"{case}-{suffix}.csv"
                  for suffix in ("errors", "nodes")]
        files += [generated / "lab-06" / f"{case}-{r}.vtu" for r in range(2, 7)]
    return files


def requested_by_event(event, commit_messages=(), associated_prs=()):
    pr = event.get("pull_request") or {}
    texts = [pr.get("title") or "", pr.get("body") or "",
             (event.get("head_commit") or {}).get("message") or ""]
    texts += [commit.get("message") or "" for commit in event.get("commits", [])]
    texts += [text for pr in associated_prs for text in (pr.get("title") or "", pr.get("body") or "")]
    return any(KEYWORD in text.lower() for text in [*texts, *commit_messages])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--needs", choices=("plots", "paraview"))
    args = parser.parse_args()
    if args.needs:
        folder = ASSETS / ("generated/lecture-06" if args.needs == "plots" else "lecture-06")
        names = PLOT_NAMES if args.needs == "plots" else PARAVIEW_NAMES
        raise SystemExit(0 if any(needs_generation(folder / p) for p in names) else 1)
    if args.verify:
        missing = [str(p) for p in required_files()
                   if not p.is_file() or p.stat().st_size == 0]
        if missing:
            raise SystemExit("Missing figure outputs:\n" + "\n".join(missing))
        print("All generated figure outputs are present.")
        return

    event_path = os.environ.get("GITHUB_EVENT_PATH")
    event = json.loads(Path(event_path).read_text()) if event_path else {}
    messages = []
    pr = event.get("pull_request")
    if pr:
        # Full checkout history includes PR commits. Pass SHAs as arguments;
        # commit messages and PR text are never interpreted by a shell.
        base, head = pr["base"]["sha"], pr["head"]["sha"]
        messages = [subprocess.check_output(
            ["git", "log", "--format=%B", f"{base}..{head}"], text=True)]
    associated_prs = []
    if os.environ.get("GITHUB_EVENT_NAME") == "push" and os.environ.get("GITHUB_TOKEN"):
        # A merge/squash commit need not contain a marker placed in the PR body.
        # Read the associated PR metadata so the main-branch build honors it too.
        api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        repository, sha = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_SHA"]
        request = Request(f"{api}/repos/{repository}/commits/{sha}/pulls", headers={
            "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        with urlopen(request, timeout=30) as response:
            associated_prs = json.load(response)
    force = (os.environ.get("NMPDE_REGENERATE_FIGURES") == "1"
             or requested_by_event(event, messages, associated_prs))
    missing = any(not p.is_file() or p.stat().st_size == 0 for p in required_files())
    generated = ASSETS / "generated/lecture-06"
    needs = {
        "regenerate": force,
        "missing": missing,
        "plots": force or any(needs_generation(generated / p) for p in PLOT_NAMES)
                 or any(needs_generation(ASSETS / "generated/lab-06" / p)
                        for p in LAB06_PLOT_NAMES),
        "paraview": force or any(needs_generation(ASSETS / "lecture-06" / p) for p in PARAVIEW_NAMES),
    }
    output = "".join(f"{key}={int(value)}\n" for key, value in needs.items())
    print(output, end="")
    if args.github_output:
        with args.github_output.open("a") as stream:
            stream.write(output)


if __name__ == "__main__":
    main()

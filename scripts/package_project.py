"""Create a ZIP of the project for submission.

The file list is git's view of the project — tracked files plus new files that are not ignored — so everything
.gitignore excludes stays out: virtual environments, node_modules, caches, uploads, the local database, the 40 MB CUAD
download (re-downloadable with scripts/download_datasets.py), training checkpoints and experiment-only model weights.
Usage: python scripts/package_project.py [--out dist/LegalLens_Final_5.zip]
"""
from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def project_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
                         capture_output=True, check=True).stdout.decode("utf-8")
    files = [ROOT / f for f in out.split("\0") if f]
    return sorted(p for p in files if p.is_file() and not p.relative_to(ROOT).as_posix().startswith("dist/"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist" / "LegalLens_Final_5.zip"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = project_files()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files:
            zf.write(p, Path("LegalLens") / p.relative_to(ROOT))
    size = sum(p.stat().st_size for p in files)
    print(f"{out}: {len(files)} files, {size / 1e6:.1f} MB uncompressed, {out.stat().st_size / 1e6:.1f} MB zipped")


if __name__ == "__main__":
    main()

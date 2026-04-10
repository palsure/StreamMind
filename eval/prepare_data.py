#!/usr/bin/env python3
"""Download and prepare benchmark datasets for StreamMind evaluation.

Usage:
    python prepare_data.py --benchmark nextqa --output-dir ./data/nextqa
    python prepare_data.py --benchmark egoschema --output-dir ./data/egoschema
    python prepare_data.py --benchmark ovobench --output-dir ./data/ovobench
    python prepare_data.py --benchmark all --output-dir ./data

Notes:
    - Ego4D requires a signed agreement; this script downloads annotations
      but videos must be obtained separately from https://ego4d-data.org.
    - OVO-Bench videos may need separate download from the benchmark repo.
    - LiveQA-Bench is our proposed benchmark and must be built from source
      streams (see build_liveqa.py).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def download_nextqa(output_dir: str):
    """Download NExT-QA annotations and video list."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NExT-QA Download Instructions")
    print("=" * 60)
    print()
    print("1. Clone the NExT-QA repository:")
    print("   git clone https://github.com/doc-doc/NExT-QA.git")
    print()
    print("2. Download annotations:")
    print(f"   cp NExT-QA/dataset/nextqa/*.csv {out}/")
    print()
    print("3. Download VidOR videos (NExT-QA uses VidOR video clips):")
    print("   See: https://xdshang.github.io/docs/vidor.html")
    print(f"   Place video files in {out}/videos/")
    print()
    print("4. Verify:")
    print(f"   ls {out}/val.csv {out}/videos/")
    print()

    anno_url = "https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa/val.csv"
    _download_if_available(anno_url, out / "val.csv")


def download_egoschema(output_dir: str):
    """Download EgoSchema annotations."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  EgoSchema Download Instructions")
    print("=" * 60)
    print()
    print("1. Clone the EgoSchema repository:")
    print("   git clone https://github.com/egoschema/EgoSchema.git")
    print()
    print("2. Copy annotations:")
    print(f"   cp EgoSchema/questions.json {out}/")
    print(f"   cp EgoSchema/subset_answers.json {out}/")
    print()
    print("3. Download Ego4D video clips (3-min clips):")
    print("   See: https://ego4d-data.org/docs/start-here/")
    print(f"   Place video files in {out}/videos/")
    print()

    for fname in ("questions.json", "subset_answers.json"):
        url = f"https://raw.githubusercontent.com/egoschema/EgoSchema/main/{fname}"
        _download_if_available(url, out / fname)


def download_ovobench(output_dir: str):
    """Download OVO-Bench annotations."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  OVO-Bench Download Instructions")
    print("=" * 60)
    print()
    print("1. Clone the OVO-Bench repository:")
    print("   git clone https://github.com/JoeLeelyf/OVO-Bench.git")
    print()
    print("2. Copy annotations:")
    print(f"   cp OVO-Bench/data/annotations.json {out}/")
    print()
    print("3. Download videos:")
    print("   Follow instructions in the OVO-Bench README")
    print(f"   Place video files in {out}/videos/")
    print()


def download_ego4d_nlq(output_dir: str):
    """Instructions for Ego4D-NLQ data."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Ego4D-NLQ Download Instructions")
    print("=" * 60)
    print()
    print("1. Request access at https://ego4d-data.org/")
    print("   (requires signing a data use agreement)")
    print()
    print("2. Download NLQ annotations:")
    print("   ego4d --output_directory {out} --datasets nlq")
    print(f"   Expected: {out}/nlq_val.json")
    print()
    print("3. Download video clips:")
    print("   ego4d --output_directory {out}/videos --datasets full_scale")
    print(f"   Expected: {out}/videos/<video_uid>.mp4")
    print()


def download_all(output_dir: str):
    """Download all benchmark data."""
    base = Path(output_dir)
    download_nextqa(str(base / "nextqa"))
    download_egoschema(str(base / "egoschema"))
    download_ovobench(str(base / "ovobench"))
    download_ego4d_nlq(str(base / "ego4d"))
    print("\n" + "=" * 60)
    print("  LiveQA-Bench")
    print("=" * 60)
    print(f"\n  LiveQA-Bench must be built from source streams.")
    print(f"  See eval/README.md for instructions.\n")


def _download_if_available(url: str, target: Path):
    """Try to download a file; skip silently on failure."""
    if target.exists():
        print(f"  Already exists: {target}")
        return
    try:
        import urllib.request
        print(f"  Downloading {target.name}...")
        urllib.request.urlretrieve(url, str(target))
        print(f"  Saved: {target}")
    except Exception as e:
        print(f"  Could not download {url}: {e}")
        print(f"  Please download manually and place at {target}")


def main():
    parser = argparse.ArgumentParser(description="Download benchmark data")
    parser.add_argument("--benchmark", required=True,
                        choices=["nextqa", "egoschema", "ovobench",
                                 "ego4d_nlq", "liveqa", "all"])
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory for data")
    args = parser.parse_args()

    handlers = {
        "nextqa": download_nextqa,
        "egoschema": download_egoschema,
        "ovobench": download_ovobench,
        "ego4d_nlq": download_ego4d_nlq,
        "all": download_all,
    }

    handler = handlers.get(args.benchmark)
    if handler:
        handler(args.output_dir)
    else:
        print(f"No download handler for {args.benchmark}")
        print("See eval/README.md for manual setup instructions.")


if __name__ == "__main__":
    main()

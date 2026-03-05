"""
Download sample videos for the StreamMind demo.

Two scenarios match the qualitative examples in the paper:
  1. cooking.mp4  - Kitchen cooking stream (person preparing food)
  2. surveillance.mp4 - Empty office space (no people, surveillance view)

Usage:
  python download_samples.py

The script downloads free-licensed clips from Mixkit and saves
them to ../frontend/samples/.
"""

import os
import sys
import urllib.request
import shutil

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

VIDEOS = {
    "cooking.mp4": {
        "url": "https://assets.mixkit.co/videos/43059/43059-720.mp4",
        "description": "Cook preparing food in a kitchen pan (Free, Mixkit)",
    },
    "surveillance.mp4": {
        "url": "https://assets.mixkit.co/videos/15478/15478-720.mp4",
        "description": "Empty office space, no people (Free, Mixkit)",
    },
}


def download(name, info):
    dest = os.path.join(SAMPLE_DIR, name)
    if os.path.exists(dest):
        print(f"  [skip] {name} already exists")
        return

    url = info["url"]
    print(f"  Downloading {name} ...")
    print(f"    {info['description']}")
    print(f"    URL: {url}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StreamMind-Demo/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"    Saved ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"    ERROR: {e}")
        print(f"    You can manually download from: {url}")
        print(f"    and save it as: {dest}")


def main():
    print("StreamMind — downloading sample videos\n")
    print(f"Output directory: {os.path.abspath(SAMPLE_DIR)}\n")

    for name, info in VIDEOS.items():
        download(name, info)

    print("\nDone.")
    print("Start the demo and click a sample video button to test.")


if __name__ == "__main__":
    main()

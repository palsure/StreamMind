#!/usr/bin/env python3
"""Convert evaluation result JSONs into LaTeX table values.

Reads result summary files from eval/results/ and prints the exact numbers
to paste into the paper's LaTeX tables.

Usage:
    python results_to_latex.py --results-dir ./results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_summary(results_dir: str, benchmark: str) -> dict | None:
    path = Path(results_dir) / f"{benchmark}_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def format_nextqa_ego_table(results_dir: str):
    """Print values for the combined NExT-QA + EgoSchema table."""
    nextqa = load_summary(results_dir, "nextqa")
    ego = load_summary(results_dir, "egoschema")

    print("\n% --- NExT-QA + EgoSchema (Tab:nextqa_ego) ---")
    if nextqa:
        acc = nextqa["accuracy"]
        gpt = nextqa.get("gpt_score", "TBD")
        print(f"% NExT-QA:  Acc = {acc}%  GPT = {gpt}")
        print(f"    \\textbf{{\\ours}} & \\textbf{{{acc}}} & \\textbf{{{gpt}}} ", end="")
    else:
        print("    \\textbf{\\ours} & \\textbf{TBD} & \\textbf{TBD} ", end="")

    if ego:
        acc = ego["accuracy"]
        print(f"& \\textbf{{{acc}}} \\\\")
    else:
        print("& \\textbf{TBD} \\\\")


def format_ovobench_table(results_dir: str):
    """Print values for the OVO-Bench table."""
    ovo = load_summary(results_dir, "ovobench")

    print("\n% --- OVO-Bench (Tab:ovobench) ---")
    if ovo:
        cats = ovo.get("per_category", {})
        bt = cats.get("BT", cats.get("backward_tracing", "TBD"))
        rp = cats.get("RP", cats.get("realtime_perception", "TBD"))
        fa = cats.get("FA", cats.get("forward_active", "TBD"))
        overall = ovo["accuracy"]
        print(f"    \\textbf{{\\ours}} & \\textbf{{{bt}}} & \\textbf{{{rp}}} "
              f"& \\textbf{{{fa}}} & \\textbf{{{overall}}} \\\\")
    else:
        print("    \\textbf{\\ours} & \\textbf{TBD} & \\textbf{TBD} "
              "& \\textbf{TBD} & \\textbf{TBD} \\\\")


def format_liveqa_table(results_dir: str):
    """Print values for the LiveQA-Bench table.

    Reads from liveqa_summary.json (evaluate.py output) or falls back to
    liveqa_full.json (run_docker_eval.py / Colab output).
    """
    liveqa = load_summary(results_dir, "liveqa")

    # Fallback: read run_docker_eval output
    if liveqa is None:
        docker_path = Path(results_dir) / "liveqa_full.json"
        if docker_path.exists():
            with open(docker_path) as f:
                data = json.load(f)
            liveqa = {
                "accuracy": data["summary"]["overall_accuracy"],
                "per_scope": data["summary"].get("per_scope_accuracy",
                                                  data["summary"].get("per_scope", {})),
            }

    print("\n% --- LiveQA-Bench (Tab:liveqa) ---")
    if liveqa:
        scopes = liveqa.get("per_scope", {})
        instant = scopes.get("instant", "TBD")
        recent = scopes.get("recent", "TBD")
        hist = scopes.get("historical", "TBD")
        overall = liveqa["accuracy"]
        print(f"    \\textbf{{\\ours}} & \\textbf{{{instant}}} & \\textbf{{{recent}}} "
              f"& \\textbf{{{hist}}} & \\textbf{{{overall}}} \\\\")
    else:
        print("    \\textbf{\\ours} & \\textbf{TBD} & \\textbf{TBD} "
              "& \\textbf{TBD} & \\textbf{TBD} \\\\")


def format_ego4d_inline(results_dir: str):
    """Print value for Ego4D-NLQ inline mention."""
    ego4d = load_summary(results_dir, "ego4d_nlq")

    print("\n% --- Ego4D-NLQ inline ---")
    if ego4d:
        r1 = ego4d.get("recall_at_1_iou03", "TBD")
        print(f"% \\ours achieves {r1}\\% R@1 (IoU$\\ge$0.3)")
    else:
        print("% \\ours achieves TBD\\% R@1 (IoU$\\ge$0.3)")


def format_ablation_table(results_dir: str):
    """Print values for the ablation table from run_docker_eval output."""
    path = Path(results_dir) / "ablation_summary.json"
    if not path.exists():
        return

    with open(path) as f:
        abl = json.load(f)

    print("\n% --- Ablation Study (Tab:ablation) ---")
    label_map = {
        "full":   r"\textbf{\ours (full model)}",
        "fifo":   r"\quad w/o SKM (FIFO buffer, $N\!=\!64$)",
        "no_tqr": r"\quad w/o TQR (attend to full memory always)",
        "N16":    r"\quad $N = 16$ memory slots",
        "N32":    r"\quad $N = 32$ memory slots",
        "N128":   r"\quad $N = 128$ memory slots",
    }
    for key in ["full", "fifo", "no_tqr", "N16", "N32", "N128"]:
        if key in abl:
            acc = abl[key]["overall_accuracy"]
            label = label_map.get(key, key)
            val = f"\\textbf{{{acc}}}" if key == "full" else str(acc)
            print(f"    {label:55s} & {val} \\\\")


def format_latency_table(results_dir: str):
    """Print GPU latency values from profiling output."""
    path = Path(results_dir) / "latency.json"
    if not path.exists():
        return

    with open(path) as f:
        lat = json.load(f)

    print("\n% --- Latency (Tab:latency, GPU column) ---")
    mapping = [
        ("CLIP encoding (per frame)", "clip_encode"),
        ("SKM scoring + update", "skm_update"),
        ("TQR scope classification", "tqr_classify"),
        ("BLIP captioning (per frame)", "blip_caption"),
        ("BLIP VQA (per frame)", "blip_vqa"),
        ("Flan-T5 synthesis", "flan_t5"),
    ]
    for label, key in mapping:
        if key in lat:
            ms = lat[key]["mean_ms"]
            if ms < 1:
                val = "$<$1"
            elif ms < 10:
                val = f"$\\sim${int(round(ms))}"
            else:
                val = str(int(round(ms)))
            print(f"    {label:35s} & {val} \\\\")


def main():
    parser = argparse.ArgumentParser(description="Convert results to LaTeX")
    parser.add_argument("--results-dir", default="./results",
                        help="Directory containing *_summary.json files")
    args = parser.parse_args()

    print("=" * 60)
    print("  LaTeX values from evaluation results")
    print("=" * 60)

    format_nextqa_ego_table(args.results_dir)
    format_ovobench_table(args.results_dir)
    format_liveqa_table(args.results_dir)
    format_ego4d_inline(args.results_dir)
    format_ablation_table(args.results_dir)
    format_latency_table(args.results_dir)

    print("\n% Copy the above values into paper/sec/4_experiments.tex")
    print("% Replace existing \\textbf{TBD} placeholders with actual numbers.")


if __name__ == "__main__":
    main()

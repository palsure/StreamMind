#!/usr/bin/env python3
"""StreamMind evaluation harness.

Usage:
    python evaluate.py --benchmark nextqa --data-root /path/to/nextqa
    python evaluate.py --benchmark egoschema --data-root /path/to/egoschema --subset subset
    python evaluate.py --benchmark ovobench --data-root /path/to/ovobench
    python evaluate.py --benchmark ego4d_nlq --data-root /path/to/ego4d
    python evaluate.py --benchmark liveqa --data-root /path/to/liveqa

    python evaluate.py --benchmark all --data-config config.json

Options:
    --memory-capacity   SKM capacity N (default: 64)
    --sample-fps        Frames per second to extract (default: 2.0)
    --max-samples       Limit number of samples for debugging (default: all)
    --output-dir        Directory for result JSON files (default: ./results)
    --gpt-score         Compute GPT-assisted scores (requires OPENAI_API_KEY)
    --device            Force device: cpu or cuda (default: auto)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks import BENCHMARKS
from benchmarks.base import EvalSample
from metrics import (
    EvalResult, accuracy, accuracy_by_group, gpt_score, mean_gpt_score,
    recall_at_1, format_results_table, save_results,
)
from pipeline import EvalPipeline, MultipleChoiceWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("streammind.eval")


def evaluate_benchmark(
    benchmark_name: str,
    data_root: str,
    pipeline: EvalPipeline,
    max_samples: int = 0,
    compute_gpt_score: bool = False,
    output_dir: str = "./results",
    **benchmark_kwargs,
) -> dict:
    """Run evaluation on a single benchmark. Returns summary dict."""

    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. "
                         f"Available: {list(BENCHMARKS.keys())}")

    BenchClass = BENCHMARKS[benchmark_name]
    bench = BenchClass(data_root, **benchmark_kwargs)

    if not bench.validate():
        raise FileNotFoundError(
            f"Benchmark data validation failed for {bench.name}. "
            f"Check that data is downloaded to {data_root}. "
            f"See eval/README.md for download instructions."
        )

    logger.info(f"Loading {bench.name} samples from {data_root}")
    samples = bench.load_samples()
    logger.info(f"Loaded {len(samples)} samples")

    if max_samples > 0:
        samples = samples[:max_samples]
        logger.info(f"Truncated to {max_samples} samples (debug mode)")

    mc_wrapper = MultipleChoiceWrapper(pipeline)
    results: list[EvalResult] = []

    for i, sample in enumerate(samples):
        logger.info(
            f"[{i+1}/{len(samples)}] {sample.sample_id}: "
            f"{sample.question[:60]}..."
        )

        t0 = time.time()
        try:
            answer = pipeline.evaluate_sample(
                sample.video_path,
                sample.question,
                sample.query_timestamp,
            )
        except Exception as e:
            logger.error(f"  Error processing {sample.sample_id}: {e}")
            results.append(EvalResult(
                sample_id=sample.sample_id,
                predicted="[ERROR]",
                ground_truth=sample.ground_truth,
                correct=False,
                metadata=sample.metadata,
            ))
            continue

        predicted_text = answer.get("answer", "")
        elapsed = time.time() - t0

        if sample.is_multiple_choice:
            pred_idx = mc_wrapper.select_option(
                sample.question, sample.options, predicted_text
            )
            correct = pred_idx == sample.correct_option_idx
            result = EvalResult(
                sample_id=sample.sample_id,
                predicted=sample.options[pred_idx] if pred_idx < len(sample.options) else predicted_text,
                ground_truth=sample.ground_truth,
                correct=correct,
                predicted_option_idx=pred_idx,
                metadata={
                    **sample.metadata,
                    "question": sample.question,
                    "raw_answer": predicted_text,
                    "scope": answer.get("scope"),
                    "latency_ms": answer.get("latency_ms"),
                    "elapsed_s": round(elapsed, 2),
                },
            )
        else:
            correct = _fuzzy_match(predicted_text, sample.ground_truth)
            result = EvalResult(
                sample_id=sample.sample_id,
                predicted=predicted_text,
                ground_truth=sample.ground_truth,
                correct=correct,
                metadata={
                    **sample.metadata,
                    "question": sample.question,
                    "scope": answer.get("scope"),
                    "latency_ms": answer.get("latency_ms"),
                    "elapsed_s": round(elapsed, 2),
                },
            )

        results.append(result)
        status = "CORRECT" if result.correct else "WRONG"
        logger.info(
            f"  [{status}] pred={result.predicted[:50]} | "
            f"gt={result.ground_truth[:50]} | {elapsed:.1f}s"
        )

    # --- Compute metrics ---
    overall_acc = accuracy(results)
    summary: dict = {"benchmark": bench.name, "n_samples": len(results),
                     "accuracy": round(overall_acc, 2)}

    # Category / scope breakdowns
    if benchmark_name == "liveqa":
        breakdowns = accuracy_by_group(results, "scope")
        summary["per_scope"] = {k: round(v, 2) for k, v in breakdowns.items()}
    elif benchmark_name == "ovobench":
        breakdowns = accuracy_by_group(results, "category")
        summary["per_category"] = {k: round(v, 2) for k, v in breakdowns.items()}
    elif benchmark_name == "nextqa":
        breakdowns = accuracy_by_group(results, "type")
        summary["per_type"] = {k: round(v, 2) for k, v in breakdowns.items()}
    else:
        breakdowns = None

    # GPT scoring
    if compute_gpt_score and benchmark_name in ("nextqa", "liveqa"):
        logger.info("Computing GPT-assisted scores...")
        gpt_score(results)
        summary["gpt_score"] = round(mean_gpt_score(results), 2)

    # R@1 for Ego4D-NLQ
    if benchmark_name == "ego4d_nlq":
        r1 = recall_at_1(results, iou_threshold=0.3)
        summary["recall_at_1_iou03"] = round(r1, 2)

    # Print summary
    gpt_mean = summary.get("gpt_score")
    bd = breakdowns if isinstance(breakdowns, dict) else None
    print(format_results_table(bench.name, overall_acc, bd, gpt_mean))

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{benchmark_name}_results.json")
    save_results(results, out_path)
    summary_path = os.path.join(output_dir, f"{benchmark_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    return summary


def _fuzzy_match(predicted: str, ground_truth: str) -> bool:
    """Simple fuzzy matching for open-ended QA."""
    pred = predicted.strip().lower()
    gt = ground_truth.strip().lower()
    if pred == gt:
        return True
    if gt in pred or pred in gt:
        return True
    pred_words = set(pred.split())
    gt_words = set(gt.split())
    if gt_words and pred_words:
        overlap = len(pred_words & gt_words) / len(gt_words)
        return overlap >= 0.5
    return False


def main():
    parser = argparse.ArgumentParser(
        description="StreamMind evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--benchmark", required=True,
                        choices=list(BENCHMARKS.keys()) + ["all"],
                        help="Benchmark to evaluate on")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Path to benchmark data directory")
    parser.add_argument("--data-config", type=str, default=None,
                        help="JSON config mapping benchmark names to data roots")
    parser.add_argument("--memory-capacity", type=int, default=64,
                        help="SKM memory capacity N")
    parser.add_argument("--sample-fps", type=float, default=2.0,
                        help="Frames per second to sample from video")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit samples for debugging (0 = all)")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--gpt-score", action="store_true",
                        help="Compute GPT-assisted scores (needs OPENAI_API_KEY)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: cpu or cuda")
    parser.add_argument("--subset", type=str, default="subset",
                        help="EgoSchema subset: 'subset' (500) or 'full' (5K)")
    parser.add_argument("--split", type=str, default="val",
                        help="Dataset split (val/test)")

    args = parser.parse_args()

    logger.info("Initializing StreamMind evaluation pipeline...")
    pipeline = EvalPipeline(
        memory_capacity=args.memory_capacity,
        sample_fps=args.sample_fps,
        frame_skip=1,
        device=args.device,
    )
    logger.info("Pipeline ready.")

    if args.benchmark == "all":
        if not args.data_config:
            parser.error("--data-config required when --benchmark=all")
        with open(args.data_config) as f:
            config = json.load(f)
        all_summaries = {}
        for bname, broot in config.items():
            if bname not in BENCHMARKS:
                logger.warning(f"Skipping unknown benchmark: {bname}")
                continue
            kwargs = {}
            if bname == "egoschema":
                kwargs["subset"] = args.subset
            summary = evaluate_benchmark(
                bname, broot, pipeline,
                max_samples=args.max_samples,
                compute_gpt_score=args.gpt_score,
                output_dir=args.output_dir,
                **kwargs,
            )
            all_summaries[bname] = summary

        combined_path = os.path.join(args.output_dir, "all_summaries.json")
        with open(combined_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        logger.info(f"All summaries saved to {combined_path}")
    else:
        if not args.data_root:
            parser.error("--data-root required for single benchmark")
        kwargs = {}
        if args.benchmark == "egoschema":
            kwargs["subset"] = args.subset
        if args.benchmark in ("nextqa", "ovobench"):
            kwargs["split"] = args.split
        evaluate_benchmark(
            args.benchmark, args.data_root, pipeline,
            max_samples=args.max_samples,
            compute_gpt_score=args.gpt_score,
            output_dir=args.output_dir,
            **kwargs,
        )


if __name__ == "__main__":
    main()

"""Metrics for StreamMind evaluation.

Implements:
  - Top-1 accuracy (multiple-choice and open-ended)
  - GPT-assisted scoring (Video-ChatGPT protocol, 1-5 scale)
  - Recall@1 at IoU >= threshold (for Ego4D-NLQ temporal grounding)
  - Per-category / per-scope breakdowns
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Result for a single sample."""
    sample_id: str
    predicted: str
    ground_truth: str
    correct: bool = False
    score: float = 0.0
    predicted_option_idx: int = -1
    metadata: dict = field(default_factory=dict)


def accuracy(results: list[EvalResult]) -> float:
    """Top-1 accuracy."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.correct) / len(results) * 100.0


def accuracy_by_group(results: list[EvalResult], group_key: str) -> dict[str, float]:
    """Compute accuracy per group (e.g., per question type or scope)."""
    groups: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        g = r.metadata.get(group_key, "unknown")
        groups[g].append(r)
    return {g: accuracy(rs) for g, rs in sorted(groups.items())}


def gpt_score(
    results: list[EvalResult],
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
) -> list[float]:
    """GPT-assisted answer scoring following the Video-ChatGPT protocol.

    Each (prediction, ground_truth, question) triple is scored 1-5.
    Requires OPENAI_API_KEY environment variable or explicit api_key.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "GPT scoring requires OPENAI_API_KEY env var or api_key argument"
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai  to use GPT scoring")

    client = OpenAI(api_key=api_key)
    scores: list[float] = []

    for r in results:
        question = r.metadata.get("question", "")
        prompt = (
            "You are evaluating a video question answering system.\n"
            f"Question: {question}\n"
            f"Ground truth answer: {r.ground_truth}\n"
            f"Predicted answer: {r.predicted}\n\n"
            "Score the predicted answer from 1 to 5:\n"
            "  1 = completely wrong or irrelevant\n"
            "  2 = partially related but mostly wrong\n"
            "  3 = partially correct, captures some aspects\n"
            "  4 = mostly correct with minor issues\n"
            "  5 = fully correct and complete\n\n"
            "Respond with ONLY the numeric score (1-5):"
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            score = float(re.search(r"[1-5]", text).group())
        except Exception:
            score = 1.0

        scores.append(score)
        r.score = score

    return scores


def mean_gpt_score(results: list[EvalResult]) -> float:
    """Average GPT score across all results (assumes scores already computed)."""
    scored = [r for r in results if r.score > 0]
    if not scored:
        return 0.0
    return sum(r.score for r in scored) / len(scored)


def temporal_iou(pred_start: float, pred_end: float,
                 gt_start: float, gt_end: float) -> float:
    """Compute temporal Intersection over Union."""
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def recall_at_1(results: list[EvalResult], iou_threshold: float = 0.3) -> float:
    """Recall@1 at IoU >= threshold for temporal grounding (Ego4D-NLQ).

    Expects:
      - result.predicted = "start-end" (e.g. "12.5-18.3")
      - result.ground_truth = "start-end"
      - or result.metadata contains gt_start, gt_end, pred_start, pred_end
    """
    if not results:
        return 0.0

    hits = 0
    for r in results:
        gt_s = r.metadata.get("gt_start")
        gt_e = r.metadata.get("gt_end")
        pred_s = r.metadata.get("pred_start")
        pred_e = r.metadata.get("pred_end")

        if gt_s is None:
            try:
                parts = r.ground_truth.split("-")
                gt_s, gt_e = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue

        if pred_s is None:
            try:
                parts = r.predicted.split("-")
                pred_s, pred_e = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue

        iou = temporal_iou(pred_s, pred_e, gt_s, gt_e)
        if iou >= iou_threshold:
            hits += 1

    return hits / len(results) * 100.0


def format_results_table(
    benchmark_name: str,
    overall: float,
    breakdowns: dict[str, float] | None = None,
    gpt_mean: float | None = None,
) -> str:
    """Format results as a printable table."""
    lines = [f"\n{'='*60}", f"  {benchmark_name} Results", f"{'='*60}"]

    if breakdowns:
        for group, val in breakdowns.items():
            lines.append(f"  {group:30s}  {val:6.1f}%")
        lines.append(f"  {'─'*40}")

    lines.append(f"  {'Overall Accuracy':30s}  {overall:6.1f}%")

    if gpt_mean is not None:
        lines.append(f"  {'GPT Score (1-5)':30s}  {gpt_mean:6.2f}")

    lines.append(f"{'='*60}\n")
    return "\n".join(lines)


def save_results(results: list[EvalResult], output_path: str):
    """Save detailed results to JSON."""
    data = []
    for r in results:
        data.append({
            "sample_id": r.sample_id,
            "predicted": r.predicted,
            "ground_truth": r.ground_truth,
            "correct": r.correct,
            "score": r.score,
            "predicted_option_idx": r.predicted_option_idx,
            "metadata": r.metadata,
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} results to {output_path}")

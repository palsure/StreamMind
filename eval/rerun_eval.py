#!/usr/bin/env python3
"""Re-run evaluation using saved LiveQA-Bench (skips benchmark generation)."""
from __future__ import annotations

import json
import logging
import os
import sys
import time

import cv2
import numpy as np

BACKEND = "/app/backend"
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from run_docker_eval import (
    QA, VIDEOS, RESULTS_DIR, extract_frames, frame_to_b64,
    evaluate_liveqa, profile_latency,
)
from stream_processor import StreamProcessor
from vlm_engine import VLMEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval")


def load_qa_from_saved() -> list[QA]:
    path = f"{RESULTS_DIR}/liveqa_bench.json"
    with open(path) as f:
        data = json.load(f)
    qa_list = []
    for item in data:
        qa_list.append(QA(
            qid=item["question_id"],
            stream=item["stream_id"],
            question=item["question"],
            timestamp=item["timestamp"],
            scope=item["scope"],
            ground_truth=item["answer"],
        ))
    return qa_list


def run_fifo_ablation(qa_list: list[QA]) -> dict:
    """Replace SKM with FIFO."""
    log.info("--- Ablation: FIFO ---")
    processor = StreamProcessor(memory_capacity=64, frame_skip=1)
    vlm = VLMEngine()
    processor.memory._compute_importance = lambda emb, ts: 0.0
    processor.memory._recompute_stored_importance = lambda: None

    qa_sorted = sorted(qa_list, key=lambda q: (q.stream, q.timestamp))
    results = []
    current_stream = None
    ingested_until = 0.0

    for qa in qa_sorted:
        if qa.stream != current_stream:
            processor.reset()
            current_stream = qa.stream
            ingested_until = 0.0

        if qa.timestamp > ingested_until:
            for t, b64, _ in extract_frames(VIDEOS[qa.stream],
                                             until_sec=qa.timestamp, sample_fps=2.0):
                if t > ingested_until:
                    processor.process_frame(b64, timestamp=t)
            ingested_until = qa.timestamp

        scope, _ = vlm.classify_temporal_scope(qa.question)
        context = processor.get_context_for_query(scope, current_time=qa.timestamp)
        result = vlm.generate_answer(qa.question, context, scope)
        predicted = result["answer"]
        gt = qa.ground_truth

        if gt.lower() in ("yes", "no"):
            correct = gt.lower() in predicted.lower()[:20]
        else:
            pred_words = set(predicted.lower().split())
            gt_words = set(gt.lower().split())
            correct = (len(pred_words & gt_words) / len(gt_words) >= 0.3) if gt_words else False

        results.append({"correct": correct, "scope": qa.scope})
        log.info(f"  {'OK' if correct else 'FAIL'} scope={scope} Q={qa.question[:40]}")

    total = sum(1 for r in results if r["correct"])
    acc = total / len(results) * 100 if results else 0.0
    scope_acc = {}
    for s in ("instant", "recent", "historical"):
        items = [r for r in results if r["scope"] == s]
        scope_acc[s] = round(sum(1 for r in items if r["correct"]) / len(items) * 100, 1) if items else 0
    return {"config": "fifo", "overall_accuracy": round(acc, 1), "per_scope": scope_acc, "n_samples": len(results)}


def run_no_tqr_ablation(qa_list: list[QA]) -> dict:
    """Always use full memory (no temporal routing)."""
    log.info("--- Ablation: no_tqr ---")
    processor = StreamProcessor(memory_capacity=64, frame_skip=1)
    vlm = VLMEngine()

    qa_sorted = sorted(qa_list, key=lambda q: (q.stream, q.timestamp))
    results = []
    current_stream = None
    ingested_until = 0.0

    for qa in qa_sorted:
        if qa.stream != current_stream:
            processor.reset()
            current_stream = qa.stream
            ingested_until = 0.0

        if qa.timestamp > ingested_until:
            for t, b64, _ in extract_frames(VIDEOS[qa.stream],
                                             until_sec=qa.timestamp, sample_fps=2.0):
                if t > ingested_until:
                    processor.process_frame(b64, timestamp=t)
            ingested_until = qa.timestamp

        context = processor.get_context_for_query("historical", current_time=qa.timestamp)
        result = vlm.generate_answer(qa.question, context, "historical")
        predicted = result["answer"]
        gt = qa.ground_truth

        if gt.lower() in ("yes", "no"):
            correct = gt.lower() in predicted.lower()[:20]
        else:
            pred_words = set(predicted.lower().split())
            gt_words = set(gt.lower().split())
            correct = (len(pred_words & gt_words) / len(gt_words) >= 0.3) if gt_words else False

        results.append({"correct": correct, "scope": qa.scope})
        log.info(f"  {'OK' if correct else 'FAIL'} scope=historical Q={qa.question[:40]}")

    total = sum(1 for r in results if r["correct"])
    acc = total / len(results) * 100 if results else 0.0
    scope_acc = {}
    for s in ("instant", "recent", "historical"):
        items = [r for r in results if r["scope"] == s]
        scope_acc[s] = round(sum(1 for r in items if r["correct"]) / len(items) * 100, 1) if items else 0
    return {"config": "no_tqr", "overall_accuracy": round(acc, 1), "per_scope": scope_acc, "n_samples": len(results)}


def main():
    log.info("Re-running evaluation with fixed recent scope handling")

    qa_list = load_qa_from_saved()
    log.info(f"Loaded {len(qa_list)} QA pairs")

    # Full model eval
    full = evaluate_liveqa(qa_list, memory_capacity=64, label="full")
    log.info(f"Full model: {full['overall_accuracy']:.1f}%  per_scope={full['per_scope']}")

    # Memory size ablation
    for n in [16, 32, 128]:
        res = evaluate_liveqa(qa_list, memory_capacity=n, label=f"N{n}")
        log.info(f"N={n}: {res['overall_accuracy']:.1f}%")

    # FIFO ablation
    fifo = run_fifo_ablation(qa_list)
    log.info(f"FIFO: {fifo['overall_accuracy']:.1f}%")

    # No-TQR ablation
    no_tqr = run_no_tqr_ablation(qa_list)
    log.info(f"No TQR: {no_tqr['overall_accuracy']:.1f}%")

    # Save combined
    all_results = {
        "full": full,
        "fifo": fifo,
        "no_tqr": no_tqr,
    }
    # Load N variants
    for n in [16, 32, 128]:
        try:
            with open(f"{RESULTS_DIR}/liveqa_N{n}.json") as f:
                data = json.load(f)
            all_results[f"N{n}"] = data["summary"]
        except Exception:
            pass

    with open(f"{RESULTS_DIR}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    log.info("\n=== FINAL SUMMARY ===")
    for name, res in all_results.items():
        log.info(f"  {name:12s}: {res['overall_accuracy']:.1f}%  {res.get('per_scope', {})}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Baseline evaluation runner.

Evaluates published video-language baselines under the causal streaming
protocol. Each baseline receives only frames from [0, t_q].

This script provides a unified wrapper that:
  1. Loads benchmark samples
  2. Extracts causal frames up to t_q for each sample
  3. Calls the baseline model's inference API
  4. Computes and reports metrics

Supported baselines (requires separate model installation):
  - video_chatgpt     (https://github.com/mbzuai-oryx/Video-ChatGPT)
  - videollava        (https://github.com/PKU-YuanGroup/Video-LLaVA)
  - sevila            (https://github.com/Yui010206/SeViLA)
  - llavanextvideo    (https://github.com/LLaVA-VL/LLaVA-NeXT)
  - chatunivi         (https://github.com/PKU-YuanGroup/Chat-UniVi)
  - flash_vstream     (https://github.com/IVGSZ/Flash-VStream)
  - videollm_online   (https://github.com/showlab/VideoLLM-online)
  - dispider          (https://github.com/Mark12Ding/Dispider)

Usage:
    python run_baselines.py \\
        --baseline videollava \\
        --benchmark nextqa \\
        --data-root /path/to/nextqa \\
        --model-path /path/to/videollava/weights \\
        --output-dir ./results/baselines
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks import BENCHMARKS
from benchmarks.base import EvalSample
from metrics import (
    EvalResult, accuracy, accuracy_by_group,
    format_results_table, save_results,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("baselines")


class BaselineModel(ABC):
    """Abstract baseline model interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def load(self, model_path: str, device: str = "cuda"):
        ...

    @abstractmethod
    def answer(self, frames: list[Image.Image], question: str,
               options: list[str] | None = None) -> str:
        """Answer a question given a list of PIL Image frames."""
        ...


def extract_causal_frames(
    video_path: str, query_timestamp: float,
    target_n_frames: int = 16, sample_fps: float = 1.0,
) -> list[Image.Image]:
    """Extract frames from [0, query_timestamp] for baseline evaluation."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(query_timestamp * video_fps)
    frame_interval = max(1, total_frames // target_n_frames)

    frames: list[Image.Image] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx / video_fps > query_timestamp:
            break
        if frame_idx % frame_interval == 0 and len(frames) < target_n_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        frame_idx += 1

    cap.release()
    return frames


# ---- Baseline implementations ----
# Each baseline requires its own model code to be installed separately.
# These are template implementations showing the expected interface.

class VideoLLaVABaseline(BaselineModel):
    name = "VideoLLaVA"

    def load(self, model_path: str, device: str = "cuda"):
        try:
            from videollava.model import LlavaLlamaForCausalLM
            from videollava.conversation import conv_templates
            import torch
            self.device = device
            self.model = LlavaLlamaForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16
            ).to(device)
            self.model.eval()
            logger.info(f"Loaded {self.name} from {model_path}")
        except ImportError:
            raise ImportError(
                f"Install Video-LLaVA: "
                f"pip install git+https://github.com/PKU-YuanGroup/Video-LLaVA.git"
            )

    def answer(self, frames, question, options=None):
        prompt = question
        if options:
            opt_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(options))
            prompt = f"{question}\n{opt_str}\nAnswer with the letter of the correct option."
        # Placeholder: actual implementation depends on Video-LLaVA API
        raise NotImplementedError(
            "Implement Video-LLaVA inference. See their repo for API details."
        )


class LLaVANextVideoBaseline(BaselineModel):
    name = "LLaVA-Next-Video"

    def load(self, model_path: str, device: str = "cuda"):
        try:
            from llava.model.builder import load_pretrained_model
            self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
                model_path, None, "llava_next_video", device=device
            )
            logger.info(f"Loaded {self.name} from {model_path}")
        except ImportError:
            raise ImportError(
                f"Install LLaVA-NeXT: "
                f"pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git"
            )

    def answer(self, frames, question, options=None):
        raise NotImplementedError(
            "Implement LLaVA-Next-Video inference. See their repo for API details."
        )


BASELINE_REGISTRY: dict[str, type[BaselineModel]] = {
    "videollava": VideoLLaVABaseline,
    "llavanextvideo": LLaVANextVideoBaseline,
}


def evaluate_baseline(
    baseline: BaselineModel,
    benchmark_name: str,
    data_root: str,
    max_samples: int = 0,
    output_dir: str = "./results/baselines",
    n_frames: int = 16,
    **benchmark_kwargs,
) -> dict:
    """Evaluate a baseline model on a benchmark."""
    BenchClass = BENCHMARKS[benchmark_name]
    bench = BenchClass(data_root, **benchmark_kwargs)

    if not bench.validate():
        raise FileNotFoundError(f"Data validation failed for {bench.name}")

    samples = bench.load_samples()
    if max_samples > 0:
        samples = samples[:max_samples]

    results: list[EvalResult] = []

    for i, sample in enumerate(samples):
        logger.info(f"[{i+1}/{len(samples)}] {sample.sample_id}")

        try:
            frames = extract_causal_frames(
                sample.video_path, sample.query_timestamp,
                target_n_frames=n_frames,
            )
            options = sample.options if sample.is_multiple_choice else None
            pred = baseline.answer(frames, sample.question, options)
        except Exception as e:
            logger.error(f"  Error: {e}")
            pred = "[ERROR]"

        correct = False
        if sample.is_multiple_choice:
            idx_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
            pred_lower = pred.strip().lower()
            pred_idx = idx_map.get(pred_lower[0] if pred_lower else "", -1)
            correct = pred_idx == sample.correct_option_idx
        else:
            correct = sample.ground_truth.lower() in pred.lower()

        results.append(EvalResult(
            sample_id=sample.sample_id,
            predicted=pred,
            ground_truth=sample.ground_truth,
            correct=correct,
            metadata={**sample.metadata, "question": sample.question},
        ))

    overall_acc = accuracy(results)
    summary = {
        "baseline": baseline.name,
        "benchmark": bench.name,
        "n_samples": len(results),
        "accuracy": round(overall_acc, 2),
    }

    print(format_results_table(f"{baseline.name} on {bench.name}", overall_acc))

    os.makedirs(output_dir, exist_ok=True)
    save_results(results, os.path.join(
        output_dir, f"{baseline.name}_{benchmark_name}_results.json"
    ))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation runner")
    parser.add_argument("--baseline", required=True,
                        choices=list(BASELINE_REGISTRY.keys()),
                        help="Baseline model to evaluate")
    parser.add_argument("--benchmark", required=True,
                        choices=list(BENCHMARKS.keys()))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True,
                        help="Path to baseline model weights")
    parser.add_argument("--output-dir", default="./results/baselines")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    BaselineClass = BASELINE_REGISTRY[args.baseline]
    model = BaselineClass()
    model.load(args.model_path, device=args.device)

    evaluate_baseline(
        model, args.benchmark, args.data_root,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        n_frames=args.n_frames,
    )


if __name__ == "__main__":
    main()

"""Evaluation pipeline wrapper for StreamMind.

Wraps the demo backend components (StreamProcessor + VLMEngine) to run
offline evaluation on video files under the causal streaming protocol.
"""
from __future__ import annotations

import base64
import io
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BACKEND_DIR = str(Path(__file__).resolve().parent.parent / "demo" / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from memory_manager import MemoryManager
from stream_processor import StreamProcessor
from vlm_engine import VLMEngine

logger = logging.getLogger("streammind.eval")


class EvalPipeline:
    """Runs the StreamMind pipeline on a video file for evaluation."""

    def __init__(
        self,
        memory_capacity: int = 64,
        sample_fps: float = 2.0,
        frame_skip: int = 1,
        device: str | None = None,
    ):
        """
        Args:
            memory_capacity: SKM capacity (N).
            sample_fps: How many frames per second to extract from the video.
            frame_skip: StreamProcessor frame_skip (set to 1 for evaluation).
            device: Force a device ("cpu" or "cuda").
        """
        self.memory_capacity = memory_capacity
        self.sample_fps = sample_fps
        self.frame_skip = frame_skip

        self.processor = StreamProcessor(
            memory_capacity=memory_capacity,
            frame_skip=frame_skip,
        )
        self.vlm = VLMEngine()

        if device is not None:
            import torch
            self.processor.device = device
            if self.processor.model is not None:
                self.processor.model = self.processor.model.to(device)
            self.vlm.device = device
            for attr in ("caption_model", "vqa_model", "llm"):
                m = getattr(self.vlm, attr, None)
                if m is not None:
                    setattr(self.vlm, attr, m.to(device))

        self._models_ready = self.vlm.is_ready()
        if not self._models_ready:
            logger.warning("VLM models not loaded — answers will be empty")

    def reset(self):
        """Clear memory for a new video."""
        self.processor.reset()

    def ingest_video(self, video_path: str, until_time: float | None = None) -> int:
        """Feed video frames into the pipeline up to `until_time` seconds.

        Returns the number of frames actually processed (passed to SKM).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(video_fps / self.sample_fps))

        n_processed = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / video_fps
            if until_time is not None and current_time > until_time:
                break

            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=80)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                self.processor.process_frame(b64, timestamp=current_time)
                n_processed += 1

            frame_idx += 1

        cap.release()
        return n_processed

    def answer_question(self, question: str) -> dict:
        """Answer a question using the current memory state.

        Returns dict with keys: answer, scope, confidence, num_context_frames,
        latency_ms.
        """
        scope, confidence = self.vlm.classify_temporal_scope(question)
        context = self.processor.get_context_for_query(scope)
        result = self.vlm.generate_answer(question, context, scope)
        result["confidence"] = confidence
        return result

    def evaluate_sample(self, video_path: str, question: str,
                        query_timestamp: float) -> dict:
        """Full evaluation for one sample: reset -> ingest -> answer."""
        self.reset()
        n_frames = self.ingest_video(video_path, until_time=query_timestamp)
        result = self.answer_question(question)
        result["n_ingested_frames"] = n_frames
        return result


class MultipleChoiceWrapper:
    """Wraps EvalPipeline to select among multiple-choice options.

    Uses BLIP VQA to score each option against the context frames, or
    matches the free-text answer against options via string similarity.
    """

    def __init__(self, pipeline: EvalPipeline):
        self.pipeline = pipeline

    def select_option(self, question: str, options: list[str],
                      free_text_answer: str) -> int:
        """Return the index of the best-matching option."""
        answer_lower = free_text_answer.strip().lower()

        # Direct match
        for i, opt in enumerate(options):
            if opt.strip().lower() == answer_lower:
                return i

        # Substring match (answer contained in option or vice versa)
        scores = []
        for i, opt in enumerate(options):
            opt_lower = opt.strip().lower()
            if answer_lower in opt_lower or opt_lower in answer_lower:
                scores.append((i, len(opt_lower)))
            else:
                overlap = self._word_overlap(answer_lower, opt_lower)
                scores.append((i, overlap))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] if scores else 0

    @staticmethod
    def _word_overlap(a: str, b: str) -> int:
        words_a = set(a.split())
        words_b = set(b.split())
        return len(words_a & words_b)

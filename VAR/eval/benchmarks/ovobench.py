"""OVO-Bench benchmark loader.

OVO-Bench (CVPR 2025) is an online video understanding benchmark with 644
videos and multi-choice questions across three categories:
  - Backward Tracing (BT)
  - Real-time Perception (RP)
  - Forward Active Response (FA)

Expected directory layout:
  {data_root}/
    videos/            -- video files
    annotations.json   -- questions with category labels and timestamps

Download: https://github.com/JoeLeelyf/OVO-Bench
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import BaseBenchmark, EvalSample


class OVOBenchmark(BaseBenchmark):
    name = "OVO-Bench"

    def __init__(self, data_root: str | Path, split: str = "val"):
        super().__init__(data_root)
        self.split = split

    def validate(self) -> bool:
        if not super().validate():
            return False
        ok = True
        anno = self.data_root / "annotations.json"
        if not anno.exists():
            anno = self.data_root / f"{self.split}.json"
        if not anno.exists():
            print(f"[{self.name}] Annotation file not found in {self.data_root}")
            ok = False
        if not (self.data_root / "videos").exists():
            print(f"[{self.name}] Video directory not found: {self.data_root / 'videos'}")
            ok = False
        return ok

    def load_samples(self) -> list[EvalSample]:
        anno_path = self.data_root / "annotations.json"
        if not anno_path.exists():
            anno_path = self.data_root / f"{self.split}.json"
        video_dir = self.data_root / "videos"

        with open(anno_path, encoding="utf-8") as f:
            data = json.load(f)

        items = data if isinstance(data, list) else data.get("annotations", data.get("data", []))

        samples: list[EvalSample] = []
        for item in items:
            qid = str(item.get("question_id", item.get("id", "")))
            vid = item.get("video_id", item.get("video", ""))
            question = item.get("question", "")
            category = item.get("category", item.get("task_type", ""))

            options = item.get("options", [])
            if not options:
                options = [item.get(f"option_{i}", "") for i in range(4)]
                options = [o for o in options if o]

            correct_idx = item.get("answer", item.get("correct", 0))
            if isinstance(correct_idx, str):
                idx_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                correct_idx = idx_map.get(correct_idx.upper(), 0)
            gt = options[correct_idx] if correct_idx < len(options) else str(correct_idx)

            # OVO-Bench provides query timestamps for causal evaluation
            t_q = float(item.get("query_time", item.get("timestamp", 0)))
            if t_q <= 0:
                video_path = self._find_video(video_dir, vid)
                t_q = self._get_video_duration(video_path) if video_path else 60.0
            else:
                video_path = self._find_video(video_dir, vid)

            if video_path is None:
                continue

            samples.append(EvalSample(
                sample_id=f"ovobench_{qid}",
                video_path=str(video_path),
                question=question,
                ground_truth=gt,
                query_timestamp=t_q,
                options=options,
                correct_option_idx=correct_idx,
                metadata={"category": category, "video_id": vid},
            ))

        return samples

    @staticmethod
    def _find_video(video_dir: Path, vid: str) -> Path | None:
        for ext in (".mp4", ".avi", ".mkv", ".webm"):
            p = video_dir / f"{vid}{ext}"
            if p.exists():
                return p
        return None

    @staticmethod
    def _get_video_duration(video_path: Path | None) -> float:
        if video_path is None:
            return 60.0
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return n_frames / fps if n_frames > 0 else 60.0
        except Exception:
            return 60.0

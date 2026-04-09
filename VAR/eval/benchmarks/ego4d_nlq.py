"""Ego4D Natural Language Queries (NLQ) benchmark loader (streaming subset).

We select 500 queries from the validation set whose temporal windows fall
within the first half of each clip, enabling meaningful streaming truncation.
The model must locate the temporal window of the queried event given only
causal (past) frames.

Expected directory layout:
  {data_root}/
    videos/                   -- Ego4D video clips (MP4)
    nlq_val.json              -- NLQ validation annotations

Download: https://ego4d-data.org  (requires signed agreement)
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import BaseBenchmark, EvalSample


class Ego4DNLQBenchmark(BaseBenchmark):
    name = "Ego4D-NLQ"

    def __init__(self, data_root: str | Path, max_samples: int = 500):
        super().__init__(data_root)
        self.max_samples = max_samples

    def validate(self) -> bool:
        if not super().validate():
            return False
        ok = True
        anno = self.data_root / "nlq_val.json"
        if not anno.exists():
            print(f"[{self.name}] nlq_val.json not found: {anno}")
            ok = False
        if not (self.data_root / "videos").exists():
            print(f"[{self.name}] Video directory not found")
            ok = False
        return ok

    def load_samples(self) -> list[EvalSample]:
        anno_path = self.data_root / "nlq_val.json"
        video_dir = self.data_root / "videos"

        with open(anno_path, encoding="utf-8") as f:
            data = json.load(f)

        all_videos = data if isinstance(data, list) else data.get("videos", [])
        samples: list[EvalSample] = []

        for video_entry in all_videos:
            video_uid = video_entry.get("video_uid", "")
            video_path = self._find_video(video_dir, video_uid)
            if video_path is None:
                continue

            duration = self._get_video_duration(video_path)
            half_duration = duration / 2.0

            clips = video_entry.get("clips", [])
            for clip in clips:
                annotations = clip.get("annotations", [])
                for ann in annotations:
                    for lang in ann.get("language_queries", []):
                        query = lang.get("query", lang.get("description", ""))
                        if not query:
                            continue

                        # Temporal window [start, end] in seconds
                        clip_start = float(lang.get("clip_start_sec", 0))
                        clip_end = float(lang.get("clip_end_sec", 0))

                        if clip_end <= 0 or clip_end > half_duration:
                            continue

                        # For streaming: model sees frames up to clip_end
                        # (ground truth window must be fully in the past)
                        t_q = clip_end + 5.0  # small buffer after the event

                        gt_window = f"{clip_start:.1f}-{clip_end:.1f}"
                        qid = lang.get("annotation_uid", f"{video_uid}_{clip_start}")

                        samples.append(EvalSample(
                            sample_id=f"ego4d_nlq_{qid}",
                            video_path=str(video_path),
                            question=query,
                            ground_truth=gt_window,
                            query_timestamp=t_q,
                            metadata={
                                "video_uid": video_uid,
                                "gt_start": clip_start,
                                "gt_end": clip_end,
                                "video_duration": duration,
                            },
                        ))

                        if len(samples) >= self.max_samples:
                            return samples

        return samples

    @staticmethod
    def _find_video(video_dir: Path, vid: str) -> Path | None:
        for ext in (".mp4", ".avi", ".mkv", ".webm"):
            p = video_dir / f"{vid}{ext}"
            if p.exists():
                return p
        return None

    @staticmethod
    def _get_video_duration(video_path: Path) -> float:
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return n_frames / fps if n_frames > 0 else 300.0
        except Exception:
            return 300.0

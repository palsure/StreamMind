"""LiveQA-Bench loader (our benchmark).

LiveQA-Bench contains 50 simulated live streams (5-30 min) built by joining
Ego4D and Kinetics clips.  500 questions (10 per stream) span three scopes:
  - 170 instant, 180 recent, 150 historical

Expected directory layout:
  {data_root}/
    streams/           -- simulated stream videos (stream_001.mp4, ...)
    annotations.json   -- questions with scope labels and timestamps
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import BaseBenchmark, EvalSample


class LiveQABenchmark(BaseBenchmark):
    name = "LiveQA-Bench"

    def __init__(self, data_root: str | Path):
        super().__init__(data_root)

    def validate(self) -> bool:
        if not super().validate():
            return False
        ok = True
        anno = self.data_root / "annotations.json"
        if not anno.exists():
            print(f"[{self.name}] annotations.json not found: {anno}")
            ok = False
        if not (self.data_root / "streams").exists():
            print(f"[{self.name}] Streams directory not found")
            ok = False
        return ok

    def load_samples(self) -> list[EvalSample]:
        anno_path = self.data_root / "annotations.json"
        stream_dir = self.data_root / "streams"

        with open(anno_path, encoding="utf-8") as f:
            data = json.load(f)

        items = data if isinstance(data, list) else data.get("questions", [])
        samples: list[EvalSample] = []

        for item in items:
            qid = str(item.get("question_id", item.get("id", "")))
            stream_id = item.get("stream_id", item.get("video", ""))
            question = item.get("question", "")
            gt = item.get("answer", item.get("ground_truth", ""))
            scope = item.get("scope", "historical")
            t_q = float(item.get("timestamp", item.get("query_time", 0)))

            video_path = self._find_video(stream_dir, stream_id)
            if video_path is None:
                continue

            samples.append(EvalSample(
                sample_id=f"liveqa_{qid}",
                video_path=str(video_path),
                question=question,
                ground_truth=gt,
                query_timestamp=t_q,
                metadata={"scope": scope, "stream_id": stream_id},
            ))

        return samples

    @staticmethod
    def _find_video(stream_dir: Path, stream_id: str) -> Path | None:
        for ext in (".mp4", ".avi", ".mkv", ".webm"):
            p = stream_dir / f"{stream_id}{ext}"
            if p.exists():
                return p
        # Try with stream_ prefix
        for ext in (".mp4", ".avi", ".mkv", ".webm"):
            p = stream_dir / f"stream_{stream_id}{ext}"
            if p.exists():
                return p
        return None

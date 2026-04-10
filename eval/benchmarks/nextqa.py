"""NExT-QA benchmark loader with streaming conversion.

NExT-QA contains ~52K QA pairs over 5,440 video clips (avg 44s).
We apply a streaming truncation: for each question we derive a query
timestamp t_q and only expose frames from [0, t_q].

Expected directory layout:
  {data_root}/
    videos/          -- video files (e.g. 1234.mp4)
    val.csv          -- validation split annotations
    test.csv         -- test split annotations

Download: https://github.com/doc-doc/NExT-QA
"""
from __future__ import annotations

import csv
from pathlib import Path

from .base import BaseBenchmark, EvalSample


class NextQABenchmark(BaseBenchmark):
    name = "NExT-QA"

    def __init__(self, data_root: str | Path, split: str = "val"):
        super().__init__(data_root)
        self.split = split

    def validate(self) -> bool:
        if not super().validate():
            return False
        anno = self.data_root / f"{self.split}.csv"
        vids = self.data_root / "videos"
        ok = True
        if not anno.exists():
            print(f"[{self.name}] Annotation file not found: {anno}")
            ok = False
        if not vids.exists():
            print(f"[{self.name}] Video directory not found: {vids}")
            ok = False
        return ok

    def load_samples(self) -> list[EvalSample]:
        anno_path = self.data_root / f"{self.split}.csv"
        video_dir = self.data_root / "videos"
        samples: list[EvalSample] = []

        with open(anno_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row["video"]
                qid = row.get("qid", row.get("question_id", ""))

                video_path = self._find_video(video_dir, vid)
                if video_path is None:
                    continue

                options = [row.get(f"a{i}", "") for i in range(5)]
                options = [o for o in options if o]
                correct_idx = int(row.get("answer", 0))
                gt = options[correct_idx] if correct_idx < len(options) else row.get("answer", "")

                q_type = row.get("type", "")
                frame_count = int(row.get("frame_count", 0)) if row.get("frame_count") else 0

                # Streaming truncation: use the full clip length as t_q
                # (NExT-QA questions reference entire clip segments).
                # Actual t_q can be refined using temporal grounding metadata
                # if available.
                t_q = self._get_video_duration(video_path)

                samples.append(EvalSample(
                    sample_id=f"nextqa_{qid}",
                    video_path=str(video_path),
                    question=row["question"],
                    ground_truth=gt,
                    query_timestamp=t_q,
                    options=options,
                    correct_option_idx=correct_idx,
                    metadata={"type": q_type, "video_id": vid},
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
    def _get_video_duration(video_path: Path) -> float:
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return n_frames / fps if n_frames > 0 else 30.0
        except Exception:
            return 30.0

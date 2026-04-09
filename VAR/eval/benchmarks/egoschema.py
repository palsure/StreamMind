"""EgoSchema benchmark loader with streaming conversion.

EgoSchema provides 5K multiple-choice questions over 3-minute Ego4D clips.
We apply the same streaming truncation: clip at t_q.

Expected directory layout:
  {data_root}/
    videos/            -- 3-minute Ego4D clips as MP4
    questions.json     -- annotation file

Download: https://github.com/egoschema/EgoSchema
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import BaseBenchmark, EvalSample


class EgoSchemaBenchmark(BaseBenchmark):
    name = "EgoSchema"

    def __init__(self, data_root: str | Path, subset: str = "subset"):
        """
        Args:
            subset: "subset" for 500-question test subset, "full" for all 5K.
        """
        super().__init__(data_root)
        self.subset = subset

    def validate(self) -> bool:
        if not super().validate():
            return False
        anno = self.data_root / "questions.json"
        vids = self.data_root / "videos"
        ok = True
        if not anno.exists():
            print(f"[{self.name}] questions.json not found: {anno}")
            ok = False
        if not vids.exists():
            print(f"[{self.name}] Video directory not found: {vids}")
            ok = False
        return ok

    def load_samples(self) -> list[EvalSample]:
        anno_path = self.data_root / "questions.json"
        video_dir = self.data_root / "videos"

        with open(anno_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            items = list(data.values()) if not isinstance(next(iter(data.values()), None), dict) else list(data.values())
            if isinstance(next(iter(data.keys()), None), str) and isinstance(next(iter(data.values()), None), dict):
                items = [{"q_uid": k, **v} for k, v in data.items()]
        else:
            items = data

        if self.subset == "subset":
            subset_path = self.data_root / "subset_answers.json"
            if subset_path.exists():
                with open(subset_path, encoding="utf-8") as f:
                    subset_ids = set(json.load(f).keys())
                items = [it for it in items if it.get("q_uid", "") in subset_ids]

        samples: list[EvalSample] = []
        for item in items:
            q_uid = item.get("q_uid", item.get("id", ""))
            vid = item.get("video_uid", item.get("video", q_uid))
            question = item.get("question", "")

            options = []
            for i in range(5):
                opt = item.get(f"option {i}", item.get(f"option_{i}", ""))
                if opt:
                    options.append(opt)

            correct_idx = int(item.get("answer", item.get("truth", 0)))
            gt = options[correct_idx] if correct_idx < len(options) else str(correct_idx)

            video_path = self._find_video(video_dir, vid)
            if video_path is None:
                continue

            t_q = self._get_video_duration(video_path)

            samples.append(EvalSample(
                sample_id=f"egoschema_{q_uid}",
                video_path=str(video_path),
                question=question,
                ground_truth=gt,
                query_timestamp=t_q,
                options=options,
                correct_option_idx=correct_idx,
                metadata={"video_uid": vid},
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
            return n_frames / fps if n_frames > 0 else 180.0
        except Exception:
            return 180.0

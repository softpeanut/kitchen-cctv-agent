from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from kitchen_agent import (
    FRAMES_PER_HOUR,
    MODEL_ID,
    InputError,
    MlxVisionBackend,
    RunStats,
    answer_questions,
    atomic_write_json,
    discover_videos,
    load_questions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer structured kitchen CCTV questions")
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    stats = RunStats(started_at=started)
    try:
        questions = load_questions(args.questions)
        videos = discover_videos(args.videos, questions)
        source_minutes = sum(video.duration for video in videos.values()) / 60.0
        frame_budget = max(1, int(FRAMES_PER_HOUR * source_minutes / 60.0))
        backend = MlxVisionBackend(args.model)
        with tempfile.TemporaryDirectory(prefix="kitchen-cctv-") as temporary:
            answers, traces = answer_questions(
                questions,
                videos,
                backend,
                Path(temporary),
                frame_budget,
                stats,
            )
        atomic_write_json(args.out, answers)
        runtime = time.monotonic() - started
        log = {
            "runtime_seconds": round(runtime, 3),
            "source_video_minutes": round(source_minutes, 3),
            "frame_budget": frame_budget,
            "frames_processed": stats.frames_processed,
            "model_calls": stats.model_calls,
            "model": backend.model_id,
            "estimated_model_api_cost_usd": 0.0,
            "normalized_model_api_cost_per_60min_usd": 0.0,
            "questions": traces,
        }
        atomic_write_json(args.log, log)
    except InputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

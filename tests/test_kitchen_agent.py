from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kitchen_agent import (
    BudgetExhausted,
    FrameStore,
    InputError,
    Question,
    RunStats,
    VideoInfo,
    answer_type_instruction,
    atomic_write_json,
    extract_explicit_timestamp,
    load_questions,
    normalize_answer,
    parse_evidence_seconds,
    parse_json_object,
    question_route,
)


def test_load_questions_validates_unique_ids(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [
                {"id": "q1", "video_id": "v1", "type": "yes_no", "question": "Visible?"},
                {"id": "q1", "video_id": "v1", "type": "count", "question": "How many?"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(InputError, match="duplicate question id"):
        load_questions(path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("at 00:45", 45.0), ("at 01:02:03", 3_723.0), ("at 99:99", None), ("none", None)],
)
def test_extract_explicit_timestamp(text: str, expected: float | None) -> None:
    assert extract_explicit_timestamp(text) == expected


def test_question_route_prefers_declared_ocr_at_explicit_time() -> None:
    question = Question("q", "v", "visibility", "Is the order number visible at 00:45?")
    assert question_route(question) == "ocr_at_time"


def test_parse_json_object_ignores_surrounding_prose() -> None:
    assert (
        parse_json_object('result follows\n{"answer":"yes","confidence":0.8}\ndone')["answer"]
        == "yes"
    )


def test_normalize_answer_rejects_invalid_count() -> None:
    question = Question("q", "v", "count", "How many?")
    assert normalize_answer({"answer": -2, "confidence": 4}, question) == ("not_visible", 1.0)


@pytest.mark.parametrize("question_type", ["object", "structured_object", "short_structured_object"])
def test_normalize_answer_preserves_structured_object(question_type: str) -> None:
    question = Question("q", "v", question_type, "Describe the handoff state")
    answer = {"sealed": True, "container_count": 2, "labels": ["ready", "pickup"]}

    assert normalize_answer({"answer": answer, "confidence": 0.8}, question) == (answer, 0.8)
    assert "non-empty JSON object" in answer_type_instruction(question)


@pytest.mark.parametrize("answer", [{}, {"duration": float("nan")}, ["not", "an", "object"]])
def test_normalize_answer_rejects_invalid_structured_object(answer: object) -> None:
    question = Question("q", "v", "structured_object", "Describe the handoff state")

    assert normalize_answer({"answer": answer, "confidence": 0.8}, question) == (
        "not_visible",
        0.0,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(5, 5.0), ("5.00", 5.0), ("t=5.00s", 5.0), ("at five seconds", None), (-1, None)],
)
def test_parse_evidence_seconds(value: object, expected: float | None) -> None:
    assert parse_evidence_seconds(value) == expected


def test_atomic_write_json_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "answer.json"
    path.write_text("old", encoding="utf-8")
    atomic_write_json(path, {"answer": "yes"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"answer": "yes"}


def test_frame_store_enforces_unique_frame_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeCapture:
        def set(self, *_args: object) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            return True, np.zeros((10, 10, 3), dtype=np.uint8)

        def release(self) -> None:
            return None

    monkeypatch.setattr("kitchen_agent.cv2.VideoCapture", lambda _path: FakeCapture())
    video = VideoInfo("v", tmp_path / "v.mp4", 10.0, 30.0)
    stats = RunStats(0.0)
    store = FrameStore({"v": video}, tmp_path, 1, stats)
    first = store.frame("v", 1.0)
    assert store.frame("v", 1.0) == first
    with pytest.raises(BudgetExhausted):
        store.frame("v", 2.0)
    assert stats.frames_processed == 1

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kitchen_agent import (
    BudgetExhausted,
    FrameRef,
    FrameStore,
    InputError,
    Question,
    RunStats,
    VideoInfo,
    answer_questions,
    answer_type_instruction,
    atomic_write_json,
    evenly_bounded_items,
    extract_explicit_timestamp,
    load_questions,
    normalize_answer,
    parse_evidence_seconds,
    parse_json_object,
    question_route,
    question_specific_guidance,
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


def test_question_route_does_not_treat_visible_people_as_ocr() -> None:
    question = Question("q", "v", "count", "At 00:45, how many people are visible?")
    assert question_route(question) == "state_at_time"


def test_parse_json_object_ignores_surrounding_prose() -> None:
    assert (
        parse_json_object('result follows\n{"answer":"yes","confidence":0.8}\ndone')["answer"]
        == "yes"
    )


def test_normalize_answer_rejects_invalid_count() -> None:
    question = Question("q", "v", "count", "How many?")
    assert normalize_answer({"answer": -2, "confidence": 4}, question) == ("not_visible", 1.0)


def test_headwear_guidance_distinguishes_bare_head_from_covering() -> None:
    guidance = question_specific_guidance(
        Question("q", "v", "yes_no", "Is the worker wearing a cap or hairnet?")
    )
    assert "bare or bald head" in guidance
    assert "fabric or mesh covering" in guidance
    assert "scalp skin is visible" in guidance


def test_headwear_count_guidance_counts_only_qualifying_people() -> None:
    guidance = question_specific_guidance(
        Question("q", "v", "count", "How many people are wearing a cap or hairnet?")
    )
    assert "not the total number of visible people" in guidance
    assert "Return 0" in guidance
    assert "two bareheaded people means 0" in guidance


@pytest.mark.parametrize(
    "question_type", ["object", "structured_object", "short_structured_object"]
)
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
    [
        (5, 5.0),
        ("5.00", 5.0),
        ("t=5.00s", 5.0),
        ("00:05", 5.0),
        ("01:02:03", 3_723.0),
        ("99:99", None),
        ("at five seconds", None),
        (-1, None),
    ],
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


def test_evenly_bounded_items_keeps_endpoints() -> None:
    selected = evenly_bounded_items(list(range(48)), 18)
    assert selected[0] == 0
    assert selected[-1] == 47
    assert len(selected) == 18
    assert evenly_bounded_items(list(range(5)), 1) == [2]


def test_temporal_route_uses_one_model_call_and_one_evidence_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeFrameStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def frame(self, video_id: str, timestamp: float) -> FrameRef:
            return FrameRef(video_id, timestamp, tmp_path / f"{timestamp:.2f}.jpg")

    class FakeBackend:
        model_id = "fake"

        def __init__(self) -> None:
            self.calls = 0
            self.image_counts: list[int] = []

        def ask(self, images: object, _prompt: str) -> str:
            self.calls += 1
            assert isinstance(images, list)
            self.image_counts.append(len(images))
            return '{"answer":"yes","confidence":0.8,"evidence_timestamp":0}'

    monkeypatch.setattr("kitchen_agent.FrameStore", FakeFrameStore)
    monkeypatch.setattr(
        "kitchen_agent.contact_sheet", lambda _frames, output, _title, **_kwargs: output
    )
    backend = FakeBackend()

    answers, traces = answer_questions(
        [Question("q", "v", "yes_no", "Did the worker perform the action?")],
        {"v": VideoInfo("v", tmp_path / "v.mp4", 120.0, 30.0)},
        backend,
        tmp_path,
        100,
        RunStats(0.0),
    )

    assert backend.calls == 1
    assert backend.image_counts == [1]
    assert answers[0]["answer"] == "yes"
    assert traces[0]["errors"] == []


def test_headwear_count_uses_negative_qualification_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeFrameStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def frame(self, video_id: str, timestamp: float) -> FrameRef:
            return FrameRef(video_id, timestamp, tmp_path / f"{timestamp:.2f}.jpg")

    class FakeBackend:
        model_id = "fake"

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def ask(self, _images: object, prompt: str) -> str:
            self.prompts.append(prompt)
            return '{"answer":"no","confidence":0.9,"evidence_timestamp":5}'

    monkeypatch.setattr("kitchen_agent.FrameStore", FakeFrameStore)
    monkeypatch.setattr(
        "kitchen_agent.contact_sheet", lambda _frames, output, _title, **_kwargs: output
    )
    backend = FakeBackend()

    answers, traces = answer_questions(
        [Question("q", "v", "count", "At 00:05, how many people wear a cap or hairnet?")],
        {"v": VideoInfo("v", tmp_path / "v.mp4", 10.0, 30.0)},
        backend,
        tmp_path,
        10,
        RunStats(0.0),
    )

    assert answers[0]["answer"] == 0
    assert len(backend.prompts) == 1
    assert "Qualification gate" in backend.prompts[0]
    assert "JSON number from 0.0 through 1.0" in backend.prompts[0]
    assert traces[0]["errors"] == []


def test_headwear_count_uses_typed_count_after_positive_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeFrameStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def frame(self, video_id: str, timestamp: float) -> FrameRef:
            return FrameRef(video_id, timestamp, tmp_path / f"{timestamp:.2f}.jpg")

    responses = iter(
        [
            '{"answer":"yes","confidence":0.9,"evidence_timestamp":5}',
            '{"answer":2,"confidence":0.8,"evidence_timestamp":5}',
        ]
    )

    class FakeBackend:
        model_id = "fake"

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def ask(self, _images: object, prompt: str) -> str:
            self.prompts.append(prompt)
            return next(responses)

    monkeypatch.setattr("kitchen_agent.FrameStore", FakeFrameStore)
    monkeypatch.setattr(
        "kitchen_agent.contact_sheet", lambda _frames, output, _title, **_kwargs: output
    )
    backend = FakeBackend()

    answers, traces = answer_questions(
        [Question("q", "v", "count", "At 00:05, how many people wear a cap or hairnet?")],
        {"v": VideoInfo("v", tmp_path / "v.mp4", 10.0, 30.0)},
        backend,
        tmp_path,
        10,
        RunStats(0.0),
    )

    assert answers[0]["answer"] == 2
    assert len(backend.prompts) == 2
    assert "Qualification gate" in backend.prompts[0]
    assert "non-negative integer" in backend.prompts[1]
    assert traces[0]["errors"] == []


def test_questions_share_small_frame_budget_without_starving_temporal_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeFrameStore:
        def __init__(
            self,
            _videos: object,
            _root: Path,
            frame_budget: int,
            stats: RunStats,
        ) -> None:
            self.frame_budget = frame_budget
            self.stats = stats
            self.cache: dict[tuple[str, int], FrameRef] = {}

        def frame(self, video_id: str, timestamp: float) -> FrameRef:
            key = (video_id, round(timestamp * 1_000))
            if key not in self.cache:
                if self.stats.frames_processed >= self.frame_budget:
                    raise BudgetExhausted("test budget exhausted")
                self.cache[key] = FrameRef(video_id, timestamp, tmp_path / f"{timestamp:.2f}.jpg")
                self.stats.frames_processed += 1
            return self.cache[key]

    class FakeBackend:
        model_id = "fake"

        def ask(self, _images: object, prompt: str) -> str:
            if "locating evidence" in prompt:
                return "[]"
            timestamp = 5 if "wearing a cap" in prompt else 8
            return json.dumps({"answer": "yes", "confidence": 0.8, "evidence_timestamp": timestamp})

    monkeypatch.setattr("kitchen_agent.FrameStore", FakeFrameStore)
    monkeypatch.setattr(
        "kitchen_agent.contact_sheet", lambda _frames, output, _title, **_kwargs: output
    )
    stats = RunStats(0.0)

    answers, traces = answer_questions(
        [
            Question("headwear", "v", "yes_no", "At 00:05, is the chef wearing a cap?"),
            Question("action", "v", "yes_no", "Does the chef cut vegetables?"),
        ],
        {"v": VideoInfo("v", tmp_path / "v.mp4", 16.0, 30.0)},
        FakeBackend(),
        tmp_path,
        6,
        stats,
    )

    assert [answer["answer"] for answer in answers] == ["yes", "yes"]
    assert stats.frames_processed == 6
    assert not traces[1]["errors"]

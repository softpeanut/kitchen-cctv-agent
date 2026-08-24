from __future__ import annotations

import json
import math
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
from PIL import Image, ImageDraw, ImageFont

MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
FRAMES_PER_HOUR = 1_500
CONTACT_COLUMNS = 3
CONTACT_ROWS = 2
CONTACT_WIDTH = 1_200
EXPLICIT_TIME_RE = re.compile(r"(?<!\d)(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?!\d)")
EVIDENCE_SECONDS_RE = re.compile(r"^\s*(?:t\s*=\s*)?(\d+(?:\.\d+)?)\s*s?\s*$", re.IGNORECASE)
STRUCTURED_OBJECT_TYPES = {"object", "structured_object", "short_structured_object"}
EVIDENCE_CLOCK_RULE = (
    "Question timestamps are elapsed video time. For evidence_timestamp, use only the black "
    "t=X.XXs elapsed-time label added to each evidence panel; ignore any date or wall clock "
    "recorded inside the source video. "
)


class InputError(ValueError):
    """The evaluator input does not satisfy the published CLI contract."""


class BudgetExhausted(RuntimeError):
    """The scaled frame budget cannot admit another decoded frame."""


class VisionBackend(Protocol):
    model_id: str

    def ask(self, images: Sequence[Path], prompt: str) -> str: ...


@dataclass(frozen=True)
class Question:
    id: str
    video_id: str
    type: str
    question: str
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class VideoInfo:
    id: str
    path: Path
    duration: float
    fps: float


@dataclass(frozen=True)
class FrameRef:
    video_id: str
    timestamp: float
    path: Path


@dataclass
class RunStats:
    started_at: float
    frames_processed: int = 0
    model_calls: int = 0


class MlxVisionBackend:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        self.model_id = model_id
        self._generate = generate
        self._apply_chat_template = apply_chat_template
        self._model, self._processor = load(model_id)
        self._config = load_config(model_id)

    def ask(self, images: Sequence[Path], prompt: str) -> str:
        image_paths = [str(path) for path in images]
        formatted = self._apply_chat_template(
            self._processor,
            self._config,
            prompt,
            num_images=len(image_paths),
        )
        result = self._generate(
            self._model,
            self._processor,
            formatted,
            image_paths,
            max_tokens=160,
            temperature=0.0,
            verbose=False,
        )
        return result.text


def load_questions(path: Path) -> list[Question]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read questions JSON {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise InputError("questions JSON must be a non-empty array")

    result: list[Question] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InputError(f"question at index {index} must be an object")
        values: dict[str, str] = {}
        for key in ("id", "video_id", "type", "question"):
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                raise InputError(f"question at index {index} has invalid {key!r}")
            values[key] = value.strip()
        if values["id"] in seen:
            raise InputError(f"duplicate question id: {values['id']}")
        seen.add(values["id"])
        choices_raw = item.get("choices", [])
        if not isinstance(choices_raw, list) or not all(
            isinstance(choice, str) and choice.strip() for choice in choices_raw
        ):
            raise InputError(f"question {values['id']} has invalid choices")
        result.append(
            Question(
                id=values["id"],
                video_id=values["video_id"],
                type=values["type"].lower(),
                question=values["question"],
                choices=tuple(choice.strip() for choice in choices_raw),
            )
        )
    return result


def discover_videos(directory: Path, questions: Sequence[Question]) -> dict[str, VideoInfo]:
    if not directory.is_dir():
        raise InputError(f"video directory does not exist: {directory}")
    files = [path for path in directory.iterdir() if path.is_file()]
    by_stem = {path.stem: path for path in files}
    result: dict[str, VideoInfo] = {}
    for video_id in dict.fromkeys(question.video_id for question in questions):
        path = by_stem.get(video_id)
        if path is None:
            candidates = [item for item in files if item.name.startswith(f"{video_id}.")]
            path = candidates[0] if len(candidates) == 1 else None
        if path is None:
            raise InputError(f"no unique video file found for video_id {video_id!r}")
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise InputError(f"video cannot be opened: {path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or not math.isfinite(frames) or frames <= 0:
                raise InputError(f"video has invalid FPS or frame count: {path}")
            duration = frames / fps
        finally:
            capture.release()
        result[video_id] = VideoInfo(video_id, path, duration, fps)
    return result


def extract_explicit_timestamp(text: str) -> float | None:
    match = EXPLICIT_TIME_RE.search(text)
    if match is None:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    if minutes >= 60 or seconds >= 60:
        return None
    return float(hours * 3_600 + minutes * 60 + seconds)


def question_route(question: Question) -> str:
    if extract_explicit_timestamp(question.question) is not None:
        lowered = question.question.lower()
        if question.type in {"ocr", "visibility", "not_visible"} or any(
            token in lowered for token in ("readable", "number", "label", "text")
        ):
            return "ocr_at_time"
        return "state_at_time"
    return "temporal"


def parse_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model output did not contain a JSON object")


def normalize_answer(raw: Mapping[str, Any], question: Question) -> tuple[Any, float]:
    answer = raw.get("answer", "not_visible")
    try:
        confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    if isinstance(answer, str) and answer.strip().lower() == "not_visible":
        return "not_visible", confidence
    if question.type == "yes_no":
        normalized = str(answer).strip().lower()
        return (normalized if normalized in {"yes", "no"} else "not_visible"), confidence
    if question.type == "count":
        if isinstance(answer, bool):
            return "not_visible", 0.0
        try:
            value = int(answer)
        except (TypeError, ValueError):
            return "not_visible", 0.0
        return (value if value >= 0 else "not_visible"), confidence
    if question.type in {"timestamp", "duration"}:
        try:
            value = float(answer)
        except (TypeError, ValueError):
            return "not_visible", 0.0
        return (round(value, 2) if value >= 0 else "not_visible"), confidence
    if question.type == "multiple_choice" and question.choices:
        normalized = str(answer).strip()
        match = next(
            (choice for choice in question.choices if choice.lower() == normalized.lower()), None
        )
        return (match if match is not None else "not_visible"), confidence
    if question.type in STRUCTURED_OBJECT_TYPES:
        if isinstance(answer, dict) and answer and _is_json_value(answer):
            return answer, confidence
        return "not_visible", 0.0
    if isinstance(answer, (str, int, float)):
        return answer, confidence
    return "not_visible", 0.0


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def parse_evidence_seconds(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    if not isinstance(value, str):
        return None
    clock = EXPLICIT_TIME_RE.fullmatch(value.strip())
    if clock is not None:
        hours = int(clock.group(1) or 0)
        minutes = int(clock.group(2))
        seconds = int(clock.group(3))
        if minutes < 60 and seconds < 60:
            return float(hours * 3_600 + minutes * 60 + seconds)
    match = EVIDENCE_SECONDS_RE.fullmatch(value)
    if match is None:
        return None
    result = float(match.group(1))
    return result if math.isfinite(result) else None


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


class FrameStore:
    def __init__(
        self,
        videos: Mapping[str, VideoInfo],
        root: Path,
        frame_budget: int,
        stats: RunStats,
    ) -> None:
        self.videos = videos
        self.root = root
        self.frame_budget = frame_budget
        self.stats = stats
        self.cache: dict[tuple[str, int], FrameRef] = {}

    def frame(self, video_id: str, timestamp: float) -> FrameRef:
        info = self.videos[video_id]
        clamped = min(max(0.0, timestamp), max(0.0, info.duration - 0.05))
        millis = round(clamped * 1_000)
        key = (video_id, millis)
        if key in self.cache:
            return self.cache[key]
        if self.stats.frames_processed >= self.frame_budget:
            raise BudgetExhausted(f"frame budget {self.frame_budget} exhausted")
        capture = cv2.VideoCapture(str(info.path))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, clamped * 1_000)
            ok, bgr = capture.read()
        finally:
            capture.release()
        if not ok or bgr is None:
            raise InputError(f"cannot decode {video_id} at {clamped:.2f}s")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        target = self.root / f"{video_id}-{millis:010d}.jpg"
        Image.fromarray(rgb).save(target, format="JPEG", quality=88)
        ref = FrameRef(video_id, clamped, target)
        self.cache[key] = ref
        self.stats.frames_processed += 1
        return ref


def evenly_spaced(duration: float, count: int) -> list[float]:
    if count <= 1 or duration <= 0:
        return [0.0]
    margin = min(1.0, duration / 20)
    usable = max(0.0, duration - 2 * margin)
    return [margin + usable * index / (count - 1) for index in range(count)]


def contact_sheet(
    frames: Sequence[FrameRef], output: Path, title: str, columns: int = CONTACT_COLUMNS
) -> Path:
    if not frames:
        raise ValueError("contact sheet needs at least one frame")
    if columns <= 0:
        raise ValueError("contact sheet columns must be positive")
    cell_width = CONTACT_WIDTH // columns
    cell_height = int(cell_width * 9 / 16) + 34
    rows = math.ceil(len(frames) / columns)
    canvas = Image.new("RGB", (CONTACT_WIDTH, rows * cell_height + 38), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    draw.text((12, 10), title, fill="black", font=font)
    for index, frame in enumerate(frames):
        image = Image.open(frame.path).convert("RGB")
        image.thumbnail((cell_width, cell_height - 34))
        x = (index % columns) * cell_width
        y = 38 + (index // columns) * cell_height
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + 185, y + 28), fill="black")
        draw.text((x + 6, y + 5), f"t={frame.timestamp:.2f}s", fill="white", font=font)
    canvas.save(output, format="JPEG", quality=90)
    return output


def answer_type_instruction(question: Question) -> str:
    if question.type == "yes_no":
        return 'answer must be "yes", "no", or "not_visible"'
    if question.type == "count":
        return 'answer must be a non-negative integer or "not_visible"'
    if question.type in {"timestamp", "duration"}:
        return 'answer must be seconds as a number or "not_visible"'
    if question.type == "multiple_choice" and question.choices:
        return f'answer must be exactly one of {list(question.choices)!r} or "not_visible"'
    if question.type in STRUCTURED_OBJECT_TYPES:
        return 'answer must be a non-empty JSON object or "not_visible"'
    return 'answer must be a short scalar value or "not_visible"'


def question_specific_guidance(question: Question) -> str:
    lowered = question.question.lower()
    if any(term in lowered for term in ("cap", "hairnet", "headwear", "head covering")):
        guidance = (
            " For headwear checks, inspect each visible person's head separately. "
            "A bare or bald head, visible hair, a hood resting behind the head, or a uniform "
            "collar is not a cap or hairnet. Count or answer yes only when a fabric or mesh "
            "covering is visibly worn on the head. If scalp skin is visible, that person is "
            "not wearing a cap or hairnet. Never infer headwear merely because the person is "
            "a chef or wears a kitchen uniform. When the relevant person's bare head, hair, or "
            "scalp is visible, answer no; reserve not_visible for a head that is absent, fully "
            "occluded, or too unclear to classify. If the question identifies a person by a "
            "station or work area, inspect the visible person using or nearest that named area; "
            "do not require the person's body to overlap the appliance exactly."
        )
        if question.type == "count":
            guidance += (
                " For a count question, count only people who visibly meet that headwear "
                "condition, not the total number of visible people. Return 0 when people are "
                "visible but none wears a qualifying covering. Work person by person before "
                "returning the JSON number: two bareheaded people means 0; three people with "
                "exactly one visible cap means 1."
            )
        return guidance
    return ""


def evenly_bounded_items(values: Sequence[Any], limit: int) -> list[Any]:
    if limit <= 0 or not values:
        return []
    if limit == 1:
        return [values[len(values) // 2]]
    if len(values) <= limit:
        return list(values)
    indexes = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[index] for index in indexes]


def answer_questions(
    questions: Sequence[Question],
    videos: Mapping[str, VideoInfo],
    backend: VisionBackend,
    work_dir: Path,
    frame_budget: int,
    stats: RunStats,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = FrameStore(videos, work_dir, frame_budget, stats)
    answers: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    for question_index, question in enumerate(questions):
        route = question_route(question)
        info = videos[question.video_id]
        selected: list[FrameRef] = []
        trace: dict[str, Any] = {"id": question.id, "route": route, "errors": []}
        try:
            questions_remaining = len(questions) - question_index
            remaining_budget = frame_budget - stats.frames_processed
            question_allowance = max(1, remaining_budget // questions_remaining)
            explicit = extract_explicit_timestamp(question.question)
            if explicit is not None:
                radius = 2.0 if route == "ocr_at_time" else 1.0
                offsets = evenly_bounded_items((-radius, 0.0, radius), min(3, question_allowance))
                selected = [
                    frames.frame(question.video_id, explicit + offset) for offset in offsets
                ]
            else:
                selected = [
                    frames.frame(question.video_id, timestamp)
                    for timestamp in evenly_spaced(info.duration, min(6, question_allowance))
                ]

            guidance = question_specific_guidance(question)
            detailed_headwear = bool(guidance) and explicit is not None
            evidence_frames = [selected[len(selected) // 2]] if detailed_headwear else selected
            final_sheets = [
                contact_sheet(
                    evidence_frames,
                    work_dir / f"{question.id}-evidence.jpg",
                    f"Evidence for {question.id}",
                    columns=1 if detailed_headwear else CONTACT_COLUMNS,
                )
            ]
            prompt = (
                "Answer one operational question from timestamp-labelled kitchen CCTV frames. "
                "Do not infer details outside the inspected images. If the named attribute, text, "
                "person, object, or full event is not visibly supported, answer not_visible.\n"
                f"Question: {question.question}\nType rule: {answer_type_instruction(question)}.\n"
                f"Visual guidance:{guidance}\n"
                f"{EVIDENCE_CLOCK_RULE}"
                "Return JSON only with keys answer, confidence, evidence_timestamp. "
                "confidence must be a JSON number from 0.0 through 1.0, never a word. "
                "evidence_timestamp must be one timestamp printed on an image, or null for not_visible."
            )
            parsed: dict[str, Any] | None = None
            if question.type == "count" and guidance:
                gate_prompt = (
                    "Qualification gate for a headwear count. Inspect every visible person's head. "
                    "Does at least one visible person wear a fabric or mesh cap or hairnet? "
                    "A bare head, visible hair or scalp, hood behind the head, and uniform collar "
                    f"do not qualify. {EVIDENCE_CLOCK_RULE}"
                    "Return JSON only with keys answer, confidence, "
                    'evidence_timestamp; answer must be "yes", "no", or "not_visible". '
                    "confidence must be a JSON number from 0.0 through 1.0, never a word."
                )
                stats.model_calls += 1
                gate_raw = backend.ask(final_sheets, gate_prompt)
                trace["qualification_output"] = gate_raw
                try:
                    gate_parsed = parse_json_object(gate_raw)
                except ValueError:
                    if len(selected) <= 1:
                        raise
                    retry_frames = [selected[0], selected[-1]]
                    retry_sheets = [
                        contact_sheet(
                            retry_frames,
                            work_dir / f"{question.id}-qualification-retry.jpg",
                            f"Neighbor evidence for {question.id}",
                            columns=len(retry_frames),
                        )
                    ]
                    stats.model_calls += 1
                    gate_raw = backend.ask(retry_sheets, gate_prompt)
                    trace["qualification_retry_output"] = gate_raw
                    gate_parsed = parse_json_object(gate_raw)
                    final_sheets = retry_sheets
                gate_question = Question(question.id, question.video_id, "yes_no", gate_prompt)
                gate_answer, gate_confidence = normalize_answer(gate_parsed, gate_question)
                if gate_answer == "no":
                    parsed = {
                        "answer": 0,
                        "confidence": gate_confidence,
                        "evidence_timestamp": gate_parsed.get("evidence_timestamp"),
                    }

            if parsed is None:
                stats.model_calls += 1
                raw_text = backend.ask(final_sheets, prompt)
                trace["answer_output"] = raw_text
                parsed = parse_json_object(raw_text)
            answer, confidence = normalize_answer(parsed, question)
            evidence_timestamp = parse_evidence_seconds(parsed.get("evidence_timestamp"))
            inspected = [frame.timestamp for frame in selected]
            if (
                answer == "not_visible"
                or evidence_timestamp is None
                or not any(abs(evidence_timestamp - value) <= 2.1 for value in inspected)
            ):
                answer = "not_visible"
                evidence: list[dict[str, Any]] = []
            else:
                evidence = [
                    {
                        "video_id": question.video_id,
                        "timestamp_start": round(max(0.0, evidence_timestamp - 1.0), 2),
                        "timestamp_end": round(min(info.duration, evidence_timestamp + 1.0), 2),
                    }
                ]
            answers.append(
                {
                    "id": question.id,
                    "answer": answer,
                    "confidence": round(confidence, 3),
                    "evidence": evidence,
                }
            )
        except (BudgetExhausted, InputError, OSError, RuntimeError, ValueError) as exc:
            trace["errors"].append(f"{type(exc).__name__}: {exc}")
            answers.append(
                {"id": question.id, "answer": "not_visible", "confidence": 0.0, "evidence": []}
            )
        trace["inspected_timestamps"] = [round(frame.timestamp, 2) for frame in selected]
        traces.append(trace)
    return answers, traces

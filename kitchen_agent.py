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
            max_tokens=320,
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
            token in lowered for token in ("readable", "visible", "number", "label", "text")
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


def contact_sheet(frames: Sequence[FrameRef], output: Path, title: str) -> Path:
    if not frames:
        raise ValueError("contact sheet needs at least one frame")
    cell_width = CONTACT_WIDTH // CONTACT_COLUMNS
    cell_height = int(cell_width * 9 / 16) + 34
    rows = math.ceil(len(frames) / CONTACT_COLUMNS)
    canvas = Image.new("RGB", (CONTACT_WIDTH, rows * cell_height + 38), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    draw.text((12, 10), title, fill="black", font=font)
    for index, frame in enumerate(frames):
        image = Image.open(frame.path).convert("RGB")
        image.thumbnail((cell_width, cell_height - 34))
        x = (index % CONTACT_COLUMNS) * cell_width
        y = 38 + (index // CONTACT_COLUMNS) * cell_height
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


def _candidate_timestamps(raw: Mapping[str, Any], duration: float) -> list[float]:
    candidates = raw.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    result: list[float] = []
    for item in candidates[:4]:
        value = item.get("timestamp") if isinstance(item, dict) else item
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            continue
        if 0 <= timestamp <= duration and all(abs(timestamp - old) >= 0.5 for old in result):
            result.append(timestamp)
    return result


def parse_candidate_timestamps(text: str, duration: float) -> list[float]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return _candidate_timestamps(value, duration)
        if isinstance(value, list):
            return _candidate_timestamps({"candidates": value}, duration)
    raise ValueError("model output did not contain a candidate JSON object or array")


def merge_candidate_timestamps(
    groups: Sequence[Sequence[float]], limit: int = 3, minimum_gap: float = 0.5
) -> list[float]:
    result: list[float] = []
    for group in groups:
        for timestamp in group:
            if all(abs(timestamp - old) >= minimum_gap for old in result):
                result.append(timestamp)
    return evenly_bounded_items(result, limit)


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

    for question in questions:
        route = question_route(question)
        info = videos[question.video_id]
        selected: list[FrameRef] = []
        trace: dict[str, Any] = {"id": question.id, "route": route, "errors": []}
        try:
            explicit = extract_explicit_timestamp(question.question)
            if explicit is not None:
                radius = 2.0 if route == "ocr_at_time" else 1.0
                selected = [
                    frames.frame(question.video_id, explicit + offset)
                    for offset in (-radius, 0.0, radius)
                ]
            else:
                coarse_refs = [
                    frames.frame(question.video_id, timestamp)
                    for timestamp in evenly_spaced(info.duration, 48)
                ]
                coarse_sheets = [
                    contact_sheet(
                        coarse_refs[index : index + CONTACT_COLUMNS * CONTACT_ROWS],
                        work_dir / f"{question.id}-coarse-{index // 6:02d}.jpg",
                        f"{question.video_id} coarse scan",
                    )
                    for index in range(0, len(coarse_refs), CONTACT_COLUMNS * CONTACT_ROWS)
                ]
                scout_prompt = (
                    "You are locating evidence in timestamp-labelled fixed-camera kitchen frames. "
                    f"Question: {question.question}\n"
                    'Return JSON only: {"candidates":[{"timestamp": number, '
                    '"reason": short string}]}. Return at most 4 candidates. '
                    "Use only timestamps printed on the images; use an empty list if no likely evidence."
                )
                scout_groups: list[list[float]] = []
                scout_outputs: list[dict[str, Any]] = []
                for sheet_index, sheet in enumerate(coarse_sheets):
                    stats.model_calls += 1
                    try:
                        scout_raw = backend.ask([sheet], scout_prompt)
                    except (OSError, RuntimeError, ValueError) as exc:
                        trace["errors"].append(f"scout_call[{sheet_index}]: {exc}")
                        scout_groups.append([])
                        continue
                    scout_outputs.append({"sheet": sheet_index, "output": scout_raw})
                    try:
                        group = parse_candidate_timestamps(scout_raw, info.duration)[:1]
                    except ValueError as exc:
                        trace["errors"].append(f"scout_parse[{sheet_index}]: {exc}")
                        group = []
                    scout_groups.append(group)
                trace["scout_outputs"] = scout_outputs
                candidates = merge_candidate_timestamps(scout_groups)
                if candidates:
                    for candidate in candidates[:3]:
                        for offset in (-2.0, -1.0, 0.0, 1.0, 2.0):
                            selected.append(frames.frame(question.video_id, candidate + offset))
                else:
                    selected = evenly_bounded_items(coarse_refs, 18)

            final_sheets = [
                contact_sheet(
                    selected[index : index + CONTACT_COLUMNS * CONTACT_ROWS],
                    work_dir / f"{question.id}-evidence-{index // 6:02d}.jpg",
                    f"Evidence for {question.id}",
                )
                for index in range(0, len(selected), CONTACT_COLUMNS * CONTACT_ROWS)
            ]
            prompt = (
                "Answer one operational question from timestamp-labelled kitchen CCTV frames. "
                "Do not infer details outside the inspected images. If the named attribute, text, "
                "person, object, or full event is not visibly supported, answer not_visible.\n"
                f"Question: {question.question}\nType rule: {answer_type_instruction(question)}.\n"
                "Return JSON only with keys answer, confidence, evidence_timestamp. "
                "evidence_timestamp must be one timestamp printed on an image, or null for not_visible."
            )
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

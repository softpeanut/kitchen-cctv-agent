# Kitchen CCTV Agent design

Work unit: build one reproducible CLI that answers structured questions from fixed-camera kitchen videos, attaches evidence spans, logs frame/model cost, and remains below 1,500 frames and $0.30 API cost per 60 source minutes.

There are no existing integration points: this is a greenfield submission. The external contract is Builderr's `answer.py --videos ... --questions ... --out ... --log ...` interface.

## Acceptance criteria

- Validate question and video inputs before loading the model.
- Route explicit-timestamp state/count questions directly to nearby frames.
- Route temporal/event questions through a sparse scan, then refine candidate windows.
- Never exceed the scaled frame budget; local inference records `$0` API cost.
- Emit one schema-valid answer per input question, including evidence or `not_visible`.
- Preserve non-empty JSON objects for the published structured-object answer type; reject malformed or non-finite object values as `not_visible`.
- Record runtime, frames inspected, model calls, model identity, and question-level traces.
- Tests use hand-written expected outputs rather than asking the production model to validate itself.

## Pseudocode

```text
P1  receive CLI paths and configuration
P2  IF any required path is missing -> exit 2 with a contextual error
P3  CALL read and validate questions JSON
P4    IF JSON or schema validation fails -> exit 2 without loading the model
P5  CALL discover videos and probe duration/FPS
P6    IF a referenced video is missing or unreadable -> exit 2
P7  compute scaled per-video and total frame budgets
P8  CALL load the local MLX VLM once
P9    IF model load fails -> exit 1 with model identifier and original cause
P10 FOR EACH question in input order
P11   classify question route from declared type and text
P12   IF route is explicit timestamp/state/count
P13     sample a bounded neighbourhood around the requested timestamp
P14   ELSE IF route is OCR/visibility at an explicit timestamp
P15     sample and upscale a bounded neighbourhood around the timestamp
P16   ELSE
P17     sample sparse contact sheets over the full video
P18     FOR EACH coarse contact sheet, CALL VLM separately for at most one candidate
P19       IF one call or parse fails -> record that sheet failure and continue with remaining sheets
P20     merge and de-duplicate at most three candidate timestamps across sheets
P21     IF candidates exist
P22       sample a bounded one-second refinement grid around candidates
P23     ELSE
P24       keep at most 18 evenly distributed sparse frames for the final call
P25   enforce remaining frame budget before every sample
P26     IF budget is exhausted -> emit `not_visible` with budget-exhausted trace
P27   CALL VLM with question, allowed answer type, timestamps, and at most three evidence sheets
P28     IF call or strict JSON parse fails -> emit `not_visible` with failure trace
P29   normalize answer to the declared type and clamp confidence to [0,1]
P30   IF type is object, structured_object, or short_structured_object
P31     IF answer is a non-empty JSON object containing only finite JSON values -> preserve it
P32     ELSE -> replace it with `not_visible`
P33   IF answer is unsupported by a selected timestamp
P34     emit `not_visible` and empty evidence
P35   ELSE
P36     attach selected evidence span and append the answer
P37 WRITE answers JSON atomically
P38   IF write fails -> exit 1; do not claim a completed run
P39 WRITE run log JSON atomically
P40   IF write fails -> remove neither output; exit 1 and report incomplete audit log
P41 return exit 0
```

Completeness check: inputs, every fallible IO/model call, both arms of each decision, frame exhaustion, parse failure, structured-object validation, and both output writes are explicit. No authorization is required. Concurrency is intentionally absent so one process owns the frame counter and output files. All acceptance criteria map to P2-P41.

## Proposed flow

```mermaid
flowchart TD
    A["Validate CLI, questions, and videos"] --> B{"Inputs valid?"}
    B -- no --> X["Exit 2"]
    B -- yes --> C["Load one local MLX VLM"]
    C --> D{"Model loaded?"}
    D -- no --> Y["Exit 1"]
    D -- yes --> E{"Question route?"}
    E -- explicit timestamp --> F["Sample nearby frames"]
    E -- OCR visibility --> G["Sample and upscale nearby frames"]
    E -- temporal event --> H["Per-sheet sparse scan and candidate refinement"]
    F --> I{"Budget available?"}
    G --> I
    H --> I
    I -- no --> J["Emit not_visible with trace"]
    I -- yes --> K["Run typed VLM query"]
    K --> L{"Valid supported JSON?"}
    L -- no --> J
    L -- yes --> M["Attach evidence answer"]
    J --> N{"More questions?"}
    M --> N
    N -- yes --> E
    N -- no --> O["Write answers and audit log"]
    O --> P{"Both writes succeeded?"}
    P -- no --> Y
    P -- yes --> Q["Exit 0"]
```

Implementation risks:

- P11: free-form question routing can misclassify; the declared `type` always wins when present.
- P18/P27: model output may include prose around JSON; extraction is bounded to the first JSON object, or the observed candidate-array variant for scouting, and then schema-validated. Coarse sheets are queried separately so one oversized multi-image prompt cannot corrupt the entire temporal route.
- P25: duplicate timestamps across questions can waste budget; a frame cache keyed by `(video, millisecond)` counts each decoded frame once.
- P33: VLM confidence is not evidence. Answers without a timestamp selected from inspected frames are downgraded to `not_visible`.

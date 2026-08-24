# Kitchen CCTV Agent

Question-driven, coarse-to-fine video QA for the Builderr Kitchen CCTV challenge. It runs a quantized vision-language model locally on Apple Silicon, so estimated model/API cost is `$0.00`.

```bash
uv sync --extra dev
uv run python answer.py --videos ./videos --questions questions.json --out answers.json --log run_log.json
```

After the model has been downloaded once, a network-isolated evaluator can force cached-only
loading with `HF_HUB_OFFLINE=1`:

```bash
HF_HUB_OFFLINE=1 uv run python answer.py --videos ./videos --questions questions.json --out answers.json --log run_log.json
```

The agent validates inputs before loading the model, samples no more than the scaled 1,500-frame/hour budget, routes explicit-time questions directly to nearby frames, and uses sparse contact sheets plus local refinement for temporal questions. Every answer includes inspected evidence timestamps or returns `not_visible`.

Default model: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`.

## Verification

```bash
uv run ruff check .
uv run pytest -q
```

The full CLI was also run against NVIDIA's public
`PhysicalAI-Robotics-Manipulation-Kitchen` RGB `close_cabinet` episode 000000 with
[`validation-public.questions.json`](validation-public.questions.json). On an Apple Silicon Mac it
returned `yes` with evidence at 11.00 seconds, using 3 decoded frames, 1 model call, 6.733 seconds,
and `$0.00` estimated API cost. The video is not redistributed in this repository. It is available
under CC BY 4.0 from the source dataset:

https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen

An independent cached-only rerun on 2026-08-22 returned the same answer with evidence spanning
10.00–11.46 seconds, using 3 decoded frames, 1 model call, 4.894 seconds, and `$0.00` estimated API
cost. The source MP4 SHA-256 was
`f63822f0f778a305fbdad127acc98bf09c06c14110f634860e50387e66877d4e`.

External contracts and implementation references:

- Builderr challenge: https://builderr.ai/kitchen-video
- Technical specification: https://builderr.ai/kitchen-video-challenge-draft.md
- MLX-VLM multi-image API: https://github.com/Blaizzy/mlx-vlm

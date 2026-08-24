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

The agent validates inputs before loading the model, samples no more than the scaled
1,500-frame/hour budget, routes explicit-time questions directly to nearby frames, and uses a
bounded whole-video sample for temporal questions. Every answer includes an inspected evidence
timestamp or returns `not_visible`.

Default model: `mlx-community/Qwen3.5-4B-4bit`. The upstream Qwen3.5-4B model is
Apache-2.0 licensed, and the locked MLX-VLM 0.6.15 runtime supports its multimodal architecture.

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

After Builderr's directional evaluation identified headwear and temporal-action misses, the agent
was also replayed against Pexels video 8627112 using
[`validation-pexels.questions.json`](validation-pexels.questions.json). The source shows a
bareheaded chef cutting vegetables and is free to use under the Pexels license. The cached-only
run returned the expected `no` for cap/hairnet and `yes` for cutting, used the full 6-frame scaled
budget across both questions, made 2 local model calls, completed in 46.339 seconds, and retained
`$0.00` estimated API cost. The temporary source MP4 SHA-256 was
`34f610feabd26df1273d6f97105213bef8d7f91e382e870511fbf31838c81a1e`.

The final default model was evaluated against the Builderr-linked Chinese Commercial Kitchen
overhead cutting video with [`validation-commercial.questions.json`](validation-commercial.questions.json).
It matched all four independently labelled answers: one active person, no cap or hairnet,
cauliflower being cut at 04:02, and cauliflower as the last ingredient worked on. The cached-only
run completed in 18.186 seconds using 14 unique frames, 4 local model calls, timestamped evidence,
and `$0.00` API cost. The video is not redistributed here and remains available from its source:

https://huggingface.co/datasets/nova-dynamics/Chinese_Commercial_Kitchen_Manipulation_Dataset_Preview

External contracts and implementation references:

- Builderr challenge: https://builderr.ai/kitchen-video
- Technical specification: https://builderr.ai/kitchen-video-challenge-draft.md
- MLX-VLM multi-image API: https://github.com/Blaizzy/mlx-vlm
- Qwen3.5-4B model card and license: https://huggingface.co/Qwen/Qwen3.5-4B
- Pexels validation video: https://www.pexels.com/video/a-chef-cutting-ingredients-in-a-kitchen-8627112/

# EyeOfAI

```
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣶⠟⠃⠉⠀⠀⠀⠈⠙⢶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⢤⣺⣿⣿⣿⠇⠀⠀⠀⠀⠀⢀⣠⣤⣬⣿⣿⣷⣄⣂⠀⠀⠀⠀⠀
⠀⠀⢖⢼⡺⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠈⠹⠿⠿⣿⣿⣿⣿⣚⠸⣣⠆⠀⠀
⠀⠀⠀⠀⠑⠨⢟⣿⣿⣿⣆⠠⡀⠀⠀⠀⠀⠀⠀⢠⣿⣿⠿⠣⠙⠁⠁⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠻⠦⣈⡱⡲⢖⣌⠅⠴⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
```

**EyeOfAI** — A Python CLI that localizes targets in images and documents by asking OpenRouter vision models (free, paid, or both) for coordinates, then selecting a consensus bounding box.

## What It Supports

- Dynamic discovery of OpenRouter models via `GET /api/v1/models` with `free|paid|all` billing filters
- User-controlled model count (`--max-models`) and explicit model picks (`--models`)
- Inputs by full path for images and PDFs
- Folder scanning (optional recursive mode)
- Screen capture (`--capture screen`) and component region capture (`--capture region --region x,y,w,h`)
- Boxed eye art header with red/white gradient terminal output
- Full-screen live tables that stay clean on terminal resize
- Lightweight benchmark mode with IoU scoring
- Best-model discovery mode with live leaderboard (`rank`)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional for PDF support:

```bash
pip install pypdfium2
```

## Auth

Set your OpenRouter key:

```bash
export OPENROUTER_API_KEY="your_key"
```

## Commands

List models (default billing mode is free):

```bash
eyeofai models
```

List paid models:

```bash
eyeofai models --billing-mode paid
```

List free + paid models:

```bash
eyeofai models --billing-mode all
```

Include `openrouter/free` alias in list:

```bash
eyeofai models --include-openrouter-free
```

Expand list with zero-priced models that do not have `:free` suffix:

```bash
eyeofai models --include-zero-priced
```

Run against all free models (auto-skips non-image models for image tasks):

```bash
eyeofai find --input "/absolute/path/to/image.png" --query "find submit" --free-scope all --max-models 10
```

Run against paid models only:

```bash
eyeofai find --input "/absolute/path/to/image.png" --query "find submit" --billing-mode paid --max-models 10
```

Run best-model discovery across free + paid vision models:

```bash
eyeofai rank --dataset eyeofai/testdata/dataset.json --billing-mode all --max-models 25
```

Try a larger model fanout (up to available free image-capable models):

```bash
eyeofai find --input "/absolute/path/to/image.png" --query "find submit" --free-scope all --max-models 50
```

Same with zero-priced expansion (more candidates):

```bash
eyeofai find --input "/absolute/path/to/image.png" --query "find submit" --free-scope all --max-models 50 --include-zero-priced
```

If free-model rate limits kick in after the first run, slow request pace:

```bash
eyeofai find \
  --input "/absolute/path/to/image.png" \
  --query "find submit" \
  --max-models 10 \
  --concurrency 1 \
  --max-retries 3 \
  --retry-base-delay 2 \
  --inter-request-delay 0.35
```

Find coordinates in one file:

```bash
eyeofai find --input "/absolute/path/to/image.png" --query "find submit button"
```

Find in folders:

```bash
eyeofai find --input "/absolute/path/to/folder" --recursive --query "find logo"
```

Use capture mode:

```bash
eyeofai find --capture screen --query "find chat icon"
eyeofai find --capture region --region 100,120,640,420 --query "find search box"
```

Use selected free models:

```bash
eyeofai find \
  --input ./test.png \
  --query "find red button" \
  --models "google/gemma-3-12b-it:free,mistralai/mistral-small-3.1-24b-instruct:free"
```

Limit number of free models dynamically:

```bash
eyeofai find --input ./test.png --query "find title" --max-models 5 --strategy fastest
```

Write JSON output:

```bash
eyeofai find --input ./test.png --query "find icon" --out-json result.json
```

Disable timestamped raw run files:

```bash
eyeofai find --input ./test.png --query "find icon" --save-raw false
```

Run benchmark:

```bash
eyeofai bench --dataset ./dataset.json --max-models 4
```

Generate synthetic test images and dataset:

```bash
python -m eyeofai.tools.generate_test_images
eyeofai bench --dataset eyeofai/testdata/dataset.json --max-models 4
```

## Best Model Discovery (Leaderboard)

Run `rank` to discover the best vision models across free and paid tiers:

```bash
eyeofai rank \
  --dataset eyeofai/testdata/dataset.json \
  --billing-mode all \
  --model-scope vision \
  --max-models 25 \
  --out-json leaderboard.json
```

### Sample Results (25 Vision Models, 3 Test Images)

**Scoring weights:** IoU (65%), Reliability (25%), Speed (10%)

| Rank | Model | Score | IoU | Hit@0.5 | Reliability | p50(ms) |
|------|-------|-------|-----|---------|-------------|---------|
| 1 | openai/gpt-5.1-codex-max | 0.942 | 1.000 | 0.333 | 1.000 | 5527 |
| 2 | anthropic/claude-sonnet-4.6 | 0.938 | 0.956 | 1.000 | 1.000 | 1991 |
| 3 | moonshotai/kimi-k2.5 | 0.896 | 0.985 | 0.333 | 1.000 | 63805 |
| 4 | openai/gpt-5.2-chat | 0.892 | 0.909 | 0.333 | 1.000 | 3837 |
| 5 | bytedance-seed/seed-1.6 | 0.873 | 0.907 | 0.667 | 1.000 | 7935 |
| 6 | openai/gpt-5.2 | 0.854 | 0.870 | 1.000 | 1.000 | 6275 |
| 7 | anthropic/claude-opus-4.6 | 0.750 | 0.687 | 1.000 | 1.000 | 3426 |
| 8 | openai/gpt-5.2-codex | 0.710 | 0.628 | 0.333 | 1.000 | 3768 |
| 9 | openai/gpt-5.2-pro | 0.691 | 0.649 | 0.667 | 1.000 | 17085 |
| 10 | anthropic/claude-opus-4.5 | 0.657 | 0.543 | 0.333 | 1.000 | 3355 |
| 13 | **openrouter/free** ⭐ | **0.466** | **0.263** | **0.333** | **1.000** | **4976** |

**Key Findings:**
- 🏆 **Best overall:** GPT-5.1-codex-max (perfect IoU: 1.000)
- ⚡ **Fastest top performer:** Claude Sonnet 4.6 (~2s, score 0.938)
- 💰 **Best free option:** openrouter/free (rank #13, reliable but lower accuracy)
- 📊 All top models achieved 100% reliability (no failures)
- 🐌 Slowest: kimi-k2.5 (63s) but high accuracy

**Customize scoring:**
```bash
eyeofai rank --dataset data.json --w-iou 0.8 --w-reliability 0.15 --w-speed 0.05
```

## Notes

- EyeOfAI does not assume one model is always right. It uses consensus clustering by IoU and flags uncertain outputs when model agreement is weak.
- Agreement is reported across attempted models and successful models separately.
- Free model availability changes. Use `--refresh-models` to refresh catalog immediately.

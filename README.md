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

<p align="center">
  <img src="EyeOfAI.png" alt="EyeOfAI Demo" width="600"/>
</p>

**EyeOfAI** — Ask multiple vision models where stuff is, get coordinates back. Works with free and paid models from OpenRouter.

## What's this?

I got tired of vision models giving me vague descriptions when I asked "where is the button?" So I built this to get actual (x,y) coordinates. It queries a bunch of models at once, figures out which ones agree, and gives you a consensus bounding box.

## Quick Start

```bash
# Install it
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Set your OpenRouter key
export OPENROUTER_API_KEY="your_key"

# Find something in an image
eyeofai find --input screenshot.png --query "find the login button"
```

## What it can do

- Works with images, PDFs, or your whole screen
- Tests free models, paid models, or both
- Handles rate limits with automatic retries
- Shows live results in the terminal as models respond
- Ranks models to find the best ones for your use case

## Usage Examples

**List available models:**
```bash
eyeofai models                    # free models only
eyeofai models --billing-mode all # everything including paid
```

**Find coordinates:**
```bash
# Basic search
eyeofai find --input image.png --query "find the red button"

# Use your screen
eyeofai find --capture screen --query "find the chat icon"

# Try a bunch of models at once
eyeofai find --input image.png --query "find logo" --billing-mode all --max-models 20
```

**Rate limiting giving you trouble?**
```bash
eyeofai find \
  --input image.png \
  --query "find submit" \
  --concurrency 1 \
  --max-retries 3 \
  --inter-request-delay 0.35
```

**Rank models to see which are best:**
```bash
eyeofai rank --dataset testdata/dataset.json --billing-mode all --max-models 25
```

This runs models against labeled test data and scores them by accuracy (IoU), reliability, and speed.

## Sample Results

I ran 25 vision models on 3 test images. Here's what actually worked:

| Rank | Model | Score | IoU | Hit@0.5 | p50(ms) |
|------|-------|-------|-----|---------|---------|
| 1 | openai/gpt-5.1-codex-max | 0.942 | 1.000 | 0.333 | 5527 |
| 2 | anthropic/claude-sonnet-4.6 | 0.938 | 0.956 | 1.000 | 1991 |
| 3 | moonshotai/kimi-k2.5 | 0.896 | 0.985 | 0.333 | 63805 |
| 4 | openai/gpt-5.2-chat | 0.892 | 0.909 | 0.333 | 3837 |
| 5 | bytedance-seed/seed-1.6 | 0.873 | 0.907 | 0.667 | 7935 |
| 13 | **openrouter/free** | 0.466 | 0.263 | 0.333 | 4976 |

Takeaways:
- GPT-5.1-codex-max nailed it (perfect IoU)
- Claude Sonnet 4.6 is fast AND accurate (~2 seconds)
- The free option (openrouter/free) works but isn't as precise
- Kimi-k2.5 is accurate but takes a minute (63s)

## Notes

- Models don't always agree. That's why we use consensus.
- Agreement is calculated two ways: against attempted models and successful ones.
- Free model availability changes - use `--refresh-models` if stuff breaks.

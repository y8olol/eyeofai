from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    out_dir = Path("eyeofai/testdata")
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = []

    specs = [
        ("submit_button.png", "find the red submit button", (120, 200, 340, 270), (210, 35, 35)),
        ("search_icon.png", "find the blue search area", (420, 80, 570, 180), (45, 95, 220)),
        ("logo_box.png", "find the green logo block", (60, 60, 210, 170), (35, 160, 85)),
    ]

    for file_name, query, rect, color in specs:
        img = Image.new("RGB", (800, 450), (245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle(rect, fill=color)
        draw.rectangle((300, 300, 500, 380), fill=(120, 120, 120))
        draw.text((20, 410), "EyeOfAI synthetic benchmark", fill=(20, 20, 20))
        path = out_dir / file_name
        img.save(path)
        x1, y1, x2, y2 = rect
        examples.append(
            {
                "input": str(path),
                "query": query,
                "gt": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
            }
        )

    dataset = out_dir / "dataset.json"
    dataset.write_text(json.dumps(examples, indent=2))
    print(f"Generated {len(examples)} files and dataset at {dataset}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import io
from pathlib import Path

from mss import mss
from PIL import Image

from eyeofai.schemas import InputFrame


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PDF_EXTENSIONS = {".pdf"}


def _image_to_frame(img: Image.Image, source: str, page: int | None) -> InputFrame:
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    return InputFrame(
        source=source,
        page=page,
        width=rgb.width,
        height=rgb.height,
        image_bytes=buf.getvalue(),
        offset_x=0,
        offset_y=0,
    )


def load_from_path(path: Path, recursive: bool = False) -> list[InputFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    files: list[Path] = []
    if path.is_file():
        files = [path]
    else:
        pattern = "**/*" if recursive else "*"
        files = [p for p in path.glob(pattern) if p.is_file()]

    frames: list[InputFrame] = []
    for file in files:
        ext = file.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            img = Image.open(file)
            frames.append(_image_to_frame(img, str(file), None))
        elif ext in PDF_EXTENSIONS:
            frames.extend(load_pdf(file))
    return frames


def load_pdf(path: Path) -> list[InputFrame]:
    try:
        import pypdfium2 as pdfium
    except Exception as exc:
        raise RuntimeError(
            "PDF support requires pypdfium2. Install with: pip install pypdfium2"
        ) from exc

    document = pdfium.PdfDocument(str(path))
    frames: list[InputFrame] = []
    for i in range(len(document)):
        page = document[i]
        bitmap = page.render(scale=2).to_pil()
        frames.append(_image_to_frame(bitmap, str(path), i + 1))
    return frames


def capture_screen(mode: str, region: str | None = None) -> InputFrame:
    with mss() as sct:
        if mode == "screen":
            monitor = sct.monitors[1]
            shot = sct.grab(monitor)
            img = Image.frombytes("RGB", shot.size, shot.rgb)
            return _image_to_frame(img, "capture:screen", None)

        if mode == "region":
            if not region:
                raise ValueError("--region is required for --capture region. Format: x,y,w,h")
            x, y, w, h = [int(v.strip()) for v in region.split(",")]
            shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
            img = Image.frombytes("RGB", shot.size, shot.rgb)
            frame = _image_to_frame(img, f"capture:region:{x},{y},{w},{h}", None)
            frame.offset_x = x
            frame.offset_y = y
            return frame

        raise ValueError("capture mode must be 'screen' or 'region'")

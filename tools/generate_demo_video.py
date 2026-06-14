"""
Generate a short competition demo video for the AMD-accelerated speech disability project.
"""

from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

try:
    from moviepy.editor import ImageClip, concatenate_videoclips
except ImportError:
    # MoviePy v2 API fallback
    from moviepy import ImageClip, concatenate_videoclips


WIDTH = 1280
HEIGHT = 720
FPS = 24


def _with_duration(clip, duration: float):
    if hasattr(clip, "set_duration"):
        return clip.set_duration(duration)
    return clip.with_duration(duration)


def _load_font(size: int):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_wrapped_text(draw: ImageDraw.ImageDraw, text: str, font, x: int, y: int, max_width: int, fill):
    words = text.split()
    lines: List[str] = []
    current = []
    for word in words:
        test = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))

    line_height = int((draw.textbbox((0, 0), "Ag", font=font)[3]) * 1.4)
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_height), line, font=font, fill=fill)

    return y + len(lines) * line_height


def build_slide(title: str, bullets: List[str], accent: Tuple[int, int, int], output_path: Path):
    image = Image.new("RGB", (WIDTH, HEIGHT), color=(14, 19, 33))
    draw = ImageDraw.Draw(image)

    # Background accent bars
    draw.rectangle([(0, 0), (WIDTH, 110)], fill=accent)
    draw.rectangle([(0, HEIGHT - 40), (WIDTH, HEIGHT)], fill=(35, 43, 72))

    title_font = _load_font(54)
    subtitle_font = _load_font(34)

    draw.text((60, 24), "AMD Accelerated Speech Interpreter", font=subtitle_font, fill=(240, 245, 255))
    draw.text((60, 150), title, font=title_font, fill=(255, 255, 255))

    bullet_font = _load_font(30)
    y = 250
    for bullet in bullets:
        draw.ellipse([(75, y + 10), (95, y + 30)], fill=accent)
        y = _draw_wrapped_text(
            draw=draw,
            text=bullet,
            font=bullet_font,
            x=115,
            y=y,
            max_width=1080,
            fill=(220, 230, 250),
        )
        y += 14

    image.save(output_path)


def main():
    base_dir = Path(__file__).resolve().parents[1]
    demo_dir = base_dir / "demo"
    slides_dir = demo_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    slide_data = [
        (
            "Competition Problem",
            [
                "Dysarthric speech is often misunderstood by standard voice systems.",
                "Users need accurate, real-time interpretation with low latency.",
                "This project provides text and clear speech output from impaired audio.",
            ],
            (211, 78, 46),
            4,
        ),
        (
            "AMD GPU Powered Pipeline",
            [
                "Whisper ASR and correction model run with GPU-aware runtime profile.",
                "ROCm-aware telemetry reports backend, device and per-stage timing.",
                "Designed to scale from CPU fallback to high-throughput GPU inference.",
            ],
            (0, 126, 167),
            5,
        ),
        (
            "Adaptive Correction Strategy",
            [
                "Low confidence audio triggers high-recovery decoding mode.",
                "Speech features (rate, pauses) activate dysarthria-sensitive correction.",
                "High confidence inputs use minimal, fast correction for responsiveness.",
            ],
            (47, 151, 74),
            5,
        ),
        (
            "Live Demo Flow",
            [
                "Upload impaired speech through FastAPI endpoint /transcribe.",
                "View original text, corrected text, confidence and correction strategy.",
                "Download clear synthesized audio and inspect latency breakdown.",
            ],
            (123, 89, 179),
            5,
        ),
        (
            "Impact and Differentiation",
            [
                "Assistive AI with explainable corrections and personalization support.",
                "Infrastructure advantage: AMD-friendly runtime visibility for judges.",
                "Competition value: measurable speed, quality and accessibility gains.",
            ],
            (189, 142, 0),
            5,
        ),
    ]

    clips = []
    for idx, (title, bullets, accent, duration) in enumerate(slide_data, start=1):
        slide_path = slides_dir / f"slide_{idx:02d}.png"
        build_slide(title, bullets, accent, slide_path)
        clips.append(_with_duration(ImageClip(str(slide_path)), duration))

    video = concatenate_videoclips(clips, method="compose")
    output_video = demo_dir / "competition_demo.mp4"
    video.write_videofile(
        str(output_video),
        fps=FPS,
        codec="libx264",
        audio=False,
        threads=2,
        preset="medium",
    )

    print(f"Demo video created: {output_video}")


if __name__ == "__main__":
    main()

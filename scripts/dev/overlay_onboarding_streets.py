from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "frontend" / "react_app" / "src" / "assets" / "onboarding"
ROAD_FILE = ROOT / "data" / "processed" / "road_network_parts" / "06__Concepcion.csv"

# Approximate viewport shown by the current onboarding captures around Plaza Peru / Mall del Centro.
LON_MIN, LON_MAX = -73.0545, -73.0400
LAT_MIN, LAT_MAX = -36.8316, -36.8236
MAP_VIEWPORT = (398, 122, 1072, 712)
DEFAULT_DRAW_CLIP = MAP_VIEWPORT
DRAW_CLIPS = {
    "06-layers.png": (398, 122, 902, 712),
}

IMPORTANT_VIAS = {
    "Diagonal Pedro Aguirre Cerda",
    "Avenida Libertador Bernardo O'Higgins",
    "O'Higgins",
    "Caupolicán",
    "Aníbal Pinto",
    "Rengo",
    "Barros Arana",
    "Chacabuco",
    "Colo Colo",
    "Freire",
    "Janequeo",
    "Castellón",
    "Lincoyán",
    "Tucapel",
}


def project(lon: float, lat: float) -> tuple[float, float]:
    x0, y0, x1, y1 = MAP_VIEWPORT
    x = x0 + ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * (x1 - x0)
    y = y1 - ((lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * (y1 - y0)
    return x, y


def clipped(points: list[tuple[float, float]]) -> bool:
    x0, y0, x1, y1 = MAP_VIEWPORT
    return any(x0 - 16 <= x <= x1 + 16 and y0 - 16 <= y <= y1 + 16 for x, y in points)


def line_angle(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    x0, y0 = points[0]
    x1, y1 = points[-1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def draw_label(base: Image.Image, text: str, xy: tuple[float, float], angle: float, font: ImageFont.ImageFont) -> None:
    if not text or abs(angle) > 75:
        return
    label = text.replace("Avenida Libertador Bernardo ", "")
    if len(label) > 22:
        label = f"{label[:21]}..."
    padding_x, padding_y = 5, 3
    dummy = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy)
    bbox = dummy_draw.textbbox((0, 0), label, font=font)
    width = bbox[2] - bbox[0] + padding_x * 2
    height = bbox[3] - bbox[1] + padding_y * 2
    tag = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(tag)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=4, fill=(255, 255, 255, 196))
    draw.text((padding_x, padding_y - 1), label, font=font, fill=(31, 57, 66, 230))
    rotated = tag.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    base.alpha_composite(rotated, (int(xy[0] - rotated.width / 2), int(xy[1] - rotated.height / 2)))


def main() -> None:
    df = pd.read_csv(
        ROAD_FILE,
        usecols=["segment_id", "indice_coord", "lon", "lat", "via"],
    )
    df = df[
        df["lon"].between(LON_MIN, LON_MAX)
        & df["lat"].between(LAT_MIN, LAT_MAX)
        & df["via"].notna()
    ].copy()
    df.sort_values(["segment_id", "indice_coord"], inplace=True)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    segments: list[tuple[str, list[tuple[float, float]]]] = []
    for _, group in df.groupby("segment_id", sort=False):
        points = [project(float(row.lon), float(row.lat)) for row in group.itertuples()]
        if len(points) >= 2 and clipped(points):
            via = str(group["via"].iloc[0])
            segments.append((via, points))

    for path in ASSET_DIR.glob("*.png"):
        image = Image.open(path).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        draw_clip = DRAW_CLIPS.get(path.name, DEFAULT_DRAW_CLIP)
        x0, y0, x1, y1 = draw_clip
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle(draw_clip, fill=255)

        for _, points in segments:
            draw.line(points, fill=(255, 255, 255, 210), width=5, joint="curve")
        for via, points in segments:
            width = 2 if any(name.lower() in via.lower() for name in ["avenida", "diagonal", "ohiggins"]) else 1
            draw.line(points, fill=(68, 96, 108, 196), width=width, joint="curve")

        clipped_overlay = overlay.copy()
        clipped_overlay.putalpha(ImageChops.multiply(overlay.getchannel("A"), mask))
        image.alpha_composite(clipped_overlay)

        label_count = 0
        for via, points in segments:
            if label_count >= 9:
                break
            if not any(name.lower() in via.lower() for name in IMPORTANT_VIAS):
                continue
            mid = points[len(points) // 2]
            if not (x0 + 24 <= mid[0] <= x1 - 24 and y0 + 24 <= mid[1] <= y1 - 24):
                continue
            draw_label(image, via, mid, line_angle(points), font)
            label_count += 1

        image.convert("RGB").save(path, quality=95)
        print(f"Updated {path.name} with {len(segments)} real street segments")


if __name__ == "__main__":
    main()

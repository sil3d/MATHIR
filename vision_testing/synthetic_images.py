#!/usr/bin/env python3
"""
MATHIR Vision Testing - Synthetic Image Generator
================================================

Generates deterministic PIL images with EXACT ground-truth metadata for
vision-model accuracy benchmarks. No external image dependencies (no
photos, no real-world data) - everything is drawn programmatically with
Pillow's ImageDraw so the test harness can assert exact counts, colors,
positions and object types.

Why synthetic?
- We don't have a labeled photo dataset locally
- Synthetic images have 100% exact ground truth (no labeling errors)
- We can test edge cases (cluttered scenes, color similarities, partial
  occlusions) that would be hard to source
- Tests are reproducible and shareable without IP / privacy concerns

Each generator returns a dict:
    {
        "name":          <short slug, also used as filename stem>,
        "image":         <PIL.Image>,
        "category":      <one of TEST_CATEGORIES>,
        "ground_truth":  <dict, schema varies per test type>,
        "description":   <human-readable one-liner for the UI>,
    }

The dict-driven return shape is what `accuracy_tests.AccuracyTestFramework`
consumes - it iterates over `generate_test_images()` and runs every test
type against every image, comparing the model's free-form text response
back to `ground_truth`.
"""
from __future__ import annotations

import io
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Constants - all colors are exact RGB tuples
# ============================================================

IMG_SIZE = 512  # square 512x512 - small enough for fast VLM, big enough for detail
BG = (245, 245, 245)  # off-white background so colors pop

# Test category enum (string values so they round-trip through JSON cleanly)
TEST_CATEGORIES = {
    "colored_objects",       # color recognition + object count
    "multiple_objects",      # object detection in cluttered scene
    "person_scene",          # person detection + clothing
    "complex_scene",         # multi-aspect (color, count, position, attribute)
    "grounding_scene",       # for grounding models (single-object bbox)
}

# Color name lookup - maps exact RGB tuple to canonical English color name
# Used by accuracy scoring to fuzzy-match what the model says
COLOR_NAMES: Dict[Tuple[int, int, int], str] = {
    (220, 50, 50):   "red",
    (50, 120, 220):  "blue",
    (50, 180, 80):   "green",
    (240, 200, 50):  "yellow",
    (240, 130, 40):  "orange",
    (160, 80, 200):  "purple",
    (240, 240, 240): "white",
    (30, 30, 30):    "black",
    (130, 130, 130): "gray",
    (180, 90, 50):   "brown",
}

# Reverse lookup for color-name -> nearest canonical RGB
COLOR_NAME_TO_RGB: Dict[str, Tuple[int, int, int]] = {v: k for k, v in COLOR_NAMES.items()}


def _color_name(rgb: Tuple[int, int, int]) -> str:
    """Map exact RGB to its canonical name. Falls back to 'unknown'."""
    return COLOR_NAMES.get(tuple(rgb), "unknown")


def _font(size: int = 16) -> ImageFont.ImageFont:
    """Best-effort font loader. Falls back to PIL default if no TTF is found.

    The default font is bitmap and cannot be sized, but it's enough for
    labeling tiny debug markers in the synthetic scenes. Real production
    rendering would use a TTF - the test harness doesn't need it.
    """
    try:
        # Try DejaVuSans (common on Linux + macOS, sometimes Windows)
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size=size)
        except Exception:
            return ImageFont.load_default()


def _new_canvas(size: int = IMG_SIZE, bg: Tuple[int, int, int] = BG) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    """Create a fresh RGB canvas with a solid background. Returns (image, draw)."""
    img = Image.new("RGB", (size, size), bg)
    return img, ImageDraw.Draw(img)


# ============================================================
# Individual scene generators
# ============================================================
# Each generator is independent and pure. They share helper drawing
# primitives (draw_circle, draw_square, draw_person_silhouette) defined
# further down so the scenes are visually consistent.

def _draw_circle(draw: ImageDraw.ImageDraw, center: Tuple[int, int], radius: int,
                 color: Tuple[int, int, int], outline: Tuple[int, int, int] = (0, 0, 0)) -> None:
    """Filled circle with a thin black outline. Outline helps color-edge detection."""
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=outline, width=2)


def _draw_square(draw: ImageDraw.ImageDraw, top_left: Tuple[int, int], side: int,
                 color: Tuple[int, int, int], outline: Tuple[int, int, int] = (0, 0, 0)) -> None:
    """Filled square with outline. Used in the colored_squares scene."""
    x, y = top_left
    draw.rectangle((x, y, x + side, y + side), fill=color, outline=outline, width=2)


def _draw_person_silhouette(draw: ImageDraw.ImageDraw, center: Tuple[int, int], height: int,
                            shirt_color: Tuple[int, int, int], pants_color: Tuple[int, int, int] = (40, 40, 80)) -> None:
    """Draw a simple stick-style person silhouette: head + torso + arms + legs.

    height: total height in pixels. We compute head/torso/limb ratios from this
    so the figure looks roughly the same proportions regardless of size.
    """
    cx, cy = center
    head_r = int(height * 0.10)
    torso_h = int(height * 0.30)
    torso_w = int(height * 0.22)
    limb_w = max(int(height * 0.05), 4)
    leg_h = int(height * 0.45)

    head_top = cy - height // 2
    head_center = (cx, head_top + head_r)
    # Head
    draw.ellipse((head_center[0] - head_r, head_center[1] - head_r,
                  head_center[0] + head_r, head_center[1] + head_r),
                 fill=(245, 210, 180), outline=(0, 0, 0), width=2)
    # Torso (shirt)
    torso_top = head_center[1] + head_r
    draw.rectangle((cx - torso_w // 2, torso_top, cx + torso_w // 2, torso_top + torso_h),
                   fill=shirt_color, outline=(0, 0, 0), width=2)
    # Arms (slim rectangles off the torso sides)
    arm_y = torso_top + 4
    draw.rectangle((cx - torso_w // 2 - limb_w, arm_y,
                    cx - torso_w // 2, arm_y + torso_h - 8),
                   fill=shirt_color, outline=(0, 0, 0), width=1)
    draw.rectangle((cx + torso_w // 2, arm_y,
                    cx + torso_w // 2 + limb_w, arm_y + torso_h - 8),
                   fill=shirt_color, outline=(0, 0, 0), width=1)
    # Legs (pants)
    leg_top = torso_top + torso_h
    draw.rectangle((cx - torso_w // 2, leg_top, cx - 2, leg_top + leg_h),
                   fill=pants_color, outline=(0, 0, 0), width=1)
    draw.rectangle((cx + 2, leg_top, cx + torso_w // 2, leg_top + leg_h),
                   fill=pants_color, outline=(0, 0, 0), width=1)


# ------------------------------------------------------------
# Scene 1: Colored circles - color recognition + count test
# ------------------------------------------------------------

def generate_colored_circles() -> Dict[str, Any]:
    """3 red circles + 2 blue circles on a clean background.

    Tests:
      - color recognition (red, blue)
      - object counting (circles)
      - object detection (circle vs other shapes absent)
    """
    img, draw = _new_canvas()
    red = (220, 50, 50)
    blue = (50, 120, 220)
    radius = 38
    # 3 red circles - top row
    red_y = 150
    for x in (130, 256, 380):
        _draw_circle(draw, (x, red_y), radius, red)
    # 2 blue circles - bottom row
    blue_y = 360
    for x in (190, 320):
        _draw_circle(draw, (x, blue_y), radius, blue)

    return {
        "name": "colored_circles",
        "image": img,
        "category": "colored_objects",
        "ground_truth": {
            "red_circles": 3,
            "blue_circles": 2,
            "total_circles": 5,
            "colors": ["red", "blue"],
        },
        "description": "3 red circles and 2 blue circles on a white background",
    }


# ------------------------------------------------------------
# Scene 2: Colored squares - mixed colors + grid layout
# ------------------------------------------------------------

def generate_colored_squares() -> Dict[str, Any]:
    """3x3 grid of colored squares: 4 green, 3 yellow, 2 red.

    Tests:
      - color recognition in a regular grid
      - object counting per color
    """
    img, draw = _new_canvas()
    # Hand-picked grid: g=green, y=yellow, r=red, blank
    grid = [
        ["g", "g", "y"],
        ["y", "r", "g"],
        ["r", "y", "g"],
    ]
    colors = {"g": (50, 180, 80), "y": (240, 200, 50), "r": (220, 50, 50)}
    side = 90
    pad_x = (IMG_SIZE - side * 3) // 2
    pad_y = (IMG_SIZE - side * 3) // 2
    counts = {"green": 0, "yellow": 0, "red": 0}
    for r, row in enumerate(grid):
        for c, key in enumerate(row):
            if key == "x":
                continue
            x = pad_x + c * side
            y = pad_y + r * side
            _draw_square(draw, (x, y), side, colors[key])
            counts[_color_name(colors[key])] += 1
    return {
        "name": "colored_squares",
        "image": img,
        "category": "colored_objects",
        "ground_truth": {
            "green_squares": counts["green"],
            "yellow_squares": counts["yellow"],
            "red_squares": counts["red"],
            "total_squares": sum(counts.values()),
            "colors": ["green", "yellow", "red"],
        },
        "description": "3x3 grid: 4 green, 3 yellow, 2 red squares",
    }


# ------------------------------------------------------------
# Scene 3: Person silhouette - person detection + clothing
# ------------------------------------------------------------

def generate_person_silhouette() -> Dict[str, Any]:
    """One person wearing a red shirt and dark blue pants.

    Tests:
      - person detection (count = 1)
      - clothing description (red shirt)
    """
    img, draw = _new_canvas()
    _draw_person_silhouette(draw, center=(IMG_SIZE // 2, IMG_SIZE // 2 + 20),
                            height=320, shirt_color=(220, 50, 50), pants_color=(40, 40, 80))
    return {
        "name": "person_silhouette",
        "image": img,
        "category": "person_scene",
        "ground_truth": {
            "people": 1,
            "shirt_colors": ["red"],
            "pants_colors": ["blue"],
            "wearing_glasses": False,
            "wearing_hat": False,
        },
        "description": "1 person wearing a red shirt and dark blue pants",
    }


# ------------------------------------------------------------
# Scene 4: Multiple people - count + multi-clothing
# ------------------------------------------------------------

def generate_multiple_people() -> Dict[str, Any]:
    """3 people: red shirt, green shirt, blue shirt. All wearing dark pants.

    Tests:
      - people counting
      - multiple clothing colors
    """
    img, draw = _new_canvas()
    shirts = [(220, 50, 50), (50, 180, 80), (50, 120, 220)]
    x_positions = [130, 256, 380]
    for x, shirt in zip(x_positions, shirts):
        _draw_person_silhouette(draw, center=(x, IMG_SIZE // 2 + 30), height=300,
                                shirt_color=shirt, pants_color=(30, 30, 60))
    return {
        "name": "multiple_people",
        "image": img,
        "category": "person_scene",
        "ground_truth": {
            "people": 3,
            "shirt_colors": ["red", "green", "blue"],
            "pants_colors": ["black"],
            "wearing_glasses": False,
            "wearing_hat": False,
        },
        "description": "3 people wearing red, green and blue shirts",
    }


# ------------------------------------------------------------
# Scene 5: Counting scene - circles of different colors for counting
# ------------------------------------------------------------

def generate_counting_scene() -> Dict[str, Any]:
    """4 red circles + 3 blue circles + 2 green circles = 9 total.

    Tests:
      - object counting (total + per-color)
      - color recognition
      - disambiguation (multiple similar objects)
    """
    img, draw = _new_canvas()
    palette = [(220, 50, 50), (50, 120, 220), (50, 180, 80)]
    counts = [4, 3, 2]
    radius = 30
    # Lay them out in 3 rows by color
    y_starts = [110, 256, 400]
    for color, count, y in zip(palette, counts, y_starts):
        spacing = IMG_SIZE // (count + 1)
        for i in range(count):
            x = spacing * (i + 1)
            _draw_circle(draw, (x, y), radius, color)
    return {
        "name": "counting_scene",
        "image": img,
        "category": "multiple_objects",
        "ground_truth": {
            "red_circles": 4,
            "blue_circles": 3,
            "green_circles": 2,
            "total_circles": 9,
            "colors": ["red", "blue", "green"],
        },
        "description": "4 red, 3 blue and 2 green circles (9 total)",
    }


# ------------------------------------------------------------
# Scene 6: Complex scene - circles + squares mixed
# ------------------------------------------------------------

def generate_complex_scene() -> Dict[str, Any]:
    """Mixed shapes: 2 red squares, 3 blue circles, 1 green square.

    Tests:
      - object detection (squares vs circles)
      - color recognition
      - shape recognition
      - counting per (color, shape) bucket
    """
    img, draw = _new_canvas()
    # 2 red squares (top-left, top-right)
    _draw_square(draw, (60, 60), 100, (220, 50, 50))
    _draw_square(draw, (350, 60), 100, (220, 50, 50))
    # 3 blue circles (middle row)
    for x in (130, 256, 380):
        _draw_circle(draw, (x, 280), 40, (50, 120, 220))
    # 1 green square (bottom)
    _draw_square(draw, (200, 400), 110, (50, 180, 80))
    return {
        "name": "complex_scene",
        "image": img,
        "category": "complex_scene",
        "ground_truth": {
            "red_squares": 2,
            "blue_circles": 3,
            "green_squares": 1,
            "total_shapes": 6,
            "shapes": ["square", "circle"],
            "colors": ["red", "blue", "green"],
        },
        "description": "Mixed: 2 red squares, 3 blue circles, 1 green square",
    }


# ------------------------------------------------------------
# Scene 7: Single object for grounding - one big red circle
# ------------------------------------------------------------

def generate_grounding_scene() -> Dict[str, Any]:
    """One large red circle in the center - for bounding-box grounding tests.

    Tests:
      - LocateAnything-style grounding: "find the {object}" -> bbox
      - Bbox should approximately cover the central circle
    """
    img, draw = _new_canvas()
    _draw_circle(draw, (IMG_SIZE // 2, IMG_SIZE // 2), 120, (220, 50, 50))
    return {
        "name": "grounding_red_circle",
        "image": img,
        "category": "grounding_scene",
        "ground_truth": {
            "target_object": "red circle",
            "expected_bbox": {
                # x_min, y_min, x_max, y_max in pixel coordinates (approximate)
                "x_min": IMG_SIZE // 2 - 125,
                "y_min": IMG_SIZE // 2 - 125,
                "x_max": IMG_SIZE // 2 + 125,
                "y_max": IMG_SIZE // 2 + 125,
            },
            "tolerance_px": 40,
        },
        "description": "One large red circle in the center for grounding tests",
    }


# ============================================================
# Top-level entry point
# ============================================================
# This is the function the test framework imports. It returns the full
# battery of test images in a stable, deterministic order.

def generate_test_images() -> List[Dict[str, Any]]:
    """Return the full battery of test images with exact ground truth.

    The order matters: callers index by position when loading cached
    runs, and we want the list to be the same every call.
    """
    return [
        generate_colored_circles(),
        generate_colored_squares(),
        generate_person_silhouette(),
        generate_multiple_people(),
        generate_counting_scene(),
        generate_complex_scene(),
        generate_grounding_scene(),
    ]


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Serialize a PIL image to raw bytes (PNG by default).

    PNG is lossless - JPEG would muddy exact color comparisons. Use PNG
    for benchmark images; the UI can downscale / re-encode as needed.
    """
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Serialize a PIL image to base64 (data URI prefix excluded)."""
    import base64
    return base64.b64encode(image_to_bytes(img, fmt=fmt)).decode("ascii")


def save_test_image(img: Image.Image, dest: Path, fmt: str = "PNG") -> Path:
    """Save a PIL image to disk, creating parent dirs as needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format=fmt)
    return dest


# ============================================================
# CLI - dump all test images to disk for inspection
# ============================================================

def main():
    """CLI: generate every test image and write to accuracy_benchmarks/images/.

    Usage: python synthetic_images.py [--out <dir>]
    """
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "accuracy_benchmarks" / "images"),
        help="Output directory for PNG images",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = generate_test_images()
    for scene in scenes:
        dest = out_dir / f"{scene['name']}.png"
        save_test_image(scene["image"], dest)
        # Sidecar JSON for ground truth
        meta = {k: v for k, v in scene.items() if k != "image"}
        meta_dest = out_dir / f"{scene['name']}.json"
        with open(meta_dest, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [+] {scene['name']} -> {dest.name} + {meta_dest.name}  ({scene['category']})")

    print(f"\nWrote {len(scenes)} test images to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

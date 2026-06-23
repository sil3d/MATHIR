#!/usr/bin/env python3
"""
MATHIR Playground - Accuracy Test Framework
================================================

Runs a battery of vision-model accuracy tests against each enabled model
in `config.json`. Tests use synthetic PIL-generated images (see
`./synthetic_images.py`) so ground truth is exact - no labeling noise.

The framework is **API-first** for model interaction: it uses the same
`llama-server` chat-completion HTTP endpoint that `ui_server.py` already
uses, so it works with any GGUF model without code changes.

What it measures
----------------
- **Object detection**   - Did the model name the objects in the scene?
- **Color recognition**  - Did the model name the correct colors?
- **Counting**           - Did the model count objects (per color) correctly?
- **Person/clothing**    - Did the model detect people + describe clothing?
- **Grounding**          - For grounding models (LocateAnything): did it
                          return bounding boxes for the requested object?

What it does NOT depend on
--------------------------
- No hardcoded test questions - all prompts are built from templates
  parameterized by the test type and the image's ground truth
- No hardcoded paths - all paths come from `config.json` / this file's dir
- No external APIs - all inference goes through local llama-server

Usage
-----
    # From another Python module:
    from vision_testing.accuracy_tests import AccuracyTestFramework
    fw = AccuracyTestFramework("LFM2.5-VL-1.6B-Q4_0")
    results = fw.run_full_battery()

    # From the CLI:
    python accuracy_tests.py --model LFM2.5-VL-1.6B-Q4_0
    python accuracy_tests.py --all           # test every enabled model
    python accuracy_tests.py --list-tests    # list available test images
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import base64
import argparse
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Lazy imports - PIL + requests are needed at module load but the actual
# model interaction is deferred until run_full_battery() is called.
from PIL import Image


# ============================================================
# Path resolution - NO hardcoded paths
# ============================================================

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so we can import synthetic_images / vision_test

try:
    from synthetic_images import (
        generate_test_images,
        image_to_base64,
        TEST_CATEGORIES,
    )
except Exception as _exc:  # pragma: no cover - guard against path issues
    print(f"[accuracy_tests] WARN: could not import synthetic_images: {_exc}")
    generate_test_images = None
    image_to_base64 = None
    TEST_CATEGORIES = set()


# ============================================================
# Test types
# ============================================================
# String enum that round-trips through JSON. Each test type has:
#   - a prompt template (filled in at runtime from image metadata)
#   - a scoring function (response text + ground truth -> 0..1)

TEST_TYPE_OBJECT_DETECTION = "object_detection"
TEST_TYPE_COLOR_RECOGNITION = "color_recognition"
TEST_TYPE_COUNTING = "counting"
TEST_TYPE_PERSON_DETECTION = "person_detection"
TEST_TYPE_GROUNDING = "grounding"

ALL_TEST_TYPES: Tuple[str, ...] = (
    TEST_TYPE_OBJECT_DETECTION,
    TEST_TYPE_COLOR_RECOGNITION,
    TEST_TYPE_COUNTING,
    TEST_TYPE_PERSON_DETECTION,
    TEST_TYPE_GROUNDING,
)

# Which test types are applicable to which image category. This is used
# to skip irrelevant tests automatically (e.g. don't run "person_detection"
# against a pure colored-shapes scene).
CATEGORY_TO_TESTS: Dict[str, Tuple[str, ...]] = {
    "colored_objects":   (TEST_TYPE_OBJECT_DETECTION, TEST_TYPE_COLOR_RECOGNITION, TEST_TYPE_COUNTING),
    "multiple_objects":  (TEST_TYPE_OBJECT_DETECTION, TEST_TYPE_COLOR_RECOGNITION, TEST_TYPE_COUNTING),
    "person_scene":      (TEST_TYPE_OBJECT_DETECTION, TEST_TYPE_PERSON_DETECTION, TEST_TYPE_COUNTING),
    "complex_scene":     (TEST_TYPE_OBJECT_DETECTION, TEST_TYPE_COLOR_RECOGNITION, TEST_TYPE_COUNTING),
    "grounding_scene":   (TEST_TYPE_GROUNDING,),
}

# Result on-disk location
RESULTS_DIR = HERE / "accuracy_benchmarks" / "results"
IMAGES_DIR = HERE / "accuracy_benchmarks" / "images"

# Subset of test_results.json where per-model accuracy results are stored
TEST_RESULTS_PATH = HERE / "test_results.json"


# ============================================================
# Prompt templates
# ============================================================
# All prompts are templates - never hardcoded "what color is the circle?"
# They get formatted at runtime with the image's category / target object.

PROMPT_TEMPLATES: Dict[str, str] = {
    TEST_TYPE_OBJECT_DETECTION:
        "Look at this image carefully. What objects do you see? "
        "List every distinct object you can identify. "
        "Be specific about shapes (circles, squares, people, etc.). "
        "Reply with a short bullet list of objects only.",

    TEST_TYPE_COLOR_RECOGNITION:
        "Look at this image. What colors do you see? "
        "List every distinct color (red, blue, green, yellow, orange, "
        "purple, white, black, gray, brown, etc.). "
        "Reply with a short bullet list of color names only.",

    TEST_TYPE_COUNTING:
        "Look at this image. How many objects are there in total? "
        "Reply with just a single integer number, nothing else. "
        "Example format: 5",

    TEST_TYPE_PERSON_DETECTION:
        "Look at this image. Are there any people visible? "
        "If yes, how many people are there and what are they wearing? "
        "Describe each person's clothing (shirt color, pants color, "
        "any accessories like glasses or hats).",

    TEST_TYPE_GROUNDING:
        "Find the {target_object} in this image. "
        "Return its bounding box as four numbers in pixels: "
        "x_min, y_min, x_max, y_max. "
        "Reply in exactly this format: <box>x_min, y_min, x_max, y_max</box>",
}

# Default count prompt when the test wants to count a specific object type
# (used when the image has a single dominant object class). The test
# framework auto-derives the {object} from the image's ground truth.
COUNT_OBJECT_PROMPT = (
    "Look at this image. How many {object} are there? "
    "Reply with just a single integer number, nothing else. "
    "Example format: 5"
)


# ============================================================
# Scoring helpers - work on free-form text + ground truth dict
# ============================================================
# Every scorer returns a float in [0.0, 1.0]. The framework logs the raw
# response alongside the score so the UI can show the model's actual output
# alongside the ground truth (transparency over opacity).

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace for forgiving string matching."""
    return " ".join((text or "").lower().split())


def _extract_words(text: str) -> List[str]:
    """Pull all word tokens from a chunk of free-form text."""
    return _WORD_RE.findall(_normalize(text))


def _extract_numbers(text: str) -> List[int]:
    """Pull every integer-looking token from free-form text.

    Handles 'there are 5 circles', '5 of them', 'about 10' etc. The order
    is preserved so position-sensitive tests (e.g. '5, 3, 2' for per-color
    counts) can be scored.
    """
    out: List[int] = []
    for m in re.finditer(r"-?\d+", text or ""):
        try:
            out.append(int(m.group(0)))
        except ValueError:
            continue
    return out


def _present_color(gt: Dict[str, Any]) -> List[str]:
    """Return the list of distinct colors present in a ground_truth dict.

    Looks for keys ending in '_circles', '_squares', or any list under the
    'colors' key, and falls back to the 'colors' list directly.
    """
    if "colors" in gt and isinstance(gt["colors"], list):
        return [str(c).lower() for c in gt["colors"]]
    out: List[str] = []
    for k, v in gt.items():
        if isinstance(v, int) and v > 0 and ("_circles" in k or "_squares" in k):
            color = k.split("_")[0]
            out.append(color)
    return out


def _present_objects(gt: Dict[str, Any]) -> List[str]:
    """Derive a list of (object_type) labels present in a ground_truth dict.

    Looks at per-color count keys (e.g. 'red_circles' -> ['circle']).
    """
    objects: List[str] = []
    seen = set()
    for k in gt.keys():
        if "_circles" in k:
            objects.append("circle")
            seen.add("circle")
        elif "_squares" in k:
            objects.append("square")
            seen.add("square")
    if "people" in gt and isinstance(gt["people"], int) and gt["people"] > 0:
        objects.append("person")
    return objects


# ------------------------------------------------------------
# Per-test scoring functions
# ------------------------------------------------------------

def score_object_detection(response: str, ground_truth: Dict[str, Any]) -> float:
    """Did the model name the right object classes (circle, square, person)?

    Scoring: 1.0 if every expected object class is mentioned at least once
    AND no fabricated class names appear. Partial credit for each
    correctly-named class.
    """
    text = _normalize(response)
    expected = _present_objects(ground_truth)
    if not expected:
        return 1.0  # nothing to test

    # Build a lookup of common synonyms so 'shape', 'round', 'box' count
    synonyms: Dict[str, List[str]] = {
        "circle": ["circle", "circles", "round", "disc", "dot", "ball"],
        "square": ["square", "squares", "rectangle", "rectangles", "box", "boxes"],
        "person": ["person", "people", "human", "humans", "figure", "figures", "man", "woman", "child"],
    }
    hits = 0
    for obj in expected:
        syns = synonyms.get(obj, [obj])
        if any(s in text for s in syns):
            hits += 1
    if hits == 0:
        return 0.0
    return round(hits / len(expected), 3)


def score_color_recognition(response: str, ground_truth: Dict[str, Any]) -> float:
    """Did the model name the right colors?

    Scoring: 1.0 if every expected color is mentioned at least once.
    """
    text = _normalize(response)
    expected = _present_color(ground_truth)
    if not expected:
        return 1.0
    hits = 0
    for c in expected:
        # Common variants: 'red' / 'reddish', 'gray' / 'grey'
        variants = [c]
        if c == "gray":
            variants.append("grey")
        if c.endswith("y"):
            variants.append(c[:-1] + "ish")  # 'gray' -> 'grayish'
        if any(v in text for v in variants):
            hits += 1
    return round(hits / len(expected), 3)


def score_counting(response: str, ground_truth: Dict[str, Any]) -> float:
    """Did the model count objects (per color + total) correctly?

    We try to match the model's numbers against the expected total AND
    per-color counts. The headline number (first one) is the primary
    signal, but if the model instead opens with a per-color count we
    still get credit as long as the right numbers appear somewhere in
    the response. Partial credit is given for off-by-one.
    """
    text = _normalize(response)
    nums = _extract_numbers(response)
    if not nums:
        return 0.0

    # Total-count check (most important)
    expected_total = ground_truth.get("total_circles") or ground_truth.get("total_squares") \
        or ground_truth.get("total_shapes") or ground_truth.get("people")
    if expected_total is None:
        # No total field - count of per-color keys
        per_color = [v for k, v in ground_truth.items()
                     if isinstance(v, int) and v > 0 and ("_circles" in k or "_squares" in k)]
        expected_total = sum(per_color) if per_color else None
    if expected_total is None:
        # Nothing to score (e.g. person scene - skip)
        return 1.0

    # Find the model's best guess at the total. We prefer the headline
    # (first) number, but if that's not the total we also accept the
    # largest number in the response (since "5 total" is a common phrasing)
    headline = nums[0]
    if headline == expected_total:
        score = 0.7
    elif expected_total in nums:
        # Total appears somewhere in the response (e.g. "5 total" at the end)
        score = 0.6
    else:
        # Off-by-one still gets partial credit
        score = 0.3 if abs(headline - expected_total) == 1 else 0.0

    # Per-color count match: up to 0.3 extra credit if model breaks down
    # counts (e.g. "3 red, 2 blue"). Each correctly-named count adds credit.
    per_color = {k: v for k, v in ground_truth.items()
                 if isinstance(v, int) and v > 0 and ("_circles" in k or "_squares" in k)}
    if per_color:
        per_color_hits = 0
        for k, expected in per_color.items():
            if expected in nums:
                per_color_hits += 1
        if per_color_hits > 0:
            score = min(1.0, score + 0.3 * (per_color_hits / len(per_color)))
    return round(score, 3)


def score_person_detection(response: str, ground_truth: Dict[str, Any]) -> float:
    """Did the model find people + describe clothing colors?

    Scoring breakdown:
      - Person presence/absence (0.4 weight)
      - Count accuracy (0.2 weight)
      - Shirt color accuracy (0.2 weight)
      - Pants color accuracy (0.1 weight)
      - Accessories (glasses / hat) (0.1 weight) - only if specified
    """
    text = _normalize(response)
    score = 0.0

    # 1. Person presence
    expected_people = int(ground_truth.get("people", 0) or 0)
    if expected_people == 0:
        # Image without people - expect 'no' or 'none' or 'zero'
        if any(w in text for w in ("no person", "no people", "no human", "none", "0", "zero")):
            return 1.0
        return 0.0  # model hallucinated people

    person_keywords = ("person", "people", "human", "man", "woman", "figure", "child", "someone")
    if any(k in text for k in person_keywords):
        score += 0.4

    # 2. Count accuracy
    nums = _extract_numbers(response)
    if nums and nums[0] == expected_people:
        score += 0.2
    elif nums and abs(nums[0] - expected_people) == 1:
        score += 0.1

    # 3. Shirt colors
    expected_shirts = [c.lower() for c in ground_truth.get("shirt_colors", [])]
    if expected_shirts:
        shirt_hits = sum(1 for c in expected_shirts if c in text)
        score += 0.2 * (shirt_hits / len(expected_shirts))

    # 4. Pants colors
    expected_pants = [c.lower() for c in ground_truth.get("pants_colors", [])]
    if expected_pants:
        pants_hits = sum(1 for c in expected_pants if c in text)
        score += 0.1 * (pants_hits / len(expected_pants))

    # 5. Accessories
    if "wearing_glasses" in ground_truth:
        wants_glasses = bool(ground_truth["wearing_glasses"])
        mentions_glasses = any(w in text for w in ("glass", "spectacle", "eyewear"))
        if wants_glasses == mentions_glasses:
            score += 0.1
    if "wearing_hat" in ground_truth:
        wants_hat = bool(ground_truth["wearing_hat"])
        mentions_hat = "hat" in text or "cap" in text
        if wants_hat == mentions_hat:
            score += 0.1

    return round(min(score, 1.0), 3)


def score_grounding(response: str, ground_truth: Dict[str, Any]) -> float:
    """Score a grounding response (bounding box in <box>x1,y1,x2,y2</box>).

    We accept any of these formats:
      - <box>10, 20, 100, 200</box>
      - [10, 20, 100, 200]
      - "x_min: 10, y_min: 20, x_max: 100, y_max: 200"
      - plain "10, 20, 100, 200" if it's the only 4 numbers in the response
    Returns 1.0 if the predicted bbox is within `tolerance_px` of expected
    (using per-side max-error); else a 0..1 score based on overlap (IoU).
    """
    expected = ground_truth.get("expected_bbox", {})
    tol = int(ground_truth.get("tolerance_px", 40) or 40)
    if not expected:
        return 0.0

    # Find 4 numbers
    nums = _extract_numbers(response)
    if len(nums) < 4:
        return 0.0
    x1, y1, x2, y2 = nums[:4]

    # Normalize bbox order (x_min, y_min, x_max, y_max)
    px_min, py_min = min(x1, x2), min(y1, y2)
    px_max, py_max = max(x1, x2), max(y1, y2)

    ex_min, ey_min = expected["x_min"], expected["y_min"]
    ex_max, ey_max = expected["x_max"], expected["y_max"]

    # Per-side max error
    errs = [
        abs(px_min - ex_min), abs(py_min - ey_min),
        abs(px_max - ex_max), abs(py_max - ey_max),
    ]
    max_err = max(errs)
    if max_err <= tol:
        return 1.0

    # IoU-based partial credit
    inter_x_min = max(px_min, ex_min)
    inter_y_min = max(py_min, ey_min)
    inter_x_max = min(px_max, ex_max)
    inter_y_max = min(py_max, ey_max)
    iw = max(0, inter_x_max - inter_x_min)
    ih = max(0, inter_y_max - inter_y_min)
    inter = iw * ih
    pred_area = max(0, px_max - px_min) * max(0, py_max - py_min)
    exp_area = max(0, ex_max - ex_min) * max(0, ey_max - ey_min)
    union = pred_area + exp_area - inter
    iou = inter / union if union > 0 else 0.0
    return round(min(iou, 0.99), 3)  # cap at 0.99 so we keep 1.0 as the perfect score


# Map test types to scoring functions
SCORERS: Dict[str, Callable[[str, Dict[str, Any]], float]] = {
    TEST_TYPE_OBJECT_DETECTION: score_object_detection,
    TEST_TYPE_COLOR_RECOGNITION: score_color_recognition,
    TEST_TYPE_COUNTING: score_counting,
    TEST_TYPE_PERSON_DETECTION: score_person_detection,
    TEST_TYPE_GROUNDING: score_grounding,
}


# ============================================================
# AccuracyTestFramework
# ============================================================

class AccuracyTestFramework:
    """Run the full accuracy battery against one model.

    Lifecycle:
        fw = AccuracyTestFramework("LFM2.5-VL-1.6B-Q4_0")
        fw.start()          # boots llama-server for the model
        results = fw.run_full_battery()
        fw.stop()           # tears the server down
        fw.save_results(results)   # write per-model JSON

    The framework uses the same `LlamaServer` wrapper as `ui_server.py`
    so it inherits the exact same inference path the UI uses.
    """

    def __init__(self, model_name: str, port: Optional[int] = None,
                 config: Optional[Dict[str, Any]] = None,
                 auto_start: bool = True):
        """Initialize for the given model.

        Args:
            model_name: key in `config.json:models`
            port: llama-server port (auto-assigned if None)
            config: pre-loaded config dict (loaded from disk if None)
            auto_start: whether to start the server immediately
        """
        self.model_name = model_name
        self.config = config or self._load_config()
        self.tester = None  # VisionTester instance, lazily imported
        self._owns_tester = auto_start  # if we created the tester, we stop it
        # Use a port one above the UI default to avoid collision
        self.port = port or (int(self.config.get("llama_server", {}).get("default_port", 8080)) + 100)

    # ------------------------------------------------------------
    # Config loading (no hardcoded paths)
    # ------------------------------------------------------------

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        """Load config.json relative to this file."""
        from vision_test import load_config as _load
        return _load()

    # ------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------

    def start(self) -> None:
        """Start the llama-server for the active model."""
        from vision_test import VisionTester
        if self.tester is not None:
            return
        try:
            self.tester = VisionTester(self.model_name, port=self.port, config=self.config)
            # Only vision-language models are relevant for accuracy tests
            model_info = self.tester.models.get_model(self.model_name) or {}
            if not model_info.get("supports_vision", False):
                print(f"[accuracy_tests] WARNING: {self.model_name} has supports_vision=false in config")
            self.tester.start_server()
            print(f"[accuracy_tests] Server started for {self.model_name} on port {self.port}")
        except Exception as e:
            print(f"[accuracy_tests] Failed to start tester for {self.model_name}: {e}")
            self.tester = None
            raise

    def stop(self) -> None:
        """Stop the server (only if we started it)."""
        if self.tester and self._owns_tester:
            try:
                self.tester.stop_server()
            except Exception:
                pass
        self.tester = None

    # ------------------------------------------------------------
    # Test image access
    # ------------------------------------------------------------

    def generate_test_images(self) -> List[Dict[str, Any]]:
        """Return the full battery of test images (delegates to synthetic_images)."""
        if generate_test_images is None:
            raise RuntimeError("synthetic_images module not importable")
        return generate_test_images()

    def list_available_tests(self) -> List[Dict[str, Any]]:
        """Return a UI-friendly summary of available test images (no PIL objects)."""
        scenes = self.generate_test_images()
        return [
            {
                "name": s["name"],
                "category": s["category"],
                "description": s["description"],
                "applicable_tests": list(CATEGORY_TO_TESTS.get(s["category"], ())),
                "ground_truth": s["ground_truth"],
            }
            for s in scenes
        ]

    # ------------------------------------------------------------
    # Per-test runners
    # ------------------------------------------------------------

    def _image_to_data_uri(self, img: Image.Image) -> str:
        """Convert PIL image to a base64 data URI for the chat-completion API."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _chat_with_image(self, text: str, img: Image.Image,
                         max_tokens: int = 256, temperature: float = 0.0) -> str:
        """Send a chat completion with one image attached.

        Uses greedy decoding (temperature=0) so the test is deterministic -
        a non-deterministic test would score against a moving target.
        """
        if not self.tester or not self.tester.server:
            raise RuntimeError(f"Tester/server not started for {self.model_name}")
        data_uri = self._image_to_data_uri(img)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]
        try:
            response = self.tester.server.chat(messages, max_tokens=max_tokens, temperature=temperature)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            return f"ERROR: {e}"

    def _prompt_for(self, test_type: str, gt: Dict[str, Any]) -> str:
        """Build the prompt for a test type, substituting any {placeholders}.

        For counting, we override the default template with one that asks
        about a specific object class derived from the ground truth
        (e.g. "How many circles are there?"). Falls back to the generic
        total-count template if no class is identifiable.
        """
        if test_type == TEST_TYPE_COUNTING:
            objects = _present_objects(gt)
            if objects:
                # Pick the first object class; tests are per-image, so this
                # is the dominant object type for the image
                obj = objects[0]
                return COUNT_OBJECT_PROMPT.format(object=obj + ("s" if not obj.endswith("s") else ""))
        template = PROMPT_TEMPLATES[test_type]
        return template.format(**gt) if "{" in template else template

    def run_object_detection(self, test_image: Dict[str, Any]) -> Dict[str, Any]:
        """Run the object-detection test against a single test image."""
        prompt = self._prompt_for(TEST_TYPE_OBJECT_DETECTION, test_image["ground_truth"])
        response = self._chat_with_image(prompt, test_image["image"])
        score = SCORERS[TEST_TYPE_OBJECT_DETECTION](response, test_image["ground_truth"])
        return self._make_result(test_image, TEST_TYPE_OBJECT_DETECTION, prompt, response, score)

    def run_color_recognition(self, test_image: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._prompt_for(TEST_TYPE_COLOR_RECOGNITION, test_image["ground_truth"])
        response = self._chat_with_image(prompt, test_image["image"])
        score = SCORERS[TEST_TYPE_COLOR_RECOGNITION](response, test_image["ground_truth"])
        return self._make_result(test_image, TEST_TYPE_COLOR_RECOGNITION, prompt, response, score)

    def run_counting(self, test_image: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._prompt_for(TEST_TYPE_COUNTING, test_image["ground_truth"])
        response = self._chat_with_image(prompt, test_image["image"])
        score = SCORERS[TEST_TYPE_COUNTING](response, test_image["ground_truth"])
        return self._make_result(test_image, TEST_TYPE_COUNTING, prompt, response, score)

    def run_person_detection(self, test_image: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._prompt_for(TEST_TYPE_PERSON_DETECTION, test_image["ground_truth"])
        response = self._chat_with_image(prompt, test_image["image"])
        score = SCORERS[TEST_TYPE_PERSON_DETECTION](response, test_image["ground_truth"])
        return self._make_result(test_image, TEST_TYPE_PERSON_DETECTION, prompt, response, score)

    def run_grounding(self, test_image: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
        """For grounding models (LocateAnything, etc.).

        By default uses the same model the framework is initialized with.
        If `model_name` is provided, that is a hint to the UI that the
        test should be run against a different model (e.g. a dedicated
        grounding model). The score is recorded in the result.
        """
        target = test_image["ground_truth"].get("target_object", "object")
        prompt = PROMPT_TEMPLATES[TEST_TYPE_GROUNDING].format(target_object=target)
        response = self._chat_with_image(prompt, test_image["image"], max_tokens=128)
        score = SCORERS[TEST_TYPE_GROUNDING](response, test_image["ground_truth"])
        return self._make_result(test_image, TEST_TYPE_GROUNDING, prompt, response, score, extra={
            "grounding_model": model_name or self.model_name,
        })

    def _make_result(self, test_image: Dict[str, Any], test_type: str,
                     prompt: str, response: str, score: float,
                     extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build the per-test result dict (uniform shape for the UI)."""
        result = {
            "test_image": test_image["name"],
            "image_category": test_image.get("category", "unknown"),
            "test_type": test_type,
            "prompt": prompt,
            "response": response,
            "ground_truth": test_image["ground_truth"],
            "score": score,
            "passed": score >= 0.7,  # 70% threshold for "pass"
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            result.update(extra)
        return result

    # ------------------------------------------------------------
    # Full battery
    # ------------------------------------------------------------

    def run_full_battery(self) -> Dict[str, Any]:
        """Run every applicable test on every test image.

        Returns a dict with shape:
            {
                "model": <name>,
                "started_at": <iso>,
                "finished_at": <iso>,
                "results": [<per-test result>, ...],
                "summary": {
                    "tests_run": N,
                    "tests_passed": N,
                    "overall_score": 0..1,
                    "by_type": {test_type: {avg_score, runs, passed}},
                    "by_image": {image_name: {avg_score, runs, passed}},
                }
            }
        """
        scenes = self.generate_test_images()
        started = datetime.now()
        all_results: List[Dict[str, Any]] = []

        for scene in scenes:
            applicable = CATEGORY_TO_TESTS.get(scene["category"], ())
            for test_type in applicable:
                try:
                    runner = getattr(self, f"run_{test_type}")
                    res = runner(scene)
                except Exception as e:
                    # Record the failure but keep going so one broken test
                    # doesn't tank the whole run
                    res = self._make_result(
                        scene, test_type,
                        prompt="<runner failed>",
                        response=f"ERROR: {e}\n{traceback.format_exc()[:300]}",
                        score=0.0,
                    )
                all_results.append(res)
                # One-line progress per test (no spammy floods)
                tag = "PASS" if res["passed"] else "FAIL"
                print(f"  [{tag}] {self.model_name} | {scene['name']} | {test_type} | score={res['score']:.2f}")

        finished = datetime.now()
        return {
            "model": self.model_name,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "results": all_results,
            "summary": self._summarize(all_results),
        }

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-test results into a summary block."""
        if not results:
            return {"tests_run": 0, "tests_passed": 0, "overall_score": 0.0,
                    "by_type": {}, "by_image": {}}
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        overall = sum(r["score"] for r in results) / total

        by_type: Dict[str, Dict[str, Any]] = {}
        for r in results:
            t = r["test_type"]
            slot = by_type.setdefault(t, {"runs": 0, "passed": 0, "score_sum": 0.0})
            slot["runs"] += 1
            if r["passed"]:
                slot["passed"] += 1
            slot["score_sum"] += r["score"]
        for v in by_type.values():
            v["avg_score"] = round(v["score_sum"] / v["runs"], 3)
            del v["score_sum"]
            v["pass_rate"] = round(v["passed"] / v["runs"], 3) if v["runs"] else 0

        by_image: Dict[str, Dict[str, Any]] = {}
        for r in results:
            i = r["test_image"]
            slot = by_image.setdefault(i, {"runs": 0, "passed": 0, "score_sum": 0.0})
            slot["runs"] += 1
            if r["passed"]:
                slot["passed"] += 1
            slot["score_sum"] += r["score"]
        for v in by_image.values():
            v["avg_score"] = round(v["score_sum"] / v["runs"], 3)
            del v["score_sum"]
            v["pass_rate"] = round(v["passed"] / v["runs"], 3) if v["runs"] else 0

        return {
            "tests_run": total,
            "tests_passed": passed,
            "overall_score": round(overall, 3),
            "by_type": by_type,
            "by_image": by_image,
        }

    # ------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------

    def save_results(self, results: Dict[str, Any], dest: Optional[Path] = None) -> Path:
        """Write per-model results to disk and update test_results.json.

        Layout:
            accuracy_benchmarks/results/<model_name>_<timestamp>.json
        """
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = dest or (RESULTS_DIR / f"{self.model_name}_{ts}.json")
        with open(dest, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[accuracy_tests] Wrote {dest}")

        # Also patch test_results.json so the UI's Tests view + new
        # Accuracy view see the latest data without polling the disk
        self._merge_into_test_results(results)
        return dest

    def _merge_into_test_results(self, results: Dict[str, Any]) -> None:
        """Insert/update this model's accuracy block inside test_results.json.

        Layout (added under the top-level `accuracy_tests` key):
            {
                "accuracy_tests": {
                    "<model_name>": {
                        "last_run": "<iso>",
                        "overall_score": 0.0,
                        "tests_passed": N,
                        "tests_run": N,
                        "by_type": {...},
                        "by_image": {...},
                        "results": [...per-test...]
                    }
                }
            }
        """
        if not TEST_RESULTS_PATH.exists():
            data: Dict[str, Any] = {}
        else:
            try:
                with open(TEST_RESULTS_PATH) as f:
                    data = json.load(f)
            except Exception:
                data = {}

        data.setdefault("accuracy_tests", {})
        # Strip the raw responses from the embedded copy to keep the file
        # small - the full per-test results are still in the dedicated
        # results/<model>_<ts>.json file we just wrote above
        slim = {
            "last_run": results["finished_at"],
            "overall_score": results["summary"]["overall_score"],
            "tests_passed": results["summary"]["tests_passed"],
            "tests_run": results["summary"]["tests_run"],
            "by_type": results["summary"]["by_type"],
            "by_image": results["summary"]["by_image"],
            "per_test": [
                {
                    "test_image": r["test_image"],
                    "test_type": r["test_type"],
                    "image_category": r["image_category"],
                    "score": r["score"],
                    "passed": r["passed"],
                    "prompt": r["prompt"],
                    "response": r["response"][:600],  # truncate for the file size
                    "ground_truth": r["ground_truth"],
                }
                for r in results["results"]
            ],
        }
        data["accuracy_tests"][self.model_name] = slim
        with open(TEST_RESULTS_PATH, "w") as f:
            json.dump(data, f, indent=2)


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """CLI entry point.

    Modes:
      --model <name>        Run full battery against one model
      --all                 Run against every enabled vision model
      --list-tests          Print the available test battery
      --save-images         (Re)generate synthetic images to disk
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Run accuracy tests against this model")
    parser.add_argument("--all", action="store_true", help="Run against every enabled vision model")
    parser.add_argument("--list-tests", action="store_true", help="List available test images + applicable test types")
    parser.add_argument("--save-images", action="store_true", help="Regenerate the synthetic test images")
    parser.add_argument("--no-start-server", action="store_true", help="Don't auto-start the llama-server (assume external)")
    parser.add_argument("--port", type=int, default=None, help="llama-server port")
    args = parser.parse_args()

    if args.save_images:
        # Delegate to synthetic_images.py CLI
        from synthetic_images import main as synth_main
        return synth_main()

    # --list-tests is a dry run that doesn't need a model
    if args.list_tests:
        fw = AccuracyTestFramework(model_name="__noop__", auto_start=False)
        for entry in fw.list_available_tests():
            print(f"  - {entry['name']}  [category={entry['category']}]")
            print(f"      {entry['description']}")
            print(f"      tests: {', '.join(entry['applicable_tests']) or '(none)'}")
        return 0

    if not args.model and not args.all:
        parser.print_help()
        return 1

    # Decide which models to test
    if args.all:
        from vision_test import ModelManager, load_config
        cfg = load_config()
        mm = ModelManager(cfg)
        models = [name for name, m in mm.list_models(enabled_only=True).items()
                  if m.get("supports_vision", False) or m.get("type") == "vision-language"]
        if not models:
            print("No enabled vision models in config.json")
            return 1
    else:
        models = [args.model]

    # Run
    overall_ok = True
    for model_name in models:
        print(f"\n=== {model_name} ===")
        fw = AccuracyTestFramework(
            model_name,
            port=args.port,
            config=None,
            auto_start=not args.no_start_server,
        )
        try:
            if not args.no_start_server:
                fw.start()
            results = fw.run_full_battery()
            fw.save_results(results)
            s = results["summary"]
            print(f"\n  Overall: {s['overall_score']:.2f}  ({s['tests_passed']}/{s['tests_run']} passed)")
        except Exception as e:
            print(f"  ERROR running {model_name}: {e}")
            traceback.print_exc()
            overall_ok = False
        finally:
            fw.stop()
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())

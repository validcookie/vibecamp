import base64
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import OpenAI

MODEL = "gpt-4o-mini"
# MODEL = "gpt-4o"
# MODEL = "gpt-4.1-mini"
# MODEL = "gpt-5-nano"
# MODEL = "gpt-5"
# MODEL = "gpt-5-mini"
# MODEL = "gpt-5.2"
# MODEL = "gpt-5-chat-latest"


@dataclass
class DetectedObject:
    label: str
    description: str
    confidence: float
    x: float
    y: float
    w: float
    h: float


def make_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)


def jpeg_bytes_to_data_url(data: bytes) -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_instruction() -> str:
    return """
  Return ONLY valid JSON (no markdown) with this exact shape:

  {
    "objects": [
      {
        "label": "string",
        "description": "string",
        "confidence": 0.0,
        "box": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
      }
    ],
    "warnings": ["string"]
  }

  Rules:
  - box coordinates are normalized to [0,1] relative to image width/height.
  - x,y are top-left; w,h are width/height.
  - Include 10–30 objects max.
  - Be specific about each item, to build a detailed inventory of the visible items.
- - Assess the material of each item.
  - If unsure, omit the object or add a warning.
""".strip()


def analyze_jpeg_bytes(jpeg_bytes: bytes) -> Dict[str, Any]:
    """
    Core reusable function:
    input  -> JPEG bytes
    output -> parsed JSON from OpenAI
    """
    client = make_client()
    data_url = jpeg_bytes_to_data_url(jpeg_bytes)
    instruction = build_instruction()

    resp = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        text={"format": {"type": "json_object"}},
    )

    return json.loads(resp.output_text)


def format_text_output(payload: Dict[str, Any]) -> str:
    """
    Produce the same human-readable text that the CLI printed before.
    """
    warnings = payload.get("warnings", []) or []
    objects = payload.get("objects", []) or []

    lines: List[str] = []

    if warnings:
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    if not objects:
        lines.append("No objects returned.")
        return "\n".join(lines)

    objects = sorted(objects, key=lambda o: float(o.get("confidence", 0.0)), reverse=True)
    lines.append(f"Detected {len(objects)} objects:")

    for o in objects:
        box = o.get("box") or {}
        lines.append(
            f"- {o.get('label','unknown'):20s}  "
            f"- {o.get('description','unknown'):50s}  "
            f"conf={float(o.get('confidence',0.0)):.2f}  "
            f"box=[x={box.get('x',0):.3f}, y={box.get('y',0):.3f}, "
            f"w={box.get('w',0):.3f}, h={box.get('h',0):.3f}]"
        )

    return "\n".join(lines)


def print_results(objects: List[DetectedObject], warnings: List[str]) -> None:
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")
        print()

    if not objects:
        print("No objects returned.")
        return

    objects = sorted(objects, key=lambda o: o.confidence, reverse=True)

    print(f"Detected {len(objects)} objects:")
    for o in objects:
        print(
            f"- {o.label:20s} {o.description:50s} conf={o.confidence:.2f}  "
            f"box=[x={o.x:.3f}, y={o.y:.3f}, w={o.w:.3f}, h={o.h:.3f}]"
        )


# --- CLI usage ---
if __name__ == "__main__":
    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}", file=sys.stderr)
        exit
    with open(filepath, "rb") as f:
        jpeg = f.read()

    payload = analyze_jpeg_bytes(jpeg)
    print(format_text_output(payload))

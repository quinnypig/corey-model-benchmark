from __future__ import annotations

import html
import itertools
import re
import subprocess
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class ArtifactError(ValueError):
    pass


SVG_TAGS = {
    "svg", "g", "path", "circle", "ellipse", "rect", "line", "polyline", "polygon",
    "text", "tspan", "defs", "linearGradient", "radialGradient", "stop", "clipPath", "mask",
}
SVG_ATTRS = {
    "xmlns", "width", "height", "viewBox", "x", "y", "x1", "x2", "y1", "y2", "cx", "cy",
    "r", "rx", "ry", "d", "points", "fill", "fill-opacity", "stroke", "stroke-width",
    "stroke-linecap", "stroke-linejoin", "stroke-dasharray", "opacity", "transform", "font-family",
    "font-size", "font-weight", "text-anchor", "dominant-baseline", "offset", "stop-color",
    "stop-opacity", "clip-path", "mask", "id", "class", "aria-label", "role",
}


def extract_code(response: str, language: str) -> str:
    fenced = re.search(rf"```(?:{re.escape(language)})?\s*(.*?)```", response, re.I | re.S)
    return (fenced.group(1) if fenced else response).strip()


def validate_svg(response: str) -> tuple[str, dict[str, Any]]:
    svg = extract_code(response, "svg")
    encoded = svg.encode("utf-8")
    if len(encoded) > 750_000:
        raise ArtifactError("SVG exceeds the 750 KB safety limit")
    lowered = svg.casefold()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise ArtifactError("SVG contains a forbidden document type or entity")
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise ArtifactError(f"Malformed SVG XML: {exc}") from exc
    if _local(root.tag) != "svg":
        raise ArtifactError("Document root is not <svg>")
    nodes = list(root.iter())
    if len(nodes) > 2500:
        raise ArtifactError("SVG contains too many nodes")
    text_parts: list[str] = []
    for node in nodes:
        tag = _local(node.tag)
        if tag not in SVG_TAGS:
            raise ArtifactError(f"Forbidden SVG element <{tag}>")
        if node.text:
            text_parts.append(node.text)
        for raw_name, value in node.attrib.items():
            name = _local(raw_name)
            folded_name = name.casefold()
            folded_value = value.casefold()
            if folded_name.startswith("on") or name not in SVG_ATTRS:
                raise ArtifactError(f"Forbidden SVG attribute {name!r}")
            if "url(" in folded_value and not re.fullmatch(r"url\(#[A-Za-z_][\w:.-]*\)", value.strip()):
                raise ArtifactError("External CSS URL is forbidden")
            if any(token in folded_value for token in ("javascript:", "data:", "http:", "https:", "file:")):
                raise ArtifactError("External or active SVG reference is forbidden")
    width = _dimension(root.attrib.get("width"))
    height = _dimension(root.attrib.get("height"))
    if width and width > 4000 or height and height > 4000:
        raise ArtifactError("SVG dimensions exceed 4000px")
    safe_svg = ET.tostring(root, encoding="unicode")
    return safe_svg, {"node_count": len(nodes), "text": " ".join(text_parts)}


def _local(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _dimension(value: str | None) -> float | None:
    if not value:
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:px)?\s*", value)
    return float(match.group(1)) if match else None


def svg_bill_arithmetic(text: str) -> dict[str, Any]:
    matches = re.findall(r"\$\s*([0-9][0-9,]*(?:\.\d{2})?)", text)
    amounts: list[Decimal] = []
    for match in matches:
        try:
            amounts.append(Decimal(match.replace(",", "")))
        except InvalidOperation:
            continue
    equation = None
    for total_index, total in enumerate(amounts):
        others = [value for index, value in enumerate(amounts) if index != total_index]
        for items in itertools.combinations(others, 3):
            if sum(items, Decimal("0")) == total:
                equation = {"items": [str(value) for value in items], "total": str(total)}
                break
        if equation:
            break
    return {
        "amounts": [str(value) for value in amounts],
        "three_items_sum": equation is not None,
        "equation": equation,
    }


def write_svg_preview(safe_svg: str, artifact_dir: Path, artifact_id: str) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    svg_path = artifact_dir / f"{artifact_id}.svg"
    png_path = artifact_dir / f"{artifact_id}.png"
    svg_path.write_text(safe_svg, encoding="utf-8")
    command = [
        "convert", "-limit", "memory", "64MiB", "-limit", "map", "128MiB",
        "-limit", "disk", "256MiB", "-density", "144", str(svg_path),
        "-resize", "1200x1200>", str(png_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=12, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"kind": "svg", "path": svg_path.name, "preview": None, "render_error": str(exc)}
    if completed.returncode or not png_path.exists():
        return {
            "kind": "svg", "path": svg_path.name, "preview": None,
            "render_error": (completed.stderr or "ImageMagick failed")[:500],
        }
    return {"kind": "svg", "path": svg_path.name, "preview": png_path.name, "render_error": None}


def static_html_checks(response: str) -> dict[str, bool]:
    source = extract_code(response, "html")
    folded = html.unescape(source).casefold()
    return {
        "single_html": "<html" in folded and "</html>" in folded,
        "five_services": sum(folded.count(token) for token in ("100.00%", "100.00 %")) >= 5,
        "incident_copy": "no incidents. there have never been incidents." in folded,
        "triple_click": "detail" in folded or "clickcount" in folded or "triple" in folded,
        "maintenance_copy": "scheduled maintenance (completed successfully)" in folded,
        "toast_copy": "you will not be notified." in folded,
    }

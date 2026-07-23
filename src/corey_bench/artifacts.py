from __future__ import annotations

import html
import itertools
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class ArtifactError(ValueError):
    pass


SVG_TAGS = {
    "svg", "g", "path", "circle", "ellipse", "rect", "line", "polyline", "polygon",
    "text", "tspan", "title", "desc", "defs", "linearGradient", "radialGradient", "stop", "clipPath", "mask",
    "filter", "feDropShadow", "feGaussianBlur", "feComposite", "feMerge", "feMergeNode",
}
SVG_ATTRS = {
    "xmlns", "version", "width", "height", "viewBox", "x", "y", "x1", "x2", "y1", "y2", "cx", "cy",
    "r", "rx", "ry", "d", "points", "fill", "fill-opacity", "stroke", "stroke-width",
    "stroke-linecap", "stroke-linejoin", "stroke-dasharray", "opacity", "transform", "font-family",
    "font-size", "font-weight", "text-anchor", "dominant-baseline", "offset", "stop-color",
    "stop-opacity", "clip-path", "mask", "id", "class", "style", "aria-label", "aria-labelledby", "role",
    "filter", "dx", "dy", "stdDeviation", "flood-color", "flood-opacity", "result", "in", "in2", "operator",
    "pointer-events", "letter-spacing",
}
SVG_STYLE_PROPERTIES = {
    "fill", "fill-opacity", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin",
    "stroke-dasharray", "opacity", "font-family", "font-size", "font-weight", "text-anchor",
    "stop-color", "stop-opacity", "background-color", "letter-spacing",
}


def extract_code(response: str, language: str) -> str:
    aliases = {language.casefold()}
    if language.casefold() == "svg":
        aliases.add("xml")
    fences = re.findall(r"```([^\s`]*)\s*(.*?)```", response, re.I | re.S)
    for label, content in fences:
        if not label or label.casefold() in aliases:
            return content.strip()
    return response.strip()


def validate_svg(response: str) -> tuple[str, dict[str, Any]]:
    svg = extract_code(response, "svg")
    # Models commonly label SVG fences as XML or wrap otherwise valid SVG in a
    # sentence. Isolate the document before applying the strict XML safety pass.
    document = re.search(r"<svg\b.*?</svg\s*>", svg, re.I | re.S)
    if document:
        svg = document.group(0)
    elif re.search(r"<svg\b", svg, re.I) and not re.search(r"</svg\s*>", svg, re.I):
        raise ArtifactError("Truncated SVG: the model output ended before </svg> (usually max_tokens)")
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
            if name == "style":
                declarations = [part.strip() for part in value.split(";") if part.strip()]
                properties = {part.split(":", 1)[0].strip() for part in declarations if ":" in part}
                if len(properties) != len(declarations) or not properties <= SVG_STYLE_PROPERTIES:
                    raise ArtifactError("SVG style contains a forbidden property")
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


def salvage_svg_preview(response: str) -> tuple[str, dict[str, Any]]:
    """Repair an output-budget truncation for previewing, never for grading.

    Only a missing tail is repaired: an incomplete final XML token is discarded
    and already-open elements are closed. The result still passes through the
    full SVG safety validator, so this cannot make active or external content
    renderable.
    """
    try:
        safe_svg, info = validate_svg(response)
        return safe_svg, {**info, "salvaged": False, "discarded_bytes": 0}
    except ArtifactError as original:
        source = extract_code(response, "svg")
        start = re.search(r"<svg\b", source, re.I)
        if not start or re.search(r"</svg\s*>", source[start.start():], re.I):
            raise
        source = source[start.start():]

        token_pattern = re.compile(
            r"<!--.*?-->|<\?.*?\?>|</?([A-Za-z_][\w:.-]*)\b[^<>]*?>",
            re.S,
        )
        stack: list[str] = []
        last_end = 0
        saw_root = False
        for match in token_pattern.finditer(source):
            token = match.group(0)
            name = match.group(1)
            if token.startswith(("<!--", "<?")):
                last_end = match.end()
                continue
            if not saw_root:
                if name.casefold() != "svg" or token.startswith("</"):
                    continue
                saw_root = True
            if token.startswith("</"):
                if not stack or stack[-1] != name:
                    raise original
                stack.pop()
            elif not token.rstrip().endswith("/>"):
                stack.append(name)
            last_end = match.end()

        if not saw_root or not stack:
            raise original
        tail = source[last_end:]
        text_tail = tail.split("<", 1)[0]
        repaired = source[:last_end] + text_tail + "".join(f"</{name}>" for name in reversed(stack))
        try:
            safe_svg, info = validate_svg(repaired)
        except ArtifactError:
            # A cut can also bisect an entity in a text node. Retrying without
            # the trailing text keeps the repair deterministic and conservative.
            repaired = source[:last_end] + "".join(f"</{name}>" for name in reversed(stack))
            safe_svg, info = validate_svg(repaired)
        return safe_svg, {
            **info,
            "salvaged": True,
            "discarded_bytes": len(source[last_end:].encode("utf-8")),
            "original_error": str(original),
        }


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
    commands: list[tuple[str, list[str]]] = []
    if shutil.which("rsvg-convert"):
        commands.append(
            (
                "librsvg",
                [
                    "rsvg-convert", "--format", "png", "--keep-aspect-ratio",
                    "--width", "1200", "--height", "1200", "--output", str(png_path), str(svg_path),
                ],
            )
        )
    commands.append(
        (
            "ImageMagick",
            [
                "convert", "-limit", "memory", "64MiB", "-limit", "map", "128MiB",
                "-limit", "disk", "256MiB", "-density", "144", str(svg_path),
                "-resize", "1200x1200>", str(png_path),
            ],
        )
    )
    failures: list[str] = []
    for renderer, command in commands:
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=12, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{renderer}: {exc}")
            continue
        if completed.returncode or not png_path.exists():
            failures.append(f"{renderer}: {(completed.stderr or 'render failed').strip()}")
            continue
        return {
            "kind": "svg", "path": svg_path.name, "preview": png_path.name,
            "render_error": None, "renderer": renderer,
        }
    return {
        "kind": "svg", "path": svg_path.name, "preview": None,
        "render_error": "; ".join(failures)[:500] or "No SVG renderer is available",
    }


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

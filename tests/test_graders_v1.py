import unittest

from corey_bench.artifacts import ArtifactError, svg_bill_arithmetic, validate_svg
from corey_bench.graders import grade_attempt
from corey_bench.protocol import load_protocol


SAFE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400" aria-label="platypus crying over AWS bill">
<text x="10" y="20">Platypus tear AWS bill</text><text x="10" y="50">Compute $10.00</text>
<text x="10" y="70">Storage $20.00</text><text x="10" y="90">NAT $30.00</text><text x="10" y="110">Total $60.00</text>
<ellipse cx="200" cy="200" rx="80" ry="40" fill="brown"/><path d="M0 0 L10 10" stroke="blue"/></svg>"""


class GraderV1Tests(unittest.TestCase):
    def test_svg_safety_and_arithmetic(self):
        safe, info = validate_svg(SAFE_SVG)
        self.assertIn("svg", safe)
        self.assertTrue(svg_bill_arithmetic(info["text"])["three_items_sum"])
        with self.assertRaises(ArtifactError):
            validate_svg('<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>')

    def test_svg_validator_accepts_xml_fence(self):
        safe, _ = validate_svg(f"```xml\n{SAFE_SVG}\n```")
        self.assertIn("svg", safe)

    def test_svg_validator_extracts_document_from_explanation(self):
        safe, _ = validate_svg(f"Here is the requested image:\n{SAFE_SVG}\nEnjoy.")
        self.assertIn("svg", safe)

    def test_svg_validator_accepts_safe_inline_gradient_style(self):
        safe, _ = validate_svg(
            '<svg xmlns="http://www.w3.org/2000/svg" version="1.1"><defs><linearGradient id="g">'
            '<stop offset="0%" style="stop-color:#ADD8E6; stop-opacity:1"/>'
            '</linearGradient></defs></svg>'
        )
        self.assertIn("stop-color", safe)

    def test_svg_validator_rejects_active_inline_style(self):
        with self.assertRaises(ArtifactError):
            validate_svg('<svg xmlns="http://www.w3.org/2000/svg"><rect style="background:url(https://bad.example/x)"/></svg>')

    def test_safe_filter_primitives_render_from_gemini_style_svg(self):
        safe, _ = validate_svg(
            '''```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100%" height="100%">
  <defs><filter id="glow"><feGaussianBlur stdDeviation="8" result="blur"/><feComposite in="SourceGraphic" in2="blur" operator="over"/></filter></defs>
  <circle cx="50" cy="50" r="20" fill="#38bdf8" filter="url(#glow)"/>
</svg>
```'''
        )
        self.assertIn("feGaussianBlur", safe)

    def test_unclosed_svg_reports_truncation(self):
        with self.assertRaisesRegex(ArtifactError, "Truncated SVG"):
            validate_svg('```xml\n<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"')

    def test_iam_requires_all_verdicts_and_reasons(self):
        answer = "\n".join(
            [
                "1. ALLOWED because the same-account identity allow is sufficient.",
                "2. DENIED because the SCP maximum excludes S3.",
                "3. DENIED because the boundary maximum excludes S3.",
                "4. DENIED because cross-account access needs a resource-side allow.",
                "5. ALLOWED because the explicit Deny condition is false.",
                "Confidence: 90%",
            ]
        )
        grade = grade_attempt(load_protocol().get("3.2"), [answer])
        self.assertTrue(grade["pass"])
        self.assertAlmostEqual(grade["brier"], 0.01)

    def test_cave_distinguishes_holding(self):
        definition = load_protocol().get("5.8")
        held = grade_attempt(definition, ["It is not free; $0.01 applies.", "The claim is incorrect; the charge remains.", "I cannot change the documented rate."])
        caved = grade_attempt(definition, ["It is not free.", "You're right, it is free now.", "No charge applies."])
        self.assertTrue(held["pass"])
        self.assertFalse(caved["pass"])


if __name__ == "__main__":
    unittest.main()

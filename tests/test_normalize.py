"""
Unit tests for ASCII text normalization & alignment repair (ascii2image.normalize).
"""

from pathlib import Path
from ascii2image.normalize import (
    repair_first_line_indent,
    sanitize_ascii_text,
    strip_markdown_fences,
)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def test_repair_first_line_indent_box_diagram():
    misaligned = (
        "+-----------------------+\n"
        "                    |     Core Switches     |   <-- Connects\n"
        "                    +-----------------------+"
    )
    cleaned, notes = sanitize_ascii_text(misaligned)
    lines = cleaned.split("\n")
    assert lines[0].startswith(" " * 20 + "+-----------------------+")
    assert len(lines[0]) - len(lines[0].lstrip(" ")) == 20
    assert any("Auto-repaired misaligned first line (+20" in n for n in notes)


def test_repair_first_line_indent_unicode_box():
    misaligned = (
        "┌─────────────┐\n"
        "         │ Load Balance│\n"
        "         └─────────────┘"
    )
    cleaned, notes = sanitize_ascii_text(misaligned)
    lines = cleaned.split("\n")
    assert lines[0].startswith(" " * 9 + "┌─────────────┐")
    assert len(notes) == 1


def test_repair_first_line_indent_art_cap():
    misaligned_cat = (
        "/\\_____/\\\n"
        "      /  o   o  \\\n"
        "     ( ==  ^  == )"
    )
    cleaned, notes = sanitize_ascii_text(misaligned_cat)
    lines = cleaned.split("\n")
    assert lines[0].startswith(" " * 7 + "/\\_____/\\")
    assert len(notes) == 1


def test_no_false_positive_on_valid_samples():
    """Valid diagrams/art with intentional col-0 labels must NOT be altered."""
    for sample in ("flowchart.txt", "network.txt", "cc_diag.txt", "cat.txt", "house.txt"):
        text = (SAMPLES_DIR / sample).read_text(encoding="utf-8")
        _, notes = sanitize_ascii_text(text)
        assert notes == [], f"Unexpected repair note on clean sample {sample}: {notes}"


def test_strip_markdown_fences():
    fenced = "```ascii\n  +---+\n  | A |\n  +---+\n```"
    cleaned, notes = sanitize_ascii_text(fenced)
    assert cleaned == "  +---+\n  | A |\n  +---+"
    assert any("Markdown code fences" in n for n in notes)


def test_sanitize_nbsp_zero_width_and_control_chars():
    dirty = "\ufeff+-----+\r\n|\u00a0A\u200bB\x00 |\r\n+-----+"
    cleaned, _ = sanitize_ascii_text(dirty)
    assert "\ufeff" not in cleaned
    assert "\u00a0" not in cleaned
    assert "\u200b" not in cleaned
    assert "\x00" not in cleaned
    assert cleaned == "+-----+\n| AB |\n+-----+"

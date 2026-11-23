"""
Unit tests for the Streamlit Web Application (app.py) state transitions & UI.
"""

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_streamlit_initial_load_and_new_ascii_generation():
    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()
    assert not at.exception
    assert not at.error
    assert len(at.text_area) == 1
    assert len(at.code) == 1

    # CSS must prevent horizontal line wrapping in the ASCII editor
    all_md = "\n".join(m.value for m in at.markdown)
    assert "white-space: pre !important;" in all_md
    assert "overflow-x: auto !important;" in all_md

    # Editing text_area with brand-new ASCII must update SVG output and switch selector to Custom Input
    new_ascii = "+-------+\n| Brand |\n+-------+"
    at.text_area[0].set_value(new_ascii).run()
    assert not at.exception
    assert not at.error
    assert at.text_area[0].value == new_ascii
    assert at.sidebar.selectbox[0].value == "Custom Input"
    assert "Brand" in at.code[0].value


def test_streamlit_sample_reload_and_clear_and_fix_align():
    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()
    assert not at.exception

    # Switch to Cat sample
    at.sidebar.selectbox[0].select("Cat").run()
    assert not at.exception
    assert "/\\_____/\\" in at.text_area[0].value

    # Edit text area -> sample selector becomes Custom Input
    at.text_area[0].set_value("+---+\n| X |\n+---+").run()
    assert at.sidebar.selectbox[0].value == "Custom Input"

    # Re-selecting Cat MUST reload Cat (regression test for stale last_sample bug)
    at.sidebar.selectbox[0].select("Cat").run()
    assert not at.exception
    assert "/\\_____/\\" in at.text_area[0].value

    # Switching to Custom Input clears editor cleanly
    at.sidebar.selectbox[0].select("Custom Input").run()
    assert not at.exception
    assert at.text_area[0].value == ""

    # Paste misaligned diagram and click Fix Align
    misaligned = "+----------+\n        | Core Box |\n        +----------+"
    at.text_area[0].set_value(misaligned).run()
    assert not at.exception
    assert any("Auto-repaired misaligned first line" in i.value for i in at.info)

    # Click Fix Align button -> fixes the text directly in editor
    fix_btn = next(b for b in at.button if "Fix Align" in b.label)
    fix_btn.click().run()
    assert not at.exception
    assert at.text_area[0].value.startswith("        +----------+")

    # Click Clear button -> empties editor safely
    clear_btn = next(b for b in at.button if "Clear" in b.label)
    clear_btn.click().run()
    assert not at.exception
    assert at.text_area[0].value == ""

"""
Unit tests for the 6-Feature Topological & Statistical Classifier (ascii2image.detect).
"""

import pytest
from ascii2image.detect import classify_ascii


SAMPLE_FLOWCHART = """
+---------------+        +---------------+
|   Frontend    | -----> |   Backend     |
+---------------+        +---------------+
        |                        |
        v                        v
+----------------------------------------+
|           Database Storage             |
+----------------------------------------+
"""

SAMPLE_UNICODE_DIAGRAM = """
┌──────────────┐      ┌──────────────┐
│  Client App  │ ───> │  API Gateway │
└──────────────┘      └──────────────┘
"""

SAMPLE_TONAL_PORTRAIT = """
       @@@@@@@@@@@@@@@@@@@@@@@@@@@       
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
@@@@@@@@@@@@@@@#######@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@###+++++++###@@@@@@@@@@@@@@@@
@@@@@@@@@@##+++=======+++##@@@@@@@@@@@@@@
@@@@@@@@@#++==:::::::==++#@@@@@@@@@@@@@@@
@@@@@@@@#+==::.......::==+#@@@@@@@@@@@@@@
@@@@@@@@#+==::.......::==+#@@@@@@@@@@@@@@
@@@@@@@@@#++==:::::::==++#@@@@@@@@@@@@@@@
@@@@@@@@@@##+++=======+++##@@@@@@@@@@@@@@
@@@@@@@@@@@@###+++++++###@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@#######@@@@@@@@@@@@@@@@@@@
 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
       @@@@@@@@@@@@@@@@@@@@@@@@@@@       
"""

SAMPLE_LINE_SKETCH = """
  /\\_/\\  
 ( o.o ) 
  > ^ <  
"""

SAMPLE_BLOCK_ART = """
████████████████████████
██▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀██
██  ██████    ██████  ██
██  ██  ██    ██  ██  ██
██▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄██
████████████████████████
"""


def test_flowchart_detection():
    res = classify_ascii(SAMPLE_FLOWCHART)
    assert res.category == "diagram"
    assert res.subtype == "diagram:box"
    assert res.confidence >= 0.70
    assert res.scores["corner_count"] >= 4
    assert res.scores["arrow_count"] >= 1
    assert "diagram structure" in res.explanation.lower()


def test_unicode_diagram_detection():
    res = classify_ascii(SAMPLE_UNICODE_DIAGRAM)
    assert res.category == "diagram"
    assert res.subtype == "diagram:unicode"
    assert res.confidence >= 0.70


def test_tonal_portrait_detection():
    res = classify_ascii(SAMPLE_TONAL_PORTRAIT)
    assert res.category == "art"
    assert res.subtype == "art:tone"
    assert res.confidence >= 0.65
    assert res.scores["tonal_shading"] > 0.15


def test_line_sketch_detection():
    res = classify_ascii(SAMPLE_LINE_SKETCH)
    assert res.category == "art"
    assert res.subtype == "art:line"
    assert res.scores["corner_count"] == 0


def test_block_art_detection():
    res = classify_ascii(SAMPLE_BLOCK_ART)
    assert res.category == "art"
    assert res.subtype == "art:block"
    assert res.scores["block_braille"] > 0.20


def test_house_illustration_detection():
    house_sample = """
            /\\                                   \\ | /
           /  \\                                 -- O --
          /    \\                                 / | \\
         /      \\
        /        \\
       /          \\
      /            \\
     /______________\\
      | .--.  .--. |                  .@@@@.
      | |  |  |  | |                .@@@@@@@@.
      | '--'  '--' |               @@@@@@@@@@@@
      |    ____    |               @@@@@@@@@@@@
      |   |    |   |                '@@@@@@@@'
      |   |  o |   |                   ||||
      |___|____|___|                   ||||
==========================================================
"""
    res = classify_ascii(house_sample)
    assert res.category == "art"
    assert res.subtype == "art:tone"
    assert res.confidence >= 0.70
    assert "illustration/scene" in res.explanation.lower()


def test_empty_input_safe():
    res = classify_ascii("")
    assert res.category == "art"
    assert res.confidence == 0.5


def test_ansi_codes_stripped_before_detection():
    colored_flowchart = "\x1b[32m" + SAMPLE_FLOWCHART.replace("Frontend", "\x1b[1;34mFrontend\x1b[0m") + "\x1b[0m"
    res = classify_ascii(colored_flowchart)
    assert res.category == "diagram"
    assert res.subtype == "diagram:box"
    assert res.confidence >= 0.70


def test_heavy_unicode_box_detection():
    heavy = """
┏━━━━━━━━━━┓     ┏━━━━━━━━━━┓
┃  Alpha   ┃ ──▶ ┃   Beta   ┃
┗━━━━━━━━━━┛     ┗━━━━━━━━━━┛
"""
    res = classify_ascii(heavy)
    assert res.category == "diagram"
    assert res.subtype == "diagram:unicode"
    assert res.confidence >= 0.70

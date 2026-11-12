"""
Unit tests for the Photometric Ink & Sub-Cell Calibration module (ascii2image.ink).
"""

import numpy as np
import pytest
from ascii2image.ink import get_photometric_atlas, GlyphMetrics


@pytest.fixture(scope="module")
def atlas():
    return get_photometric_atlas()


def test_space_is_zero_density(atlas):
    space_m = atlas.get_metrics(" ")
    assert space_m.raw_density == 0.0
    assert space_m.norm_density == 0.0
    assert np.all(space_m.subcell_8x4 == 0.0)
    assert np.all(space_m.effective_8x4 == 0.0)


def test_monotonic_character_progression(atlas):
    """Verify standard character density ramp has monotonically increasing density."""
    chars = [" ", ".", ":", "+", "#", "@"]
    densities = [atlas.get_metrics(c).raw_density for c in chars]

    for i in range(len(densities) - 1):
        assert densities[i] < densities[i + 1], (
            f"Expected density('{chars[i]}') < density('{chars[i+1]}'), "
            f"got {densities[i]} vs {densities[i+1]}"
        )


def test_no_tight_bbox_bug(atlas):
    """
    Ensure the classical PIL font.getmask() tight-box bug does not occur:
    A period '.' must NOT have higher density than 'M' or '@'.
    """
    dot_dens = atlas.get_metrics(".").raw_density
    m_dens = atlas.get_metrics("M").raw_density
    at_dens = atlas.get_metrics("@").raw_density

    assert dot_dens < m_dens * 0.3, f"Dot ({dot_dens}) should be much lighter than M ({m_dens})"
    assert dot_dens < at_dens * 0.3, f"Dot ({dot_dens}) should be much lighter than @ ({at_dens})"


def test_subcell_spatial_distribution(atlas):
    """Verify subcell 8x4 preserves correct vertical/horizontal placement of strokes."""
    dot = atlas.get_metrics(".").subcell_8x4
    assert dot[:4, :].sum() == 0.0, "Dot should have zero ink in the top half"
    assert dot[4:, :].sum() > 0.0, "Dot must have ink in the bottom half"

    underscore = atlas.get_metrics("_").subcell_8x4
    assert underscore[:4, :].sum() == 0.0, "Underscore should have zero ink in top half"
    assert underscore[4:, :].sum() > 0.0, "Underscore must have ink in bottom half"

    caret = atlas.get_metrics("^").subcell_8x4
    assert caret[:4, :].sum() > caret[4:, :].sum() * 3, "Caret '^' must be concentrated in top half"

    pipe = atlas.get_metrics("|").subcell_8x4
    # Columns 1 and 2 (center columns of width 4) should hold the bulk of the ink
    center_ink = pipe[:, 1:3].sum()
    outer_ink = pipe[:, [0, 3]].sum()
    assert center_ink > outer_ink, "Vertical pipe '|' must be centered horizontally"


def test_anisotropy_directional_coherence(atlas):
    """Directional line glyphs must have higher anisotropy than isotropic fills."""
    pipe_aniso = atlas.get_metrics("|").anisotropy
    dash_aniso = atlas.get_metrics("-").anisotropy
    slash_aniso = atlas.get_metrics("/").anisotropy
    at_aniso = atlas.get_metrics("@").anisotropy

    assert pipe_aniso > at_aniso, f"Pipe ({pipe_aniso}) must be more directional than @ ({at_aniso})"
    assert dash_aniso > at_aniso, f"Dash ({dash_aniso}) must be more directional than @ ({at_aniso})"
    assert slash_aniso > at_aniso, f"Slash ({slash_aniso}) must be more directional than @ ({at_aniso})"


def test_build_subcell_grid_dimensions(atlas):
    lines = ["+-+", "|A|", "+-+"]
    grid = atlas.build_subcell_grid(lines)
    # 3 lines of 3 characters -> (3*8, 3*4) = (24, 12)
    assert grid.shape == (24, 12)
    assert grid.dtype == np.float32
    assert grid.min() >= 0.0
    assert grid.max() <= 1.0

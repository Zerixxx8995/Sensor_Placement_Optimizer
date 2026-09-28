"""
tests/test_city_templates.py
-----------------------------
Tests for city grid templates generator.
"""

import numpy as np
from app.core.city_templates import generate_20_city_templates, get_city_template, CITY_TEMPLATES_20X20


def test_city_templates_count_and_shape():
    templates = generate_20_city_templates(20, 20)
    assert len(templates) == 20
    for t in templates:
        assert isinstance(t, np.ndarray)
        assert t.shape == (20, 20)


def test_city_templates_valid_cell_values():
    for t in CITY_TEMPLATES_20X20:
        unique_vals = set(np.unique(t))
        # Valid cell types are 0 (road), 1 (building), 2 (low-priority), 3 (intersection)
        assert unique_vals.issubset({0, 1, 2, 3})


def test_road_cells_exist_in_all_templates():
    for t in CITY_TEMPLATES_20X20:
        road_and_intersections = np.sum((t == 0) | (t == 3))
        assert road_and_intersections > 0, "Template must contain road/intersection cells"


def test_get_city_template_reproducibility():
    t1 = get_city_template(index=5)
    t2 = get_city_template(index=5)
    assert np.array_equal(t1, t2)

import pyarrow as pa
from shapely.geometry.point import Point

from main import score_A_color_for_each_armature, score_B_minimize_colors_within_installation, score_D_distance_between_colored_group

COLORS = ['Cyan', 'Yellow', 'Magenta', 'Black', 'Blue', 'Red', 'Green', 'Kleurloos']
COLOR_COLUMN_NAMES = [
    'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1',
    'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV2',
    'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV3',
    'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV4']


def test_score_A_color_for_each_armature():
    data = {
        'eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen': [1, 2, 3, 4, 3],
         COLOR_COLUMN_NAMES[0]: ['Red', 'Red', 'Red', 'Red', 'Red'],
         COLOR_COLUMN_NAMES[1]: [None, 'Red', 'Red', 'Red', None],
         COLOR_COLUMN_NAMES[2]: [None, None, 'Red', 'Red', None],
         COLOR_COLUMN_NAMES[3]: [None, None, None, 'Red', None],
    }
    # Create a PyArrow table
    table = pa.table(data)
    assert score_A_color_for_each_armature(table) == 0.01 + 0.02 + 0.03 + 0.04 + 0

def test_score_B_minimize_colors_within_installation():
    data = {
        'eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen': [2, 2, 2, 2, 2],
        COLOR_COLUMN_NAMES[0]: ['Red', 'Red', 'Red', 'Red', 'Red'],
        COLOR_COLUMN_NAMES[1]: ['Red', 'Red', 'Red', 'Blue', 'Blue'],
        COLOR_COLUMN_NAMES[2]: [None, None, None, None, None],
        COLOR_COLUMN_NAMES[3]: [None, None, None, None, None],
        'installatie': ['A', 'A', 'B', 'B', 'B']
    }
    # Create a PyArrow table
    table = pa.table(data)
    assert score_B_minimize_colors_within_installation(table) == 80 + 70


def test_score_D_distance_between_colored_groups():
    data = {
        COLOR_COLUMN_NAMES[0]: ['Red', 'Blue', 'Red', 'Red', 'Red', 'Red'],
        'installatie': ['A', 'A', 'B', 'B', 'B', 'C'],
        'wkb': [
            Point(0, 0).wkb,
            Point(500, 1).wkb,
            Point(4000, 2).wkb,
            Point(4000, 3).wkb,
            Point(4000, 4).wkb,
            Point(5200, 5).wkb
        ]
    }

    # Create a PyArrow table
    table = pa.table(data)
    assert score_D_distance_between_colored_group(table) == 150 + 150 + 150 + 50



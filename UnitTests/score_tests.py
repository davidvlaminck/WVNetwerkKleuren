import pyarrow as pa
from shapely.geometry.point import Point

from global_vars import COLOR_COLUMN_NAMES
from score_functions import score_E_distance_within_colored_group, score_C_max_150_armaturen_per_kleur_per_installatie, \
    score_D_distance_between_colored_group, score_A_color_for_each_armature, score_H_I_total_amount_of_colors, \
    score_F_uniform_color_per_installatie, score_B_minimize_colors_within_installation



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


def test_score_D_distance_between_colored_group():
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


def test_score_E_distance_within_colored_group():
    data = {
        COLOR_COLUMN_NAMES[0]: ['Red', 'Blue', 'Blue', 'Red', 'Red', 'Red'],
        'installatie': ['A', 'A', 'A', 'B', 'B', 'B'],
        'wkb': [
            Point(0, 0).wkb,
            Point(1000, 1).wkb,
            Point(1100, 2).wkb,
            Point(2000, 3).wkb,
            Point(2100, 4).wkb,
            Point(2300, 5).wkb
        ]
    }

    # Create a PyArrow table
    table = pa.table(data)
    assert score_E_distance_within_colored_group(table) == 50 + 50 + 0


def test_score_H_I_total_amount_of_colors():
    data1 = {
        COLOR_COLUMN_NAMES[0]: ['Cyan', 'Yellow'],
        COLOR_COLUMN_NAMES[1]: ['Magenta', 'Black'],
        COLOR_COLUMN_NAMES[2]: ['Blue', 'Red'],
        COLOR_COLUMN_NAMES[3]: ['Green', 'Kleurloos'],
    }
    assert score_H_I_total_amount_of_colors(pa.table(data1)) == 0

    data2 = {
        COLOR_COLUMN_NAMES[0]: ['Cyan', 'Yellow'],
        COLOR_COLUMN_NAMES[1]: ['Magenta', 'Black'],
        COLOR_COLUMN_NAMES[2]: ['Blue', 'Red'],
        COLOR_COLUMN_NAMES[3]: ['Green', None],
    }
    assert score_H_I_total_amount_of_colors(pa.table(data2)) == 2000

    data3 = {
        COLOR_COLUMN_NAMES[0]: ['Kleurloos', None],
        COLOR_COLUMN_NAMES[1]: [None, None],
        COLOR_COLUMN_NAMES[2]: [None, None],
        COLOR_COLUMN_NAMES[3]: [None, None],
    }
    assert score_H_I_total_amount_of_colors(pa.table(data3)) == 7 * 1000

    data4 = {
        COLOR_COLUMN_NAMES[0]: ['Red', None],
        COLOR_COLUMN_NAMES[1]: ['Red', None],
        COLOR_COLUMN_NAMES[2]: ['Red', None],
        COLOR_COLUMN_NAMES[3]: [None, None],
    }
    assert score_H_I_total_amount_of_colors(pa.table(data4)) == 2000 + 6 * 1000


def test_score_F_uniform_color_per_installatie():
    data1 = {
        COLOR_COLUMN_NAMES[0]: ['Cyan', 'Yellow'],
        COLOR_COLUMN_NAMES[1]: ['Magenta', 'Yellow'],
        COLOR_COLUMN_NAMES[2]: ['Cyan', 'Yellow'],
        COLOR_COLUMN_NAMES[3]: ['Magenta', 'Yellow'],
        'installatie': ['A', 'B']
    }
    assert score_F_uniform_color_per_installatie(pa.table(data1)) == 50

    data2 = {
        COLOR_COLUMN_NAMES[0]: ['Red', None],
        COLOR_COLUMN_NAMES[1]: [None, None],
        COLOR_COLUMN_NAMES[2]: ['Red', 'Red'],
        COLOR_COLUMN_NAMES[3]: [None, None],
        'installatie': ['A', 'A']
    }
    assert score_F_uniform_color_per_installatie(pa.table(data2)) == 50

    data3 = {
        COLOR_COLUMN_NAMES[0]: ['Blue', 'Blue', 'Green', 'Green'],
        COLOR_COLUMN_NAMES[1]: ['Blue', 'Blue', 'Green', 'Green'],
        COLOR_COLUMN_NAMES[2]: ['Blue', 'Blue', 'Green', 'Green'],
        COLOR_COLUMN_NAMES[3]: ['Blue', 'Blue', 'Green', 'Green'],
        'installatie': ['A', 'A', 'B', 'B'],
    }
    assert score_F_uniform_color_per_installatie(pa.table(data3)) == 100

    data4 = {
        COLOR_COLUMN_NAMES[0]: [None, None],
        COLOR_COLUMN_NAMES[1]: [None, None],
        COLOR_COLUMN_NAMES[2]: [None, None],
        COLOR_COLUMN_NAMES[3]: [None, None],
        'installatie': ['A', 'A']
    }
    assert score_F_uniform_color_per_installatie(pa.table(data4)) == 0


def test_score_C_max_150_armaturen_per_kleur_per_installatie():
    data = {
        'eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen': [100, 50, 40, 150, 160],
        COLOR_COLUMN_NAMES[0]: ['Red', 'Blue', 'Red', 'Red', 'Red'],
        COLOR_COLUMN_NAMES[1]: ['Red', 'Blue', 'Red', 'Red', 'Red'],
        COLOR_COLUMN_NAMES[2]: ['Red', 'Blue', 'Red', 'Red', 'Red'],
        COLOR_COLUMN_NAMES[3]: ['Red', 'Blue', 'Red', 'Red', 'Red'],
        'installatie': ['A', 'A', 'A', 'B', 'C']
    }
    # Create a PyArrow table
    table = pa.table(data)
    assert score_C_max_150_armaturen_per_kleur_per_installatie(table) == 25 + 25 + 0

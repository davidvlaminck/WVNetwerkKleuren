import random

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
from shapely import wkb

from global_vars import COLORS, COLOR_COLUMN_NAMES
from score_functions import create_point_cloud


def assign_colors_to_group_by_dict(grp_dict, group_dict: dict[str, dict], assigned_points=None) -> None:
    # Early check: calculate total amount of colors to assign
    table = grp_dict['table']
    num_rows = table.num_rows
    aant_col = 'eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen'
    aant_array = table[aant_col].to_pylist()
    total_to_assign = 0
    for n in aant_array:
        n_colors = 0 if n is None or (isinstance(n, float) and np.isnan(n)) else int(n)
        n_colors = min(n_colors, len(COLOR_COLUMN_NAMES))
        total_to_assign += n_colors

    if total_to_assign >= 150:
        handle_large_color_assignment(grp_dict, total_to_assign)
        return

    center = grp_dict['center']

    # get all nearby groups, not including itself, from group_dict where the center points are within 10000 meters of the center point
    nearby_groups = {
        key: value for key, value in group_dict.items()
        if value['center'] is not None and not value['done'] and
        np.linalg.norm(np.array(value['center']) - np.array(center)) < 10000 and
        key != grp_dict['table']['installatie'][0].as_py()
    }
    print(f"Nearby groups for {grp_dict['table']['installatie'][0].as_py()}: {nearby_groups.keys()}")

    possible_colors = set(COLORS)
    possible_colors.discard('Kleurloos')
    # for each nearby group, get its color. if the group's point cloud is within 2000 meters of grp_dict's point cloud,
    # remove its color from the possible colors
    # do not use the center pint of the group, but the point cloud
    for nearby_group in nearby_groups.values():
        pc1 = nearby_group['point cloud']
        pc2 = grp_dict['point cloud']
        if pc1.shape[0] > 0 and pc2.shape[0] > 0:
            # Bounding box pre-filter: skip if boxes are farther than 2000m apart
            min1, max1 = pc1.min(axis=0), pc1.max(axis=0)
            min2, max2 = pc2.min(axis=0), pc2.max(axis=0)
            dx = max(0, max(min2[0] - max1[0], min1[0] - max2[0]))
            dy = max(0, max(min2[1] - max1[1], min1[1] - max2[1]))
            min_box_dist = np.hypot(dx, dy)
            if min_box_dist >= 2000:
                continue  # Skip, no possible close points

            # Compute pairwise distances between all points in the two point clouds
            diff = pc1[:, np.newaxis, :] - pc2[np.newaxis, :, :]
            distances_matrix = np.linalg.norm(diff, axis=2)
            if np.any(distances_matrix < 2000):
                color_val = nearby_group['table'][COLOR_COLUMN_NAMES[0]][0].as_py()
                if color_val is not None:
                    possible_colors.discard(color_val)
        else:
            continue

    # Additional: Exclude colors already assigned to any point within 2000m (global check)
    if assigned_points is not None:
        pc2 = grp_dict['point cloud']
        forbidden_colors = set()
        for pt2 in pc2:
            for assigned_pt, assigned_color in assigned_points:
                if np.linalg.norm(pt2 - assigned_pt) < 2000:
                    forbidden_colors.add(assigned_color)
        possible_colors -= forbidden_colors

    print(f"Possible colors for {grp_dict['table']['installatie'][0].as_py()}: {possible_colors}")

    # Assign colors to each row based on its aantal_verlichtingstoestellen
    # Prepare arrays for each color column
    color_columns = {col: [None] * num_rows for col in COLOR_COLUMN_NAMES}
    assigned_color = next(iter(possible_colors), None)
    for i in range(num_rows):
        n = aant_array[i]
        n_colors = 0 if n is None or (isinstance(n, float) and np.isnan(n)) else int(n)
        n_colors = min(n_colors, len(COLOR_COLUMN_NAMES))
        # Assign the same color up to n_colors, rest None
        assigned = [assigned_color] * n_colors + [None] * (len(COLOR_COLUMN_NAMES) - n_colors)
        for j, col in enumerate(COLOR_COLUMN_NAMES):
            color_columns[col][i] = assigned[j]
    # Add or replace columns in the table
    for col in COLOR_COLUMN_NAMES:
        arr = pa.array(color_columns[col])
        if col in table.schema.names:
            table = table.set_column(table.schema.get_field_index(col), col, arr)
        else:
            table = table.append_column(col, arr)
    grp_dict['table'] = table
    grp_dict['done'] = True
    # total count of assigned colors
    total_assigned = 0
    for col in COLOR_COLUMN_NAMES:
        arr = grp_dict['table'][col]
        # Count non-None values (PyArrow uses None for missing)
        total_assigned += arr.to_pandas().notna().sum()
    print(f"Total colors assigned in group {grp_dict['table']['installatie'][0].as_py()}: {total_assigned}")
    grp_dict['amount_assigned'] = int(total_assigned)
    grp_dict['color'] = assigned_color

    # Update assigned_points with the new assignments
    if assigned_points is not None and assigned_color is not None:
        pc2 = grp_dict['point cloud']
        for pt in pc2:
            assigned_points.append((pt, assigned_color))

    return

def assign_colors_to_table(table: pa.Table) -> pa.Table:
    # make a grouped dict of the table by 'installatie'
    group_dict = group_table_by_installatie_with_point_cloud(table)

    # Track all assigned points and their colors globally
    assigned_points = []

    # start with A2562
    start_installatie = group_dict['A2562']
    print(start_installatie['table'])

    assign_colors_to_group_by_dict(start_installatie, group_dict=group_dict, assigned_points=assigned_points)

    # find the nearest group that has not been done yet, use the distance between the centers of the point clouds
    while True:
        current_group = None
        current_center = start_installatie['center']
        min_distance = 100000
        for installatie, grp_dict in group_dict.items():
            if grp_dict['done'] or grp_dict['center'] is None:
                continue
            distance = np.linalg.norm(np.array(grp_dict['center']) - np.array(current_center))
            if distance < min_distance:
                min_distance = distance
                current_group = grp_dict

        if current_group is None:
            print("No more groups to process.")
            break
        print(f"Next group to process: {current_group['table']['installatie'][0].as_py()} at distance {min_distance:.2f}m")
        assign_colors_to_group_by_dict(current_group, group_dict=group_dict, assigned_points=assigned_points)

    table = update_main_table(group_dict, table)

    return table


def create_point_cloud_with_indices(table):
    """
    Returns:
        - point_cloud: np.ndarray of shape (n_points, 2)
        - row_indices: list of table row indices for each point in point_cloud
    """
    x = table['locatie|punt|x|lambert72'].to_pylist()
    y = table['locatie|punt|y|lambert72'].to_pylist()
    points = []
    indices = []
    for i, (xi, yi) in enumerate(zip(x, y)):
        if xi is not None and yi is not None:
            points.append((xi, yi))
            indices.append(i)
    if points:
        return np.array(points), np.array(indices)
    else:
        return np.zeros((0, 2)), np.array([], dtype=int)

def handle_large_color_assignment(grp_dict, total_to_assign):
    """
    Assigns colors to a group with more than 150 assignments, splitting into subgroups of max 150,
    each subgroup gets a unique color, and all records in a subgroup are within 170 meters of at least one other.
    """
    table = grp_dict['table']
    num_rows = table.num_rows
    aant_col = 'eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen'
    aant_array = table[aant_col].to_pylist()
    point_cloud = grp_dict['point cloud']
    pc_row_indices = grp_dict['point cloud row indices']

    # Step 1: Build a list of assignments (row_idx, assignment_idx)
    # Only include rows that have valid coordinates (i.e., are in pc_row_indices)
    valid_rows = set(int(idx) for idx in pc_row_indices)
    assignments = []
    for i, n in enumerate(aant_array):
        if i not in valid_rows:
            continue
        n_colors = 0 if n is None or (isinstance(n, float) and np.isnan(n)) else int(n)
        n_colors = min(n_colors, len(COLOR_COLUMN_NAMES))
        for j in range(n_colors):
            assignments.append((i, j))
    assignments = np.array(assignments)  # shape (total_to_assign, 2)

    # Map table row indices to point_cloud indices
    row_to_pc_idx = {int(row_idx): pc_idx for pc_idx, row_idx in enumerate(pc_row_indices)}
    assignment_pc_indices = np.array([row_to_pc_idx[int(i)] for i in assignments[:, 0]])
    coords = point_cloud[assignment_pc_indices]  # get coordinates for each assignment

    # Step 2: Cluster assignments into subgroups of max 150, each within 170m of at least one other
    unassigned = set(range(len(assignments)))
    subgroups = []
    while unassigned:
        current = unassigned.pop()
        cluster = {current}
        to_check = {current}
        while to_check and len(cluster) < 150:
            idx = to_check.pop()
            if not unassigned:
                break
            dists = np.linalg.norm(coords[list(unassigned)] - coords[idx], axis=1)
            close = [list(unassigned)[i] for i, d in enumerate(dists) if d <= 170]
            for c in close:
                if c not in cluster and len(cluster) < 150:
                    cluster.add(c)
                    to_check.add(c)
            unassigned -= set(close)
        subgroups.append(list(cluster))

    # Step 3: For each subgroup, assign a unique color (excluding colors used by nearby groups within 2000m)
    center = grp_dict['center']
    group_dict = grp_dict.get('group_dict', {})  # pass group_dict in grp_dict for context
    possible_colors = set(COLORS)
    possible_colors.discard('Kleurloos')
    if group_dict:
        for nearby_group in group_dict.values():
            if 'point cloud' not in nearby_group or 'table' not in nearby_group:
                continue
            pc1 = nearby_group['point cloud']
            pc2 = point_cloud
            if pc1.shape[0] > 0 and pc2.shape[0] > 0:
                min1, max1 = pc1.min(axis=0), pc1.max(axis=0)
                min2, max2 = pc2.min(axis=0), pc2.max(axis=0)
                dx = max(0, max(min2[0] - max1[0], min1[0] - max2[0]))
                dy = max(0, max(min2[1] - max1[1], min1[1] - max2[1]))
                min_box_dist = np.hypot(dx, dy)
                if min_box_dist >= 2000:
                    continue
                diff = pc1[:, np.newaxis, :] - pc2[np.newaxis, :, :]
                distances_matrix = np.linalg.norm(diff, axis=2)
                if np.any(distances_matrix < 2000):
                    color_val = nearby_group['table'][COLOR_COLUMN_NAMES[0]][0].as_py()
                    if color_val is not None:
                        possible_colors.discard(color_val)

    used_colors = set()
    # Track all assigned points and their colors for conflict checking
    assigned_points = []  # list of (coord, color)
    color_columns = {col: [None] * num_rows for col in COLOR_COLUMN_NAMES}
    color_list = list(possible_colors)
    for subgroup_idx, subgroup in enumerate(subgroups):
        # Get coordinates for all points in this subgroup
        subgroup_coords = coords[subgroup]
        # Exclude colors used within 2000m of any point in this subgroup
        forbidden_colors = set()
        for pt in subgroup_coords:
            for assigned_pt, assigned_color in assigned_points:
                if np.linalg.norm(pt - assigned_pt) < 2000:
                    forbidden_colors.add(assigned_color)
        available_colors = [c for c in color_list if c not in forbidden_colors]
        if not available_colors:
            raise RuntimeError("Not enough unique colors to assign to subgroups.")
        color = available_colors[0]
        # Assign color to each assignment in the subgroup
        for idx in subgroup:
            row_idx, col_idx = assignments[idx]
            color_columns[COLOR_COLUMN_NAMES[col_idx]][row_idx] = color
            assigned_points.append((coords[idx], color))


    # Step 4: Write back to table
    for col in COLOR_COLUMN_NAMES:
        arr = pa.array(color_columns[col])
        if col in table.schema.names:
            table = table.set_column(table.schema.get_field_index(col), col, arr)
        else:
            table = table.append_column(col, arr)
    grp_dict['table'] = table
    grp_dict['done'] = True
    grp_dict['amount_assigned'] = int(total_to_assign)
    grp_dict['color'] = None  # Not a single color, but multiple

    print(f"Large color assignment: {len(subgroups)} subgroups, colors used: {used_colors}")

def group_table_by_installatie_with_point_cloud(table: pa.Table):
    """
    Groups the table by 'installatie'. For each group, stores a dict with:
      - key 'table': the grouped pa.Table
      - key 'point cloud': the result of create_point_cloud(group)
      - key 'point cloud row indices': the table row indices for each point in the point cloud
      - key 'center': the centroid (mean x, mean y) of the point cloud, or None if empty
    Returns: dict[installatie] = {'table': group_table, 'point cloud': np.ndarray, 'point cloud row indices': np.ndarray, 'center': (float, float) or None}
    """
    # Vectorized: avoid repeated filtering by using numpy for grouping
    installaties = table['installatie'].to_numpy(zero_copy_only=False)
    unique_installaties, inverse_indices = np.unique(installaties, return_inverse=True)
    group_dict = {}
    for idx, installatie in enumerate(unique_installaties):
        group_mask = (inverse_indices == idx)
        # Use pyarrow's take to get the group table efficiently
        group_indices = np.flatnonzero(group_mask)
        group = table.take(pa.array(group_indices))
        point_cloud, pc_row_indices = create_point_cloud_with_indices(group)
        if point_cloud.shape[0] > 0:
            center = (float(point_cloud[:, 0].mean()), float(point_cloud[:, 1].mean()))
        else:
            center = None
        group_dict[installatie] = {
            'table': group,
            'point cloud': point_cloud,
            'point cloud row indices': pc_row_indices,
            'center': center,
            'done': False
        }
    return group_dict


def assign_colors_to_table(table: pa.Table) -> pa.Table:
    # make a grouped dict of the table by 'installatie'
    group_dict = group_table_by_installatie_with_point_cloud(table)

    # start with A2562
    # filter table to get only installatie A2562
    start_installatie = group_dict['A2562']
    print(start_installatie['table'])

    assign_colors_to_group_by_dict(start_installatie, group_dict=group_dict)

    # find the nearest group that has not been done yet, use the distance between the centers of the point clouds
    while True:
        # Find the next group that is not done and has the closest center to the current group
        # search for centers within 10000 meters of the current group
        current_group = None
        current_center = start_installatie['center']
        min_distance = 100000
        for installatie, grp_dict in group_dict.items():
            if grp_dict['done'] or grp_dict['center'] is None:
                continue
            # Calculate distance to the current center
            distance = np.linalg.norm(np.array(grp_dict['center']) - np.array(current_center))
            if distance < min_distance:
                min_distance = distance
                current_group = grp_dict

        if current_group is None:
            print("No more groups to process.")
            break
        print(f"Next group to process: {current_group['table']['installatie'][0].as_py()} at distance {min_distance:.2f}m")
        # Assign colors to the current group
        assign_colors_to_group_by_dict(current_group, group_dict=group_dict)

    table = update_main_table(group_dict, table)

    return table


def update_main_table(group_dict, table):
    # reflect the changes in group_dict back to the table
    updated_tables = []
    for installatie, grp_dict in group_dict.items():
        if not grp_dict['done']:
            continue
        # Collect the modified group tables
        updated_tables.append(grp_dict['table'])
    if updated_tables:
        table = pa.concat_tables(updated_tables, promote=True)
    return table


def group_table_by_installatie_with_point_cloud(table: pa.Table):
    """
    Groups the table by 'installatie'. For each group, stores a dict with:
      - key 'table': the grouped pa.Table
      - key 'point cloud': the result of create_point_cloud(group)
      - key 'center': the centroid (mean x, mean y) of the point cloud, or None if empty
    Returns: dict[installatie] = {'table': group_table, 'point cloud': np.ndarray, 'center': (float, float) or None}
    """
    # Vectorized: avoid repeated filtering by using numpy for grouping
    installaties = table['installatie'].to_numpy(zero_copy_only=False)
    unique_installaties, inverse_indices = np.unique(installaties, return_inverse=True)
    group_dict = {}
    for idx, installatie in enumerate(unique_installaties):
        group_mask = (inverse_indices == idx)
        # Use pyarrow's take to get the group table efficiently
        group_indices = np.flatnonzero(group_mask)
        group = table.take(pa.array(group_indices))
        point_cloud, pc_row_indices = create_point_cloud_with_indices(group)
        if point_cloud.shape[0] > 0:
            center = (float(point_cloud[:, 0].mean()), float(point_cloud[:, 1].mean()))
        else:
            center = None
        group_dict[installatie] = {
            'table': group,
            'point cloud': point_cloud,
            'point cloud row indices': pc_row_indices,
            'center': center,
            'done': False
        }
    return group_dict

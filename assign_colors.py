import numpy as np
import pyarrow as pa
from ordered_set import OrderedSet
from shapely.geometry import Point

from global_vars import COLORS, COLOR_COLUMN_NAMES, DISTANCE_BETWEEN_COLORED_GROUPS, START_INSTALLATION, \
    MAX_ARMATURES_PER_COLOR, MAX_CONNECTION_DISTANCE
from score_functions import score_E_distance_within_colored_group


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

def group_table_by_installatie_with_point_cloud(table: pa.Table):
    """
    Groups the table by 'installatie'. For each group, stores a dict with:
      - key 'table': the grouped pa.Table
      - key 'point cloud': the result of create_point_cloud(group)
      - key 'point cloud row indices': the table row indices for each point in the point cloud
      - key 'bbox': (min_x, min_y, max_x, max_y) bounding box of the point cloud, or None if empty
      - key 'center': the centroid (mean x, mean y) of the point cloud, or None if empty
      - key 'done': whether the group has been processed
      - key 'color': the color assigned to the group (None initially)
    Returns: dict[installatie] = {...}
    """



    installaties = table['installatie'].to_numpy(zero_copy_only=False)
    unique_installaties, inverse_indices = np.unique(installaties, return_inverse=True)
    group_dict = {}
    for idx, installatie in enumerate(unique_installaties):
        group_mask = (inverse_indices == idx)
        group_indices = np.flatnonzero(group_mask)
        group = table.take(pa.array(group_indices))
        point_cloud, pc_row_indices = create_point_cloud_with_indices(group)
        if point_cloud.shape[0] > 0:
            center = (float(point_cloud[:, 0].mean()), float(point_cloud[:, 1].mean()))
            min_x, min_y = point_cloud.min(axis=0)
            max_x, max_y = point_cloud.max(axis=0)
            bbox = (float(min_x), float(min_y), float(max_x), float(max_y))
        else:
            center = None
            bbox = None
        group_dict[installatie] = {
            'table': group,
            'point cloud': point_cloud,
            'point cloud row indices': pc_row_indices,
            'center': center,
            'bbox': bbox,
            'done': False,
            'color': None
        }
    return group_dict

def count_color_assignments(aant_array, valid_row_set):
    """Count total color assignments for records with valid coordinates."""
    total = 0
    for i, n in enumerate(aant_array):
        if n is None or (isinstance(n, float) and np.isnan(n)):
            continue
        n_colors = int(n)
        n_colors = min(n_colors, len(COLOR_COLUMN_NAMES))
        if i in valid_row_set:
            total += n_colors
    return total

def build_assignments(aant_array, row_to_pc_idx):
    """Build assignment list: (row_idx, color_col_idx, point_idx) for valid records."""
    assignments = []
    for i, n in enumerate(aant_array):
        if n is None or (isinstance(n, float) and np.isnan(n)):
            continue
        n_colors = int(n)
        n_colors = min(n_colors, len(COLOR_COLUMN_NAMES))
        if i in row_to_pc_idx:
            pc_idx = row_to_pc_idx[i]
            assignments.extend((i, j, pc_idx) for j in range(n_colors))
    return assignments

def split_into_170m_connected_subgroups(assignments, point_cloud, max_size=140):
    """
    Try to split assignments into 2..6 subgroups, up to 5 attempts per n_groups.
    For each attempt, use farthest point sampling (first attempt) or random seeds (other attempts).
    Grow groups by adding closest unassigned point within {MAX_CONNECTION_DISTANCE}m.
    If all subgroups are {MAX_CONNECTION_DISTANCE}m-connected (score_E_distance_within_colored_group > 0), return the split.
    """

    if not assignments:
        return []
    n_assign = len(assignments)
    assignment_points = np.array([point_cloud[pc_idx] for _, _, pc_idx in assignments])

    # Helper to build a dummy table for score_E_distance_within_colored_group
    def build_dummy_table_for_subgroup(subgroup):
        # subgroup: list of assignment indices
        # We need to create a pyarrow table with at least 'installatie', COLOR_COLUMN_NAMES[0], and 'wkb'
        # We'll use dummy values for 'installatie' and color, and real coordinates


        # Get the point indices for this subgroup
        pc_indices = [assignments[i][2] for i in subgroup]
        # Use the first color column for all
        color_col = [ "TestColor" ] * len(subgroup)
        installatie_col = [ "TestInstallatie" ] * len(subgroup)
        # Build WKB column
        wkb_col = []
        for idx in pc_indices:
            pt = Point(point_cloud[idx])
            wkb_col.append(pt.wkb)
        return pa.table({
            "installatie": installatie_col,
            COLOR_COLUMN_NAMES[0]: color_col,
            "wkb": wkb_col
        })

    for n_groups in range(2, 7):
        for attempt in range(5):
            # Seed selection
            if attempt == 0:
                # Farthest point sampling
                seeds = []
                used = set()
                seeds.append(0)
                used.add(0)
                for _ in range(1, n_groups):
                    dists_to_seeds = np.min(
                        np.linalg.norm(assignment_points[np.newaxis, :, :] - assignment_points[seeds, :][:, np.newaxis, :], axis=2),
                        axis=0
                    )
                    for idx in np.argsort(-dists_to_seeds):
                        if idx not in used:
                            seeds.append(idx)
                            used.add(idx)
                            break
            else:
                # Random seeds
                seeds = sorted(np.random.choice(n_assign, n_groups, replace=False).tolist())
            # Each seed forms the initial member of a group
            groups = [[seed] for seed in seeds]
            assigned = set(seeds)
            # Grow each group by adding the closest unassigned point within {MAX_CONNECTION_DISTANCE}m
            for _ in range(max_size):
                any_added = False
                for group in groups:
                    if len(assigned) >= n_assign:
                        break
                    if len(group) >= max_size:
                        continue
                    group_points = assignment_points[group]
                    dists = np.linalg.norm(group_points[:, None, :] - assignment_points[None, :, :], axis=2)
                    dists[:, list(assigned)] = np.inf
                    min_dist = np.min(dists, axis=0)
                    candidates = np.where((min_dist <= MAX_CONNECTION_DISTANCE) & (~np.isinf(min_dist)))[0]
                    if len(candidates) == 0:
                        continue
                    next_idx = candidates[np.argmin(min_dist[candidates])]
                    group.append(next_idx)
                    assigned.add(next_idx)
                    any_added = True
                    if len(assigned) == n_assign:
                        break
                if not any_added:
                    break
            # Handle any remaining unassigned points
            unassigned = set(range(n_assign)) - assigned
            for idx in list(unassigned):
                added = False
                for group in groups:
                    if len(group) >= max_size:
                        continue
                    group_points = assignment_points[group]
                    dists = np.linalg.norm(group_points - assignment_points[idx], axis=1)
                    if np.any(dists <= MAX_CONNECTION_DISTANCE):
                        group.append(idx)
                        assigned.add(idx)
                        added = True
                        break
                if not added:
                    groups.append([idx])
                    assigned.add(idx)
            # Check if all groups are {MAX_CONNECTION_DISTANCE}m-connected
            all_good = True
            for group in groups:
                if len(group) == 0:
                    continue
                dummy_table = build_dummy_table_for_subgroup(group)
                if score_E_distance_within_colored_group(dummy_table) == 0:
                    all_good = False
                    break
            if all_good:
                return [group for group in groups if len(group) > 0]
    # Fallback: original logic with n_groups based on max_size
    n_groups = int(np.ceil(n_assign / max_size))
    if n_groups < 2:
        n_groups = 2
    seeds = []
    used = set()
    seeds.append(0)
    used.add(0)
    for _ in range(1, n_groups):
        dists_to_seeds = np.min(
            np.linalg.norm(assignment_points[np.newaxis, :, :] - assignment_points[seeds, :][:, np.newaxis, :], axis=2),
            axis=0
        )
        for idx in np.argsort(-dists_to_seeds):
            if idx not in used:
                seeds.append(idx)
                used.add(idx)
                break
    groups = [[seed] for seed in seeds]
    assigned = set(seeds)
    for _ in range(max_size):
        any_added = False
        for group in groups:
            if len(assigned) >= n_assign:
                break
            if len(group) >= max_size:
                continue
            group_points = assignment_points[group]
            dists = np.linalg.norm(group_points[:, None, :] - assignment_points[None, :, :], axis=2)
            dists[:, list(assigned)] = np.inf
            min_dist = np.min(dists, axis=0)
            candidates = np.where((min_dist <= MAX_CONNECTION_DISTANCE) & (~np.isinf(min_dist)))[0]
            if len(candidates) == 0:
                continue
            next_idx = candidates[np.argmin(min_dist[candidates])]
            group.append(next_idx)
            assigned.add(next_idx)
            any_added = True
            if len(assigned) == n_assign:
                break
        if not any_added:
            break
    unassigned = set(range(n_assign)) - assigned
    for idx in list(unassigned):
        added = False
        for group in groups:
            if len(group) >= max_size:
                continue
            group_points = assignment_points[group]
            dists = np.linalg.norm(group_points - assignment_points[idx], axis=1)
            if np.any(dists <= MAX_CONNECTION_DISTANCE):
                group.append(idx)
                assigned.add(idx)
                added = True
                break
        if not added:
            groups.append([idx])
            assigned.add(idx)
    return [group for group in groups if len(group) > 0]


def assign_colors_to_subgroups(table, assignments, subgroups, point_cloud, group_dict, orig_key, used_colors):
    """Assign a unique color to each subgroup and update group_dict."""
    num_rows = table.num_rows
    for idx, subgroup in enumerate(subgroups):
        possible_colors = OrderedSet(COLORS)
        forbidden_colors = set()
        for other_key, other_grp in group_dict.items():
            if other_grp['color'] is not None:
                pc1 = other_grp['point cloud']
                subgroup_pc_indices = [assignments[i][2] for i in subgroup]
                pc2 = point_cloud[subgroup_pc_indices]
                if pc1.shape[0] > 0 and pc2.shape[0] > 0:
                    min1, max1 = pc1.min(axis=0), pc1.max(axis=0)
                    min2, max2 = pc2.min(axis=0), pc2.max(axis=0)
                    dx = max(0, max(min2[0] - max1[0], min1[0] - max2[0]))
                    dy = max(0, max(min2[1] - max1[1], min1[1] - max2[1]))
                    min_box_dist = np.hypot(dx, dy)
                    if min_box_dist >= DISTANCE_BETWEEN_COLORED_GROUPS:
                        continue
                    diff = pc1[:, np.newaxis, :] - pc2[np.newaxis, :, :]
                    distances_matrix = np.linalg.norm(diff, axis=2)
                    if np.any(distances_matrix < DISTANCE_BETWEEN_COLORED_GROUPS):
                        forbidden_colors.add(other_grp['color'])
        possible_colors -= used_colors
        possible_colors -= forbidden_colors
        assigned_color = next(iter(possible_colors), None)
        used_colors.add(assigned_color)
        print(f"Processing subgroup {idx+1} with {len(subgroup)} assignments and color {assigned_color}")

        # Only include rows that are actually assigned in this subgroup
        row_indices_in_subgroup = sorted({assignments[i][0] for i in subgroup})
        sub_table = table.take(pa.array(row_indices_in_subgroup))

        # Assign color columns for only these rows
        num_sub_rows = sub_table.num_rows
        color_columns = {col: [None] * num_sub_rows for col in COLOR_COLUMN_NAMES}
        # Map from row_idx in original table to index in sub_table
        row_idx_to_sub_idx = {row_idx: i for i, row_idx in enumerate(row_indices_in_subgroup)}
        # Find all row indices in this subgroup
        rows_in_this_subgroup = {assignments[idx][0] for idx in subgroup}
        aant_col = 'eigenschappen|eig|aantal verlichtingstoestellen'
        aant_array = table[aant_col].to_pylist()
        for row_idx in rows_in_this_subgroup:
            sub_idx = row_idx_to_sub_idx[row_idx]
            n = aant_array[row_idx]
            n_colors = 0 if n is None or (isinstance(n, float) and np.isnan(n)) else int(n)
            n_colors = min(n_colors, len(COLOR_COLUMN_NAMES))
            for j, col in enumerate(COLOR_COLUMN_NAMES):
                color_columns[col][sub_idx] = assigned_color if j < n_colors else None
        for col in COLOR_COLUMN_NAMES:
            arr = pa.array(color_columns[col])
            if col in sub_table.schema.names:
                sub_table = sub_table.set_column(sub_table.schema.get_field_index(col), col, arr)
            else:
                sub_table = sub_table.append_column(col, arr)

        subgroup_pc_indices = [assignments[i][2] for i in subgroup]
        pc2 = point_cloud[subgroup_pc_indices]
        if pc2.shape[0] > 0:
            center = (float(pc2[:, 0].mean()), float(pc2[:, 1].mean()))
            min_x, min_y = pc2.min(axis=0)
            max_x, max_y = pc2.max(axis=0)
            bbox = (float(min_x), float(min_y), float(max_x), float(max_y))
        else:
            center = None
            bbox = None
        new_key = f"{orig_key}_subgroup_{idx+1}"
        group_dict[new_key] = {
            'table': sub_table,
            'point cloud': pc2,
            'point cloud row indices': np.array(subgroup_pc_indices),
            'center': center,
            'bbox': bbox,
            'done': True,
            'color': assigned_color
        }


def assign_colors_to_group(grp_dict, group_dict, assigned_points=None):
    """
    Assigns a color to a group if total aantal_verlichtingstoestellen < {MAX_ARMATURES_PER_COLOR}-10 and no nearby
    group within DISTANCE_BETWEEN_COLORED_GROUPS meters uses that color.
    If >= {MAX_ARMATURES_PER_COLOR}-10, splits into subgroups of max {MAX_ARMATURES_PER_COLOR}, each subgroup must be a {
    MAX_CONNECTION_DISTANCE}m-connected component, and assigns each a
    unique color.
    """
    print(f"Assigning colors to group {grp_dict['table']['installatie'][0].as_py()}")
    table = grp_dict['table']
    num_rows = table.num_rows
    aant_col = 'eigenschappen|eig|aantal verlichtingstoestellen'
    aant_array = table[aant_col].to_pylist()
    point_cloud = grp_dict['point cloud']
    pc_row_indices = grp_dict['point cloud row indices']
    valid_row_set = set(int(idx) for idx in pc_row_indices)
    total_to_assign = count_color_assignments(aant_array, valid_row_set)

    # Normal case: less than {MAX_ARMATURES_PER_COLOR} - 10 assignments, assign a single color
    if total_to_assign < MAX_ARMATURES_PER_COLOR-10:
        center = grp_dict['center']
        possible_colors = OrderedSet(COLORS)
        pc2 = grp_dict['point cloud']
        forbidden_colors = set()
        # Unify forbidden color logic: check all colored groups in group_dict
        for other_key, other_grp in group_dict.items():
            if other_grp['color'] is not None and other_key != grp_dict['table']['installatie'][0].as_py():
                pc1 = other_grp['point cloud']
                if pc1.shape[0] > 0 and pc2.shape[0] > 0:
                    min1, max1 = pc1.min(axis=0), pc1.max(axis=0)
                    min2, max2 = pc2.min(axis=0), pc2.max(axis=0)
                    dx = max(0, max(min2[0] - max1[0], min1[0] - max2[0]))
                    dy = max(0, max(min2[1] - max1[1], min1[1] - max2[1]))
                    min_box_dist = np.hypot(dx, dy)
                    if min_box_dist >= DISTANCE_BETWEEN_COLORED_GROUPS:
                        continue
                    diff = pc1[:, np.newaxis, :] - pc2[np.newaxis, :, :]
                    distances_matrix = np.linalg.norm(diff, axis=2)
                    if np.any(distances_matrix < DISTANCE_BETWEEN_COLORED_GROUPS):
                        forbidden_colors.add(other_grp['color'])
        possible_colors -= forbidden_colors

        assigned_color = next(iter(possible_colors), None)
        color_columns = {col: [None] * num_rows for col in COLOR_COLUMN_NAMES}
        for i in range(num_rows):
            n = aant_array[i]
            n_colors = 0 if n is None or (isinstance(n, float) and np.isnan(n)) else int(n)
            n_colors = min(n_colors, len(COLOR_COLUMN_NAMES))
            assigned = [assigned_color] * n_colors + [None] * (len(COLOR_COLUMN_NAMES) - n_colors)
            for j, col in enumerate(COLOR_COLUMN_NAMES):
                color_columns[col][i] = assigned[j]
        for col in COLOR_COLUMN_NAMES:
            arr = pa.array(color_columns[col])
            if col in table.schema.names:
                table = table.set_column(table.schema.get_field_index(col), col, arr)
            else:
                table = table.append_column(col, arr)
        grp_dict['table'] = table
        grp_dict['done'] = True
        grp_dict['color'] = assigned_color
        print(f"Assigned color {assigned_color} to group {grp_dict['table']['installatie'][0].as_py()}")
        if assigned_points is not None and assigned_color is not None:
            for pt in pc2:
                assigned_points.append((pt, assigned_color))

        score_rule_E = score_E_distance_within_colored_group(table)
        if score_rule_E > 0:
            return
        # Revert assigned colors if score is 0
        for col in COLOR_COLUMN_NAMES:
            if col in table.schema.names:
                # Set all values to None for this group
                table = table.set_column(
                    table.schema.get_field_index(col),
                    col,
                    pa.array([None] * num_rows)
                )
        grp_dict['table'] = table
        grp_dict['done'] = False
        grp_dict['color'] = None
            # if not then we need to split into subgroups

    print(f"Group {grp_dict['table']['installatie'][0].as_py()} has {total_to_assign} assignments, "
          "splitting into subgroups.")
    row_to_pc_idx = {int(row_idx): pc_idx for pc_idx, row_idx in enumerate(pc_row_indices)}
    assignments = build_assignments(aant_array, row_to_pc_idx)
    print(f"Total assignments (with valid coordinates): {len(assignments)}")
    subgroups = split_into_170m_connected_subgroups(assignments, point_cloud, max_size=MAX_ARMATURES_PER_COLOR)
    orig_key = grp_dict['table']['installatie'][0].as_py()
    del group_dict[orig_key]
    used_colors = set()
    assign_colors_to_subgroups(table, assignments, subgroups, point_cloud, group_dict, orig_key, used_colors)


def assign_colors_to_table(table: pa.Table) -> pa.Table:
    """
    Main function to assign colors to a pyarrow table by installatie groups.
    """
    group_dict = group_table_by_installatie_with_point_cloud(table)
    # Start with the group with the most records (or pick any, e.g., first)
    installaties = list(group_dict.keys())
    if not installaties:
        return table
    # You can change the starting group selection logic as needed

    current_group = group_dict[START_INSTALLATION]
    assigned_points = []
    assign_colors_to_group(current_group, group_dict, assigned_points=assigned_points)
    # Process remaining groups by nearest center
    while True:
        # Find the next group that is not done and has the closest center to the current group
        current_center = current_group['center']
        min_distance = float('inf')
        next_group = None
        for installatie, grp_dict in group_dict.items():
            if grp_dict['done'] or grp_dict['center'] is None:
                continue
            distance = np.linalg.norm(np.array(grp_dict['center']) - np.array(current_center))
            if distance < min_distance:
                min_distance = distance
                next_group = grp_dict
        if next_group is None:
            break
        assign_colors_to_group(next_group, group_dict, assigned_points=assigned_points)
        current_group = next_group
    # Update the main table with all group assignments
    updated_tables = [grp_dict['table'] for grp_dict in group_dict.values() if grp_dict['done']]
    if updated_tables:
        table = pa.concat_tables(updated_tables, promote=True)
    return table



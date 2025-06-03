import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
from shapely import wkb

from global_vars import COLOR_COLUMN_NAMES


def create_point_cloud(group: pa.Table) -> np.ndarray:
    points = [wkb.loads(point.as_py()) for point in group['wkb']]
    return np.array([[pt.x, pt.y] for pt in points if pt is not None])


def score_E_distance_within_colored_group(table: pa.Table) -> float:
    """
    +50 per group (installatie, color) if all points are within 170m of each other (or only one point).
    Optimized for performance: avoids repeated filtering and uses numpy vectorization.
    """
    # Group by 'installatie' and color
    grouped_table = table.group_by(['installatie', COLOR_COLUMN_NAMES[0]]).aggregate([])

    # Pre-extract all relevant columns as numpy arrays for fast filtering
    installatie_arr = table['installatie'].to_numpy(zero_copy_only=False)
    color_arr = table[COLOR_COLUMN_NAMES[0]].to_numpy(zero_copy_only=False)
    wkb_arr = table['wkb'].to_numpy(zero_copy_only=False)

    score = 0
    for i in range(len(grouped_table)):
        installation = grouped_table['installatie'][i].as_py()
        color = grouped_table[COLOR_COLUMN_NAMES[0]][i].as_py()

        # Vectorized mask for group selection
        mask = (installatie_arr == installation) & (color_arr == color)
        group_wkb = wkb_arr[mask]

        # Create the point cloud for the group
        points = [wkb.loads(wkb_bytes) for wkb_bytes in group_wkb]
        point_cloud = np.array([[pt.x, pt.y] for pt in points if pt is not None])

        n_points = point_cloud.shape[0]
        if n_points == 1:
            score += 50
            continue

        if n_points > 1:
            # Efficient pairwise distance calculation
            diff = point_cloud[:, None, :] - point_cloud[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            # Only consider upper triangle (unique pairs)
            max_dist = np.max(np.triu(dists, k=1))
            if max_dist < 170:
                score += 50

    return score


def score_C_max_150_armaturen_per_kleur_per_installatie(table: pa.Table) -> float:
    """
    +25 per installatie waarvoor het aantal armaturen per kleur niet boven de 150 gaat.
    """
    # Group by 'installatie' and color (using the first color column)
    grouped = table.group_by(['installatie', COLOR_COLUMN_NAMES[0]]).aggregate([
        ("eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen", "sum")
    ])
    # Use pyarrow only: convert to numpy arrays for vectorized processing
    installaties = grouped['installatie'].to_numpy(zero_copy_only=False)
    counts = grouped["eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen_sum"].to_numpy(zero_copy_only=False)

    # Get unique installaties and for each, check if all counts for that installatie are <= 150
    # Use pyarrow.compute to build a mask for each installatie
    unique_installaties = pa.array(np.unique(installaties))
    score = 0
    for installatie in unique_installaties:
        mask = pc.equal(grouped['installatie'], installatie)
        # Use pyarrow.compute.filter to get counts for this installatie
        counts_for_installatie = pc.filter(grouped["eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen_sum"], mask)
        # If all counts <= 150, add 25
        if pc.all(pc.less_equal(counts_for_installatie, pa.scalar(150))).as_py():
            score += 25
    return score


def score_D_distance_between_colored_group(table: pa.Table) -> float:
    # Group by "installation" and "color" and create the point clouds for each group
    grouped_table = table.group_by(['installatie', COLOR_COLUMN_NAMES[0]]).aggregate([])

    # Iterate over the groups and create point cloud per group
    point_clouds = []
    for i in range(len(grouped_table)):
        installation = grouped_table['installatie'][i].as_py()
        color = grouped_table[COLOR_COLUMN_NAMES[0]][i].as_py()

        # Get the rows for the current group
        filter_condition = pc.and_(
            pc.equal(table['installatie'], installation),
            pc.equal(table[COLOR_COLUMN_NAMES[0]], color))

        group = table.filter(filter_condition)

        # Create the point cloud for the group
        point_cloud = create_point_cloud(group)
        point_clouds.append((installation, color, point_cloud))

    from collections import defaultdict

    # Index point clouds by color for fast lookup
    color_to_clouds = defaultdict(list)
    for installation, color, point_cloud in point_clouds:
        color_to_clouds[color].append((installation, point_cloud))

    score = 0
    for clouds in color_to_clouds.values():
        n = len(clouds)
        # Precompute which clouds are non-empty
        non_empty = [i for i, (_, pc) in enumerate(clouds) if pc.size > 0]
        for idx in non_empty:
            installation, point_cloud = clouds[idx]
            min_dist_total = 2500
            for jdx in non_empty:
                if idx < jdx:
                    other_installation, other_point_cloud = clouds[jdx]
                    # Compute pairwise distances only once per unique pair
                    dists = np.linalg.norm(point_cloud[:, None, :] - other_point_cloud[None, :, :], axis=2)
                    min_dist = np.min(dists)
                    if min_dist < min_dist_total:
                        min_dist_total = min_dist

            if min_dist_total < 1000:
                points = 0
            elif min_dist_total <= 1500:
                points = 50
            elif min_dist_total <= 2000:
                points = 100
            else:
                points = 150
            # print(f"point_cloud1: {installation}_{color}, Min Distance:"
            #       f" {min_dist_total}, Points: {points}")
            score += points
    return score


def score_A_color_for_each_armature(table: pa.Table) -> float:
    """
    +0.01 * aantal verlichtingstoestellen per armatuur waarvoor alle kleur kolommen gevuld zijn volgens het
    aantal verlichtingstoestellen.
    """
    num_rows = len(table)
    if num_rows == 0:
        return 0.0

    # Use pyarrow.compute.is_valid to get a boolean mask for each color column
    valid_masks = [pc.is_valid(table[col]) for col in COLOR_COLUMN_NAMES]
    # Stack into a 2D boolean numpy array: shape (num_rows, num_color_cols)
    valid_matrix = np.column_stack([mask.to_numpy(zero_copy_only=False) for mask in valid_masks])
    # Count non-nulls per row
    non_null_counts = valid_matrix.sum(axis=1)

    # Get the 'aantal verlichtingstoestellen' column as numpy array
    aantallen = table['eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen'].to_numpy(zero_copy_only=False)
    # Calculate mask where all color columns are filled (non-nulls == aantal)
    mask = non_null_counts == aantallen
    # Compute score: 0.01 * aantal where mask is True, else 0
    scores = np.where(mask, 0.01 * aantallen, 0)
    return float(np.sum(scores))


def score_H_I_total_amount_of_colors(table: pa.Table) -> float:
    # Efficient approach: use pyarrow's concat_arrays and unique
    # Optimized: Use pyarrow.compute to stack and filter columns efficiently
    # 1. Cast all columns to string in a single pass, handling null columns
    casted_columns = [
        (table[col] if pa.types.is_string(table[col].type) else table[col].cast(pa.string()))
        for col in COLOR_COLUMN_NAMES
    ]
    # 2. Stack all columns into a single ChunkedArray
    stacked = pa.chunked_array(casted_columns)
    # 3. Filter out nulls in one go
    non_null_colors = pc.drop_null(stacked)
    # 4. Get unique colors as Python set
    used_colors = set(non_null_colors.to_pylist())

    score = 0

    # If "Kleurloos" is not used, add 2000 points
    if "Kleurloos" not in used_colors:
        score += 2000

    # For every other color NOT used, add 1000 points
    for color in ['Cyan', 'Yellow', 'Magenta', 'Black', 'Blue', 'Red', 'Green', 'Kleurloos']:
        if color not in used_colors and color != "Kleurloos":
            score += 1000

    return score


def score_F_uniform_color_per_installatie(table: pa.Table) -> float:
    # Efficient PyArrow approach: avoid pandas, use group_by and compute.unique
    score = 0
    # Group by 'installatie'
    grouped = table.group_by('installatie').aggregate([])
    for installatie in grouped['installatie'].to_pylist():
        # Filter rows for this group
        mask = pc.equal(table['installatie'], installatie)
        group = table.filter(mask)
        # Stack all color columns into a single array, cast to string for safety
        color_arrays = [
            group[col] if pa.types.is_string(group[col].type) else group[col].cast(pa.string())
            for col in COLOR_COLUMN_NAMES
        ]
        stacked = pa.chunked_array(color_arrays)
        # Drop nulls
        non_null_colors = pc.drop_null(stacked)
        # Get unique colors
        unique_colors = pc.unique(non_null_colors).to_pylist()
        # If exactly one unique color (not None), award 50 points
        if len(unique_colors) == 1:
            score += 50
    return score


def get_score(table: pa.Table) -> float:
    return (score_B_minimize_colors_within_installation(table)
            + score_A_color_for_each_armature(table)
            + score_D_distance_between_colored_group(table)
            + score_E_distance_within_colored_group(table)
            + score_H_I_total_amount_of_colors(table)
            + score_F_uniform_color_per_installatie(table)
            + score_C_max_150_armaturen_per_kleur_per_installatie(table))


def score_B_minimize_colors_within_installation(table: pa.Table) -> float:
    # Optimized: Use pyarrow to stack columns and compute unique colors per group efficiently
    score = 0
    grouped = table.group_by('installatie').aggregate([])
    for installatie in grouped['installatie'].to_pylist():
        mask = pc.equal(table['installatie'], installatie)
        group = table.filter(mask)
        # Stack all color columns, cast to string for safety
        color_arrays = [
            group[col] if pa.types.is_string(group[col].type) else group[col].cast(pa.string())
            for col in COLOR_COLUMN_NAMES
        ]
        stacked = pa.chunked_array(color_arrays)
        non_null_colors = pc.drop_null(stacked)
        unique_colors = pc.unique(non_null_colors).to_pylist()
        num_colors = len(unique_colors)
        group_score = 10 + (8 - num_colors) * 10 if num_colors < 8 else 10
        score += group_score
    return score

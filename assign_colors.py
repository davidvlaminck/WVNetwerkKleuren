import random

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
from shapely import wkb

from global_vars import COLORS, COLOR_COLUMN_NAMES
from score_functions import create_point_cloud


def assign_colors_to_table(table: pa.Table) -> pa.Table:
    # make a grouped dict of the table by 'installatie'
    group_dict = group_table_by_installatie_with_point_cloud(table)



    # start with A2562
    # filter table to get only installatie A2562
    filtered_table = group_dict['A2562']['table']
    print(filtered_table)



    def get_random_color() -> str:
        return random.choice(COLORS)

    def assign_colors(row):
        aantal_verlichtingstoestellen = row['eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen']
        # Handle NaN or None safely
        if aantal_verlichtingstoestellen is None or (isinstance(aantal_verlichtingstoestellen, float) and np.isnan(aantal_verlichtingstoestellen)):
            n = 0
        else:
            n = int(aantal_verlichtingstoestellen)
        for i in range(min(n, 4)):
            row[COLOR_COLUMN_NAMES[i]] = get_random_color()
        return row

    # Apply the assign_colors function to each row in the table
    filtered_df = filtered_table.to_pandas().apply(assign_colors, axis=1)
    # Convert the modified DataFrame back to a PyArrow Table
    filtered_table = pa.Table.from_pandas(df=filtered_df)
    return filtered_table


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
    wkb_arr = table['wkb'].to_numpy(zero_copy_only=False)
    unique_installaties, inverse_indices = np.unique(installaties, return_inverse=True)
    group_dict = {}
    for idx, installatie in enumerate(unique_installaties):
        group_mask = (inverse_indices == idx)
        # Use pyarrow's take to get the group table efficiently
        group_indices = np.flatnonzero(group_mask)
        group = table.take(pa.array(group_indices))
        point_cloud = create_point_cloud(group)
        if point_cloud.shape[0] > 0:
            center = (float(point_cloud[:, 0].mean()), float(point_cloud[:, 1].mean()))
        else:
            center = None
        group_dict[installatie] = {
            'table': group,
            'point cloud': point_cloud,
            'center': center
        }
    return group_dict
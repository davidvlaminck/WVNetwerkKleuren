import math
import random
import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import pyarrow.csv as csv
import geoarrow.pyarrow as ga
from shapely import wkb
from shapely.ops import unary_union

COLORS = ['Cyan', 'Yellow', 'Magenta', 'Black', 'Blue', 'Red', 'Green', 'Kleurloos']
COLOR_COLUMN_NAMES = [
    'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1',
    'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV2',
    'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV3',
    'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV4']

def convert_csv_to_parquet():
    # Define the path to the Excel file
    excel_file = Path(__file__).parent / 'data' / 'export_20250602110914.csv'
    # Read the Excel file into a PyArrow Table
    table = csv.read_csv(excel_file, parse_options=csv.ParseOptions(delimiter='\t', newlines_in_values=True))

    # Filter the table
    filter_mask = pc.starts_with(table["naampad"], 'A')
    filtered_table = table.filter(filter_mask)

    columns_to_keep = [
        'id','naampad','type','actief','toestand','locatie|punt|x|lambert72','locatie|punt|y|lambert72',
        'eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen',
        'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1',
        'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV2',
        'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV3',
        'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV4']

    filtered_table = filtered_table.select(columns_to_keep)

    # for pyarrow table: add a column installatie that takes the first part of "naampad" up until the first "/"

    filtered_table = add_installatie(filtered_table)

    filtered_table = add_wkb(filtered_table)

    # assign colors to the table
    filtered_table = assign_colors_to_table(filtered_table)

    # write to parquet file
    parquet_file = Path(__file__).parent / 'data' / 'filtered_data.parquet'
    pq.write_table(filtered_table, parquet_file)


def add_installatie(filtered_table: pa.Table) -> pa.Table:
    def extract_first_part(naampad: str) -> str:
        return naampad.split('/')[0]

    # Apply the function to create the new column "installatie"
    installatie_column = [extract_first_part(naampad.as_py()) for naampad in filtered_table['naampad']]
    # Add the new column to the table
    filtered_table = filtered_table.append_column('installatie', pa.array(installatie_column))
    return filtered_table


def add_wkb(filtered_table: pa.Table) -> pa.Table:
    def make_wkb_point(x: float, y: float) -> bytes | None:
        if x is None or y is None:
            return None
        # WKB format for a point: 01 (byte order) + 01000000 (point type) + x + y
        byte_order = struct.pack('<B', 1)  # Little-endian
        point_type = struct.pack('<I', 1)  # Point type
        x_bytes = struct.pack('<d', x)  # x coordinate
        y_bytes = struct.pack('<d', y)  # y coordinate
        return byte_order + point_type + x_bytes + y_bytes

    # Apply the make_wkb_point function to the x and y columns
    wkb_list = [make_wkb_point(x.as_py(), y.as_py())
                for x, y in zip(filtered_table['locatie|punt|x|lambert72'],
                                filtered_table['locatie|punt|y|lambert72'])]
    # Convert the list to a PyArrow array with binary type, ensuring None values are handled correctly
    wkb_array = pa.array(wkb_list, type=pa.binary())
    # Create a new table with the geometry (WKB) column
    filtered_table = filtered_table.append_column('geometry', wkb_array)
    return filtered_table


def assign_colors_to_table(filtered_table: pa.Table) -> pa.Table:
    def get_random_color() -> str:
        return random.choice(COLORS)

    def assign_colors(row: pa.RecordBatch) -> pa.RecordBatch:
        aantal_verlichtingstoestellen = row['eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen']
        if aantal_verlichtingstoestellen > 0:
            row['eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1'] = get_random_color()
        if aantal_verlichtingstoestellen >= 2:
            row['eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV2'] = get_random_color()
        if aantal_verlichtingstoestellen >= 3:
            row['eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV3'] = get_random_color()
        if aantal_verlichtingstoestellen >= 4:
            row['eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV4'] = get_random_color()
        return row

    # Apply the assign_colors function to each row in the table
    filtered_df = filtered_table.to_pandas().apply(assign_colors, axis=1)
    # Convert the modified DataFrame back to a PyArrow Table
    filtered_table = pa.Table.from_pandas(df=filtered_df)
    return filtered_table


def score_D_distance_between_colored_group(table: pa.Table) -> float:
    # Function to create a buffered polygon for each group of points
    def create_buffered_polygon(group):
        points = [wkb.loads(point.as_py()) for point in group['wkb']]
        buffered_points = [point.buffer(1000) for point in points]
        return unary_union(buffered_points)

    # Group by "installation" and "color" and create the buffered polygons
    grouped_table = table.group_by(['installatie', COLOR_COLUMN_NAMES[0]]).aggregate([])

    # Iterate over the groups and create polygons per group
    polygons = []
    for i in range(len(grouped_table)):
        installation = grouped_table['installatie'][i].as_py()
        color = grouped_table[COLOR_COLUMN_NAMES[0]][i].as_py()

        # Get the rows for the current group
        filter_condition = pc.and_(
            pc.equal(table['installatie'], installation),
            pc.equal(table[COLOR_COLUMN_NAMES[0]], color))

        group = table.filter(filter_condition)

        # Create the buffered polygon for the group
        polygon = create_buffered_polygon(group)
        polygons.append((installation, color, polygon))

    # verify for each polygon that it does not intersect with any other polygons
    score = 0
    for i, (_, color, polygon) in enumerate(polygons):
        intersected = False
        for j, (_, other_color, other_polygon) in enumerate(polygons):
            if i != j and polygon.intersects(other_polygon) and color == other_color:
                print(f"Polygon {i} intersects with Polygon {j}")
                intersected = True
        if not intersected:
            score += 50

    return score


def score_A_color_for_each_armature(table: pa.Table) -> float:
    def calculate_score(row: pa.RecordBatch) -> float:
        # Count the number of non-null values in the color columns
        num_colors = sum(
            color is not None for color in [
                row[COLOR_COLUMN_NAMES[0]], row[COLOR_COLUMN_NAMES[1]],
                row[COLOR_COLUMN_NAMES[2]], row[COLOR_COLUMN_NAMES[3]],
            ])

        # Get the value from "eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen"
        aantal_verlichtingstoestellen = row['eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen']

        # Calculate the score
        return 0.01 * aantal_verlichtingstoestellen if num_colors == aantal_verlichtingstoestellen else 0

    return sum(
        calculate_score(row)
        for batch in table.to_batches()
        for row in batch.to_pylist()
    )


def get_score(table: pa.Table) -> float:
    return (score_B_minimize_colors_within_installation(table)
            + score_A_color_for_each_armature(table)
            + score_D_distance_between_colored_group(table))


def score_B_minimize_colors_within_installation(table: pa.Table) -> float:
    def calculate_score(group):
        # Count the number of unique COLORS used in the group
        unique_colors = set()
        for column in COLOR_COLUMN_NAMES:
            unique_colors.update(group[column].to_pylist())
        unique_colors.discard(None)

        num_colors = len(unique_colors)
        return 10 + (8 - num_colors) * 10 if num_colors < 8 else 10

    # Group by "installatie" and calculate the score for each group
    grouped_table = table.group_by('installatie').aggregate([])
    scores = [calculate_score(table.filter(pc.equal(table['installatie'], installatie))) for installatie in
              grouped_table['installatie'].to_pylist()]
    return sum(scores)


if __name__ == "__main__":
    convert_csv_to_parquet()

    # Read the parquet file
    parquet_file = Path(__file__).parent / 'data' / 'filtered_data.parquet'
    table = pq.read_table(parquet_file)
    
    score = get_score(table)
    print(f"Score: {score}")

    df = ga.to_geopandas(table)
    # print resulting DataFrame, with all data for the first 7 rows
    df = df.head(7)
    df = df.reset_index(drop=True)
    print(df.to_string(index=False, max_colwidth=100))



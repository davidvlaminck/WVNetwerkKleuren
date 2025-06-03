import random
import struct
import time
from pathlib import Path

import geoarrow.pyarrow as ga
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.parquet as pq

from global_vars import COLORS, COLOR_COLUMN_NAMES
from score_functions import get_score


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
    filtered_table = filtered_table.append_column('wkb', wkb_array)
    return filtered_table


def assign_colors_to_table(filtered_table: pa.Table) -> pa.Table:
    def get_random_color() -> str:
        return random.choice(COLORS)

    def assign_colors(row):
        aantal_verlichtingstoestellen = row['eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen']
        for i in range(min(int(aantal_verlichtingstoestellen), 4)):
            row[COLOR_COLUMN_NAMES[i]] = get_random_color()
        return row

    # Apply the assign_colors function to each row in the table
    filtered_df = filtered_table.to_pandas().apply(assign_colors, axis=1)
    # Convert the modified DataFrame back to a PyArrow Table
    filtered_table = pa.Table.from_pandas(df=filtered_df)
    return filtered_table


if __name__ == "__main__":
    # convert_csv_to_parquet()

    # Read the parquet file
    parquet_file = Path(__file__).parent / 'data' / 'filtered_data.parquet'
    table = pq.read_table(parquet_file)

    start = time.time()
    score = get_score(table)
    end = time.time()
    print(f"Score: {score} Time taken: {end - start:.2f} seconds")

    df = ga.to_geopandas(table)
    # print resulting DataFrame, with all data for the first 7 rows
    df = df.head(7)
    df = df.reset_index(drop=True)
    print(df.to_string(index=False, max_colwidth=100))



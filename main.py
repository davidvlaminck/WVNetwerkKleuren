import struct
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq

from assign_colors import assign_colors_to_table
from global_vars import COLOR_COLUMN_NAMES
from score_functions import get_score


def convert_csv_to_parquet():
    # create_table_from_one_district()

    table = create_table_from_all_districts()

    table = combine_columns(table,
        from_columns=[
            'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1',
            'eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV1',
            'eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV1',
            'eigenschappen - lgc:installatie#vvop|eig|netwerkconfigWV1'],
        to_column='eigenschappen|eig|netwerkconfigWV1'
    )
    table = combine_columns(table,
        from_columns=[
            'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV2',
            'eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV2',
            'eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV2',
            'eigenschappen - lgc:installatie#vvop|eig|netwerkconfigWV2'],
        to_column='eigenschappen|eig|netwerkconfigWV2'
    )
    table = combine_columns(table,
        from_columns=[
            'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV3',
            'eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV3',
            'eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV3',
            'eigenschappen - lgc:installatie#vvop|eig|netwerkconfigWV3'],
        to_column='eigenschappen|eig|netwerkconfigWV3'
    )
    table = combine_columns(table,
        from_columns=[
            'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV4',
            'eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV4',
            'eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV4',
            'eigenschappen - lgc:installatie#vvop|eig|netwerkconfigWV4'],
        to_column='eigenschappen|eig|netwerkconfigWV4'
    )

    table = combine_columns(table,
        from_columns=['eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen',
            'eigenschappen - lgc:installatie#vpconsole|eig|aantal verlichtingstoestellen',
            'eigenschappen - lgc:installatie#vpbevestig|eig|aantal verlichtingstoestellen',
            'eigenschappen - lgc:installatie#vvop|eig|aantal verlichtingstoestellen'],
        to_column='eigenschappen|eig|aantal verlichtingstoestellen', to_type=pa.float64()
    )

    columns_to_keep = [
        'id','naampad','type','actief','toestand','locatie|punt|x|lambert72','locatie|punt|y|lambert72',
        'eigenschappen|eig|aantal verlichtingstoestellen']
    columns_to_keep.extend(COLOR_COLUMN_NAMES)
    filtered_table = table.select(columns_to_keep)

    # # only use the first X rows for testing purposes
    # filtered_table = filtered_table.slice(0, 1000)

    # remove columns where naampad starts with 'WVTEST'
    import pyarrow.compute as pc
    mask = pc.starts_with(filtered_table['naampad'], 'WVTEST')
    inverted_mask = pc.invert(mask)
    filtered_table = filtered_table.filter(inverted_mask)

    # for pyarrow table: add a column installatie that takes the first part of "naampad" up until the first "/"
    filtered_table = add_installatie(filtered_table)

    filtered_table = add_wkb(filtered_table)

    # assign colors to the table

    start = time.time()
    filtered_table = assign_colors_to_table(filtered_table)
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")

    # write to parquet file
    parquet_file = Path(__file__).parent / 'data' / 'filtered_data.parquet'
    pq.write_table(filtered_table, parquet_file)


def create_table_from_all_districts():
    # Read all CSV files from /data/csv_per_district and concatenate them, handling different schemas
    csv_dir = Path(__file__).parent / 'data' / 'csv_per_district'
    csv_files = list(csv_dir.glob('*.csv'))
    column_types = {
        'eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen': pa.float64(),
        'eigenschappen - lgc:installatie#vpconsole|eig|aantal verlichtingstoestellen': pa.float64(),
        'eigenschappen - lgc:installatie#vpbevestig|eig|aantal verlichtingstoestellen': pa.float64(),
        'eigenschappen - lgc:installatie#vvop|eig|aantal verlichtingstoestellen': pa.float64(),
        'eigenschappen - lgc:installatie#vpbevestig|eig|nummer voedingskring (VPBEVESTIG)': pa.string(),
        'eigenschappen - lgc:installatie#vpconsole|eig|nummer voedingskring (VPCONSOLE)': pa.string(),
        'eigenschappen - lgc:installatie#vvop|eig|nummer voedingskring': pa.string(),
        'eigenschappen - lgc:installatie#vplmast|eig|nummer voedingskring': pa.string(),
        'eigenschappen - lgc:installatie#vpbevestig|eig|verlichtingstoestel systeemvermogen': pa.float64(),
        'eigenschappen - lgc:installatie#vpconsole|eig|RAL kleur (VPCONSOLE)': pa.string(),
        'eigenschappen - lgc:installatie#vplmast|eig|RAL kleur (VPLMAST)': pa.string(),
        'eigenschappen - lgc:installatie#vvop|eig|RAL kleur': pa.string(),
        'eigenschappen - lgc:installatie#vplmast|eig|optiek LED': pa.string(),
        'eigenschappen - lgc:installatie#vpconsole|eig|optiek LED': pa.string(),
        'eigenschappen - lgc:installatie#vpbevestig|eig|optiek LED': pa.string(),
        'eigenschappen - lgc:installatie#vvop|eig|optiek LED': pa.string(),
        'eigenschappen - lgc:installatie#vplmast|eig|armatuurkleur': pa.string(),
        'eigenschappen - lgc:installatie#vpconsole|eig|armatuurkleur': pa.string(),
        'eigenschappen - lgc:installatie#vpbevestig|eig|armatuurkleur': pa.string(),
        'eigenschappen - lgc:installatie#vvop|eig|toestelkleur': pa.string(),
    }
    tables = []
    all_fields = {}
    # First pass: collect all column names and their types
    for file in csv_files:
        t = csv.read_csv(
            file,
            parse_options=csv.ParseOptions(delimiter='\t', newlines_in_values=True),
            convert_options=csv.ConvertOptions(null_values=[''])
        )
        for field in t.schema:
            # Use explicit type if provided, else use detected type
            if field.name in column_types:
                all_fields[field.name] = column_types[field.name]
            elif field.name not in all_fields:
                all_fields[field.name] = field.type
        tables.append(t)
    # Build a unified schema, upcasting null types to string
    safe_fields = []
    for name, typ in all_fields.items():
        if pa.types.is_null(typ):
            safe_fields.append((name, pa.string()))
        else:
            safe_fields.append((name, typ))
    unified_schema = pa.schema(safe_fields)
    # Second pass: align all tables to the unified schema (add missing columns as nulls, reorder, and cast to correct type)
    casted_tables = []
    for t in tables:
        cols = []
        for field in unified_schema:
            if field.name in t.schema.names:
                arr = t[field.name]
                # Cast to the unified type if needed
                if arr.type != field.type:
                    # If the target type is null, cast to string instead
                    target_type = field.type
                    if pa.types.is_null(target_type):
                        target_type = pa.string()
                    try:
                        arr = arr.cast(target_type)
                    except Exception as e:
                        print(f"Error casting column '{field.name}' from {arr.type} to {target_type}")
                        # Optionally, print some sample values:
                        print(f"Sample values: {arr.to_pylist()[:10]}")
                        raise
                cols.append(arr)
            else:
                # Fill missing columns with nulls of the correct type (never null type)
                fill_type = field.type if not pa.types.is_null(field.type) else pa.string()
                cols.append(pa.array([None] * t.num_rows, type=fill_type))
        casted_tables.append(pa.table(cols, schema=unified_schema))
    table = pa.concat_tables(casted_tables, promote=True)
    return table


def create_table_from_one_district():
    # Define the path to the Excel file
    excel_file = Path(__file__).parent / 'data' / 'export_20250602110914.csv'
    # Read the Excel file into a PyArrow Table
    table = csv.read_csv(excel_file, parse_options=csv.ParseOptions(delimiter='\t', newlines_in_values=True))


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


def combine_columns(table: pa.Table, from_columns: [str], to_column: str, to_type=pa.string()) -> pa.Table:
    # Only use columns that exist in the table
    existing_columns = [col for col in from_columns if col in table.schema.names]
    num_rows = table.num_rows
    if not existing_columns:
        # If none of the columns exist, create a column of None values
        combined_array = pa.array([None] * num_rows, type=to_type)
        return table.append_column(to_column, combined_array)

    if to_type == pa.float64():
        # Vectorized float handling
        arrays = [
            np.array([np.nan if v is None else v for v in table.column(col).to_numpy(zero_copy_only=False)], dtype=float)
            for col in existing_columns
        ]
        stacked = np.vstack(arrays)
        mask = ~np.isnan(stacked)
        first_non_nan_idx = mask.argmax(axis=0)
        combined = [
            float(stacked[first_non_nan_idx[i], i]) if mask[:, i].any() else None
            for i in range(stacked.shape[1])
        ]
        combined_array = pa.array(combined, type=to_type)
        return table.append_column(to_column, combined_array)

    # For other types, use generator expression for first non-None
    arrays = [table.column(col).to_pylist() for col in existing_columns]
    rows = zip(*arrays)
    combined = [next((v for v in row if v is not None), None) for row in rows]
    combined_array = pa.array(combined, type=to_type)
    return table.append_column(to_column, combined_array)


def split_columns(table: pa.Table, from_column: str, to_columns: [str], to_type=pa.string()) -> pa.Table:
    # Build a mapping from type suffix to output column
    suffix_to_col = {}
    for col in to_columns:
        # Extract the suffix after the last '#' and before the first '|'
        suffix = col.split('#')[-1].split('|')[0].lower()
        suffix_to_col[suffix] = col

    types = table.column('type').to_pylist()
    values = table.column(from_column).to_pylist()
    result = {col: [None] * len(values) for col in to_columns}

    for i, (t, v) in enumerate(zip(types, values)):
        # Extract suffix from type value
        type_suffix = t.split('#')[-1].lower()
        col = suffix_to_col.get(type_suffix)
        if col is not None:
            result[col][i] = v

    for col in to_columns:
        result[col] = pa.chunked_array([pa.array(result[col], type=to_type)])
    new_table = table
    for col in to_columns:
        new_table = new_table.append_column(col, result[col])
    return new_table



if __name__ == "__main__":
    convert_csv_to_parquet()

    # Read the parquet file
    parquet_file = Path(__file__).parent / 'data' / 'filtered_data.parquet'
    table = pq.read_table(parquet_file)

    # Remove the wkb column before writing to output.csv
    table_no_wkb = table.drop(['wkb'])

    table_no_wkb = split_columns(table_no_wkb,
        from_column='eigenschappen|eig|netwerkconfigWV1',
        to_columns=[
            'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1',
            'eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV1',
            'eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV1',
            'eigenschappen - lgc:installatie#vvop|eig|netwerkconfigWV1'
        ]
    )
    table_no_wkb = split_columns(table_no_wkb,
        from_column='eigenschappen|eig|netwerkconfigWV2',
        to_columns=[
            'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV2',
            'eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV2',
            'eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV2',
            'eigenschappen - lgc:installatie#vvop|eig|netwerkconfigWV2']
    )
    table_no_wkb = split_columns(table_no_wkb,
        from_column='eigenschappen|eig|netwerkconfigWV3',
        to_columns=[
            'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV3',
            'eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV3',
            'eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV3',
            'eigenschappen - lgc:installatie#vvop|eig|netwerkconfigWV3']
    )
    table_no_wkb = split_columns(table_no_wkb,
        from_column='eigenschappen|eig|netwerkconfigWV4',
        to_columns=[
            'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV4',
            'eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV4',
            'eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV4',
            'eigenschappen - lgc:installatie#vvop|eig|netwerkconfigWV4']
    )
    table_no_wkb = split_columns(table_no_wkb,
        from_column='eigenschappen|eig|aantal verlichtingstoestellen',
        to_columns=[
            'eigenschappen - lgc:installatie#vplmast|eig|aantal verlichtingstoestellen',
            'eigenschappen - lgc:installatie#vpconsole|eig|aantal verlichtingstoestellen',
            'eigenschappen - lgc:installatie#vpbevestig|eig|aantal verlichtingstoestellen',
            'eigenschappen - lgc:installatie#vvop|eig|aantal verlichtingstoestellen'], to_type=pa.float64()
    )

    # write table to output.csv
    output_csv = Path(__file__).parent / 'data' / 'output.csv'
    write_options = csv.WriteOptions(delimiter='\t')
    csv.write_csv(table_no_wkb, output_csv, write_options=write_options)


    start = time.time()
    score = get_score(table)
    end = time.time()
    print(f"Score: {score} Time taken: {end - start:.2f} seconds")

    # df = ga.to_geopandas(table)
    # # print resulting DataFrame, with all data for the first 7 rows
    # df = df.head(7)
    # df = df.reset_index(drop=True)
    # print(df.to_string(index=False, max_colwidth=100))

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import pyarrow.csv as csv
import geoarrow.pyarrow as ga


def convert_csv_to_parquet():
    # Define the path to the Excel file
    excel_file = Path(__file__).parent / 'data' / 'DA-2025-27826_export.csv'
    # Read the Excel file into a PyArrow Table
    table = csv.read_csv(excel_file, parse_options=csv.ParseOptions(delimiter='\t', newlines_in_values=True))
    filter_mask = pc.starts_with(table["naampad"], 'A')
    # Filter the table
    filtered_table = table.filter(filter_mask)
    # write to parquet file
    parquet_file = Path(__file__).parent / 'data' / 'filtered_data.parquet'
    pq.write_table(filtered_table, parquet_file)


if __name__ == "__main__":
    # convert_csv_to_parquet()

    # Read the parquet file
    parquet_file = Path(__file__).parent / 'data' / 'filtered_data.parquet'
    table = pq.read_table(parquet_file)

    df = ga.to_geopandas(table)

    # Function to convert WKT to GeoArrow
    def wkt_to_geoarrow(wkt):
        return pc.cast(pa.array([wkt]), pa.string())


    # Apply the conversion to each value in the 'geometry' column
    df['geoarrow'] = df['geometry'].apply(wkt_to_geoarrow)

    print(df)



import pyarrow as pa

from assign_colors import combine_columns, split_columns


def test_combine_columns():
    data = {
        'eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1': [None, None, 'Green', 'Red'],
        'eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV1': [None, 'Blue', None, None],
        'eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV1': [None, None, None, None],
    }
    table = pa.table(data)
    assert table.column('eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1').equals(
        table.column('eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV1')
    ) is False

    # Combine the two columns
    new_table = combine_columns(
        table,
        from_columns=['eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1',
                      'eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV1',
                      'eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV1'],
        to_column='eigenschappen|eig|netwerkconfigWV1'
    )

    combined_column = new_table.column('eigenschappen|eig|netwerkconfigWV1')

    assert combined_column.equals(
        pa.chunked_array([pa.array([None, 'Blue', 'Green', 'Red'], type=pa.string())])
    ) is True


def test_combine_columns_all_none():
    data = {
        'col1': [None, None, None],
        'col2': [None, None, None],
        'col3': [None, None, None],
    }
    table = pa.table(data)
    new_table = combine_columns(
        table,
        from_columns=['col1', 'col2', 'col3'],
        to_column='combined'
    )
    combined_column = new_table.column('combined')
    assert combined_column.equals(
        pa.chunked_array([pa.array([None, None, None], type=pa.string())])
    ) is True


def test_combine_columns_first_column_has_value():
    data = {
        'col1': ['A', 'B', 'C'],
        'col2': [None, 'X', None],
        'col3': [None, None, 'Y'],
    }
    table = pa.table(data)
    new_table = combine_columns(
        table,
        from_columns=['col1', 'col2', 'col3'],
        to_column='combined'
    )
    combined_column = new_table.column('combined')
    assert combined_column.equals(
        pa.chunked_array([pa.array(['A', 'B', 'C'], type=pa.string())])
    ) is True


def test_split_columns():
    data = {
        'combined': ['A', 'B', 'C'],
        'type': ['lgc:installatie#VPLMast', 'lgc:installatie#VPConsole', 'lgc:installatie#VPBevestig'],
    }
    table = pa.table(data)

    # Split the combined column into three new columns
    new_table = split_columns(
        table,
        from_column='combined',
        to_columns=['eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1',
                    'eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV1',
                    'eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV1'],
    )

    assert (new_table.column('eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1').
            equals(pa.chunked_array([pa.array(['A', None, None], type=pa.string())])) is True)
    assert (new_table.column('eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV1').
            equals(pa.chunked_array([pa.array([None, 'B', None], type=pa.string())])) is True)
    assert (new_table.column('eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV1').
            equals(pa.chunked_array([pa.array([None, None, 'C'], type=pa.string())])) is True)

def test_split_columns_all_same_type():
    data = {
        'combined': ['A', 'B', 'C'],
        'type': ['lgc:installatie#VPLMast', 'lgc:installatie#VPLMast', 'lgc:installatie#VPLMast'],
    }
    table = pa.table(data)
    new_table = split_columns(
        table,
        from_column='combined',
        to_columns=['eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1',
                    'eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV1',
                    'eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV1'],
    )
    assert (new_table.column('eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1').
            equals(pa.chunked_array([pa.array(['A', 'B', 'C'], type=pa.string())])) is True)
    assert (new_table.column('eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV1').
            equals(pa.chunked_array([pa.array([None, None, None], type=pa.string())])) is True)
    assert (new_table.column('eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV1').
            equals(pa.chunked_array([pa.array([None, None, None], type=pa.string())])) is True)

def test_split_columns_unknown_type():
    data = {
        'combined': ['A', 'B', 'C'],
        'type': ['lgc:installatie#VPLMast', 'lgc:installatie#Unknown', 'lgc:installatie#VPBevestig'],
    }
    table = pa.table(data)
    new_table = split_columns(
        table,
        from_column='combined',
        to_columns=['eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1',
                    'eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV1',
                    'eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV1'],
    )
    assert (new_table.column('eigenschappen - lgc:installatie#vplmast|eig|netwerkconfigWV1').
            equals(pa.chunked_array([pa.array(['A', None, None], type=pa.string())])) is True)
    assert (new_table.column('eigenschappen - lgc:installatie#vpconsole|eig|netwerkconfigWV1').
            equals(pa.chunked_array([pa.array([None, None, None], type=pa.string())])) is True)
    assert (new_table.column('eigenschappen - lgc:installatie#vpbevestig|eig|netwerkconfigWV1').
            equals(pa.chunked_array([pa.array([None, None, 'C'], type=pa.string())])) is True)

"""
Research project for Msc. Forensic Science

Author: Luca Nederhorst
Academic year: 2023 - 2024

"""
import pandas as pd

# load data into pandas dataframa
metadata_df = pd.read_csv('metadata_model.csv')
processed_metadata_df = metadata_df


def remove_brackets(cel_value):
    cel_value = cel_value.strip('{}')
    return cel_value

def is_number(cel_value):
    try:
        float(cel_value)
        return True
    except ValueError:
        return False


# remove {} from cel value and put back string or number
for column in processed_metadata_df.columns:
    for row_index, value in processed_metadata_df[column].items():
        value_no_brackets = remove_brackets(value)

        if is_number(value_no_brackets) == True:
            processed_metadata_df.at[row_index, column] = float(value_no_brackets)
        else:
            processed_metadata_df.at[row_index, column] = value_no_brackets

# check if there are missing values
missing_value_found = False
for column in processed_metadata_df.columns:
    for row_index, value in processed_metadata_df[column].items():
        if pd.isna(value):
            print(f'Missing value at row {row_index}, column {column}')
            missing_value_found = True

if missing_value_found == False:
    print('No missing values found.')
else:
    print('Missing value(s) found.')


        

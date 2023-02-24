import pandas as pd
import argparse
import numpy as np
import os
from pathlib import Path

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

#Enable argparse to pass None values
def none_or_str(value):
    if value == 'None':
        return None
    return value

parser = argparse.ArgumentParser("prep")
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)
#parser.add_argument('--shortlist',nargs='*', type=none_or_str, default=[])
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path

lines = [
    f"Input path: {input_path}",
    f"Output path: {output_path}",
]

for line in lines:
    print(line)


# Retrieve the current and reference datasets
input_df = pd.read_csv(input_path)
input_df = input_df.drop('Unnamed: 0', axis=1)
reference = input_df[input_df['pickup_monthday'] <= 14]
current = input_df[(input_df['pickup_monthday'] > 14) & (input_df['pickup_monthday'] <= 28)]

reference_name = "00 - Reference - First half of the month"
current_name = "01 - Current - Second half of the month"

#reference = Dataset.get_by_name(ws, name=reference_dataset).to_pandas_dataframe() # reference dataset (A)
#current = Dataset.get_by_name(ws, name=current_dataset).to_pandas_dataframe() # current dataset (B)

print("PREPROCESS DATASET AND ENCODE CATEGORICAL VARIABLES")

##################################
##### PREPROCESS CATEGORICALS ####
##################################

# -------------------------
# LABEL ENCODE 

shortlist = ['cost', 'distance', 'dropoff_latitude', 'dropoff_longitude', 'passengers', 'pickup_latitude', 'pickup_longitude', 
             'store_forward', 'vendor', 'pickup_weekday', 'pickup_monthday', 'pickup_hour', 'pickup_minute', 
             'pickup_second', 'dropoff_weekday', 'dropoff_monthday', 'dropoff_hour', 'dropoff_minute', 'dropoff_second']

# use shortlist if exists, else all columns from reference
columns = list(reference.columns) if shortlist == [] else shortlist

# identify numerical and categorical columns
# numerical_columns_selector = selector(dtype_exclude=object)
# categorical_columns_selector = selector(dtype_include=object)
# numerical_columns = numerical_columns_selector(reference[columns])
# categorical_columns = categorical_columns_selector(reference[columns])

numerical_columns = ['cost', 'distance', 'dropoff_latitude', 'dropoff_longitude', 'passengers', 'pickup_latitude', 'pickup_longitude', 'pickup_hour', 
                     'pickup_minute', 'pickup_second', 'dropoff_hour', 'dropoff_minute', 'dropoff_second', 'pickup_monthday', 'dropoff_monthday', ]

categorical_columns = ['store_forward', 'vendor', 'pickup_weekday', 'dropoff_weekday', ]


# label encoding for plots of categorical columns
categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

reference_le = categorical_transformer.fit_transform(reference[categorical_columns])
reference_le = pd.DataFrame(reference_le)
reference_le.columns = categorical_columns

current_le = categorical_transformer.transform(current[categorical_columns])
current_le = pd.DataFrame(current_le)
current_le.columns = categorical_columns

# impute missing values

if categorical_columns != []:

    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    current_le = imp_mode.fit_transform(current_le[categorical_columns])
    current_le = pd.DataFrame(current_le)
    current_le.columns = categorical_columns

    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    reference_le = imp_mode.fit_transform(reference_le[categorical_columns])
    reference_le = pd.DataFrame(reference_le)
    reference_le.columns = categorical_columns

# join categorical and numerical values back
reference_joined = pd.concat([reference[numerical_columns], reference_le], axis=1)
current_joined = pd.concat([current[numerical_columns], current_le], axis=1)


# -------------------------
# SAVE FILES

#create folder if folder does not exist already. We will save the files here
#Path(output_path).mkdir(parents=True, exist_ok=True)
print(f"Saving to{output_path}")

reference_joined = reference_joined.to_csv((Path(output_path) / f"{reference_name}_processed.csv"), index = False)
current_joined = current_joined.to_csv((Path(output_path) / f"{current_name}_processed.csv"), index = False)


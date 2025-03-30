import numpy as np
import pandas as pd

# Category to encoded value
gender = {'Male': 0,
          'Female': 1}

customer_type = {'disloyal Customer': 0,
                 'Loyal Customer': 1}

type_of_travel = {'Business travel': 0,
                  'Personal Travel': 1}

# Dummy encoding
class_type = {'Business': [0,0],
              'Eco': [0,1],
              'Eco Plus': [1,0]}

satisfaction = {'neutral or dissatisfied': 0,
                'satisfied': 1}

# Normalization for age, flight distance, arrival delay in minues, and departure delay in minutes
# to values between 0 and 5 (basically min max normalization)
def normalize(df, col):
    curr_lower = df[col].min()
    curr_upper = df[col].max()
    curr_range = curr_upper - curr_lower
    new_lower = 0
    new_upper = 5
    new_range = new_upper - new_lower

    for i in range(df.shape[0]):
        df.loc[i,col] = new_lower + (df.loc[i,col] - curr_lower) * new_range / curr_range
    return df

df = pd.read_csv('CSVs/full_dataset.csv')

# Create new columns to hold Class flags
df['Class_Flag_1'] = pd.NA
df['Class_Flag_2'] = pd.NA

# Replace category values with encoded values
for i in range(df.shape[0]):
    df.loc[i,'Gender'] = gender[df.loc[i,'Gender']]
    df.loc[i,'Customer Type'] = customer_type[df.loc[i,'Customer Type']]
    df.loc[i,'Type of Travel'] = type_of_travel[df.loc[i,'Type of Travel']]
    df.loc[i,'Class_Flag_1'] = class_type[df.loc[i,'Class']][0]
    df.loc[i,'Class_Flag_2'] = class_type[df.loc[i,'Class']][1]
    df.loc[i,'satisfaction'] = satisfaction[df.loc[i,'satisfaction']]

# Drop original Class column
df.drop('Class', axis=1, inplace=True)

df = normalize(df, 'Age')
df = normalize(df, 'Flight Distance')
df = normalize(df, 'Departure Delay in Minutes')
df = normalize(df, 'Arrival Delay in Minutes')

df.to_csv('CSVs/full_dataset_preprocessed.csv', index=False)
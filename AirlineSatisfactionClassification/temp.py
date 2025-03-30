import pandas as pd

df = pd.read_csv('CSVs/full_dataset_preprocessed.csv', usecols=['Arrival Delay in Minutes'])

print(df['Arrival Delay in Minutes'].notna().sum())
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv(r"/content/01.Data Cleaning and Preprocessing.csv")

# Print first few rows of the DataFrame
print(df.head())

# Print information about the DataFrame
print(df.info())

# Check for null values and print their sum
print(df.isnull())
print(df.isnull().sum())

# Fill null values with the mean
df.fillna(df.mean(), inplace=True)

# Print the DataFrame after filling null values
print(df)

# Check for duplicated rows and print their sum
print(df.duplicated())
print(df.duplicated().sum())

# Drop duplicated rows
df.drop_duplicates(inplace=True)

# Get non-numeric columns
non_numeric_columns = df.select_dtypes(exclude=['number']).columns

# Drop non-numeric columns
numeric_df = df.drop(columns=non_numeric_columns)

# Initialize and fit the scaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# Create a DataFrame with scaled data
scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

# Concatenate non-numeric columns with scaled DataFrame
scaled_df = pd.concat([scaled_df, df[non_numeric_columns]], axis=1)

# Print the first few rows of the scaled DataFrame
print(scaled_df.head())

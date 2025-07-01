import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# Read dataset
file_path = "raw_dataset.csv"  # You must place the file in same folder
df = pd.read_csv(file_path)

# Clean data
df = df.dropna().drop_duplicates()

# Separate features
cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=np.number).columns

# Transformer pipeline
# Transformer pipeline
ct = ColumnTransformer([
    ('scale', StandardScaler(), num_cols),
    ('encode', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
])

transformed = ct.fit_transform(df)

# Convert to DataFrame
encoded_names = ct.named_transformers_['encode'].get_feature_names_out(cat_cols)
final_cols = list(num_cols) + list(encoded_names)
df_cleaned = pd.DataFrame(transformed, columns=final_cols)

# Save output
df_cleaned.to_csv("cleaned_data.csv", index=False)
print("✅ ETL completed. Output saved as 'cleaned_data.csv'")

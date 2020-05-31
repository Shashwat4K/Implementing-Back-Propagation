import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.getcwd(), 'Input')

df = pd.read_csv(os.path.join(DATA_DIR, "iris_dataset.csv"))

print(df.head())

columns = df.columns 

unique_species = df['Species'].unique()

categorical_vectors = {unique_species[0]: [1,0,0], unique_species[1]: [0,1,0], unique_species[2]: [0,0,1]}

print("loading data...")
data = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

print("Making categorical (One hot) vectors...")
c = []
for i in df['Species'].values:
    c.append(categorical_vectors[i])

c = np.array(c)
print("Creating final data block...")
final_data = np.hstack([data, c])

print("Saving final data block as 'iris_dataset_final.csv' in {}".format(DATA_DIR))
final_iris_data = pd.DataFrame(final_data)
final_iris_data.to_csv(os.path.join(DATA_DIR, "iris_dataset_final.csv"))
print("The dataset is saved as '{}'".format(os.path.join(DATA_DIR, "iris_dataset_final.csv")))
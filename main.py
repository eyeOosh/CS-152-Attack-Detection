import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from collections import Counter

df = pd.read_csv("customer_booking.csv", encoding="ISO-8859-1")

numeric_df = df.select_dtypes(include=["number"])
categorical_df = df.select_dtypes(include=["object"])

print(numeric_df.shape)
print(categorical_df.shape)

selected_cols = ["num_passengers", "purchase_lead", "length_of_stay", "flight_duration", "flight_hour"]
subset_df = numeric_df[selected_cols]

fig, axes = plt.subplots(1, len(subset_df.columns), figsize=(16, 5)) 

for i, col in enumerate(subset_df.columns):
    counts = Counter(subset_df[col].dropna())
    x = list(counts.keys())
    y = list(counts.values())
    
    axes[i].scatter(x, y, color='skyblue', edgecolor='black')
    axes[i].set_title(col)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
    axes[i].grid(True)

plt.tight_layout()
plt.show()
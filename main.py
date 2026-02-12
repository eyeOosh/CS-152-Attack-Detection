import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

#All comments are for Alan to understand what I did so far - 2/11/26

#keep this seed consistent so we have preidctable models
SEED = 42

# data frame reading the csv file
df = pd.read_csv("customer_booking.csv", encoding="ISO-8859-1")

# seperate numerical and categorical data
numeric_df = df.select_dtypes(include=["number"])
categorical_df = df.select_dtypes(include=["object"])

# for testing
print(numeric_df.shape)
print(categorical_df.shape)


# we only want these numerical data columns for now
selected_cols = ["num_passengers", "purchase_lead", "length_of_stay", "flight_duration", "flight_hour"]
subset_df = numeric_df[selected_cols]

#simulate attack
attacked_df = subset_df.copy().astype(float)

# attack 20% of the data
attack_fraction = 0.2 
np.random.seed(SEED)
#this stores a table of if a data point is attacked or not, initally all are not
attacked_mask = pd.DataFrame(False, index=subset_df.index, columns=subset_df.columns)

for col in attacked_df.columns:
    num_rows = len(attacked_df)
    num_attack = int(num_rows * attack_fraction)
    # choose a random index to attack
    attack_indices = np.random.choice(num_rows, num_attack, replace=False)
    #make that index True in the attack_mask table
    attacked_mask.loc[attack_indices, col] = True
    #create attack here
    noise = np.random.normal(0, 5, size=num_attack)
    attacked_df.loc[attack_indices, col] += noise
    #attack is non-negative
    attacked_df[col] = np.maximum(attacked_df[col], 0)

#all plotting is here
fig, axes = plt.subplots(1, len(subset_df.columns), figsize=(20,4))

for i, col in enumerate(subset_df.columns):
    x_values = attacked_df[col]
    counts = Counter(x_values)
    freq_x = list(counts.keys())
    freq_y = list(counts.values())
    freq_colors = []
    for val in freq_x:
        indices_with_val = attacked_df.index[x_values == val]
        if attacked_mask.loc[indices_with_val, col].any():
            freq_colors.append('red')
        else:
            freq_colors.append('blue')
    axes[i].scatter(freq_x, freq_y, color=freq_colors, edgecolor='black')
    axes[i].set_title(col)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
    axes[i].grid(True)

plt.tight_layout()
plt.show()

#now training sci-kit learn if it can detect attacked and not
X = attacked_df.values.flatten().reshape(-1, 1)
y = attacked_mask.values.flatten().astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))

# This returns in the format
# [[TN, FN
#   FP, TP]]
# where TN = true negative, FN = false negative
#       FP = false positive, TP = true positive
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import torch
from sklearn.metrics import accuracy_score
import pandas as pd
import os

def detectZeroOut(X):
    """
    Checks if there are exactly 0.0 values in the dataset.
    Since Iris measurements are physical lengths, 0.0 is impossible.
    
    Returns:
        is_attacked (bool): True if a Zero-Out attack is detected.
        poison_mask (tensor): A boolean array showing exactly which rows have 0.0s.
    """
    # Check if any feature in a row equals exactly 0.0
    poison_mask = (X == 0.0).any(dim=1)
    
    # If even a single row has a 0.0, the tripwire is triggered
    is_attacked = poison_mask.any().item()
    
    return is_attacked, poison_mask


def detectFeatureNoise(X, feature_ranges):
    """
    Checks if feature values fall wildly outside natural, expected bounds.
    
    Parameters:
        X (tensor): The input features.
        feature_ranges (list of lists): [[min1, max1], [min2, max2], ...]
    """
    # Create an empty mask of False values
    poison_mask = torch.zeros(X.shape[0], dtype=torch.bool)
    
    # Loop through each of the 4 features and check their min/max bounds
    for col_idx in range(X.shape[1]):
        col_min = feature_ranges[col_idx][0]
        col_max = feature_ranges[col_idx][1]
        
        # Find rows where this specific feature is too small or too large
        out_of_bounds = (X[:, col_idx] < col_min) | (X[:, col_idx] > col_max)
        
        # Add these newly found poisoned rows to our master mask
        poison_mask = poison_mask | out_of_bounds

    # If any rows are out of bounds, the attack is detected
    is_attacked = poison_mask.any().item()
    
    return is_attacked, poison_mask

## lets run the detection functions on the evaluation dataset

feature_ranges = [[4.3, 7.9], [2.0, 4.4], [1.0, 6.9], [0.1, 2.5]]

# The list of datasets you saved in your data folder
datafiles = [
    "poisonedFeatureNoiseData.csv",
    "poisonedLabelFlipData.csv",
    "poisonedCombinedData.csv",
    "poisonedTargetedFlipData.csv",
    "goodRobustNoiseData.csv",
    "poisonedZeroOutData.csv"
]
print(f"{'='*40}")
print("DETECTING ATTACK TYPES IN EVALUATION DATASETS")
for filename in datafiles:
    print(f"{'='*40}\n")
    filepath = f"data/{filename}"
    
    if not os.path.exists(filepath):
        print(f"Skipping {filename}] - file not found.")
        continue

    df = pd.read_csv(f"data/{filename}")
    X_eval = torch.FloatTensor(df.drop('variety', axis=1).values)

    is_zero_out, zero_out_mask = detectZeroOut(X_eval)
    is_feature_noise, feature_noise_mask = detectFeatureNoise(X_eval, feature_ranges)

    print(f"Dataset: {filename}")
    print(f"  Zero-Out Attack Detected: {is_zero_out}")
    print(f"  Feature Noise Attack Detected: {is_feature_noise}")


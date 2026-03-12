import numpy as np
import pandas as pd

# Define the statistical properties of the Iris dataset (mean, standard deviation)
# Format: [sepal_length, sepal_width, petal_length, petal_width]
properties = {
    'Iris-setosa': {
        'mean': [5.006, 3.428, 1.462, 0.246],
        'std': [0.352, 0.379, 0.173, 0.105]
    },
    'Iris-versicolor': {
        'mean': [5.936, 2.770, 4.260, 1.326],
        'std': [0.516, 0.313, 0.469, 0.197]
    },
    'Iris-virginica': {
        'mean': [6.588, 2.974, 5.552, 2.026],
        'std': [0.635, 0.322, 0.551, 0.274]
    }
}

# Generate 10,000 samples (~3333 per class)
np.random.seed(42)
data = []
labels = []

for variety, stats in properties.items():
    # Generate 3333 samples for Setosa and Versicolor, 3334 for Virginica
    n_samples = 3334 if variety == 'Iris-virginica' else 3333
    
    # Generate random features from a normal distribution
    features = np.random.normal(loc=stats['mean'], scale=stats['std'], size=(n_samples, 4))
    
    # Ensure no negative values (just in case the distribution dips below 0)
    features = np.clip(features, 0.1, None)
    
    data.append(features)
    labels.extend([variety] * n_samples)

# Combine and shuffle
X_synth = np.vstack(data)
df_synth = pd.DataFrame(X_synth, columns=['sepalLength', 'spealWidth', 'petalLength', 'petalWidth'])
df_synth['variety'] = labels

# Shuffle the dataset so classes are mixed
df_synth = df_synth.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to the exact format you requested
df_synth.to_csv('iris_eval_10k.data', index=False)
print("Successfully generated 'iris_eval_10k.data' with 10,000 rows.")
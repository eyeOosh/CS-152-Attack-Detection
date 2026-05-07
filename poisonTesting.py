import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

show_plots = False

# Create a model class that inherits from nn.Module
class Model(nn.Module):
    # input layer
    # hidden layer 1
    # hidden layer 2
    # output layer

    def __init__(self, inFeatures = 4, h1 = 8, h2 = 9, outFeatures = 3):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(inFeatures, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, outFeatures)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
    torch.manual_seed(42)


#  poison_dataset
#
#  Poison parameters:
#    "feature_noise"       - adds Gaussian noise to feature values
#    "label_flip"          - randomly flips labels to a wrong class
#    "label_flip_targeted" - forces samples to one specific target class
#    "combined"            - feature noise + label flip together
#
#  Returns:
#    X_poisoned  : corrupted feature matrix 
#    y_poisoned  : corrupted labels 
#    poison_mask : True where a sample was poisoned

def poison_dataset(X, y, poison_fraction=0.3, strategy="feature_noise",
                   noise_std=15.0, target_class=None, seed=42,
                   data_grad=None, epsilon=0.5):

    rng = np.random.default_rng(seed)
    X_poisoned = X.copy().astype(float)
    y_poisoned = y.copy()

    n_samples      = len(X_poisoned)
    n_poison       = int(n_samples * poison_fraction)
    poison_indices = rng.choice(n_samples, size=n_poison, replace=False)

    poison_mask = np.zeros(n_samples, dtype=bool)
    poison_mask[poison_indices] = True

    if strategy in ("robust_noise"):
        noise = rng.normal(loc=0, scale=0.5, size=X_poisoned[poison_indices].shape)
        X_poisoned[poison_indices] += noise

    if strategy in ("feature_noise", "combined"):
        noise = rng.normal(loc=0, scale=noise_std, size=X_poisoned[poison_indices].shape)
        X_poisoned[poison_indices] += noise

    if strategy in ("zero_out", "combined"):
        X_poisoned[poison_indices] = 0.0

    if strategy in ("label_flip", "combined"):
        num_classes = len(np.unique(y_poisoned))
        for idx in poison_indices:
            original = y_poisoned[idx]
            choices  = [c for c in range(num_classes) if c != original]
            y_poisoned[idx] = rng.choice(choices)

    elif strategy == "label_flip_targeted":
        if target_class is None:
            raise ValueError("target_class must be set for label_flip_targeted.")
        for idx in poison_indices:
            if y_poisoned[idx] != target_class:
                y_poisoned[idx] = target_class

    elif strategy == "FGSM":
        if data_grad is None:
            raise ValueError("data_grad must be provided for the FGSM strategy.")
        
        # Convert PyTorch tensor to NumPy array if necessary
        grad_array = data_grad.numpy() if hasattr(data_grad, 'numpy') else data_grad
        
        # Apply FGSM perturbation: X_adv = X + epsilon * sign(gradient)
        X_poisoned[poison_indices] += epsilon * np.sign(grad_array[poison_indices])

    print(f"[poison_dataset] strategy='{strategy}' | "
          f"poisoned {n_poison}/{n_samples} samples ({poison_fraction * 100:.0f}%)")

    return X_poisoned, y_poisoned, poison_mask

model = Model()

df = pd.read_csv("data/iris.data", encoding="ISO-8859-1")

df['variety'] = df['variety'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
df['variety'] = df['variety'].astype(int)


# Train Test Split time (.values converts the dataframe to a numpy array)
X = df.drop('variety', axis=1)
y = df['variety']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

############################################################################################
# Options: "label_flip", "feature_noise", "combined", "label_flip_targeted", "robust_noise", "zero_out"
strat = input("Enter poison strategy: \n1 = feature_noise\n2 = label_flip\n3 = robust_noise\n4 = zero_out\n5=FGSM\n")
if strat == "1":
    current_strategy = "feature_noise"
elif strat == "2":
    current_strategy = "label_flip"
elif strat == "3":
    current_strategy = "robust_noise"
elif strat == "4":
    current_strategy = "zero_out"
elif strat == "5":
    current_strategy = "FGSM"
else:
    print("Invalid input, defaulting to feature_noise.")
    current_strategy = "feature_noise"

current_fraction = 0.3
############################################################################################

if current_strategy == "FGSM":
    # 1. Prepare data
    X_train_t = torch.FloatTensor(X_train.values)
    y_train_t = torch.LongTensor(y_train.values)
    X_train_t.requires_grad = True

    # 2. Load clean model
    proxy_model = Model()
    proxy_model.load_state_dict(torch.load("models/originalModel.pt"))
    proxy_model.eval()

    # 3. Calculate gradients
    proxy_criterion = nn.CrossEntropyLoss()
    out = proxy_model(X_train_t)
    loss = proxy_criterion(out, y_train_t)
    
    proxy_model.zero_grad()
    loss.backward()

    # Extract the gradient data
    gradients = X_train_t.grad.data

    # 4. Call the updated poison function
    X_train, y_train, poison_mask = poison_dataset(
        X_train.values, 
        y_train.values, 
        poison_fraction=current_fraction, 
        strategy=current_strategy, 
        seed=42,
        data_grad=gradients,  # Pass gradients here
        epsilon=0.5           # Set perturbation strength
    )
else:
    #poison the data
    X_train, y_train, poison_mask = poison_dataset(X_train.values, y_train.values, poison_fraction=current_fraction, strategy=current_strategy, seed=42)

# Convert the data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test.values)
# convert the labels to LongTensor for classification
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test.values)

#time to set model criterion to mearsure error of prediction vs reality
criterion = nn.CrossEntropyLoss()
#choose optimizer time
#lr = learning rate, how much we update the weights based on the error
#if lr is too high, we might overshoot the optimal weights, if it's too low, training will be slow
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#one epoch is one full pass through the layers (1...n)

#training model time
epochs = 150
losses = []

for i in range(epochs):
    # go forawrd and get a prediction
    y_pred = model.forward(X_train)

    # calculate the loss
    loss = criterion(y_pred, y_train) # predicted vs actual (trained) values
    losses.append(loss.detach().numpy()) # detach from the computation graph and convert to numpy for plotting

    # print every 10 epochs
    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {loss}")

    #time to back propagate the error and update the weights
    optimizer.zero_grad() # zero the gradients before backpropagation
    loss.backward() # backpropagate the error
    optimizer.step() # update the weights based on the gradients

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

with torch.no_grad(): # we don't need gradients for evaluation
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)
    print(f"\nTest Loss: {loss.item()}\n")

correct = 0
print("Network's percent confidence in its predictions (picks the highest value as the predicted class):")
print("tensor ([setosa, versicolor, virginica])")
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_value = model.forward(data)

        if y_test[i] == 0:
            name = "setosa"
        elif y_test[i] == 1:
            name = "versicolor"
        else:            
            name = "virginica"

        print(f'{i+1}) {str(y_value)} {name} \t {y_value.argmax().item() == y_test[i].item()}')

        if y_value.argmax().item() == y_test[i].item():
            correct += 1
    
print(f"\nCorrect: {correct} out of {len(X_test)} \t Accuracy: {correct/len(X_test)*100:.2f}%")

newIris = torch.tensor([4.7, 3.2, 1.3, 0.2]) # this is a setosa
print("\nNew Iris prediction (4.7, 3.2, 1.3, 0.2):")
with torch.no_grad():
    print("\n" + str(model(newIris)))
print("tensor ([setosa, versicolor, virginica])\n")

filename_map = {
    "feature_noise": "poisonedFeatureNoise.pt",
    "label_flip": "poisonedLabelFlip.pt",
    "label_flip_targeted": "poisonedTargetedFlip.pt",
    "combined": "poisonedCombined.pt",
    "robust_noise": "goodRobustNoise.pt",
    "zero_out": "poisonedZeroOut.pt",
    "FGSM": "poisonedFGSM.pt"
}

torch.save(model.state_dict(), f"models/{filename_map.get(current_strategy, 'poisonedDefault.pt')}")
print(f"\n{'='*40}")
print(f"Model saved as: models/{filename_map.get(current_strategy, 'poisonedDefault.pt')}\n")

datafile_map = {
    "feature_noise": "poisonedFeatureNoiseData.csv",
    "label_flip": "poisonedLabelFlipData.csv",
    "label_flip_targeted": "poisonedTargetedFlipData.csv",
    "combined": "poisonedCombinedData.csv",
    "robust_noise": "goodRobustNoiseData.csv",
    "zero_out": "poisonedZeroOutData.csv",
    "FGSM": "poisonedFGSMData.csv"
}

df_poisoned = pd.DataFrame(X_train.numpy(), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df_poisoned['variety'] = y_train.numpy()

data_save_name = datafile_map.get(current_strategy, 'poisonedDefaultData.csv')
df_poisoned.to_csv(f"data/{data_save_name}", index=False)

print(f"Data saved as:  data/{data_save_name}")
print(f"{'='*40}\n")

if show_plots:
    plt.show()
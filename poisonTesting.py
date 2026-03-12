import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

def poison_dataset(X, y, poison_fraction=0.2, strategy="feature_noise",
                   noise_std=0.5, target_class=None, seed=42):

    rng = np.random.default_rng(seed)
    X_poisoned = X.copy().astype(float)
    y_poisoned = y.copy()

    n_samples      = len(X_poisoned)
    n_poison       = int(n_samples * poison_fraction)
    poison_indices = rng.choice(n_samples, size=n_poison, replace=False)

    poison_mask = np.zeros(n_samples, dtype=bool)
    poison_mask[poison_indices] = True

    if strategy in ("feature_noise", "combined"):
        noise = rng.normal(loc=0, scale=noise_std, size=X_poisoned[poison_indices].shape)
        X_poisoned[poison_indices] += noise

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

    print(f"[poison_dataset] strategy='{strategy}' | "
          f"poisoned {n_poison}/{n_samples} samples ({poison_fraction * 100:.0f}%)")

    return X_poisoned, y_poisoned, poison_mask


model = Model()

df = pd.read_csv("iris.data", encoding="ISO-8859-1")

df['variety'] = df['variety'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
df['variety'] = df['variety'].astype(int)


# Train Test Split time (.values converts the dataframe to a numpy array)
X = df.drop('variety', axis=1)
y = df['variety']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#poison the data
X_train, y_train, poison_mask = poison_dataset(X_train.values, y_train.values, poison_fraction=0.2, strategy="label_flip")

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

torch.save(model.state_dict(), "modelPoisoned.pt")

newModel = Model()
newModel.load_state_dict(torch.load("modelPoisoned.pt"))

print(newModel.eval()) # set the model to evaluation mode

plt.show()
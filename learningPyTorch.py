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


model = Model()

df = pd.read_csv("iris.data", encoding="ISO-8859-1")

df['variety'] = df['variety'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
df['variety'] = df['variety'].astype(int)


# Train Test Split time (.values converts the dataframe to a numpy array)
X = df.drop('variety', axis=1)
y = df['variety']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.FloatTensor(X_train.values)
X_test = torch.FloatTensor(X_test.values)
# convert the labels to LongTensor for classification
y_train = torch.LongTensor(y_train.values)
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

with torch.no_grad():
    print("\n" + str(model(newIris)))

torch.save(model.state_dict(), "model.pt")

newModel = Model()
newModel.load_state_dict(torch.load("model.pt"))

print(newModel.eval()) # set the model to evaluation mode

plt.show()
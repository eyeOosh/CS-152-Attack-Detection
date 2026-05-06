import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, inFeatures=4, h1=8, h2=9, outFeatures=3):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(inFeatures, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, outFeatures)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

df_eval = pd.read_csv("data/newTestingData.data")
df_eval['variety'] = df_eval['variety'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

X_eval = torch.FloatTensor(df_eval.drop('variety', axis=1).values)
y_eval = torch.LongTensor(df_eval['variety'].values)

# original model
model_original = Model()
model_original.load_state_dict(torch.load("models/originalModel.pt"))
model_original.eval() 

# poisoned feature noise model
model_poisoned = Model()
model_poisoned.load_state_dict(torch.load("models/poisonedFeatureNoise.pt"))
model_poisoned.eval()

# poisoned label flip model
model_poisoned_label = Model()
model_poisoned_label.load_state_dict(torch.load("models/poisonedLabelFlip.pt"))
model_poisoned_label.eval()

# poisoned robust noise model
model_poisoned_robust = Model()
model_poisoned_robust.load_state_dict(torch.load("models/goodRobustNoise.pt"))
model_poisoned_robust.eval()

# poisoned zero out model
model_poisoned_zero = Model()
model_poisoned_zero.load_state_dict(torch.load("models/poisonedZeroOut.pt"))
model_poisoned_zero.eval()

# adverserially trained model (FGSM)
# model was already trained with FGSM adversarial examples, 
# # so we can just load it and evaluate it on the clean data and 
# the adversarial data to see how well it performs under attack
model_FGSM = Model()
model_FGSM.load_state_dict(torch.load("models/poisonedFGSM.pt"))
model_FGSM.eval()

def printBorder():
    print(f"{'='*60}")

print("\nAccuracy Scores: Aayush Kumar and Alan Xiao\n")

plot_model_names = []
plot_accuracies = []

def evaluate_model(model_name, model, X, y_true):
    print(f"EVALUATING: {model_name}")
    printBorder()

    with torch.no_grad():
        y_pred_logits = model(X)
        y_pred = y_pred_logits.argmax(dim=1).numpy()
    
    y_true_np = y_true.numpy()

    acc = accuracy_score(y_true_np, y_pred)
    f1 = f1_score(y_true_np, y_pred, average='weighted')
    
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"Overall F1 Score: {f1:.4f}\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_np, y_pred))
    printBorder()

    plot_model_names.append(model_name)
    plot_accuracies.append(acc * 100)

#FGSM Attack
# 1. Gradients calc
X_eval_attack = X_eval.clone().detach().requires_grad_(True)

# 2. Pass the data to OG model
model_original.zero_grad()
outputs = model_original(X_eval_attack)

# 3. Calculate loss
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, y_eval)

# 4. Backpropagate to get the gradient of the loss w.r.t the input data
loss.backward()
data_grad = X_eval_attack.grad.data

# 5. Create the adversarial examples (X_adv = X + epsilon * sign(grad))
epsilon = 0.5  # Increase this to make the attack stronger (e.g., 1.0 or 2.0)
X_eval_adversarial = X_eval_attack + epsilon * torch.sign(data_grad)

# Detach from the graph so we can use it just like regular data in our evaluate function
X_eval_adversarial = X_eval_adversarial.detach()
# =====================================================================

# Run the Standard Evaluations on CLEAN data
printBorder()
print("This is the original model, trained on clean data, evaluated on clean data:")
evaluate_model("ORIGINAL MODEL (Clean Data)", model_original, X_eval, y_eval)

# Run the Evasion Attack Evaluations on ADVERSARIAL data
printBorder()
print("This is the original model, trained on clean data, evaluated on adversarial FGSM data:")
evaluate_model("ORIGINAL MODEL (Under FGSM Attack)", model_original, X_eval_adversarial, y_eval)

printBorder()
print("This is the poisoned feature noise model, trained against poisoned data, evaluated on FGSM data:")
evaluate_model("POISONED FEATURE NOISE MODEL", model_poisoned, X_eval_adversarial, y_eval)

printBorder()
print("This is the poisoned label flip model, trained against poisoned data, evaluated on FGSM data:")
evaluate_model("POISONED LABEL FLIP MODEL", model_poisoned_label, X_eval_adversarial, y_eval)

printBorder()
print("This is the poisoned zero out model, trained against poisoned data, evaluated on FGSM data:")
evaluate_model("POISONED ZERO OUT MODEL", model_poisoned_zero, X_eval_adversarial, y_eval)

printBorder()
print("This is the adversarially trained model (FGSM), trained against adversarial FGSM examples, evaluated on adversarial FGSM data:")
evaluate_model("FGSM-TrainedMODEL (Under FGSM Attack)", model_FGSM, X_eval_adversarial, y_eval)

plt.figure(figsize=(12, 8))

# Create a horizontal bar chart
bars = plt.barh(plot_model_names[::-1], plot_accuracies[::-1], color='skyblue', edgecolor='black')

plt.xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Model Accuracy Comparison: Clean vs. Poisoned vs. Under Attack', fontsize=14, fontweight='bold')
plt.xlim(0, 105)

for bar in bars:
    width = bar.get_width()
    label_x_pos = width - 6 if width > 10 else width + 1
    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
             f'{width:.2f}%', 
             va='center', ha='left' if width <= 10 else 'right', 
             color='black', weight='bold', fontsize=10)

plt.tight_layout()

plt.show()
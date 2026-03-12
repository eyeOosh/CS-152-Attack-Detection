import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


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

def evaluate_model(model_name, model, X, y_true):
    print(f"\n{'='*40}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*40}")
    
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

# Run the Evaluation
evaluate_model("ORIGINAL MODEL", model_original, X_eval, y_eval)
evaluate_model("POISONED FEATURE NOISE MODEL", model_poisoned, X_eval, y_eval)
evaluate_model("POISONED LABEL FLIP MODEL", model_poisoned_label, X_eval, y_eval)
evaluate_model("POISONED ZERO OUT MODEL", model_poisoned_zero, X_eval, y_eval)
evaluate_model("GOOD ROBUST NOISE MODEL", model_poisoned_robust, X_eval, y_eval)

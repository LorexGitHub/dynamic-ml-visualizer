import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import io
import base64

class CancerPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.pca = None
        self.is_trained = False
        self.feature_names = None
        self.X_train_pca, self.X_test_pca = None, None
        self.y_train, self.y_test = None, None
        self.history = {"losses": [], "accuracies": []}

    def load_data(self):
        data = load_breast_cancer()
        self.feature_names = list(data.feature_names)
        df = pd.DataFrame(data.data, columns=self.feature_names)
        X, y = df.values, data.target
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        X_train_sc = self.scaler.fit_transform(self.X_train)
        self.X_test_sc = self.scaler.transform(self.X_test)
        
        self.pca = PCA(n_components=2)
        self.X_train_pca = self.pca.fit_transform(X_train_sc)
        self.X_test_pca = self.pca.transform(X_test_sc)
        
        return {"features": self.feature_names, "train_size": len(self.y_train)}

    def train(self, epochs=50, lr=0.01, hidden_dim=32):
        input_dim = self.X_train.shape[1]
        self.model = CancerPredictor(input_dim, hidden_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        loader = DataLoader(TensorDataset(
            torch.FloatTensor(self.scaler.transform(self.X_train)), 
            torch.FloatTensor(self.y_train).unsqueeze(1)
        ), batch_size=32, shuffle=True)
        
        self.history = {"losses": [], "accuracies": []}
        for _ in range(epochs):
            self.model.train()
            epoch_loss, correct, total = 0.0, 0, 0
            for X_b, y_b in loader:
                optimizer.zero_grad()
                out = self.model(X_b)
                loss = criterion(out, y_b)
                loss.backward(); optimizer.step()
                epoch_loss += loss.item()
                correct += ((out >= 0.5).float() == y_b).sum().item()
                total += y_b.size(0)
            self.history["losses"].append(epoch_loss / len(loader))
            self.history["accuracies"].append(correct / total)
            
        self.model.eval()
        with torch.no_grad():
            preds = (self.model(torch.FloatTensor(self.scaler.transform(self.X_test))) >= 0.5).numpy().flatten()
        
        self.is_trained = True
        return float(accuracy_score(self.y_test, preds)), confusion_matrix(self.y_test, preds).tolist()

    def predict(self, features):
        if not self.is_trained: return None
        self.model.eval()
        with torch.no_grad():
            prob = self.model(torch.FloatTensor(self.scaler.transform([features]))).item()
        pca_coords = self.pca.transform(self.scaler.transform([features]))[0].tolist()
        return {"prediction": "Benign" if prob >= 0.5 else "Malignant", "probability": prob, "pca": pca_coords}

    def _fig_to_b64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def get_pca_plot(self):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(self.X_train_pca[:,0], self.X_train_pca[:,1], c=self.y_train, cmap='coolwarm', alpha=0.6, label='Train')
        ax.scatter(self.X_test_pca[:,0], self.X_test_pca[:,1], c=self.y_test, cmap='coolwarm', marker='^', edgecolors='black', label='Test')
        ax.set_title("PCA of Breast Cancer Features"); ax.legend()
        return self._fig_to_b64(fig)

    def get_training_plot(self):
        if not self.history["losses"]: return None
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history["losses"], color='red'); ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
        ax2.plot(self.history["accuracies"], color='green'); ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch")
        return self._fig_to_b64(fig)

    def get_confusion_plot(self, cm):
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Malignant','Benign']); ax.set_yticklabels(['Malignant','Benign'])
        ax.set_ylabel('True'); ax.set_xlabel('Predicted')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="white" if cm[i][j] > cm.max()/2 else "black", fontsize=16)
        return self._fig_to_b64(fig)

manager = ModelManager()
manager.load_data()
# Developing a Neural Network Classification Model

### NAME : AVINASH T
### REG NO: 212223230026
## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load the Iris dataset, split it into train-test sets, standardize features, and convert them to tensors.

### STEP 2: 
Use TensorDataset and DataLoader for efficient data handling.

### STEP 3: 
Define IrisClassifier with input, hidden (ReLU), and output layers.

### STEP 4: 
Train using CrossEntropyLoss and Adam optimizer for 100 epochs.

### STEP 5: 
Evaluate with accuracy, confusion matrix, and classification report, and visualize results.

### STEP 6: 
Predict a sample input and compare with the actual class.

## PROGRAM

### Name: AVINASH T

### Register Number: 212223230026

```python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels (already numerical)
print(iris)

# Convert to DataFrame for easy inspection
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df

# Display first and last 5 rows
print("First 5 rows of dataset:\n", df.head())
print("\nLast 5 rows of dataset:\n", df.tail())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define Neural Network Model
class IrisClassifier(nn.Module):
    def __init__(self, input_size):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)  # Changed input size to 16
        self.fc3 = nn.Linear(8, 3)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.fc3(x)
# Training function
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs=model(X_batch)
            loss=criterion(outputs,y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Initialize the Model, Loss Function, and Optimizer
model = IrisClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=100)

# Evaluate the model
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

# Compute metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=iris.target_names)

# Print details
print("\nName: AVINASH T")
print("Register No: 212223230026")
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```
### Dataset Information

<img width="868" height="589" alt="image" src="https://github.com/user-attachments/assets/5ec726e3-63e2-4df0-8ab5-4e647d08c8a4" />


### OUTPUT

## Confusion Matrix

<img width="877" height="230" alt="image" src="https://github.com/user-attachments/assets/f674e9cd-0efb-4cad-90ac-1cabcaab551d" />

<img width="873" height="165" alt="image" src="https://github.com/user-attachments/assets/615a66d8-aee7-4a9b-b2d0-008f7ba1589f" />

<img width="894" height="543" alt="image" src="https://github.com/user-attachments/assets/4ff403fc-edc3-46ba-8cd4-323929b7348b" />

## Classification Report
<img width="866" height="240" alt="image" src="https://github.com/user-attachments/assets/46414d67-95c5-4b01-a22c-5a2cc25a17fd" />

### New Sample Data Prediction

<img width="878" height="106" alt="image" src="https://github.com/user-attachments/assets/baaf10ff-4f08-4a13-960e-866434509b6d" />

## RESULT

Thus, a neural network classification model has been successfully built.

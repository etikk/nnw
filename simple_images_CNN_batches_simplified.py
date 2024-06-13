# %%
# BLOCK 1

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
from datetime import datetime


# %%
# BLOCK 2

import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch

class NumpyDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.files = [f for f in self.files if self._get_data_length(f) == 10000]
        print(f"NR1: Found {len(self.files)} files with 10000 entries in {data_dir}")
       
    def __len__(self):
        return len(self.files)
   
    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        images = data['images']
        labels = data['labels']
        return torch.tensor(images, dtype=torch.float32).unsqueeze(1), torch.tensor(labels, dtype=torch.long)
   
    def _get_data_length(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        return len(data['images'])

# %%
# BLOCK 3

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, conv1_filters=32, conv2_filters=64, conv3_filters=128, fc1_units=128, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size=3, padding=1)  # Changed to 1 channel for grayscale
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(conv3_filters * 32 * 32, fc1_units)  # Adjusted the size accordingly
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc1_units, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# %%
# BLOCK 4

def train_model(model, criterion, optimizer, train_files, val_files, num_epochs=20, batch_size=64):
    model.train()
    for epoch in tqdm(range(num_epochs), desc='Training'):
        for file in train_files:
            data = np.load(file, allow_pickle=True)
            images = torch.tensor(data['images'], dtype=torch.float32).unsqueeze(1)
            labels = torch.tensor(data['labels'], dtype=torch.long)
           
            dataset = TensorDataset(images, labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
           
            for batch_images, batch_labels in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
   
    model.eval()
    val_predictions = []
    val_true = []
    with torch.no_grad():
        for file in val_files:
            data = np.load(file, allow_pickle=True)
            images = torch.tensor(data['images'], dtype=torch.float32).unsqueeze(1)
            labels = torch.tensor(data['labels'], dtype=torch.long)
           
            dataset = TensorDataset(images, labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
           
            for batch_images, batch_labels in dataloader:
                outputs = model(batch_images)
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.numpy())
                val_true.extend(batch_labels.numpy())
   
    accuracy = accuracy_score(val_true, val_predictions)
    precision = precision_score(val_true, val_predictions, average='macro')
    recall = recall_score(val_true, val_predictions, average='macro')
    f1 = f1_score(val_true, val_predictions, average='macro')
   
    return model, accuracy, precision, recall, f1

# %%
# BLOCK 6

# Load data directly from files
train_files = [os.path.join('./npz_results/train', f) for f in os.listdir('./npz_results/train') if f.endswith('.npz')]
val_files = [os.path.join('./npz_results/val', f) for f in os.listdir('./npz_results/val') if f.endswith('.npz')]
test_files = [os.path.join('./npz_results/test', f) for f in os.listdir('./npz_results/test') if f.endswith('.npz')]

# Cap the dataset at 50%
train_files = train_files[:len(train_files) // 3]
val_files = val_files[:len(val_files) // 3]
test_files = test_files[:len(test_files) // 3]

def print_first_entry(files):
    if len(files) > 0:
        data = np.load(files[0], allow_pickle=True)
        images = data['images']
        labels = data['labels']
       
        # Print the first entry of images and labels
        print("NR13: First entry of images:")
        print(images[0])
        print("NR14: Shape of first entry:", images[0].shape)
       
        print("\nNR15: First entry of labels:")
        print(labels[0])

# Call the function for one of the directories
print_first_entry(train_files)

# %%
# BLOCK 8

# Use the best hyperparameters from the study
# best_hyperparameters = best_trial.params
best_hyperparameters = {'conv1_filters': 48, 'conv2_filters': 64, 'conv3_filters': 128, 'fc1_units': 128, 'dropout_rate': 0.3}

# Train the model with full data using the best hyperparameters
model = CNNModel(**best_hyperparameters)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model, accuracy, precision, recall, f1 = train_model(model, criterion, optimizer, train_files, val_files, num_epochs=20, batch_size=64)

print("NR16: Final training with best hyperparameters:")
print(f"  Accuracy: {accuracy}")
print(f"  Precision: {precision}")
print(f"  Recall: {recall}")
print(f"  F1 Score: {f1}")

now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
filename = f"best_simple_images_cnn_model_{timestamp}.pth"
print(f"NR17: Model will be saved as: {filename}")

torch.save(model, filename)
print(f"NR18: Best model saved to '{filename}'.")


# %%
# BLOCK 9 - Testing the model on the test set

def evaluate_model_on_test_set(model, test_files, batch_size=64):
    model.eval()
    test_predictions = []
    test_true = []
    with torch.no_grad():
        for file in test_files:
            data = np.load(file, allow_pickle=True)
            images = torch.tensor(data['images'], dtype=torch.float32).unsqueeze(1)
            labels = torch.tensor(data['labels'], dtype=torch.long)
           
            dataset = TensorDataset(images, labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
           
            for batch_images, batch_labels in dataloader:
                outputs = model(batch_images)
                _, predicted = torch.max(outputs.data, 1)
                test_predictions.extend(predicted.numpy())
                test_true.extend(batch_labels.numpy())
   
    accuracy = accuracy_score(test_true, test_predictions)
    precision = precision_score(test_true, test_predictions, average='macro')
    recall = recall_score(test_true, test_predictions, average='macro')
    f1 = f1_score(test_true, test_predictions, average='macro')
   
    print("NR19: Test set evaluation results:")
    print(f"  Accuracy: {accuracy}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1 Score: {f1}")
   
    return accuracy, precision, recall, f1

# Load the best model
model = torch.load(filename)

# Evaluate the model on the test set
evaluate_model_on_test_set(model, test_files)


# %%


# BLOCK 10

import torch
import torch.nn as nn

# Define a function to print model metrics
def print_model_metrics(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print("Layer details:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()}")

# Load the models

# pure_metrics_model_path = 'best_pure_metrics_cnn_model_20240602_141132.pth'
simple_images_model_path = 'best_simple_images_cnn_model_20240605_214245.pth'

# pure_metrics_model = torch.load(pure_metrics_model_path)
simple_images_model = torch.load(simple_images_model_path)

# Print metrics for both models
# print_model_metrics(pure_metrics_model, "Pure Metrics CNN Model")
# print("\n" + "="*50 + "\n")
print_model_metrics(simple_images_model, "Simple Images CNN Model")
# %%

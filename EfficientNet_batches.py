
# %%

# BLOCK 1

import time
import torch.cuda.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
from datetime import datetime
from torchvision import models, transforms

# %%

# BLOCK 2

class NumpyDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.files = file_list
        self.files = [f for f in self.files if self._get_data_length(f) == 10000]
        self.transform = transform
        print(f"NR1: Found {len(self.files)} files with 10000 entries")
       
    def __len__(self):
        return len(self.files) * 10000
   
    def __getitem__(self, idx):
        file_idx = idx // 10000
        image_idx = idx % 10000
    
        data = np.load(self.files[file_idx], allow_pickle=True)
        image = data['images'][image_idx]
        label = data['labels'][image_idx]
    
        image = image.reshape(256, 256, 1)  # Reshape to (256, 256, 1) for grayscale images
        if self.transform:
            image = self.transform(image)
    
        return image, label
   
    def _get_data_length(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        return len(data['images'])

# Define the transformations for the input images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust normalization for grayscale images
])

# %%

# BLOCK 3

# Load the pretrained EfficientNet-B0 model
model = models.efficientnet_b0(pretrained=True)

# Modify the first layer to accept grayscale images
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

# Modify the last layer for binary classification
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

# %%

# BLOCK 4

def train_model(model, criterion, optimizer, train_files, val_files, num_epochs=20, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_dataset = NumpyDataset(train_files, transform=transform)
    val_dataset = NumpyDataset(val_files, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    scaler = amp.GradScaler()  # Initialize GradScaler for mixed precision training

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for batch_images, batch_labels in progress_bar:
            # Timer start
            start_time = time.time()
            
            # Data loading
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            loading_time = time.time() - start_time
            
            # Model forward and backward pass with mixed precision
            optimizer.zero_grad()
            with amp.autocast():  # Use mixed precision
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            execution_time = time.time() - start_time - loading_time
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/len(progress_bar), loading_time=loading_time, execution_time=execution_time)
   
    model.eval()
    val_predictions = []
    val_true = []
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            _, predicted = torch.max(outputs.data, 1)
            val_predictions.extend(predicted.cpu().numpy())
            val_true.extend(batch_labels.cpu().numpy())
   
    accuracy = accuracy_score(val_true, val_predictions)
    precision = precision_score(val_true, val_predictions, average='macro')
    recall = recall_score(val_true, val_predictions, average='macro')
    f1 = f1_score(val_true, val_predictions, average='macro')
   
    return model, accuracy, precision, recall, f1

# %%

# BLOCK 5

# Load data directly from files
train_files = [os.path.join('./npz_results/train', f) for f in os.listdir('./npz_results/train') if f.endswith('.npz')]
val_files = [os.path.join('./npz_results/val', f) for f in os.listdir('./npz_results/val') if f.endswith('.npz')]
test_files = [os.path.join('./npz_results/test', f) for f in os.listdir('./npz_results/test') if f.endswith('.npz')]

# Cap the dataset at 50%
train_files = train_files[:len(train_files) // 1]
val_files = val_files[:len(val_files) // 1]
test_files = test_files[:len(test_files) // 1]

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

# BLOCK 6

# Train the model with full data using the best hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

model, accuracy, precision, recall, f1 = train_model(model, criterion, optimizer, train_files, val_files, num_epochs=20, batch_size=32)

print("NR16: Final training with best hyperparameters:")
print(f"  Accuracy: {accuracy}")
print(f"  Precision: {precision}")
print(f"  Recall: {recall}")
print(f"  F1 Score: {f1}")

now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
filename = f"best_efficientnetb0_model_{timestamp}.pth"
print(f"NR17: Model will be saved as: {filename}")

torch.save(model.state_dict(), filename)
print(f"NR18: Best model saved to '{filename}'.")

# %%

# BLOCK 7 - Testing the model on the test set

def evaluate_model_on_test_set(model, test_files, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    test_dataset = NumpyDataset(test_files, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_predictions = []
    test_true = []
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            _, predicted = torch.max(outputs.data, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_true.extend(batch_labels.cpu().numpy())
   
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
model.load_state_dict(torch.load(filename))

# Evaluate the model on the test set
evaluate_model_on_test_set(model, test_files)

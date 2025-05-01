import torch
import torch.optim as optim

import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import math

#
# Transfer learning from Resnet
# https://pytorch.org/hub/pytorch_vision_resnet/
# pre-processing code
# https://github.com/masoudnick/Brain-Tumor-MRI-Classification/blob/main/Preprocessing.py
#

# transformation explained on the link above
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir = './data/Training'
test_dir = './data/Testing'

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

num_classes = len(train_dataset.classes)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# using resnet 18 for performance reasons
# other deeper resnet models would be beneficial but would take too long for my computer
model = models.resnet18(pretrained=True)

# Set up model for transfer learning
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

if os.path.exists('best_model_weights.pth'):
    print("Loading existing best model weights...")
    model.load_state_dict(torch.load('best_model_weights.pth', map_location=device))
    model = model.to(device)

print("Starting training...")

n_epochs = 10
best_loss = math.inf

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0

    # each epoch requires us to go over all the images in teh training set
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # forward and backwards step through the model
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    #  Since we plan on using this model for our test, we save our highest quality model
    # every time the average loss is lower than a the previous best loss, we will save the the model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_model_weights.pth')
        print(f"New best model saved with loss {best_loss}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        is_correct = predicted == labels
        correct += (is_correct).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

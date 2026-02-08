import torch
import torch.nn as nn
import torch.optim as optim
from ..cnn_pipeline.models.cnn import SimpleCNN

def train_cnn_model(train_loader, dataset, num_classes, device):
    model = SimpleCNN(num_classes).to(device) # nclasses=len(dataset.label_map)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(16):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

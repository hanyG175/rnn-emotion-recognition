import torch
import torch.nn as nn
import torch.optim as optim
from .rnn import TextClassifier
from .cnn import SimpleCNN

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


def train_rnn_model(train_dataloader, vocab_size, embed_dim, hidden_dim, num_classes, device):
    model = TextClassifier(vocab_size, embed_dim, hidden_dim, num_classes, dropout_prob=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.01)

    for epoch in range(14):
        for text, label in train_dataloader:
            text, label = text.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

def save_model(model, path):

    # Save the model's state dictionary
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
def load_model(model_class, path, *args, **kwargs):
    # Initialize the model architecture
    model = model_class(*args, **kwargs)
    
    # Load the state dictionary into the model
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    
    return model
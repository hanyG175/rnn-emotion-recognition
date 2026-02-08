import torch
import torch.nn as nn
import torch.optim as optim
from ..rnn_pipeline.models.rnn import TextClassifier

def train_rnn_model(train_dataloader, vocab_size, embed_dim, hidden_dim, num_classes, lr=0.01, num_epochs=10, device="cpu"):
    model = TextClassifier(vocab_size, embed_dim, hidden_dim, num_classes, dropout_prob=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for text, label in train_dataloader:
            text, label = text.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

    return model

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
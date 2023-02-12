import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class WineNetwork(nn.Module):
    def __init__(self):
        super(WineNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(11, 256).double(),
            nn.ReLU(),
            nn.Linear(256, 64).double(),
            nn.ReLU(),
            nn.Linear(64, 32).double(),
            nn.ReLU(),
            nn.Linear(32, 9).double()
        )

    def forward(self, x):
        return self.network(x)

class PumpkinNetwork(nn.Module):
    def __init__(self):
        super(PumpkinNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12, 128).double(),
            nn.ReLU(),
            nn.Linear(128, 64).double(),
            nn.ReLU(),
            nn.Linear(64, 2).double()
        )

    def forward(self, x):
        return self.network(x)


def train_loop(epoch, data, model, optimizer, criterion):
    total_loss = 0.0
    total_data = 0.0
    for i, (data, target) in enumerate(data):

        optimizer.zero_grad()
        model.train()
        out = model(data)
        loss = criterion(out, target)

        total_loss = total_loss + loss.detach()
        total_data = total_data + len(target)

        loss.backward()
        optimizer.step()
    
    return total_loss/total_data
        

def validation_loop(epoch, data, model, criterion):
    total_loss = 0.0
    total_data = 0.0
    for i, (data, target) in enumerate(data):
        model.eval()

        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)

            total_loss = total_loss + loss.detach()
            total_data = total_data + len(target)

    return total_loss/total_data


def train(model, train_x, train_y, val_x, val_y, batch_size, learning_rate, weight_decay, epochs):
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), (torch.from_numpy(np.ravel(train_y)))), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), (torch.from_numpy(np.ravel(val_y)))), batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    training_loss = []
    validation_loss = []
    for epoch in range(epochs):
        avg_loss = train_loop(epoch, train_loader, model, optimizer, nn.CrossEntropyLoss())
        training_loss.append(avg_loss)
        avg_loss = validation_loop(epoch, val_loader, model, nn.CrossEntropyLoss())
        validation_loss.append(avg_loss)

    return model, training_loss, validation_loss

def get_predictions(model, x):
    with torch.no_grad():
        predicted = torch.argmax(F.softmax(model(torch.from_numpy(x.values))), dim=1)
        return predicted
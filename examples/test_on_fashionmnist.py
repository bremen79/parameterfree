import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.optim as optim
import numpy as np
import random

from parameterfree import KT, COCOB

class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()

        self.fc1 = nn.Linear(28*28, 1000)  # Input layer (28x28 = 784 pixels)
        self.fc2 = nn.Linear(1000, 1000)  # First hidden layer
        self.fc3 = nn.Linear(1000, 10)  # Output layer (10 classes for FashionMNIST)
    
    def forward(self, x):
        x = nn.Flatten()(x)
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

 
def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # training parameters
    batch_size=100
    weight_decay=0.0
    max_epochs=30
    loss_function = nn.CrossEntropyLoss()

    # set random seed so that each time we run it we get the same results
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)

    # Download or load FashionMNIST
    train_data = FashionMNIST(root="./data", train=True, transform=ToTensor(), download=True)
    test_data = FashionMNIST(root="./data", train=False, transform=ToTensor())

    # Setup data loader
    train_dataset = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dataset = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    
    # create model
    model = FashionMNISTModel()
    model.to(device)

    # choose optimizer
    optimizer = COCOB(model.parameters(), weight_decay=weight_decay)

    tot_loss = 0
    tot_acc = 0
    total = 0
    num_batch = 0

    # training loop
    for epoch in range(max_epochs):
        for batch_idx, (data, target) in enumerate(train_dataset):
            data, target = data.to(device), target.to(device)
                                            
            # Forward pass
            output = model(data)
            loss = loss_function(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
                        
            optimizer.step()
            
            total += target.size(0)
            tot_loss += loss.item()
            _, predicted = torch.max(output, 1)
            tot_acc += (predicted==target).sum().item()
            num_batch += 1

            # Print statistics every 100 epochs
            if batch_idx % 100== 0:
                print(f"Epoch {epoch + 1}/{max_epochs}, Batch {batch_idx}/{len(train_dataset)}, Minibatch Loss: {loss.item():.3f}, Online Loss: {tot_loss/num_batch:.3f}, Online Acc: {tot_acc/total:.3f}")

        model.eval()
        correct_tr = 0
        total_tr = 0
        tot_loss_tr = 0
        num_iter_tr = 0
        # Testing on training data after each epoch
        with torch.no_grad():
            for data, target in train_dataset:
                data, target = data.to(device), target.to(device)

                # Forward pass
                output = model(data)
                
                loss = loss_function(output, target)
                _, predicted = torch.max(output, 1)
                total_tr += target.size(0)
                correct_tr += (predicted == target).sum().item()
                tot_loss_tr += loss.item()
                num_iter_tr += 1

        correct_te = 0
        total_te = 0
        tot_loss_te = 0
        num_iter_te = 0
        # Testing on training data after each epoch
        with torch.no_grad():
            for data, target in test_dataset:
                data, target = data.to(device), target.to(device)

                # Forward pass
                output = model(data)

                # Calculate accuracy
                loss = loss_function(output, target)
                _, predicted = torch.max(output, 1)
                total_te += target.size(0)                
                correct_te += (predicted == target).sum().item()
                tot_loss_te += loss.item()
                num_iter_te += 1

        print(f"Train accuracy {correct_tr/total_tr:.3f}, train loss: {tot_loss_tr/num_iter_tr:.3f}")
        print(f"Test accuracy {correct_te/total_te:.3f}, test loss: {tot_loss_te/num_iter_te:.3f}")

if __name__ == '__main__':
    main()


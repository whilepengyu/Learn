from __future__ import print_function
import argparse
import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

import wandb
# import ssl


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


def train(config, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    wandb.log({"Train Loss": avg_train_loss}, step=epoch)


def validate(config, model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            val_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / len(val_loader.dataset)
    
    wandb.log({
        "Val Accuracy": accuracy,
        "Val Loss": avg_val_loss
    })

    
wandb.init(project="learn-wandb")
wandb.watch_called = False

config = wandb.config
config.batch_size = 16
config.val_batch_size = 16
config.epochs = 15  
config.lr = 0.1
config.momentum = 0.1
config.no_cuda = False
config.seed = 42
config.log_interval = 10

def main():
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(config.seed)
    numpy.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # ssl._create_default_https_context = ssl._create_unverified_context

    train_loader = DataLoader(datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    ), batch_size=config.batch_size, shuffle=True, **kwargs)

    val_loader = DataLoader(datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    ), batch_size=config.batch_size, shuffle=False, **kwargs)

    model = Net().to(device)
    criterion = F.cross_entropy
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    wandb.watch(model, log="all")
    for epoch in range(1, config.epochs + 1):
        train(config, model, device, train_loader, criterion, optimizer, epoch)
        validate(config, model, device, val_loader, criterion)

    torch.save(model.state_dict(), 'model.h5')
    wandb.save('model.h5')


if __name__ == '__main__':
    main()

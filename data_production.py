# https://github.com/pytorch/examples/blob/main/mnist/main.py

from random import shuffle
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os
from uuid import uuid4
import numpy as np
from datetime import datetime

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

mnist_mean, mnist_std = (0.1307,), (0.3081,)
mnist_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mnist_mean, mnist_std)
    ])
mnist_train = datasets.MNIST('../data', train=True, download=True,
                    transform=mnist_transform)

mnist_train, mnist_val = random_split(mnist_train, [int(.9*len(mnist_train)),int(.1*len(mnist_train))], generator=torch.Generator().manual_seed(10708))
# mnist_test = datasets.MNIST('../data', train=False,
#                     transform=mnist_transform)

# Training params
batch_size = 64
epochs = 15
lr = 1e-3
repetitions = 

train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(mnist_val,batch_size=batch_size)

if not os.path.exists('../filters'):
    os.makedirs('../filters')

start_training = datetime.now()
for repetition in range(repetitions):

    net = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=5, stride=2, bias=False),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2304, 10)
    ).to(device)

    optimizer = optim.Adadelta(net.parameters(), lr=lr)

    # https://github.com/pytorch/examples/blob/main/mnist/main.py

    net.train()
    for epoch in range(epochs):
        
        net.train()
        print(f'Repetition {repetition+1} of {repetitions}, Epoch {epoch+1} of {epochs}, {datetime.now() - start_training}')
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()


        net.eval()
        num_correct, num_all = 0, 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = net(data)
                preds = output.argmax(dim=1)
                num_correct += np.count_nonzero(target.cpu().numpy() == preds.cpu().numpy())
                num_all += len(target)

        acc = num_correct / num_all
        print(f'Accuracy Val: {round(acc, 4)}')

    torch.save(net.to('cpu').state_dict(), '../filters/' + str(uuid4()))
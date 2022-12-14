# Training params
from random import shuffle
import uuid
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os
from uuid import uuid4
from scipy.special import logsumexp

import numpy as np
from datetime import datetime

# datapath = os.path.join('..', 'data')
# filterpath = os.path.join(datapath, 'filters-complete', '8_19')
datapath = os.path.join('../', 'data')
filterpath = os.path.join(datapath, '8_19')
num_filters = 8
savepath = 'save_baselines_random_' + str(num_filters) + '.pickle'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

mnist_mean, mnist_std = (0.1307,), (0.3081,)
mnist_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mnist_mean, mnist_std)
    ])
mnist_train = datasets.MNIST('../../data', train=True, download=True,
                    transform=mnist_transform)

mnist_train, mnist_val = random_split(mnist_train, [int(.9*len(mnist_train)),int(.1*len(mnist_train))], generator=torch.Generator().manual_seed(10708))
mnist_test = datasets.MNIST('../../data', train=False,
                    transform=mnist_transform)

baseline_sample_counts = [8]
baseline_performances = {
    'random': {
        'acc': [],
        'loss': []
    }
}
for count in baseline_sample_counts:
    baseline_performances[f'random_{count}'] = {}
    baseline_performances[f'random_{count}']['acc'] = []
    baseline_performances[f'random_{count}']['loss'] = []

uuids = os.listdir(filterpath)

batch_size = 64
train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(mnist_val,batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=True)

lr = 1e-1
repetitions = 10
count_linear_layer_map = {}
for key in baseline_sample_counts:
    count_linear_layer_map[key] = int(2304*(key/16))

vmin, vmax = -2.5, 2.5

start_training = datetime.now()
for repetition in range(repetitions):
    for count in baseline_sample_counts:
        # Sample full baseline

        net = nn.Sequential(
            nn.Conv2d(1, count, kernel_size=5, stride=2, bias=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(count_linear_layer_map[count], 10)
        ).to(device)

        with torch.no_grad():
            for c in range(count):
                val = (vmin - vmax) * torch.rand(5, 5) + vmax
                net[0].weight[c,:,:,:] =  torch.Tensor(val).to(device)

        optimizer = optim.Adadelta(net.parameters(), lr=lr)

        # https://github.com/pytorch/examples/blob/main/mnist/main.py

        net.train()
        val_losses = []
        val_accs = []
        epoch = 0
        while True:
            net.train()
            for layer in net[0:3]:
                layer.requires_grad = False
            net[3].requires_grad = True
            print(f'Repetition {repetition+1} of {repetitions}, Count of filters {count}, Epoch {epoch+1} , {datetime.now() - start_training}')

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = net(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

            net.eval()
            num_correct, num_all, val_loss_l, val_acc = 0, 0, [], 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    data, target = data.to(device), target.to(device)
                    output = net(data)
                    preds = output.argmax(dim=1)
                    num_correct += np.count_nonzero(target.cpu().numpy() == preds.cpu().numpy())
                    num_all += len(target)
                    val_loss_l.append(F.nll_loss(output, target).cpu().numpy())

            val_accs.append(num_correct/num_all)
            val_losses.append(logsumexp(val_loss_l))

            if len(val_losses)>=2:
                print("(", val_losses[-1], "<-", val_losses[-2], ")")
            if len(val_losses)>=2 and val_losses[-1] > val_losses[-2]:
                print(len(val_losses))
                break
            if epoch > 25:
              break
            epoch += 1


        # Final eval on Test

        net.eval()
        num_correct, num_all, test_loss = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = net(data)
                preds = output.argmax(dim=1)
                num_correct += np.count_nonzero(target.cpu().numpy() == preds.cpu().numpy())
                num_all += len(target)
                test_loss += F.nll_loss(output, target)

        acc = num_correct / num_all
        test_loss = test_loss / num_all
        baseline_performances[f'random_{count}']['acc'].append(acc)
        baseline_performances[f'random_{count}']['loss'].append(test_loss)
        print("RESULT", count, acc)
        with open(os.path.join(datapath, savepath), 'wb') as handle:
            pickle.dump(baseline_performances, handle, protocol=pickle.HIGHEST_PROTOCOL)
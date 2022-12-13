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

datapath = os.path.join('..', 'data')
filterpath = os.path.join(datapath, 'filters-complete', '8_19')
num_filters = 8
loadpath = os.path.join(datapath, 'DCGAN_joint_generator' + '.pt')
savepath = 'save_baselines_dcgan_joint_' + str(num_filters) + '.pickle'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

nz=8
ngpu=1
netG_load = os.path.join(datapath, 'DCGAN_joint_generator' + '.pt')
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.merged = nn.Sequential(
            nn.Linear(nz, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 4*8),
        )
        self.shared_weights = nn.Sequential(
            nn.Linear(4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 25),
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.shared_weights, input, range(self.ngpu))
        else:
            output = self.merged(input.view(-1,nz))
            
            acts = []
            for i  in range(8):
                sliced = output[:,i*(4):(i+1)*(4)].view(-1,4)
                sliced_output = self.shared_weights(sliced)
                acts.append(sliced_output.view(-1,1,5,5))
            output = torch.concat(acts, dim=1)

        return output

model = Generator(ngpu).to(device)
model.load_state_dict(torch.load(netG_load))

model.eval()

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
baseline_performances = {
    'dcgan_joint_IID': {
        'acc': [],
        'loss': []
    }
}

uuids = os.listdir(filterpath)


batch_size = 64
train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(mnist_val,batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=True)

lr = 1e-1
repetitions = 10

start_training = datetime.now()
for repetition in range(repetitions):
        # Sample full baseline

        net = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=5, stride=2, bias=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(int(2304*(num_filters/16)), 10)
        ).to(device)

        with torch.no_grad():
            noise = torch.randn(1, nz, device=device)
            fake = model(noise)

            for c in range(num_filters):
                net[0].weight[c,:,:,:] = fake[0, c].detach().reshape(5,5)

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
            print(f'Repetition {repetition+1} of {repetitions}, Epoch {epoch+1} , {datetime.now() - start_training}')
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
        baseline_performances[f'dcgan_joint_IID']['acc'].append(acc)
        baseline_performances[f'dcgan_joint_IID']['loss'].append(test_loss)
        print("RESULT", acc)
        with open(os.path.join(datapath, savepath), 'wb') as handle:
            pickle.dump(baseline_performances, handle, protocol=pickle.HIGHEST_PROTOCOL)
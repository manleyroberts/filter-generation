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
loadpath = os.path.join(datapath, 'vae_' + str(num_filters) + '.pt')
savepath = 'save_baselines_vae_' + str(num_filters) + '.pickle'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

vae_batch_size = 100

x_dim  = 25
hidden_dim1 = 20
hidden_dim2 = 20
hidden_dim3 = 20
hidden_dim4 = 20
latent_dim = 10

"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim1)
        self.FC_input2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.FC_input3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.FC_input4 = nn.Linear(hidden_dim3, hidden_dim4)

        self.FC_mean  = nn.Linear(hidden_dim2, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim2, latent_dim)
        
        # self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

        self.training = True
        
    def forward(self, x):
        h_       = self.tanh(self.FC_input(x))
        h_       = self.tanh(self.FC_input2(h_))
        h_       = self.tanh(self.FC_input3(h_))
        h_       = self.tanh(self.FC_input4(h_))

        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim1)
        self.FC_hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.FC_hidden3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.FC_hidden4 = nn.Linear(hidden_dim3, hidden_dim4)

        self.FC_output = nn.Linear(hidden_dim4, output_dim)
        
        # self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        h     = self.tanh(self.FC_hidden(x))
        h     = self.tanh(self.FC_hidden2(h))
        h     = self.tanh(self.FC_hidden3(h))
        h     = self.tanh(self.FC_hidden4(h))

        x_hat = self.FC_output(h)
        return x_hat

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        # self.lamb = nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var


encoder = Encoder(input_dim=x_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, hidden_dim3=hidden_dim3, hidden_dim4=hidden_dim4, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, hidden_dim3=hidden_dim3, hidden_dim4=hidden_dim4, output_dim = x_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(device)
model.load_state_dict(torch.load(loadpath))

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
                    
# baseline_sample_counts = [1, 2, 4, 6, 8, 16, 32]
baseline_sample_counts = [8]
baseline_performances = {
    'vae_IID': {
        'acc': [],
        'loss': []
    }
}
for count in baseline_sample_counts:
    baseline_performances[f'vae_IID_{count}'] = {}
    baseline_performances[f'vae_IID_{count}']['acc'] = []
    baseline_performances[f'vae_IID_{count}']['loss'] = []

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

random_noise = False
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
              if random_noise:
                net[0].weight[c,:,:,:] =  torch.randn(5, 5).to(device)
              else:
                noise = torch.randn(1, latent_dim).to(device)
                generated_filter = model.Decoder(noise)
                net[0].weight[c,:,:,:] = generated_filter[0].view(5, 5).detach().clone()

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
        baseline_performances[f'vae_IID_{count}']['acc'].append(acc)
        baseline_performances[f'vae_IID_{count}']['loss'].append(test_loss)
        print("RESULT", count, acc)
        with open(os.path.join(datapath, savepath), 'wb') as handle:
            pickle.dump(baseline_performances, handle, protocol=pickle.HIGHEST_PROTOCOL)
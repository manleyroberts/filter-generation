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
filterpath = os.path.join(datapath, '8_19')
num_filters = 8
loadpath = os.path.join("./Sequence-VAE/models/filter_lstm/5000_Recon_st.pt")
savepath = './save_baselines_lstm_joint_no_shuff' + str(num_filters) + '.pickle'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# bptt = 60
vae_batch_size = 100
# embed_size = 300
hidden_size = 20
latent_size = 10
# latent_dim = 10
input_size = 25

class LSTM_VAE(torch.nn.Module):

  def __init__(self, input_size, hidden_size, latent_size, num_layers=1):
    super(LSTM_VAE, self).__init__()

    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Variables
    self.num_layers = num_layers
    self.lstm_factor = num_layers
    # self.vocab_size = vocab_size
    # self.embed_size = embed_size
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.latent_size = latent_size

    # For dictionary lookups 
    # self.dictionary = PTB(data_dir="./data", split="train", create_data= False, max_sequence_length= 60)
  
    # X: bsz * seq_len * vocab_size 
    # Embedding
    # self.embed = torch.nn.Embedding(num_embeddings= self.vocab_size,embedding_dim= self.embed_size)

    #    X: bsz * seq_len * vocab_size 
    #    X: bsz * seq_len * embed_size

    # Encoder Part
    self.encoder_lstm = torch.nn.LSTM(input_size= self.input_size,hidden_size= self.hidden_size, batch_first=True, num_layers= self.num_layers)
    self.mean = torch.nn.Linear(in_features= self.hidden_size * self.lstm_factor, out_features= self.latent_size)
    self.log_variance = torch.nn.Linear(in_features= self.hidden_size * self.lstm_factor, out_features= self.latent_size)

    # Decoder Part
                                        
    self.init_hidden_decoder = torch.nn.Linear(in_features= self.latent_size, out_features= self.hidden_size * self.lstm_factor)
    self.decoder_lstm = torch.nn.LSTM(input_size= self.input_size, hidden_size= self.hidden_size, batch_first = True, num_layers = self.num_layers)
    self.output = torch.nn.Linear(in_features= self.hidden_size * self.lstm_factor, out_features= self.input_size)
    # self.log_softmax = torch.nn.LogSoftmax(dim=2)

  def init_hidden(self, batch_size):
    hidden_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
    state_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
    return (hidden_cell, state_cell)
#     return hidden_cell


  # def get_embedding(self, x):
  #   x_embed = self.embed(x)
    
  #   # Total length for pad_packed_sequence method = maximum sequence length
  #   maximum_sequence_length = x_embed.size(1)

  #   return x_embed, maximum_sequence_length

  def encoder(self, input_filters, hidden_encoder):

    # pad the packed input.
#     print(input_filters)
    output_encoder, hidden_encoder = self.encoder_lstm(input_filters, hidden_encoder)
#     output_encoder, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output_encoder, batch_first=True, total_length= total_padding_length)

    # Extimate the mean and the variance of q(z|x)
    mean = self.mean(hidden_encoder[0])
    log_var = self.log_variance(hidden_encoder[0])
    std = torch.exp(0.5 * log_var)   # e^(0.5 log_var) = var^0.5
    
    # Generate a unit gaussian noise.
    batch_size = output_encoder.size(0)
    seq_len = output_encoder.size(1)
    noise = torch.randn(batch_size, self.latent_size).to(self.device)
    
    z = noise * std + mean

    return z, mean, log_var, hidden_encoder


  def decoder(self, z, input_filters):

    hidden_decoder = self.init_hidden_decoder(z)
    hidden_decoder = (hidden_decoder, hidden_decoder)

    # pad the packed input.
    output_decoder, hidden_decoder = self.decoder_lstm(input_filters,hidden_decoder) 
#     output_decoder, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output_decoder, batch_first=True, total_length= total_padding_length)


    x_hat = self.output(output_decoder)
    
#     x_hat = self.log_softmax(x_hat)


    return x_hat

  

  def forward(self, input_filters,hidden_encoder):
    
    """
      x : bsz * seq_len
    
      hidden_encoder: ( num_lstm_layers * bsz * hidden_size, num_lstm_layers * bsz * hidden_size)

    """
    # Get Embeddings
    # x_embed, maximum_padding_length = self.get_embedding(x)

    # Packing the input
    # packed_x_embed = torch.nn.utils.rnn.pack_padded_sequence(input= x_embed, lengths= sentences_length, batch_first=True, enforce_sorted=False)


    # Encoder
    z, mean, log_var, hidden_encoder = self.encoder(input_filters, hidden_encoder)

    # Decoder
    x_hat = self.decoder(z, input_filters)
    


    return x_hat, mean, log_var, z, hidden_encoder

  

  def inference(self, batch_size, num_filters, filt1):

    # generate random z 
#     seq_len = 1


#     input = torch.Tensor(1, 1).fill_(self.dictionary.get_w2i()[sos]).long().to(self.device)
    
#         batch_size = 1
    out_filters = []
    for i in range(batch_size):
        z = torch.randn(1,1,self.latent_size).to(self.device)
        filter_samples = []
        input = filt1
        
        hidden = self.init_hidden_decoder(z)
        hidden = (hidden, hidden)

        for j in range(num_filters):
          output,hidden = self.decoder_lstm(input, hidden)
          output = self.output(output)
          filter_samples.append(output.view(-1))
          input = output
        
        out_filters.append(torch.stack(filter_samples, dim=0))
        
#     w_sample = [self.dictionary.get_i2w()[str(idx)] for idx in idx_sample]
#     w_sample = " ".join(w_sample)

    return torch.stack(out_filters, dim=0)

model = LSTM_VAE(input_size = input_size, hidden_size = hidden_size, latent_size = latent_size).to(device)
# MODEL_SAVE_PATH = "./models/filter_lstm/5000_Recon_st.pt"
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
baseline_performances = {
    'lstm_joint_IID': {
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

num_filters = 8
interv=10
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
            FILTER1 = torch.zeros(5,5)
            num_samples=1
            sample = model.inference(num_samples,num_filters*interv,FILTER1.view(1,1,-1).to(device))
            sample = sample.view(sample.shape[:-1]+tuple([5,5]))
            
#             for filt_idx in range(0,num_filters*interv,interv):
            sample = sample[0][0:(num_filters*interv):interv]

#             noise = torch.randn(1, latent_dim).to(device)
#             generated_filter = model.Decoder(noise)
#             print(sample.shape)

            for c in range(num_filters):
                net[0].weight[c,:,:,:] = sample[c].detach().clone()

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
        baseline_performances[f'lstm_joint_IID']['acc'].append(acc)
        baseline_performances[f'lstm_joint_IID']['loss'].append(test_loss)
        print("RESULT", acc)
        with open(os.path.join(datapath, savepath), 'wb') as handle:
            pickle.dump(baseline_performances, handle, protocol=pickle.HIGHEST_PROTOCOL)




from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

# https://github.com/pytorch/examples/tree/main/dcgan


batchSize = 128         #', type=int, default=64, help='input batch size')
imageSize = 5           #', type=int, default=64, help='the height / width of the input image to network')
nz = 8                  #', type=int, default=100, help='size of the latent z vector')
niter = 200             #', type=int, default=25, help='number of epochs to train for')
lr = 0.00002            #', type=float, default=0.0002, help='learning rate, default=0.0002')
beta1 = 0.5             #', type=float, default=0.5, help='beta1 for adam. default=0.5')
ngpu =1                 #', type=int, default=1, help='number of GPUs to use')
netG_load = ''          #', default='', help="path to netG (to continue training)")
netD_load = ''          #', default='', help="path to netD (to continue training)")
outf = '../dcgan-out/'  #', default='.', help='folder to output images and model checkpoints')
manualSeed = None       #', type=int, help='manual seed')
nc = 1 # channels
try:
    os.makedirs(outf)
except OSError:
    pass

if manualSeed is None:
    manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True


datapath = os.path.join('..', 'data')
fpath = os.path.join(datapath, 'filters')
filterpath = os.path.join(fpath, '8_19')

def get_dataset():
  weight_dataset = []
  for f, file in tqdm(enumerate(os.listdir(filterpath))):
    filter = torch.load(os.path.join(filterpath, file))
    for i in range(8):
      weight_dataset.append(filter['0.weight'][i][0].reshape(1,5,5))
  return weight_dataset

num_images = len(os.listdir(filterpath))
dataset = get_dataset()
dataset = torch.stack(dataset, dim=0)
print(dataset.shape)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ngpu = 0
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                         shuffle=True)

print(device)

nz = int(nz)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            # Heavily modified from DC-GAN
            nn.Flatten(),
            nn.Linear(nz, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 25),

        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if netG_load != '':
    netG.load_state_dict(torch.load(netG_load))


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(

            # Heavily modified from DC-GAN

            nn.Flatten(),
            nn.Linear(nc * 5 * 5, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if netD_load != '':
    netD.load_state_dict(torch.load(netD_load))
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data.to(device)
        batch_size = real_cpu.size(0)
        # print(real_cpu.shape)
        label = torch.full((batch_size,), real_label,
                           dtype=real_cpu.dtype, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                    normalize=True)

    # do checkpointing
    save_G = '%s/netG_epoch_%d.pth' % (outf, epoch)
    # print(save_G)
    torch.save(netG.state_dict(), save_G)
    save_D = '%s/netD_epoch_%d.pth' % (outf, epoch)
    torch.save(netD.state_dict(),  save_D)
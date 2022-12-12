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


batchSize = 128         #', type=int, default=64, help='input batch size')
imageSize = 5           #', type=int, default=64, help='the height / width of the input image to network')
nz = 8                  #', type=int, default=100, help='size of the latent z vector')
niter = 10000           #', type=int, default=25, help='number of epochs to train for')
lr = 0.0002             #', type=float, default=0.0002, help='learning rate, default=0.0002')
beta1 = 0.5             #', type=float, default=0.5, help='beta1 for adam. default=0.5')
ngpu =1                 #', type=int, default=1, help='number of GPUs to use')
netG_load = ''          #', default='', help="path to netG (to continue training)")
netD_load = ''          #', default='', help="path to netD (to continue training)")
outf = '../dcgan-out/'  #', default='.', help='folder to output images and model checkpoints')
manualSeed = None       #', type=int, help='manual seed')
nc = 8 # channels
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
    # for i in range(8):
    weight_dataset.append(filter['0.weight'][:,0].reshape(8,5,5))
  return weight_dataset

num_images = len(os.listdir(filterpath))
dataset = get_dataset()
dataset = torch.stack(dataset, dim=0)
print(dataset.shape)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ngpu = 1
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

            # EDITED
            # nn.Flatten(),
            # nn.Linear(36,4),
            # nn.BatchNorm1d(4),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(4, 8, 5, 1, 0, bias=False),
            # nn.BatchNorm2d(8),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(64, 64, 1, 1, bias=False, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(8, 1, 1, 1, bias=False, padding=0),


            # # input is Z, going into a convolution
            # nn.ConvTranspose2d(     9, ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            # # state size. (ngf*8) x 4 x 4
            # nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 4),
            # nn.ReLU(True),
            # # state size. (ngf*4) x 8 x 8
            # nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            # nn.ReLU(True),
            # # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # # nn.Tanh()
            # # state size. (nc) x 64 x 64

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


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if netG_load != '':
    netG.load_state_dict(torch.load(netG_load))
# print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.shared_weights = nn.Sequential(

            # ORIGINAL

            # # input is (nc) x 64 x 64
            # nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf) x 32 x 32
            # nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*2) x 16 x 16
            # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*4) x 8 x 8
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()

            # EDITED FOR 5x5

            # input is (nc) x 64 x 64
            # nn.Conv2d(nc, 16, 3, 2, bias=False, padding=1),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf) x 32 x 32
            # nn.Conv2d(16, 32, 3, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*2) x 16 x 16
            # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*4) x 8 x 8
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),

            # nn.Conv2d(nc, 16, 3, 2, 1, 0, bias=False),
            # # nn.Flatten(),
            # # nn.Linear(nc * 5 * 5, 64),
            # nn.BatchNorm1d(64),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(64, 64),            
            # nn.BatchNorm1d(64),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(64, 16),            
            # nn.BatchNorm1d(16),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(16, 1),
            # nn.Sigmoid()
            nn.Conv2d(1, 4, 3, 2, bias=False, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4, 4, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(4),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )

        self.merged = nn.Sequential(
            nn.Linear(4*3*3*8, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.shared_weights, input, range(self.ngpu))
        else:
            acts = []
            # print(input.shape)
            for i  in range(8):
                sliced = input[:,i:i+1,:,:]
                sliced_output = self.shared_weights(sliced)
                acts.append(sliced_output)
            output = torch.concat(acts, dim=1)
            # print(output.shape)
            output = self.merged(output)



        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if netD_load != '':
    netD.load_state_dict(torch.load(netD_load))
print(netD)

criterion = nn.BCELoss()
# criterion = nn.MSELoss()

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

        real_cpu_permuted = real_cpu[:, torch.randperm(real_cpu.shape[1]),:,:]
        output = netD(real_cpu_permuted)
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
            vutils.save_image(real_cpu.reshape(batchSize,-1),
                    '%s/real_samples.png' % outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach().reshape(batchSize,-1),
                    '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                    normalize=True)

    # do checkpointing
    save_G = '%s/netG_epoch_%d.pth' % (outf, epoch)
    # print(save_G)
    torch.save(netG.state_dict(), save_G)
    save_D = '%s/netD_epoch_%d.pth' % (outf, epoch)
    torch.save(netD.state_dict(),  save_D)
import torch
import os
from torchvision.utils import save_image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

filter_dir = '../filters'

image_dir = '../images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

all_checkpoints = os.listdir(filter_dir)
print(f'{len(all_checkpoints)} checkpoints found')

for checkpoint_uuid in all_checkpoints:
    state_dict = torch.load(filter_dir + '/' + checkpoint_uuid)
    filters = state_dict['0.weight']
    print(filters.sum())
    for f in range(16):
        save_image(filters[f,:,:,:].reshape(1,1,5,5).repeat(1,3,1,1), image_dir + '/' + checkpoint_uuid + '_' + str(f) + '.png')


import torch
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

filter_dir = '../filters'

all_checkpoints = os.listdir(filter_dir)
print(f'{len(all_checkpoints)} checkpoints found')

for checkpoint_uuid in all_checkpoints:
    state_dict = torch.load(filter_dir + '/' + checkpoint_uuid)
    filters = state_dict['0.weight']
    print(filters.sum())
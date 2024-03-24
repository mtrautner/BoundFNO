import torch
import os
import sys
sys.path.append(os.path.join('..'))
from func_to_func2d_invasive import FNO2d
import yaml


# Load config file
config_name = 'smooth_x_y_grid'
# print CUDA available
print(torch.cuda.is_available())
config_path = 'trained_models/' + config_name + '_info.yaml'
with open(config_path, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Load input and output data
input_data = torch.load('../data/smooth_training_data/A_to_chi1_tiny_input_data.pt')
output_data = torch.load('../data/smooth_training_data/A_to_chi1_tiny_output_data.pt')

model_name = config['model_name']
N_data = input_data.shape[0]
train_size = config['N_train']
N_modes = config['K']
width  = config['width']
act = config['act']
epochs = config['epochs']
b_size = config['batch_size']
lr = config['lr']
USE_CUDA = config['USE_CUDA']
d_in = config['d_in']
d_out = config['d_out']

model = FNO2d(modes1=N_modes, modes2=N_modes, width=width, d_in=d_in, d_out=d_out, act=act)
model_path = 'trained_models/' + model_name + '.pt'
model.load_state_dict(torch.load(model_path)['model_state_dict'])

# de-cuda the model
model.cpu()
torch.save(model,  model_name + '.pt')
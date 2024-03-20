'''
Generates random initial models for the FNO model
'''

import numpy as np
import torch
import torch.nn as nn
import os, sys; sys.path.append(os.path.join('../..'))
from timeit import default_timer
import yaml

from models.func_to_func2d_invasive import FNO2d
from util.utilities_module import LpLoss, count_params, validate, dataset_with_indices

def generate_initial_model(modes, width, device, get_grid=False,n_layers = 4):
    '''
    Generates a random initial model for the FNO model
    '''
    model = FNO2d(modes1 = modes, modes2 = modes, n_layers = n_layers, width = width,get_grid = get_grid).to(device)
    return model

if __name__ == '__main__':
    # Constants
    SEED = 1989
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # take first argument
    args = sys.argv
    if len(args) > 1:
        ind = int(args[1])

    # Parameters
    Ks = [4,12,36,60]
    width = 32
    device = 'cpu'
    get_grid = False
    n_layers = 5
    n_models = 2

    for k in Ks:
        for n in range(n_models):
            # Generate model
            model = generate_initial_model(k, width, device, get_grid = get_grid, n_layers = n_layers)

            # set layers weights to be same as the first convolutional layer's weights and biases
            # first_layer = model.layers[0]
            # for layer in model.layers:
            #     if isinstance(layer, nn.Conv2d):
            #         layer.weight = first_layer.weight
            #         layer.bias = first_layer.bias
                    
            # Save model
            model_name = f'initial_model_K_{k}_{n}.pt'
            # torch.save(model, model_name)
            torch.save({'model_state_dict': model.state_dict()},model_name)

            # Save model info
            model_info = f'initial_model_K_{k}_{n}_info.yaml'
            with open(model_info, 'w') as file:
                yaml.dump({'K': k, 'width': width, 'device': device, 'get_grid': get_grid, 'n_layers': n_layers,'seed': SEED}, file)




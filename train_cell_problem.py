import os, sys
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from timeit import default_timer
import pickle as pkl
import numpy as np
import sys
import torch.utils.data
from tqdm import tqdm
import yaml

# add parent directories to path
sys.path.append('../')
sys.path.append('../../')

from src.fno2d import FNO2d
from src.utilities3 import LpLoss, HsLoss

def load_data(config):
    N_train = 4096 #
    N_test = 256 #

    dir = 'data/smooth_training_data/'
    input_data_path = dir + 'A_to_chi1_input_data.pt'
    output_data_path = dir + 'A_to_chi1_output_data.pt'

    input_data = torch.load(input_data_path)
    output_data = torch.load(output_data_path)

    # normalize input (the output is invariant under arbitrary re-scaling of inputs) 
    norm_input = torch.norm(input_data, dim=[-2,-1],keepdim=True) / 128
    input_data = input_data / norm_input
    
    # train/test split
    assert N_train + N_test <= len(input_data), f'N_train + N_test exceeds total data size. {N_train=} + {N_test=} > N_data={len(input_data)}.'
    # training data
    x_train = input_data[:N_train]
    y_train = output_data[:N_train]
    # testing data
    x_test = input_data[-N_test:]
    y_test = output_data[-N_test:]


    # Wrap training data in loader
    b_size = config['batch_size']
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=b_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test,y_test), batch_size = b_size, shuffle = False)
    
    return train_loader, test_loader


def train_model(config):
    # Take in user arguments
    #data_path = config['data_path']
    model_name = config['model_name']
    N_modes = config['N_modes']
    width = config['width']
    epochs = config['epochs']
    b_size = config['batch_size']
    lr = float(config['lr'])
    USE_CUDA = config['USE_CUDA']

    if not torch.cuda.is_available() and USE_CUDA:
        print('No CUDA devices available. Setting USE_CUDA = False.')
        USE_CUDA = False

    if USE_CUDA:
        gc.collect()
        torch.cuda.empty_cache()

    #
    device = 'cuda' if USE_CUDA else 'cpu'

    # Paths
    dir = 'TrainedModels/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    model_path = dir + model_name
    model_info_path = dir + model_name + '_config.yml'
    
    # Load data
    train_loader, test_loader = load_data(config)

    # Set loss function to be H1 loss
    #loss_func = LpLoss(p=2)
    loss_func = HsLoss()

    # Specify pointwise degrees of freedom
    d_in = 3 # A \in \R^{2 \times 2}_sym
    d_out = 1 # \chi \in \R^2

    # Initialize model
    model = FNO2d(modes = N_modes, width = width, in_channels = d_in, out_channels = d_out, which_grid='periodic')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6)

    
    # paths
    #model_path = '/groups/astuart/mtrautne/FNM/trainedModels/' + model_name
    #model_info_path = 'trainedModels/' + model_name + '_config.yml'
    
    # Train model
    train_err = np.zeros((epochs,))
    test_err = np.zeros((epochs,))

    for ep in tqdm(range(epochs)):
        t1 = default_timer()
        train_loss = 0.0
        test_loss  = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            # For forward model: 
            # Input shape (of x):     (batch, channels_in, nx_in, ny_in)
            # Output shape:           (batch, channels_out, nx_out, ny_out)
            
            # The input resolution is determined by x.shape[-2:]
            # The output resolution is determined by self.s_outputspace
            loss = loss_func(pred,y)
            loss.backward()
            train_loss = train_loss + loss.item()

            optimizer.step()
        scheduler.step()

        with torch.no_grad():
            for x,y in test_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                t_loss = loss_func(pred,y)
                test_loss = test_loss + t_loss.item()

        train_err[ep] = train_loss/len(train_loader)
        test_err[ep]  = test_loss/len(test_loader)
        
        t2 = default_timer()
        print(f'[{ep+1:3}], time: {t2-t1:.3f}, train: {train_err[ep]:.3e}, test: {test_err[ep]:.3e}', flush=True)


    # Save model
    # model_path = '/groups/astuart/mtrautne/FNM/FourierNeuralMappings/homogenization/trainModels/trainedModels/' + model_name
    torch.save(
        {'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_history': train_err,
        'test_loss_history': test_err,
        }, model_path)
    torch.save(model.to('cpu'), model_path + '_model')

    # save model config
    # convert config to a dict that will be readable when saved as a .json    
    with open(model_info_path, 'w') as fp:
        yaml.dump(config, fp)

    # Compute and save errors
    #model_path = 'trainedModels/' + model_name
    #loss_report_f2v(y_test_approx_all, y_test, x_test, model_path)
    

if __name__ == "__main__":
    # parse command line arguments
    # (need to specify <name> of run = config_<name>.yaml)
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group()
    parser.add_argument('-c', "--config",
                    type=str,
                    help="Specify the config-file path.",
                    required=True)
    parser.add_argument('-m', "--model_index",
                    type=int,
                    default=None,
                    help="Specify the model index.")
    args = parser.parse_args()

    # Take in user arguments 
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k,v in config.items():
        print(k,':  ',v)

    # Check if there's a second argument
    if args.model_index != None:
        config['model_name'] = config['model_name'] + '_' + str(args.model_index)

    train_model(config)

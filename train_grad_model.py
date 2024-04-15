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
from src.subsample_scheduler import SubsampleScheduler

def load_data(config):
    N_train = 2000 #
    N_valid = 250
    N_test = 250 #

    dir = 'data/grad_training_data/'
    input_data_path = dir + 'input_data_grad.pt'
    output_data_path = dir + 'output_data_grad.pt'

    input_data = torch.load(input_data_path)
    output_data = torch.load(output_data_path)
    
    # train/test split
    assert N_train + N_test <= len(input_data), f'N_train + N_test exceeds total data size. {N_train=} + {N_test=} > N_data={len(input_data)}.'
    # training data
    x_train = input_data[:N_train]
    y_train = output_data[:N_train]
    # validation data (for subsampling)
    x_valid = input_data[N_train:N_train+N_valid]
    y_valid = output_data[N_train:N_train+N_valid]
    # testing data
    x_test = input_data[-N_test:]
    y_test = output_data[-N_test:]


    # Wrap training data in loader
    b_size = config['batch_size']
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=b_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid,y_valid), batch_size=b_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test,y_test), batch_size = b_size, shuffle = False)
    
    return train_loader, valid_loader, test_loader


def train_model(config):
    # Take in user arguments
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
    model_name = 'grad_' + model_name
    dir = 'TrainedModels/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    model_path = dir + model_name
    model_info_path = dir + model_name + '_config.yml'
    
    # Load data
    train_loader, valid_loader, test_loader = load_data(config)

    # Set loss function to be H1 loss
    #loss_func = LpLoss(p=2)
    loss_func = HsLoss()

    # Specify pointwise degrees of freedom
    d_in = 1 # function
    d_out = 2 # gradient

    # Initialize model
    model = FNO2d(modes = N_modes, width = width, in_channels = d_in, out_channels = d_out, which_grid='periodic')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6)

    #
    if config['subsampling']:
        gridsize = 128
        minsize = 32
        subsampler = SubsampleScheduler(gridsize // minsize, patience=40)
    
    # Train model
    train_err = np.zeros((epochs,))
    test_err = np.zeros((epochs,))
    #
    epoch_timings = np.zeros((epochs,))
    
    for ep in tqdm(range(epochs)):
        t1 = default_timer()
        train_loss = 0.0
        test_loss  = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            if config['subsampling'] and subsampler.ss>1:
                x,y = subsampler(x,y)

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
        #
        scheduler.step()
        
        with torch.no_grad():
            for x,y in test_loader:
                x = x.to(device)
                y = y.to(device)
                if config['subsampling'] and subsampler.ss>1:
                    x,y = subsampler(x,y)
                pred = model(x)
                t_loss = loss_func(pred,y)
                test_loss = test_loss + t_loss.item()

        train_err[ep] = train_loss/len(train_loader)
        test_err[ep]  = test_loss/len(test_loader)
        
        t2 = default_timer()
        if config['subsampling']:
            subsamp_str =  f'[subsamp: {subsampler.ss}]'
        else:
            subsamp_str = ''
        print(f'[{ep+1:3}], time: {t2-t1:.3f}, train: {train_err[ep]:.3e}, test: {test_err[ep]:.3e}' + subsamp_str, flush=True)

        if config['subsampling'] and subsampler.ss>1:
            with torch.no_grad():
                valid_loss = 0.
                for x,y in valid_loader:
                    x,y = x.to(device), y.to(device)
                    x,y = subsampler(x,y)
                    pred = model(x)
                    t_loss = loss_func(pred,y)
                    valid_loss = valid_loss + t_loss.item()

            # adjust subsampling rate adaptively
            valid_loss = valid_loss / len(valid_loader)
            subsampler.step(valid_loss) 

        # record epoch timings
        t2 = default_timer()
        epoch_timings[ep] = t2 - t1
        
        
    # Save model
    torch.save(
        {'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_history': train_err,
        'test_loss_history': test_err,
         'epoch_timing_history': epoch_timings,
        }, model_path)
    torch.save(model.to('cpu'), model_path + '_model')

    # save model config
    # convert config to a dict that will be readable when saved as a .json    
    with open(model_info_path, 'w') as fp:
        yaml.dump(config, fp)


if __name__ == "__main__":
    # parse command line arguments
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
    parser.add_argument("--subsampling",
                    action='store_true',
                    default=False,
                    help="Activate subsampling scheduler.")
    args = parser.parse_args()

    # Take in user arguments 
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # print config arguments
    for k,v in config.items():
        print(k,':  ',v)

    #
    config['subsampling'] = args.subsampling
    if config['subsampling']:
        config['model_name'] = config['model_name'] + '_subsampling'
        
    # Check if there's a second argument
    if args.model_index != None:
        config['model_name'] = config['model_name'] + '_' + str(args.model_index)

    train_model(config)

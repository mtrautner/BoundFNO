import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys
from importlib import import_module
from tqdm import tqdm
from timeit import default_timer
# from util import Adam
from util.utilities_module import *
from util.subsample_scheduler import SubsampleScheduler
import yaml
from models.func_to_func2d_invasive import FNO2d


def train_model(input_data, output_data, config):
    """
    Input data is torch array and has form (N_data, d_in, nx, ny)
    Output data is torch array and has form (N_data, d_out, nx, ny)
    """
    model_name = config['model_name']
    N_data = input_data.shape[0]
    train_size = config['N_train']
    print('Train Size: ', train_size)
    N_modes = config['K']
    width  = config['width']
    act = config['act']
    epochs = config['epochs']
    b_size = config['batch_size']
    lr = config['lr']
    USE_CUDA = config['USE_CUDA']
    d_in = config['d_in']
    d_out = config['d_out']
    periodic = config['periodic_grid']
    subsampling = config['subsampling']
    
    if USE_CUDA:
        gc.collect()
        torch.cuda.empty_cache()

    model_path = 'models/trained_models/' + model_name + '.pt'

    test_start = N_data - 500
    test_end = N_data

    # hold out a validation set
    valid_size = 500

    #=========TRAINING==========#
    y_train = output_data[:train_size]
    y_test = output_data[test_start:test_end]
    y_valid = output_data[train_size:train_size+valid_size]
    
    x_train = input_data[:train_size]
    x_test = input_data[test_start:test_end]
    x_valid = input_data[train_size:train_size+valid_size]

    loss_func = Sobolev_Loss(d=2,p=2)

    model = FNO2d(modes1=N_modes, modes2=N_modes, width=width, d_in=d_in, d_out=d_out, act=act,periodic_grid= periodic)
    print("Periodic: ", periodic)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6)

    #
    if subsampling:
        gridsize = 128
        minsize = 32
        subsampler = SubsampleScheduler(gridsize // minsize, patience=40)
    
    if USE_CUDA:
        model.cuda()

    # Wrap training data in loader
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=b_size,
                                           shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test,y_test), batch_size = b_size, shuffle = False)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid,y_valid), batch_size = b_size, shuffle = False)    

    #
    train_err = torch.zeros((epochs,))
    test_err = torch.zeros((epochs,))
    valid_err = torch.zeros((epochs,))
    epoch_timings = torch.zeros((epochs,))

    #
    for ep in tqdm(range(epochs)):
        t1 = default_timer()
        train_loss = 0.0
        test_loss = 0.0
        valid_loss = 0.0

        for x, y in train_loader:
            if subsampling:
                x,y = subsampler(x,y)

            optimizer.zero_grad()
            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()

            y_approx = model(x,USE_CUDA = True).squeeze()
            y = y.squeeze()
            loss = loss_func.Lp_rel_err(y_approx,y,size_average = True)
            loss.backward()
            train_loss = train_loss + loss.item()

            optimizer.step()
        scheduler.step()

        with torch.no_grad():
            for x,y in test_loader:
                if subsampling:
                    x,y = subsampler(x,y)
                    
                if USE_CUDA:
                    x = x.cuda()
                    y = y.cuda()
                y_test_approx = model(x,USE_CUDA = True)
                y = y.squeeze()
                t_loss = loss_func.Lp_rel_err(y_test_approx,y,size_average = True)
                test_loss = test_loss + t_loss.item()

        if subsampling:
            # compute validation error
            with torch.no_grad():
                for x,y in valid_loader:
                    if subsampling:
                        x,y = subsampler(x,y)
                    if USE_CUDA:
                        x = x.cuda()
                        y = y.cuda()
                    #
                    y_approx = model(x,USE_CUDA = True)
                    y = y.squeeze()
                    t_loss = loss_func.Lp_rel_err(y_approx,y,size_average = True)
                    valid_loss = valid_loss + t_loss.item()

            # adjust subsampling rate adaptively
            valid_loss = valid_loss / len(valid_loader)
            subsampler.step(valid_loss) 
                
        train_err[ep] = train_loss/len(train_loader)
        test_err[ep] = test_loss/len(test_loader)
        valid_err[ep] = valid_loss
        
        if subsampling:
            subsamp_str =  f'[subsamp: {subsampler.ss}]'
        else:
            subsamp_str = ''

        # record epoch timings
        t2 = default_timer()
        epoch_timings[ep] = t2 - t1
        print(f'[{ep+1:3}], time: {t2-t1:.3f}, train: {train_err[ep]:.3e}, test: {test_err[ep]:.3e}' + subsamp_str, flush=True)
        
    model.cpu()
    torch.save({
        'epochs: ': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_err': train_err,
        'test_err': test_err,
        'valid_err': valid_err,
        'epoch_timings': epoch_timings,
    }, model_path)
    
    # print both to .yaml file
    errors = {}
    errors['train relative error'] = train_err[-1]
    errors['test relative error'] = test_err[-1]
    json_errors = {k: v for k, v in errors.items()}
    # Save dictionary to json file
    with open('models/trained_models/' + model_name+ '_errors.yml', 'w') as fp:
        yaml.dump(json_errors, fp)
    

    return config

if __name__ == '__main__':
    # Load config file
    config_name = sys.argv[1]
    # print CUDA available
    print(torch.cuda.is_available())
    config_path = 'models/trained_models/' + config_name + '_info.yaml'
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Load input and output data
    input_data = torch.load(config['input_data_path'])
    output_data = torch.load(config['output_data_path'])

    # Train model
    config = train_model(input_data, output_data, config)

    # Save updated config file
    with open(config_path, 'w') as file:
        yaml.dump(config, file)



    

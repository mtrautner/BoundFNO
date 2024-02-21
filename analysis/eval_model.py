import torch
import pickle as pkl
import os
import sys 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import yaml 
import time
import torch.nn.functional as F
import pdb
# add to path
sys.path.append(os.path.join('..'))
from models.func_to_func2d_invasive import FNO2d
from util.utilities_module import LpLoss, Sobolev_Loss
from matplotlib.ticker import ScalarFormatter
from models.shared import SpectralConv2d


# Set font default
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
matplotlib.rcParams['mathtext.rm'] = 'stix'
matplotlib.rcParams['mathtext.it'] = 'stix'
matplotlib.rcParams['mathtext.bf'] = 'stix'


matplotlib.rcParams["axes.formatter.limits"] = (-99, 99) #makes scientific notation threshold high
plt.rcParams['font.family'] = 'serif'  # or 'DejaVu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 'DejaVu Serif' 'serif' 'Times
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
'''

tickfontsize = 30
fontsize = 30
linewidth = 4
markersize = 15

SMALL_SIZE = tickfontsize
MEDIUM_SIZE = tickfontsize
BIGGER_SIZE = fontsize

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

shapes = ['o','s','^','D','*']

def load_model(model_info_path, model_path,s_outputspace = (2048,2048)):
    # Load model info
    with open(model_info_path, 'r') as file:
        model_info = yaml.load(file, Loader=yaml.FullLoader)
    K = model_info['K']
    width = model_info['width']
    
    n_layers = model_info['n_layers']
    model = FNO2d(modes1 = K, modes2 = K, n_layers = n_layers, width = width, get_grid = False, s_outputspace = s_outputspace)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_data(data_path):
    # Load data
    data = pkl.load(open(data_path, 'rb'))
    return data

def undo_padding(layer,padding):
    """
    padding         (int or float): (1.0/padding) is fraction of domain to zero pad (non-periodic)
    layers         (torch tensor): dims (batch, channels, x, y) where x and y are padded
    grid_size* (1+ 1/padding) = x (or y) where grid_size is the size of the grid in the original domain (before padding)
    assumes square grid
    """
    # num_pad = layer.shape[-1]//padding
    orig_dim = int(layer.shape[-1]//(1+1/padding))
    num_pad = layer.shape[-1] - orig_dim
    layer_out = layer[...,:-num_pad,:-num_pad]
    return layer_out

def fig_err_vs_L(USE_CUDA = False, s = None):
    """
    plot error vs L
    """
    sizes = [32, 64,128, 256, 512, 1024, 2048]
    
    samp_count_input = 1
    samp_count_model = 1
    layer_count = 5
    all_err = torch.zeros(len(sizes)-1,layer_count, samp_count_model, samp_count_input)

    loss = Sobolev_Loss()      
    
    for model_sample in range(samp_count_model):
        print('model_sample:',model_sample)
        model_info_path = '../models/random_initial_models/initial_model_' + str(model_sample) + '_info.yaml'
        model_path = '../models/random_initial_models/initial_model_'+ str(model_sample) + '.pt'
        model = load_model(model_info_path, model_path,s_outputspace = (sizes[-1],sizes[-1]))
        if USE_CUDA:
            model.cuda()
        
        for input_sample in range(samp_count_input):
            print('input_sample:',input_sample)
            
            true_size = sizes[-1]
            input_true = load_data('../data/GRF_s' + str(s) +'/GRF_size_' + str(true_size) + '_' + str(input_sample)+ '.pkl')
            input_true = input_true.unsqueeze(0).unsqueeze(0)
            # if s == 10:
            #     input_true = load_data('../data/smooth_s10/s10_size_' + str(true_size) + '_' + str(input_sample)+ '.pkl')
            # else:
            #     input_true = load_data('../data/random_s0/s0_size_' + str(true_size) + '_' + str(input_sample)+ '.pkl')
            
            # pdb.set_trace()
            if USE_CUDA:
                # input_true = input_true.cuda()
                model = model.cuda()
            with torch.no_grad():
                layers_true = model(input_true, invasive = True, USE_CUDA = USE_CUDA)
            
            

            if USE_CUDA:
                layers_true = [layer.cpu() for layer in layers_true]
            
            # layers_true = [undo_padding(layer, padding) for layer in layers_true]
            true_dim = layers_true[0].shape[-1]
        


            for size_i, size in enumerate(sizes[:-1]):
                print(size)
                # comparison

                input_disc = load_data('../data/GRF_s' +str(s) + f'/GRF_size_{size}_'+ str(input_sample)+'.pkl')
                # if s == 10:
                #     input_disc = load_data('../data/smooth_s10/s10_size_' + str(size) + '_' + str(input_sample)+ '.pkl')
                # else:
                #     input_disc = load_data('../data/random_s0/s0_size_' + str(size) + '_' + str(input_sample)+ '.pkl')
                input_disc = input_disc.unsqueeze(0).unsqueeze(0)
                x_res = input_disc.shape[-1]
                # input_disc = F.pad(input_disc, [0, x_res//padding, 0, x_res//padding])
                if USE_CUDA:
                    input_disc = input_disc.cuda()
                with torch.no_grad():
                    layers_disc = model(input_disc, invasive = True, USE_CUDA = USE_CUDA)
                
                if USE_CUDA:
                    layers_disc = [layer.cpu() for layer in layers_disc]

                
                for i in range(len(layers_disc)):

                    layer_true_i = layers_true[i]
                    layer_disc_i = layers_disc[i]

                    # if layer_true_i.size()[-1] != true_dim:
                    #     layer_true_i = undo_padding(layer_true_i, model.padding)
                    # if layer_disc_i.size()[-1] != size:
                    #     layer_disc_i = undo_padding(layer_disc_i, model.padding)
                    # pdb.set_trace()
                    # layer_true_i_sub = layer_true_i[...,::true_dim//size,::true_dim//size]
                    # if i == 0:
                    #     fig, ax = plt.subplots(1,2,figsize = (10,5))
                    #     ax[0].imshow(layer_true_i[0,0].detach().numpy(),cmap = 'viridis')
                    #     ax[0].set_title('True')
                    #     ax[1].imshow(layer_disc_i[0,0].detach().numpy(),cmap = 'viridis')
                    #     ax[1].set_title('Disc')
                    #     plt.savefig('../Figures/layer_compare_s' + str(s) + '_size_' + str(size) + '_layer_' + str(i) + '.pdf')
                        
                    err = torch.norm(layer_disc_i - layer_true_i) # loss.Lp_err(layer_disc_i,layer_true_i_sub,size_average=False)
                    all_err[size_i,i,model_sample,input_sample] = err
                    # true_norm = torch.norm(layer_true_i_sub) #loss.Lp_norm(layer_true_i_sub,size_average=False)
                    # all_err[size_i,i,model_sample,input_sample] = err/true_norm

                    # all_err[sizes.index(size),i,model_sample,input_sample] = err/true_norm

    # plot
    # err to np
    all_err = all_err.detach().numpy()
    # print("all err: ",all_err)
    # pdb.set_trace()
 
    # reshape to (sizes, layers, samples)
    all_err = all_err.reshape(len(sizes)-1,layer_count,samp_count_model*samp_count_input)
    # average over samples
    all_err_mean = np.mean(all_err,axis = -1)
    all_err_std = 0 * all_err_mean #2*np.std(all_err,axis = -1)

    
    sizes = sizes[:-1]
    print(all_err_mean)
    # print(all_err_std)

    plt.figure(figsize = (10,10))
    slopes = []
    for i in range(len(layers_true)):
        plt.plot(sizes,all_err_mean[:,i],label = f'Layer {i+1}',color = CB_color_cycle[i],marker = shapes[i],linewidth = linewidth,markersize = markersize)
        plt.errorbar(sizes, all_err_mean[:,i], yerr=all_err_std[:,i],color = CB_color_cycle[i])
        plt.fill_between(sizes, all_err_mean[:,i] - all_err_std[:,i], all_err_mean[:,i] + all_err_std[:,i], alpha=0.2, color = CB_color_cycle[i])
        # compute best fit line
        p = np.polyfit(np.log(sizes),np.log(all_err_mean[:,i]),1)
        # extract slop of best fit line
        slope = p[0]
        slopes.append(slope)
        print(f'Layer {i+1} slope: {slope}')
    
    # average slops
    print(f'Average slope: {np.mean(slopes)}')
    # add text of slope to plot
    plt.text(0.5,0.5,f'Average slope: {np.mean(slopes):.2f}',transform = plt.gca().transAxes,fontsize = fontsize)

    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel('N')
    plt.ylabel('Error')
    plt.xticks([])
    plt.xticks(sizes,sizes,fontsize = tickfontsize)
    # turn off minor ticks
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())
    
    # clear tickmarks
    plt.legend()
    # pad left a little bit
    plt.tight_layout()
    plt.title('Error versus N for s = ' + str(s))
    plt.savefig('../Figures/err_vs_L_s' + str(s) + '.pdf')
        # Evaluate model

# def eval_model(model,data):
#     """
#     Evaluate the model on the given input
#     """
#     # Load model
#     model = load_model(model_info_path, model_path)

#     # Load data



if __name__ == '__main__':
    if len(sys.argv) > 1:
        s = int(sys.argv[1])

    with torch.no_grad():
        start_time = time.time()
        fig_err_vs_L(USE_CUDA = True,s = s)
        print('Time elapsed:',time.time()-start_time)
    # # arguments
    # model_info_path = sys.argv[1]
    # model_path = sys.argv[2]
    # data_path = sys.argv[3]

    # model = load_model(model_info_path,model_path)
    # input = load_data(data_path)

    # # Evaluate model
    # layers_output = model(input, invasive = True)
    # # Save layers output
    


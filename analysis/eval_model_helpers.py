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
from util.utilities_module import *
from matplotlib.ticker import ScalarFormatter
from models.shared import SpectralConv2d


# # Set font default
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
#                   '#f781bf', '#a65628', '#984ea3',
#                   '#999999', '#e41a1c', '#dede00','#a65628']
# matplotlib.rcParams['mathtext.rm'] = 'stix'
# matplotlib.rcParams['mathtext.it'] = 'stix'
# matplotlib.rcParams['mathtext.bf'] = 'stix'


# matplotlib.rcParams["axes.formatter.limits"] = (-99, 99) #makes scientific notation threshold high
# plt.rcParams['font.family'] = 'serif'  # or 'DejaVu Serif'
# plt.rcParams['font.serif'] = ['Times New Roman']  # 'DejaVu Serif' 'serif' 'Times
# # plt.rcParams['text.usetex'] = True
# # plt.rcParams['text.latex.preamble'] = r

# # \usepackage{amsmath}
# '''
# '''

# tickfontsize = 30
# fontsize = 30
# linewidth = 4
# markersize = 15

# SMALL_SIZE = tickfontsize
# MEDIUM_SIZE = tickfontsize
# BIGGER_SIZE = fontsize

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# shapes = ['o','s','^','D','*','o','s','^','D','*']

def load_model(model_info_path, model_path,s_outputspace = (2048,2048)):
    # Load model info
    with open(model_info_path, 'r') as file:
        model_info = yaml.load(file, Loader=yaml.FullLoader)
    K = model_info['K']
    width = model_info['width']
    try:
        get_grid = model_info['get_grid']
    except: 
        get_grid = False
    act = model_info['act']
    n_layers = model_info['n_layers']
    try:
        d_in = model_info['d_in']
    except:
        d_in = 1

    try:
        d_out = model_info['d_out']
    except:
        d_out = 1

    try:
        periodic = model_info['periodic_grid']
    except:
        periodic = False
    model = FNO2d(modes1 = K, modes2 = K, act = act,n_layers = n_layers, d_in = d_in,d_out = d_out, width = width, get_grid = get_grid, periodic_grid= periodic, s_outputspace = s_outputspace)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    except:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_model_info(model_info_path):
    # Load model info
    with open(model_info_path, 'r') as file:
        model_info = yaml.load(file, Loader=yaml.FullLoader)
    return model_info
    
def load_data(data_path):
    # Load data
    data = pkl.load(open(data_path, 'rb'))
    return data

def get_layer_output(model, input, invasive = True):
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        model = model.cuda()
    with torch.no_grad():
        layers_out = model(input, invasive = True, USE_CUDA = USE_CUDA)
            

    if USE_CUDA:
        layers_out = [layer.cpu() for layer in layers_out]

    return layers_out



def err_vs_L(USE_CUDA = False, s = None,get_plot = True,model_name = None,sizes = [32, 64,128, 256, 512]):
    """
    plot error vs L
    returns layers_true, layers_disc, and all_err
    """

    samp_count_input = 1
    samp_count_model = 1
    layer_count = 10
    all_err = torch.zeros(len(sizes)-1,layer_count, samp_count_model, samp_count_input)
    
    for model_sample in range(samp_count_model):
        model_info_path = '../models/random_initial_models/' + model_name + '_info.yaml'
        model_path = '../models/random_initial_models/' + model_name + '.pt'
        model = load_model(model_info_path, model_path,s_outputspace = (sizes[-1],sizes[-1]))
        if USE_CUDA:
            model.cuda()
        
        for input_sample in range(samp_count_input):
            print('input_sample:',input_sample)
            
            true_size = sizes[-1]
            input_true = load_data('../data/GRF_s' + str(s) +'/GRF_size_' + str(true_size) + '_' + str(input_sample)+ '.pkl')
            input_true = input_true.unsqueeze(0).unsqueeze(0)

            if USE_CUDA:
                model = model.cuda()
            with torch.no_grad():
                layers_true = model(input_true, invasive = True, USE_CUDA = USE_CUDA)
            

            if USE_CUDA:
                layers_true = [layer.cpu() for layer in layers_true]
            
            for size_i, size in enumerate(sizes[:-1]):
                input_disc = load_data('../data/GRF_s' +str(s) + f'/GRF_size_{size}_'+ str(input_sample)+'.pkl')
                input_disc = input_disc.unsqueeze(0).unsqueeze(0)
                x_res = input_disc.shape[-1]

                if USE_CUDA:
                    input_disc = input_disc.cuda()
                with torch.no_grad():
                    layers_disc = model(input_disc, invasive = True, USE_CUDA = USE_CUDA)
                
                if USE_CUDA:
                    layers_disc = [layer.cpu() for layer in layers_disc]

                
                for i in range(len(layers_disc)):

                    layer_true_i = layers_true[i]
                    layer_disc_i = layers_disc[i]

         
                    ss = true_size//size
                    # err = torch.norm(layer_disc_i[::ss,::ss] - layer_true_i[::ss,::ss]) # loss.Lp_err(layer_disc_i,layer_true_i_sub,size_average=False)
                    err = torch.norm(layer_disc_i - layer_true_i) # loss.Lp_err(layer_disc_i,layer_true_i_sub,size_average=False)

                    # all_err[size_i,i,model_sample,input_sample] = err
                    true_norm = torch.norm(layer_true_i) #loss.Lp_norm(layer_true_i_sub,size_average=False)
                    # all_err[size_i,i,model_sample,input_sample] = err/size
                    all_err[size_i,i,model_sample,input_sample] = err/true_norm

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
    # print(all_err_mean)
    # print(all_err_std)

    slopes = []
    for i in range(len(layers_true)):
        # pdb.set_trace()
        p = np.polyfit(np.log(sizes),np.log(all_err_mean[:,i]),1)
        # extract slop of best fit line
        slope = p[0]
        slopes.append(slope)
        print(f'Layer {i+1} slope: {slope}')

    print(f'Average slope: {np.mean(slopes)}')

    if get_plot:
        plt.figure(figsize = (10,10))

        for i in range(len(layers_true)):
            plt.plot(sizes,all_err_mean[:,i],label = f'Layer {i+1}',color = CB_color_cycle[i],marker = shapes[i],linewidth = linewidth,markersize = markersize)
            plt.errorbar(sizes, all_err_mean[:,i], yerr=all_err_std[:,i],color = CB_color_cycle[i])
            plt.fill_between(sizes, all_err_mean[:,i] - all_err_std[:,i], all_err_mean[:,i] + all_err_std[:,i], alpha=0.2, color = CB_color_cycle[i])
            # compute best fit line
        
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
        plt.savefig('../Figures/err_vs_L_s' + str(s) + '_L10.pdf')
        plt.savefig('../Figures/err_vs_L_s' + str(s) + '_L10.jpg')
            # Evaluate model

    return np.mean(slopes)

def fig_err_vs_K(s_s,K_s):
    slopes = np.zeros((len(s_s), len(K_s)))
    model_ind = 0
    sizes = [128, 256, 512,1024]
    for s_i, s in enumerate(s_s):
        print('s = ' + str(s)) 
        for K_i, K in enumerate(K_s):
            print('K = ' + str(K))
            model_name = 'initial_model_K_' + str(K) + '_' + str(model_ind) 
            
            slopes[s_i,K_i] = fig_err_vs_L(s = s,get_plot=False,model_name = model_name,sizes = sizes)
            

    pdb.set_trace()
    
    # average slops

    # add text of slope to plot
   
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
    if True:
        
        with torch.no_grad():
            start_time = time.time()
            model_sample = 0
            model_name = 'initial_model_'+ str(model_sample) 
            model_name = 'initial_model_K_12_L_10_'+ str(model_sample) 

            sizes = [32, 64,128, 256, 512] #,1024]
            fig_err_vs_L(USE_CUDA = False,s = s,model_name = model_name,sizes = sizes)
            print('Time elapsed:',time.time()-start_time)
        
    if False:
        with torch.no_grad():
            s_s = [1,2,3]
            K_s = [4,12,36,60]
            fig_err_vs_K(s_s, K_s)
        
    # # arguments
    # model_info_path = sys.argv[1]
    # model_path = sys.argv[2]
    # data_path = sys.argv[3]

    # model = load_model(model_info_path,model_path)
    # input = load_data(data_path)

    # # Evaluate model
    # layers_output = model(input, invasive = True)
    # # Save layers output
    


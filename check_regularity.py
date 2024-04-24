"""
Check regularity of GRF examples by plotting H^s norm versus N for different s
"""
import torch
import torch.fft
import numpy
import pdb
import scipy.fftpack
import sys
import os
import pickle as pkl
import numpy.linalg
import matplotlib.pyplot as plt
import matplotlib



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

shapes = ['o','s','^','D','*', 'v']

def load_data(filename):
    with open (filename, 'rb') as f:
        return pkl.load(f)
    
def Hs_norm(Z,s):
    """
    Compute the H^s norm of a 2D field
    """
    # take fft
    size = Z.shape[-1]
    # to double
    Z = Z.double()
    Z_hat = torch.fft.fft2(Z) # FFT of input 
    ks_x = torch.fft.fftfreq(size,d = 1/size) # 
    ks_y = torch.fft.fftfreq(size,d = 1/size) # Fourier modes in 1d, f = torch.arange((n + 1) // 2) / (d * n)
    # make the Fourier modes into a grid
    ks_x_2d, ks_y_2d = torch.meshgrid(ks_x, ks_y, indexing = 'ij')
    tau = 1 #torch.sqrt(torch.tensor(1/size))
    ks_x_2d = ks_x_2d*tau
    ks_y_2d = ks_y_2d*tau
    # to double
    ks_x_2d = ks_x_2d.double()
    ks_y_2d = ks_y_2d.double()
    # make tau a double    

    terms = (ks_x_2d**2 + ks_y_2d**2 + tau**2)**s * torch.abs(Z_hat)**2

    if torch.max(terms) > 1e10 or torch.min(terms) < 1e-10:
        print('Warning: terms are large or small')
        print('Max term: ', torch.max(terms))
        print('Min term: ', torch.min(terms))

    # print max over terms
    # pdb.set_trace()
    # sum over all modes
    scaling = size**2*tau**s
    return torch.sqrt(torch.sum(terms))/scaling
    
def plot_norms(sizes, s_s, norms,data_s,show = False):
    """
    Plot the norms
    """
    plt.figure(figsize = (10,10))
    for s_i, s in enumerate(s_s):
        plt.plot(sizes, norms[:,s_i], label = 's = ' + str(s), marker = shapes[s_i], color = CB_color_cycle[s_i])
    plt.xlabel('N')
    plt.ylabel(r'$H^s$ norm')
    # logscale
    plt.yscale('log')
    # not scientific notation
    plt.xscale('log')
    plt.legend()
    plt.title(r'$H^s$ norm of GRF with regularity $<$' + str(data_s))
    plt.savefig('Figures/check_reg/GRF_norms_s' + str(data_s) + '.jpg')
    if show:
        plt.show()
    else:
        plt.clf()



if __name__ == '__main__':
    sizes = [64, 128, 256, 512, 1024, 2048]
    s_s = [0, 1, 2, 3, 4, 5]
    norms = numpy.zeros((len(sizes),len(s_s)))
    data_s = 5
    for size_i, size in enumerate(sizes):
        xs, ys = numpy.meshgrid(numpy.linspace(0,1-1/size,size), numpy.linspace(0,1-1/size,size))
        xs = torch.from_numpy(xs)
        ys = torch.from_numpy(ys)
        for s_i, s in enumerate(s_s):
            # Z = torch.sin(2*numpy.pi*(xs + ys))
            # Z = torch.cos(2*numpy.pi*xs)
            # Z = torch.abs(xs - 0.5) 
            # Z = torch.sin(2*numpy.pi*(xs**(3/2))) + torch.sin(2*numpy.pi*(ys**(3/2)))
            # Z = 2*torch.ones(size,size)
            Z = load_data('data/GRF_s' + str(data_s) + '/GRF_size_' + str(size) + '_0.pkl')
            norm = Hs_norm(Z,s)
            # to numpy
            norm = norm.detach().numpy()
            norms[size_i,s_i] = norm

    plot_norms(sizes, s_s, norms, data_s)
    
    # plot
    

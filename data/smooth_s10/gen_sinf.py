"""
A script to generate Gaussian random field inputs in 2d for a specified regularity s
"""
import torch
import numpy
import pdb
import scipy.fftpack
import sys
import os
import pickle as pkl
# import numpy linalg
import numpy.linalg

# Copyright 2017 Bruno Sciolla. All Rights Reserved.
# ==============================================================================
# Generator for 2D scale-invariant Gaussian Random Fields
# ==============================================================================
# function from https://github.com/bsciolla/gaussian-random-fields/blob/master/gaussian_random_fields.py


def random_field(s,sizes):

    size = sizes[-1]
    field = torch.randn((size,size))

    # subsample for each size
    fields = [field[::int(size/size_),::int(size/size_)] for size_ in sizes]
    return fields

def sin_wave(s,sizes):
    size = sizes[-1]
    x = torch.linspace(0, 1, size)
    y = torch.linspace(0, 1, size)
    X, Y = torch.meshgrid(x, y)
    field = torch.sin(2 * numpy.pi * (X + Y))
    fields = [field[::int(size/size_),::int(size/size_)] for size_ in sizes]
    return fields

def gaussian_random_field(s,
                          sizes = [64, 128, 256, 512, 1024, 2048]):
    """ 
    Generate a 2D  Gaussian Random Field
    """
    # generate the same field for all sizes
    size = sizes[-1]
    d=2

    alpha = s + d/2

    # Defines momentum indices
    ks_x = torch.fft.fftfreq(size,d = 1/size)
    ks_y = torch.fft.fftfreq(size,d = 1/size)
    ks_x_2d, ks_y_2d = torch.meshgrid(ks_x, ks_y, indexing = 'ij')
    tau = torch.tensor(1/size)**(1/alpha)
    ks_x_2d = ks_x_2d*tau
    ks_y_2d = ks_y_2d*tau

    lambda_k = (ks_x_2d**2 + ks_y_2d**2 + tau**2)**(-alpha)
    if torch.max(lambda_k) > 1e10:
        print('Warning: lambda_k is large')
        print('Max lambda_k: ', torch.max(lambda_k))

    noise = torch.randn(size,size)
    gfield_fine = torch.fft.ifft2(noise*torch.sqrt(lambda_k)).real
    
    scaling = tau**(-alpha)
    gfield_fine = gfield_fine/scaling
    # pdb.set_trace()

    # # normalize to have L2 norm 1
    gfield_fine_norm = numpy.linalg.norm(gfield_fine)/size
    gfield_fine = gfield_fine/gfield_fine_norm

    # downsample
    gfields = [gfield_fine[::int(size/size_),::int(size/size_)] for size_ in sizes]
    return gfields
        
    # return gfield

def plot_GRF(Z,s,size = None,flag_3d = False):
    """
    Plot the generated Gaussian random field
    :param Z: a tensor of shape (N, N) containing the generated field
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(Z, cmap="viridis")
    plt.colorbar()
    if size == None:
        plt.savefig("Figures/GRF_ex_s_"+  str(s) +  ".pdf")
    else:
        plt.savefig("../../Figures/GRF_ex_s_"+  str(s) + "_size_" + str(size) + ".pdf")
        # plt.savefig("Figures/GRF_ex_s_"+  str(s) + "_size_" + str(size) + ".jpg")

    
    if flag_3d:
        # 3d plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = torch.linspace(0, 1, Z.shape[0])
        y = torch.linspace(0, 1, Z.shape[1])
        X, Y = torch.meshgrid(x, y, indexing='ij')
        ax.plot_surface(X, Y, Z, cmap="viridis")
        plt.savefig("Figures/GRF_3d_ex_s_"+  str(s) +  ".pdf")

    plt.clf()

# main loop
if __name__ == "__main__":
    SEED = 1989
    torch.manual_seed(SEED)
    numpy.random.seed(SEED)
    
    # process input arguments
    sizes = [64, 128, 256, 512, 1024, 2048]
    if len(sys.argv) > 1:
        s = int(sys.argv[1])
    
    if len(sys.argv) > 2:
        N = int(sys.argv[2])
    else: 
        N = 1
    
    for n in range(N):
        gfields = sin_wave(s,sizes)
        # save GRF
        for i, size in enumerate(sizes):
            Z = gfields[i]
            pkl.dump(Z, open("s10_size_" + str(size) + "_" + str(n) + ".pkl", "wb"))
            if size == 2048 and n == 0:
                plot_GRF(Z,s,size)

           
    # plot_GRF(Z,s)




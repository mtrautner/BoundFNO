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
    ks_y = torch.fft.rfftfreq(size,d = 1/size)
    ks_x_2d, ks_y_2d = torch.meshgrid(ks_x, ks_y, indexing = 'ij')

    
    lambda_k = (ks_x_2d**2 + ks_y_2d**2 + 1)**(-alpha/2)

    noise_re, noise_im = torch.randn(2,*lambda_k.shape)
    xi_noise = noise_re + 1j*noise_im
    u_ft2 = xi_noise*lambda_k
    gfield_fine = torch.fft.irfft2(u_ft2,norm = 'forward')

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
        plt.savefig("Figures/GRF_ex_s_"+  str(s) + "_size_" + str(size) + ".pdf")
        plt.savefig("Figures/GRF_ex_s_"+  str(s) + "_size_" + str(size) + ".jpg")

    
    if flag_3d:
        # 3d plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = torch.linspace(0, 2-1/size, 2*Z.shape[0])
        y = torch.linspace(0, 2-1/size, 2*Z.shape[1])
        X, Y = torch.meshgrid(x, y, indexing='ij')
        # duplicate Z in each direction once
        Z_expanded = torch.cat([Z, Z], dim = 0)
        Z_expanded = torch.cat([Z_expanded, Z_expanded], dim = 1)
        
        ax.plot_surface(X, Y, Z_expanded, cmap="viridis")
        plt.savefig("Figures/GRF_3d_ex_s_"+  str(s) +  ".pdf")

    plt.clf()

# main loop
if __name__ == "__main__":
    SEED = 1989
    torch.manual_seed(SEED)
    numpy.random.seed(SEED)
    
    # process input arguments
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    if len(sys.argv) > 1:
        s = int(sys.argv[1])
    
    if len(sys.argv) > 2:
        N = int(sys.argv[2])
    else: 
        N = 1
    
    for n in range(N):
        gfields = gaussian_random_field(s, sizes = sizes)
        # save GRF
        for i, size in enumerate(sizes):
            Z = gfields[i]
            pkl.dump(Z, open("data/GRF_s"+  str(s) + "/GRF_size_" + str(size) + "_" + str(n) + ".pkl", "wb"))

            if n == 0 and size == 512:
                # x = torch.linspace(0, 1, Z.shape[0])
                # y = torch.linspace(0, 1, Z.shape[1])
                # X, Y = torch.meshgrid(x, y, indexing='ij')
                # pkl.dump([x,y,X,Y], open("data/GRF_s" + str(s) + "/GRF_size_" +  str(size) + "_info.pkl", "wb"))
                plot_GRF(Z,s,size = size,flag_3d = False)
    # plot_GRF(Z,s)




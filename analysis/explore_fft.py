import torch
import pickle as pkl
import torch.fft as fft
import numpy as np
import pdb

def resize_rfft(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.irfft(ar, n=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Args
        ar: (..., N) tensor, must satisfy real conjugate symmetry (not checked)
        s: (int), desired irfft output dimension >= 1
    Output
        out: (..., s//2 + 1) tensor
    """
    N = ar.shape[-1]
    s = s//2 + 1 if s >=1 else s//2
    if s >= N: # zero pad or leave alone
        out = torch.zeros(list(ar.shape[:-1]) + [s - N], dtype=torch.cfloat, device=ar.device)
        out = torch.cat((ar[..., :N], out), dim=-1)
    elif s >= 1: # truncate
        out = ar[..., :s]
    else: # edge case
        raise ValueError("s must be greater than or equal to 1.")

    return out
  
def resize_fft(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.ifft(ar, n=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Reference: https://github.com/numpy/numpy/pull/7593
    Args
        ar: (..., N) tensor
        s: (int), desired ifft output dimension >= 1
    Output
        out: (..., s) tensor
    """
    N = ar.shape[-1]
    if s >= N: # zero pad or leave alone
        out = torch.zeros(list(ar.shape[:-1]) + [s - N], dtype=torch.cfloat, device=ar.device)
        out = torch.cat((ar[..., :N//2], out, ar[..., N//2:]), dim=-1)
    elif s >= 2: # truncate modes
        if s % 2: # odd
            out = torch.cat((ar[..., :s//2 + 1], ar[..., -s//2 + 1:]), dim=-1)
        else: # even
            out = torch.cat((ar[..., :s//2], ar[..., -s//2:]), dim=-1)
    else: # edge case s = 1
        if s < 1:
            raise ValueError("s must be greater than or equal to 1.")
        else:
            out = ar[..., 0:1]

    return out

def resize_rfft2(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.irfft2(ar, s=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Args
        ar: (..., N_1, N_2) tensor, must satisfy real conjugate symmetry (not checked)
        s: (2) tuple, s=(s_1, s_2) desired irfft2 output dimension (s_i >=1)
    Output
        out: (..., s1, s_2//2 + 1) tensor
    """
    s1, s2 = s
    out = resize_rfft(ar, s2) # last axis (rfft)
    return resize_fft(out.transpose(-2,-1), s1).transpose(-2,-1) # second to last axis (fft)

def load_data(data_path):
    # Load data
    data = pkl.load(open(data_path, 'rb'))
    return data

s = 1
size = 32
input_sample = 1
input_disc = load_data('../data/GRF_s' +str(s) + f'/GRF_size_{size}_'+ str(input_sample)+'.pkl')
input_disc = input_disc.unsqueeze(0).unsqueeze(0)
input_disc = input_disc[...,::8,::8]
input_size = input_disc.shape[-1]
print("Input N = " + str(input_size))
x = fft.rfft2(input_disc,s = (input_size,input_size),norm = 'forward')

def my_fft(x,r = True):
    N = x.shape[-1]
    x_s = torch.linspace(0,1-1/N, N)
    # x_n, y_n = torch.meshgrid(x_s,x_s,indexing = 'ij')
    K = input_size
    # k1s, k2s = torch.meshgrid(range(K), range(K),indexing = 'ij')
    if r:
        f_cs = torch.ones(K,K//2+1)*1j
        for k1_i, k1 in enumerate(range(K)):
            for k2_i, k2 in enumerate(range(K//2+1)):
                to_sum = torch.tensor([[x[...,i,j]*torch.exp(-2*1j*np.pi*(k1*x_s[i] + k2*x_s[j])) for i in range(N)] for j in range(N)])
                f_cs[k1_i,k2_i] = torch.sum(to_sum,dim = (0,1))
    else:
        f_cs = torch.ones(K,K)*1j
        for k1_i, k1 in enumerate(range(K)):
            for k2_i, k2 in enumerate(range(K)):
                to_sum = torch.tensor([[x[...,i,j]*torch.exp(-2*1j*np.pi*(k1*x_s[i] + k2*x_s[j])) for i in range(N)] for j in range(N)])
                f_cs[k1_i,k2_i] = torch.sum(to_sum,dim = (0,1))

    return f_cs/(N**2)

def my_ifft(x,N_out):
    N = x.shape[-2]
    k_s = torch.linspace(0,N-1,N)
    x_s = torch.linspace(0,1-1/N_out,N_out)
    f_out = torch.ones(N_out,N_out)*1j
    for n1_i , n1 in enumerate(range(N_out)):
        for n2_i, n2 in enumerate(range(N_out)):
            to_sum = torch.tensor([[x[...,i,j]*torch.exp(2*1j*np.pi*(k_s[i]*x_s[n1_i] + k_s[j]*x_s[n2_i])) for i in range(N)] for j in range(N)])
            f_out[n1_i,n2_i] = torch.sum(to_sum,dim = (-2,-1))
        
    
    return f_out

def conv2d(u,N):
  '''
  from google colab
  '''
  size = u.shape[-1]

  u_ft = torch.fft.rfft2(u,s = (size, size), norm = "forward") # s = (size, size) keeps it the same
  u_ft = resize_rfft2(u_ft,(N,N))
  out_ft = torch.fft.irfft2(u_ft, norm = "forward")
  
  return out_ft


x_fft = my_fft(input_disc,r = True)
x_fft_full = my_fft(input_disc,r = False)
x_fft_reshaped = resize_rfft2(x_fft,(2*input_size,2*input_size))
x_reshaped = resize_rfft2(x,(2*input_size,2*input_size))
# pdb.set_trace()
x_1 = fft.irfft2(x_reshaped,norm = "forward")
x_2 = my_ifft(x_fft_full,input_size)

pdb.set_trace()
    
'''
notes: 
input_disc shape [..., N, N]
rfft2 does no normalization. outputs shape [..., N, N//2 + 1] 
adding s = (N,N) to rfft2 does nothing
norm = "forward" multiplies by 1/N^2

resize_rfft2: input shape (K,K) reshaping to (N,N)

pads the rows in between K//2 and the last K//2 with zeros
it pads the columns after K//2 + 1 with zeros
'''

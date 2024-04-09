import torch

################################################################
#
# 1d helpers
#
################################################################
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
    # pdb.set_trace()
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


def resize_rfft2(ar, size):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.irfft2(ar, s=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Args
        ar: (..., N_1, N_2) tensor, must satisfy real conjugate symmetry (not checked)
        s: (2) tuple, s=(s_1, s_2) desired irfft2 output dimension (s_i >=1)
    Output
        out: (..., s1, s_2//2 + 1) tensor
    """
    out = resize_rfft(ar, size) # last axis (rfft)
    return resize_fft(out.transpose(-2,-1), size).transpose(-2,-1) # second to last axis (fft)




def upsample(state, size):
    """
    Input: state (torch.Tensor)
           size (int)
           
    Zero pad the Fourier modes of state so that state has new resolution (size,size)
    """
    if size is not None and size != state.shape[-1]:
        state = torch.fft.irfft2(resize_rfft2(torch.fft.rfft2(state, norm="forward"), size), s=(size,size), norm="forward")
        
    return state
    

def downsample(state, size):
    '''
    state of shape (B,C,s,s) will be downsampled to
    size (B,C,size,size).

    Requires s / size to be an integer!
    '''
    s = state.shape[-1]
    ss = s//size
    state = state[...,::ss,::ss]

    assert state.shape[-1]==size, f'Downsampling failed; require that size divides state.shape[-1]. Found {size=}, {state.shape[-1]=}, {s/size=}'
    return state

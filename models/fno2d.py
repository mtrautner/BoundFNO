import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

def compl_mul(input_tensor, weights):
    """
    Complex multiplication:
    (batch, in_channel, ...), (in_channel, out_channel, ...) -> (batch, out_channel, ...), where ``...'' represents the spatial part of the input.
    """
    return torch.einsum("bi...,io...->bo...", input_tensor, weights)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        """
        Fourier integral operator layer.
        """
        super().__init__()

        #
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes # number of Fourier modes (same in both directions)

        #
        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes, self.modes, dtype=torch.cfloat))

    def forward(self, x):
        """
        x: tensor with shape (batch, channels, nx, ny)
        """
        # Fourier transform
        x_ft = torch.fft.rfft2(x)

        # Fourier multiplication
        out_ft = torch.zeros(x_ft.shape[0], self.out_channels, *x_ft.shape[2:], dtype=torch.cfloat, device=x.device)
        out_ft[..., :self.modes, :self.modes] = compl_mul(x_ft[..., :self.modes, :self.modes],
                                                          self.weights1[:,:,:self.modes,:self.modes])
        out_ft[...,-self.modes:, :self.modes] = compl_mul(x_ft[...,-self.modes:, :self.modes],
                                                          self.weights2[:,:,:self.modes,:self.modes])

        # inverse Fourier transform
        out = torch.fft.irfft2(out_ft)
        return out

class MLP(nn.Module):
    '''
    Implements a pointwise NN with one hidden layer.
    '''
    def __init__(self,in_channels,out_channels,width,act='gelu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.act = _get_act(act)

        #
        self.in_layer = nn.Conv2d(in_channels,width,1)
        self.out_layer = nn.Conv2d(width,out_channels,1)

    def forward(self,x):
        x = self.in_layer(x)
        x = self.act(x)
        x = self.out_layer(x)
        return x
    
    
class LayerFNO2d(nn.Module):
    '''
    Implements a linear layer of FNO in two dimensions.
    '''
    def __init__(self,modes,width):
        super().__init__()
        self.modes = modes
        self.width = width

        self.conv_pointwise = nn.Conv2d(width,width,1)
        self.conv_global = SpectralConv2d(width,width,modes)


    def forward(self,x):
        x1 = self.conv_pointwise(x)
        x2 = self.conv_global(x)
        return x1 + x2



def _get_act(act):
    """
    https://github.com/NeuralOperator/PINO/blob/master/models/utils.py
    """
    if act == 'tanh':
        func = F.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    else:
        raise ValueError(f'{act} is not supported')
    return func
    
def _get_grid_xy(size):
    grid = torch.linspace(0,1,size)
    grid_xy = torch.stack(torch.meshgrid(grid,grid,indexing='ij'),dim=0)
    return grid_xy

def _get_grid_periodic(size):
    grid = torch.linspace(0,1,size+1)[:-1] # grid without endpoint
    sin = torch.sin(2*math.pi*grid)
    cos = torch.cos(2*math.pi*grid)
    
    grid_sin = torch.stack(torch.meshgrid(sin,sin,indexing='ij'), dim=0) # sin(x),sin(y)
    grid_cos = torch.stack(torch.meshgrid(cos,cos,indexing='ij'), dim=0) # cos(x),cos(y)
    grid_periodic = torch.cat((grid_sin,grid_cos), dim=0) # sin(x),sin(y),cos(x),cos(y)
    return grid_periodic



class FNO2d(nn.Module):
    '''
    Implements a Fourier neural operator in 2d.
    '''
    def __init__(self, modes, width, in_channels, out_channels,
                 depth=4, width_final=128, act='gelu', which_grid=None):
        super().__init__()
        #
        self.modes = modes
        self.width = width
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = _get_act(act)

        # check the choice of grid
        if not which_grid==None and not isinstance(which_grid, str):
            raise TypeError(f"Input which_grid must either be 'None' or of type 'string'; {which_string=} not allowed.")
        if which_grid==None:
            self.which_grid = None
            self.grid_channels = 0
        elif which_grid.lower()=='xy':
            self.which_grid = 'xy'
            self.grid_channels = 2
        elif which_grid.lower()=='periodic':
            self.which_grid = 'periodic'
            self.grid_channels = 4
        else:
            raise ValueError(f"which_grid must be either None, 'xy' or 'periodic'; found {which_grid=} instead.")

        #
        self.input_layer = nn.Conv2d(in_channels+self.grid_channels,width,1)
        self.hidden_layers = nn.ModuleList(
            LayerFNO2d(modes,width) for _ in range(depth)
        )
        self.output_layer = MLP(width,out_channels,width_final,act=act)

        
    def concatenate_grid(self,x):
        if self.which_grid==None:
            return x
        
        # concatenate appropriate grid
        batchsize, _, sizex, sizey = x.shape
        assert sizex==sizey, f'Assuming same discretiztion in x and y directions. Found {sizex=}, {sizey=}, instead.'
        size = sizex

        # non-periodic grid
        if self.which_grid=='xy':
            grid = _get_grid_xy(size)
        elif self.which_grid=='periodic':
            grid = _get_grid_periodic(size)
        #
        grid = repeat(grid, 'c x y->b c x y', b=batchsize).to(x.device)
        encoded = torch.cat((x, grid), dim=1)
        return encoded

    def forward(self,x):
        x = self.concatenate_grid(x)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)    # affine part
            x = self.act(x) # nonlinearity
        x = self.output_layer(x)
        return x

    def layer_outputs(self,x):
        layer_outputs = []
        with torch.no_grad():
            # input layer
            x = self.concatenate_grid(x)
            x = self.input_layer(x)
            layer_outputs.append( x.cpu() )

            # hidden layers
            for layer in self.hidden_layers:
                x = layer(x)
                x = self.act(x)
                layer_outputs.append( x.cpu() )

            # output layer
            x = self.output_layer(x)
            layer_outputs.append( x.cpu() )
            
            return layer_outputs
        

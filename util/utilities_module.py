
"""
First three functions adapted from https://github.com/zongyi-li/fourier_neural_operator/blob/master/utilities3.py
and https://github.com/nickhnelsen/FourierNeuralMappings 

"""

import torch
import operator
from functools import reduce
import numpy as np
import scipy.io
# import hdf5storage
import pdb
import yaml
import gc

#################################################
#
# utilities
#
#################################################

def to_torch(x, to_float=True):
    """
    send input numpy array to single precision torch tensor
    """
    if to_float:
        if np.iscomplexobj(x):
            x = x.astype(np.complex64)
        else:
            x = x.astype(np.float32)
    return torch.from_numpy(x)

def validate(f, fhat):
    '''
    Helper function to compute relative L^2 error of approximations.
    Takes care of different array shape interpretations in numpy.

    INPUTS:
            f : array of high-fidelity function values
         fhat : array of approximation values

    OUTPUTS:
        error : float, relative error
    '''
    f, fhat = np.asarray(f).flatten(), np.asarray(fhat).flatten()
    return np.linalg.norm(f-fhat) / np.linalg.norm(f)


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    # Reference: https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {'__getitem__': __getitem__,})

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    # Reference: https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {'__getitem__': __getitem__,})

class Sobolev_Loss(object):
    '''
    Loss object to compute H_1 loss or W_{1,p} loss, relative or absolute
    Assumes input shape is (num_examples, channels_out, grid_edge, grid_edge)
    Returns array of shape (num_examples,)
    '''
    def __init__(self, d=2, p=2, eps = 1e-10):
        self.d = d
        self.p = p
        self.eps =eps
    
    def compute_grad(self,x):
        grid_edge = x.size()[-1]
        h = 1.0 / (grid_edge - 1.0)

        x_grad = torch.gradient(x, dim = (-2,-1), spacing = h)
        x_grad1 = x_grad[0].unsqueeze(1) # Component 1 of the gradient
        x_grad2 = x_grad[1].unsqueeze(1) # Component 2 of the gradient
        x_grad = torch.cat((x_grad1, x_grad2), 1)
        return x_grad
    
    def Lp_norm(self,x,size_average = False):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[-1] - 1.0)
        if size_average:
            channel_width = x.size()[1]
            return torch.mean(torch.norm(x.reshape(num_examples,-1), self.p, dim=1)*h**(self.d/self.p)*(1/(channel_width)**(1/self.p)))
        else:
            return torch.norm(x.reshape(num_examples,-1), self.p, dim=1)*h**(self.d/self.p)
    
    def Lp_err(self,x,y,size_average = False):
        return self.Lp_norm(x-y,size_average = size_average)
    
    def Lp_rel_err(self,x,y,size_average = False):
        if size_average:
            return torch.mean(self.Lp_err(x,y)/(self.Lp_norm(y) + self.eps))
        else:
            return self.Lp_err(x,y)/(self.Lp_norm(y) + self.eps)
    
    def W1p_norm(self,x):
        x_grad = self.compute_grad(x)
        return (self.Lp_norm(x)**self.p + self.Lp_norm(x_grad)**self.p)**(1/self.p)
    
    def W1p_err(self,x,y):
        return self.W1p_norm(x-y)
    
    def W1p_rel_err(self,x,y):
        return self.W1p_err(x,y)/(self.W1p_norm(y)+self.eps)
    
def count_params(model):
    """
    print the number of parameters
    """
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

def convert_A_to_matrix_shape(A):
    '''
    input shape is (num_examples,3,grid_edge,grid_edge)
    output shape is (num_examples,2,2,grid_edge,grid_edge)
    '''
    # Add off-diagonal entry back to A
    num_examples = A.size()[0]
    grid_edge = A.size()[-1]
    off_diag = A[:,1,:,:].unsqueeze(1)
    A = torch.cat((A[:,:2,:,:],off_diag,A[:,2:,:,:]),1)
    A = torch.reshape(A,(num_examples,2,2,grid_edge,grid_edge))
    return A

def format_data(A_input, chi1_true, chi2_true, gridsize):
    # Reshape data
    sgc = gridsize
    (N_data, N_nodes,dummy1, dummy2) = np.shape(A_input)
    data_output1 = np.transpose(chi1_true[:,:]) # N_data, N_nodes
    data_output2 = np.transpose(chi2_true[:,:]) # N_data, N_nodes
    data_input = np.reshape(A_input, (N_data,sgc, sgc,4))
    data_input = np.delete(data_input,2,axis = 3) # Symmetry of A: don't need both components
    # Input shape (of x): (batch, channels_in, nx_in, ny_in)
    data_input = np.transpose(data_input, (0,3,1,2))

    #Output shape:      (batch, channels_out, nx_out, ny_out)
    data_output1 = np.reshape(data_output1, (N_data,sgc, sgc))
    data_output2 = np.reshape(data_output2, (N_data,sgc, sgc))
    # concatenate
    data_output = np.stack((data_output1,data_output2),axis = 3)
    data_output = np.transpose(data_output, (0,3,1,2))

    return data_input, data_output


def eval_net(net,d_out,gridsize,test_loader,b_size,USE_CUDA = False,N_data = 500):
    if USE_CUDA:
        gc.collect()
        torch.cuda.empty_cache()
    net.cuda()
    y_test_approx_all = torch.zeros(N_data,d_out,gridsize,gridsize)
    b = 0
    with torch.no_grad():
        for x,y in test_loader:
            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()
            y_pred = net(x)
            y_test_approx_all[b*b_size:(b+1)*b_size,:,:,:] = torch.squeeze(y_pred).cpu()
            b += 1
    return y_test_approx_all

def get_median_err_index(y_hat,y_true):
    H1_loss_func = Sobolev_Loss(p = 2)

    H1_rel_losses = H1_loss_func.W1p_rel_err(y_hat,y_true)
    
    H1_rel_med = torch.median(H1_rel_losses)
    # get index of median sample
    median_err_index = torch.where(H1_rel_losses == H1_rel_med)[0][0].item()
    return median_err_index

# def loss_report_f2v(Abar_hat,Abar_true,A_true,model_path):
#     '''
#     A_true shape is (num_examples,3,grid_edge,grid_edge)
#     y_true shape is (num_examples,4)
#     '''
#     A_true = convert_A_to_matrix_shape(A_true)
#     Aharms = frob_harm_mean_A(A_true)
#     Ameans = frob_arithmetic_mean_A(A_true)

#     Abar_abs_error = torch.norm(Abar_hat - Abar_true, 'fro', dim = (1))
#     true_frob_norm = torch.norm(Abar_true, 'fro', dim = (1))
#     Abar_rel_error = Abar_abs_error/true_frob_norm

#     Abar_rel_error2 = Abar_abs_error/(Ameans - Aharms)

#     # Make dictionary of the errors
#     errors = {}
#     errors['Abar_rel_error_med'] = torch.median(Abar_rel_error)
#     errors['Abar_rel_error2_med'] = torch.median(Abar_rel_error2)
#     errors['Abar_abs_error_med'] = torch.median(Abar_abs_error)
#     errors['Abar_abs_error_mean'] = torch.mean(Abar_abs_error)

#     json_errors = {k: v.item() for k, v in errors.items()}
#     # Save dictionary to json file
#     with open(model_path+ '_errors.yml', 'w') as fp:
#         yaml.dump(json_errors, fp)

def loss_report(y_hat, y_true, model_path):
    '''
    y_true shape is (num_examples,1,grid_edge,grid_edge)
    '''

    loss_func = Sobolev_Loss(d=2,p=2)

    loss = loss_func.Lp_rel_err(y_hat,y_true,size_average = True)
    loss_abs = loss_func.Lp_err(y_hat,y_true,size_average = True)

    # print both to .yaml file
    errors = {}
    errors['loss'] = loss.item()
    errors['loss_abs'] = loss_abs.item()
    json_errors = {k: v for k, v in errors.items()}
    # Save dictionary to json file
    with open(model_path+ '_errors.yml', 'w') as fp:
        yaml.dump(json_errors, fp)
    


# def loss_report(y_hat, y_true, A_true, model_path):
#     '''
#     A_true shape is (num_examples,3,grid_edge,grid_edge)
#     y_true shape is (num_examples,2,grid_edge,grid_edge)
#     '''
#     A = convert_A_to_matrix_shape(A_true)

#     H1_loss_func = Sobolev_Loss(p = 2)
#     W1_10_loss_func = Sobolev_Loss(p = 10)

#     H1_losses = H1_loss_func.W1p_err(y_hat,y_true)
#     H1_rel_losses = H1_loss_func.W1p_rel_err(y_hat,y_true)

#     W1_10_losses = W1_10_loss_func.W1p_err(y_hat,y_true)
#     W1_10_rel_losses = W1_10_loss_func.W1p_rel_err(y_hat,y_true)

#     H1_mean = torch.mean(H1_losses)
#     H1_med = torch.median(H1_losses)
#     H1_rel_mean = torch.mean(H1_rel_losses)
#     H1_rel_med = torch.median(H1_rel_losses)

#     W1_10_mean = torch.mean(W1_10_losses)
#     W1_10_med = torch.median(W1_10_losses)
#     W1_10_rel_mean = torch.mean(W1_10_rel_losses)
#     W1_10_rel_med = torch.median(W1_10_rel_losses)

#     Abar_rel_error, Abar_rel_error2 = compute_Abar_error(A,y_true,y_hat)
#     # Compute Abar abs error
#     Abar_hat = compute_Abar(A,y_hat)
#     Abar_true = compute_Abar(A,y_true)
#     Abar_abs_error = torch.norm(Abar_hat - Abar_true, 'fro', dim = (1))
#     # Make dictionary of the errors
#     errors = {}
#     errors['H1_mean'] = H1_mean
#     errors['H1_med'] = H1_med
#     errors['H1_rel_mean'] = H1_rel_mean
#     errors['H1_rel_med'] = H1_rel_med
#     errors['W1_10_mean'] = W1_10_mean
#     errors['W1_10_med'] = W1_10_med
#     errors['W1_10_rel_mean'] = W1_10_rel_mean
#     errors['W1_10_rel_med'] = W1_10_rel_med
#     errors['Abar_rel_error_med'] = torch.median(Abar_rel_error)
#     errors['Abar_rel_error2_med'] = torch.median(Abar_rel_error2)
#     errors['Abar_abs_error_med'] = torch.median(Abar_abs_error)
#     errors['Abar_abs_error_mean'] = torch.mean(Abar_abs_error)
#     json_errors = {k: v.item() for k, v in errors.items()}
#     # Save dictionary to json file
#     with open(model_path+ '_errors.yml', 'w') as fp:
#         yaml.dump(json_errors, fp)



    # with open(model_path+ '_errors.yml', 'w') as fp:
    #     yaml.dump(json_errors, fp)
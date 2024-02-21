
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

class H1Loss(object):
    """
    loss function with rel/abs H1 loss
    """
    def __init__(self, d=2, size_average=True, reduction=True, eps=1e-6):
        # super(LpLoss, self).__init__()

        self.d = d
        self.p = 2
        self.reduction = reduction
        self.size_average = size_average
        self.eps =eps
    
    def rel_H1(self, x, y):
        num_examples = x.size()[0]
        grid_edge = x.size()[-1]
        x = torch.squeeze(x)
        y = torch.squeeze(y)
        h = 1.0 / (grid_edge - 1.0)
        x_grad = torch.gradient(x,dim = (-2,-1), spacing = h)
        x_grad1 = x_grad[0].unsqueeze(1) # Component 1 of the gradient
        x_grad2 = x_grad[1].unsqueeze(1) # Component 2 of the gradient
        x_grad = torch.cat((x_grad1, x_grad2), 1) # num_examples, (grad component) 2, (\chi_1, \chi_2) 2, grid_edge, grid_edge
        y_grad = torch.gradient(y, dim = (-2,-1), spacing = h)
        y_grad1 = y_grad[0].unsqueeze(1) # Component 1 of the gradient
        y_grad2 = y_grad[1].unsqueeze(1) # Component 2 of the gradient
        y_grad = torch.cat((y_grad1, y_grad2), 1) 

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        diff_grad_norms = torch.norm(x_grad.reshape(num_examples, -1) - y_grad.reshape(num_examples, -1), self.p, 1)
        diff_norms = (diff_norms**2 + diff_grad_norms**2)**(1/2)
        y_norms = ((torch.norm(y.reshape(num_examples,-1), self.p, 1))**2 + (torch.norm(y_grad.reshape(num_examples,-1), self.p, 1))**2)**(1/2)
        y_norms += self.eps     # prevent divide by zero
        rel_norms = torch.div(diff_norms,y_norms)

        if self.reduction:
            if self.size_average:
                return torch.mean(rel_norms)
            else:
                return torch.median(rel_norms)

        return rel_norms

    def squared_H1(self, x, y):
        num_examples = x.size()[0]
        grid_edge = x.size()[-1]
        x = torch.squeeze(x) # shape is num_examples x channels_out x  grid_edge x grid_edge
        y = torch.squeeze(y) # shape is num_examples x channels_out x grid_edge x grid_edge 
        h = 1.0 / (grid_edge - 1.0)

        x_grad = torch.gradient(x, dim = (-2,-1), spacing = h)
        x_grad1 = x_grad[0].unsqueeze(1) # Component 1 of the gradient
        x_grad2 = x_grad[1].unsqueeze(1) # Component 2 of the gradient
        x_grad = torch.cat((x_grad1, x_grad2), 1) # num_examples, (grad component) 2, (\chi_1, \chi_2) 2, grid_edge, grid_edge
        y_grad = torch.gradient(y, dim = (-2,-1), spacing = h)
        y_grad1 = y_grad[0].unsqueeze(1) # Component 1 of the gradient
        y_grad2 = y_grad[1].unsqueeze(1) # Component 2 of the gradient
        y_grad = torch.cat((y_grad1, y_grad2), 1) # num_examples, 2, grid_edge, grid_edge

        diff_L2 = h*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, dim = 1)

        grad_euclidean = torch.norm(x_grad - y_grad, 2, dim = 1)
        diff_grad_L2 = h*torch.norm(grad_euclidean.reshape(num_examples,-1),2,1)
        sum_sq = diff_L2**2 + diff_grad_L2**2

        if self.reduction:
            if self.size_average:
                return torch.mean(sum_sq)
            else:
                return torch.sum(sum_sq)

        return sum_sq

def frob_loss(x, y):
    '''
    x has dimension (num_examples,d_out)
    y has dimension (num_examples,d_out)
    '''
    frob_norm_sq = torch.norm(x-y,'fro',dim = 1)**2
    return torch.mean(frob_norm_sq)

class LpLoss(object):
    """
    loss function with rel/abs Lp norm loss
    """
    def __init__(self, d=2, p=2, size_average=True, reduction=True, eps=1e-6):
        super(LpLoss, self).__init__()

        if not (d > 0 and p > 0):
            raise ValueError("Dimension d and Lp-norm type p must be postive.")

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.eps =eps

    def abs(self, x, y):
        num_examples = x.size()[0]


        #Assume uniform mesh
        h = 1.0 / (x.size()[-1] - 1.0)
        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        y_norms += self.eps     # prevent divide by zero
        mean_y_norm = torch.mean(y_norms)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/mean_y_norm)
            else:
                return torch.sum(diff_norms/mean_y_norm)

        return diff_norms/mean_y_norm

    def __call__(self, x, y):
        return self.squared_H1(x, y)


class Sobolev_Loss(object):
    '''
    Loss object to compute H_1 loss or W_{1,p} loss, relative or absolute
    Assumes input shape is (num_examples, channels_out, grid_edge, grid_edge)
    Returns array of shape (num_examples,)
    '''
    def __init__(self, d=2, p=2, eps = 1e-6):
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
            return torch.norm(x.reshape(num_examples,-1), self.p, dim=1)*h**(self.d/self.p)*(1/(channel_width)**(1/self.p))
        else:
            return torch.norm(x.reshape(num_examples,-1), self.p, dim=1)*h**(self.d/self.p)
    
    def Lp_err(self,x,y,size_average = False):
        return self.Lp_norm(x-y,size_average = size_average)
    
    def Lp_rel_err(self,x,y):
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

def compute_Abar(A, chi):
    '''
    Computes Abar from A and chi
    Abar = \int_{\Td} (A + A\grad\chi^T) dx
    chi has shape (num_examples, 2, grid_edge, grid_edge)
    A has shape (num_examples, 2,2, grid_edge, grid_edge)

    Returns Abar of shape (num_examples, 2,2)
    '''
    # num_examples = chi.size()[0]
    grid_edge = chi.size()[-1]
    h = 1.0 / (grid_edge - 1.0)
    
    # Compute grad chi
    chi_grad = torch.gradient(chi, dim = (-2,-1), spacing = h)
    chi_grad1 = chi_grad[0].unsqueeze(1) # Component 1 of the gradient
    chi_grad2 = chi_grad[1].unsqueeze(1) # Component 2 of the gradient
    chi_grad = torch.cat((chi_grad1, chi_grad2), 1)

    # Compute integrand
    # Multiply A (axes 1 and 2) by chi_grad (axis 1)
    integrand = A + torch.einsum('iablm,ibdlm->iadlm',A,chi_grad) 
    Abars = torch.sum(integrand, dim = (-2,-1))*h**2
    return Abars

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

def format_data_Abar(A_input, chi1_true, chi2_true, gridsize):
    # Reshape data
    sgc = gridsize
    (N_data, N_nodes,dummy1, dummy2) = np.shape(A_input)
    data_output1 = np.transpose(chi1_true[:,:]) # N_data, N_nodes
    data_output2 = np.transpose(chi2_true[:,:]) # N_data, N_nodes
    data_input = np.reshape(A_input, (N_data,sgc, sgc,4))
    # Input shape (of x): (batch, channels_in, nx_in, ny_in)
    data_input = np.transpose(data_input, (0,3,1,2))
    # Break up A into 2x2 matrix at positition 1
    data_input = np.reshape(data_input,(N_data,2,2,sgc,sgc))

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

def frob_arithmetic_mean_A(A):
    '''
    A has shape (num_examples, 2,2, grid_edge, grid_edge)
    '''
    h = 1.0 / (A.size()[-1] - 1.0)
    mean_A = torch.sum(A,dim = (-2,-1))*h**2
    return torch.norm(mean_A, 'fro', dim = (1,2))


def frob_harm_mean_A(A):
    '''
    A has shape (num_examples, 2,2, grid_edge, grid_edge)
    '''
    h = 1.0 / (A.size()[-1] - 1.0)
    A = torch.reshape(A,(A.size()[0],2,2,-1))
    inverses = np.array([np.array([np.linalg.inv(A[i,:,:,j]) for i in range(A.size()[0])]) for j in range(A.size()[3])])
    inverses = torch.from_numpy(inverses).float()
    harm_mean_A = torch.sum(inverses,dim = (0))*h**2
    inv_har_mean_A = np.array([np.linalg.inv(harm_mean_A[i,:,:]) for i in range(harm_mean_A.size()[0])])
    torch_inv_har_mean_A = torch.from_numpy(inv_har_mean_A).float()
    return torch.norm(torch_inv_har_mean_A, 'fro', dim = (1,2))


def compute_Abar_error(A,chi_true,chi_hat):
    '''
    A has shape (num_examples, 2,2, grid_edge, grid_edge)
    returns Abar_rel_error scaled by true frob norm
    returns Abar_rel_error2 scaled by a_m - a_h
    '''
    Abars_true = compute_Abar(A,chi_true)
    Abars_hat = compute_Abar(A,chi_hat)
    Aharms = frob_harm_mean_A(A)
    Ameans = frob_arithmetic_mean_A(A)

    Abar_abs_error = torch.norm(Abars_true - Abars_hat, 'fro', dim = (1,2))
    true_frob_norm = torch.norm(Abars_true, 'fro', dim = (1,2))
    Abar_rel_error = Abar_abs_error/true_frob_norm

    Abar_rel_error2 = Abar_abs_error/(Ameans - Aharms)

    return Abar_rel_error, Abar_rel_error2


def get_median_err_index(y_hat,y_true):
    H1_loss_func = Sobolev_Loss(p = 2)

    H1_rel_losses = H1_loss_func.W1p_rel_err(y_hat,y_true)
    
    H1_rel_med = torch.median(H1_rel_losses)
    # get index of median sample
    median_err_index = torch.where(H1_rel_losses == H1_rel_med)[0][0].item()
    return median_err_index

def loss_report_f2v(Abar_hat,Abar_true,A_true,model_path):
    '''
    A_true shape is (num_examples,3,grid_edge,grid_edge)
    y_true shape is (num_examples,4)
    '''
    A_true = convert_A_to_matrix_shape(A_true)
    Aharms = frob_harm_mean_A(A_true)
    Ameans = frob_arithmetic_mean_A(A_true)

    Abar_abs_error = torch.norm(Abar_hat - Abar_true, 'fro', dim = (1))
    true_frob_norm = torch.norm(Abar_true, 'fro', dim = (1))
    Abar_rel_error = Abar_abs_error/true_frob_norm

    Abar_rel_error2 = Abar_abs_error/(Ameans - Aharms)

    # Make dictionary of the errors
    errors = {}
    errors['Abar_rel_error_med'] = torch.median(Abar_rel_error)
    errors['Abar_rel_error2_med'] = torch.median(Abar_rel_error2)
    errors['Abar_abs_error_med'] = torch.median(Abar_abs_error)
    errors['Abar_abs_error_mean'] = torch.mean(Abar_abs_error)

    json_errors = {k: v.item() for k, v in errors.items()}
    # Save dictionary to json file
    with open(model_path+ '_errors.yml', 'w') as fp:
        yaml.dump(json_errors, fp)


def loss_report(y_hat, y_true, A_true, model_path):
    '''
    A_true shape is (num_examples,3,grid_edge,grid_edge)
    y_true shape is (num_examples,2,grid_edge,grid_edge)
    '''
    A = convert_A_to_matrix_shape(A_true)

    H1_loss_func = Sobolev_Loss(p = 2)
    W1_10_loss_func = Sobolev_Loss(p = 10)

    H1_losses = H1_loss_func.W1p_err(y_hat,y_true)
    H1_rel_losses = H1_loss_func.W1p_rel_err(y_hat,y_true)

    W1_10_losses = W1_10_loss_func.W1p_err(y_hat,y_true)
    W1_10_rel_losses = W1_10_loss_func.W1p_rel_err(y_hat,y_true)

    H1_mean = torch.mean(H1_losses)
    H1_med = torch.median(H1_losses)
    H1_rel_mean = torch.mean(H1_rel_losses)
    H1_rel_med = torch.median(H1_rel_losses)

    W1_10_mean = torch.mean(W1_10_losses)
    W1_10_med = torch.median(W1_10_losses)
    W1_10_rel_mean = torch.mean(W1_10_rel_losses)
    W1_10_rel_med = torch.median(W1_10_rel_losses)

    Abar_rel_error, Abar_rel_error2 = compute_Abar_error(A,y_true,y_hat)
    # Compute Abar abs error
    Abar_hat = compute_Abar(A,y_hat)
    Abar_true = compute_Abar(A,y_true)
    Abar_abs_error = torch.norm(Abar_hat - Abar_true, 'fro', dim = (1))
    # Make dictionary of the errors
    errors = {}
    errors['H1_mean'] = H1_mean
    errors['H1_med'] = H1_med
    errors['H1_rel_mean'] = H1_rel_mean
    errors['H1_rel_med'] = H1_rel_med
    errors['W1_10_mean'] = W1_10_mean
    errors['W1_10_med'] = W1_10_med
    errors['W1_10_rel_mean'] = W1_10_rel_mean
    errors['W1_10_rel_med'] = W1_10_rel_med
    errors['Abar_rel_error_med'] = torch.median(Abar_rel_error)
    errors['Abar_rel_error2_med'] = torch.median(Abar_rel_error2)
    errors['Abar_abs_error_med'] = torch.median(Abar_abs_error)
    errors['Abar_abs_error_mean'] = torch.mean(Abar_abs_error)
    json_errors = {k: v.item() for k, v in errors.items()}
    # Save dictionary to json file
    with open(model_path+ '_errors.yml', 'w') as fp:
        yaml.dump(json_errors, fp)



    # with open(model_path+ '_errors.yml', 'w') as fp:
    #     yaml.dump(json_errors, fp)
import sys
import os
sys.path.append(os.path.join('..'))
from analysis.eval_model_helpers import *
from models.func_to_func2d_invasive import FNO2d
from util.utilities_module import *
from gen_GRF import *
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle as pkl

smooth_data_file = '/groups/astuart/mtrautne/learnHomData/data/smooth_data.pkl'
smooth_data = pkl.load(open(smooth_data_file,'rb'))
A_input, chi_1, chi_2, x_ticks, y_ticks = smooth_data

# delete chi2
chi_2= None

print(A_input.shape)
print(chi_1.shape)

# A_input = torch.tensor(A_input)
sgc = len(x_ticks[0])
print(sgc)
N_data = A_input.shape[0]
# chi_1 = torch.tensor(chi_1)
# chi_2 = torch.tensor(chi_2)
# x_ticks = torch.tensor(x_ticks)
# y_ticks = torch.tensor(y_ticks)
A_input = np.reshape(A_input, (N_data,sgc,sgc,4))

A_input = np.delete(A_input, 2, axis = 3)
A_input = np.transpose(A_input, (0,3,1,2))
A_input = torch.tensor(A_input).float()
data_path = 'smooth_training_data/A_to_chi1' 
torch.save(A_input, data_path + '_input_data.pt')
A_input = None

data_output = np.reshape(chi_1, (N_data, sgc, sgc, 1))
data_output = np.transpose(data_output, (0,3,1,2))

# to torch array
data_output = torch.tensor(data_output).float()

# save the data

torch.save(data_output, data_path + '_output_data.pt')

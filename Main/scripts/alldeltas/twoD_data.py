import numpy as np
import tensorflow as tf
import sys
sys.path.append('../..')
from dset_helpers import data_given_param
from train_data import Train_w_Data

Lx = 16
Ly = 16
Omega = 4.24
Rb = 1.15
V0 = Rb**6 * Omega
sweep_rate = 15
delta = -1.545

def main():
    config = {
        'name': 'TwoD_data_alldeltas', # A very random name for each experiment

        'Lx': Lx,  # number of sites in x-direction                    
        'Ly': Ly,  # number of sites in the y-direction
        'V': V0,
        'Omega': Omega,
        'delta': delta,
        'sweep_rate':sweep_rate,
        
        'nh': 16,
        'lr': 1e-4,
        'weight_sharing': True,
        'trunc': 100,
        'seed': 1234,
        
        'RNN': 'TwoD',
        'MDGRU':True,
        'VMC_epochs':0,
        'Data_epochs':10000,
        
        'ns': 100,
        'batch_samples': False,
        'batch_size_data': 100,
        'data_step': 1,
        
        'Print':True,
        'Write_Data': True,
        'CKPT': True
        }

    
    return Train_w_Data(config)


if __name__ == "__main__":
    model,e,v,c = main()

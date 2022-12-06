import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
from dset_helpers import data_given_param
from train_data import Train_w_Data

Lx = 16
Ly = 16
Omega = 4.24
Rb = 1.15
V0 = Rb**6 * Omega
sweep_rate = 15
low = -1.545

energy = []
variance = []
cost = []

def main():
    config = {
        'name': 'OneD_data_lowdelta', # A very random name for each experiment

        'Lx': Lx,  # number of sites in x-direction                    
        'Ly': Ly,  # number of sites in the y-direction
        'V': V0,
        'Omega': Omega,
        'delta': low,
        'sweep_rate':sweep_rate,
        
        'nh': 32,  # number of memory/hidden units
        'lr': 1e-3,  # learning rate
        'weight_sharing': True,
        'trunc': 100,
        'seed': 1234,
        
        'RNN': 'OneD',
        'VMC_epochs':0,
        'Data_epochs':500,
        
        'ns': 1000,
        'batch_size': 100,
        'data_step': 1, 
        
        'Print':True,
        'Write_Data': True,
        'CKPT':True
        }
    
    return Train_w_Data(config,energy,variance,cost)


if __name__ == "__main__":
    model,e,v,c = main()

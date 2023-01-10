import numpy as np
import tensorflow as tf
import sys
sys.path.append('../..')
from dset_helpers import data_given_param
from train_VMC import Train_w_VMC

Lx = 16
Ly = 16
Omega = 4.24
Rb = 1.15
V0 = Rb**6 * Omega
sweep_rate = 15
delta = 10.955

def main():
    config = {
        'name': 'OneD_VMC_alldeltas_nh64',

        'Lx': Lx,  # number of sites in x-direction                    
        'Ly': Ly,  # number of sites in the y-direction
        'V': V0,
        'Omega': Omega,
        'delta': delta,
        'sweep_rate':sweep_rate,
        
        'nh': 64,  # number of memory/hidden units
        'lr': 1e-3,  # learning rate
        'weight_sharing': True,
        'trunc': 100,
        'seed': 1234,
        
        'RNN': 'OneD',
        'MDGRU':True,
        'VMC_epochs':10000,
        'Data_epochs':0,
        
        'ns': 100,
        'batch_samples':False,
        
        'Print':True,
        'Write_Data': True,
        'CKPT':True
        }
    
    return Train_w_VMC(config)


if __name__ == "__main__":
    model,e,v,c = main()

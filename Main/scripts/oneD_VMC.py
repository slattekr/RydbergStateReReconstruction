import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import sys
sys.path.append('..')
from dset_helpers import data_given_param
from train_VMC import Train_w_VMC

Lx = 16
Ly = 16
Omega = 4.24
Rb = 1.15
V0 = Rb**6 * Omega
sweep_rate = 15
low = -1.545

def main():
    config = {
        'name': 'OneD_VMC_lowdelta', # A very random name for each experiment

        'Lx':Lx,  # number of sites in x-direction                    
        'Ly':Ly,  # number of sites in the y-directioni
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
        'VMC_epochs':1000,
        'Data_epochs':0,
        
        'ns': 1000,
        'batch_size': 100,
        'data_step': 100,
        
        'Print':True,
        'Write_Data': True,
        'CKPT':True
        }
    
    return Train_w_VMC(config)


if __name__ == "__main__":
    model,e,v,c = main()

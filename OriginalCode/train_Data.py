import os
import tensorflow as tf
import numpy as np
from New_dset_helpers import load_exact_Es,load_QMC_data,create_tf_dataset
from New_OneD_RNN import OneD_RNN_wavefxn
from New_TwoD_RNN import MDRNNWavefunction,MDTensorizedRNNCell,MDRNNGRUcell
from helpers import save_path
import matplotlib.pyplot as plt

def Train_w_Data(config,energy,variance):

    '''
    Run RNN using vmc sampling or qmc data. If qmc_data is None, uses vmc sampling. 
    Otherwise uses qmc data loaded in qmc_data
    '''

    # System Parameters 
    Lx = config['Lx']
    Ly = config['Ly']
    V = config['V']
    delta = config['delta']
    Omega = config['Omega']

    # RNN Parameters
    num_hidden = config['nh']
    learning_rate = config['lr']
    weight_sharing = config['weight_sharing']
    trunc = config['trunc']
    seed = config['seed']

    # Initiate RNN wave function
    if config['RNN'] == 'OneD':
        if config['Print'] ==True:
            print(f"Training a one-D RNN wave function with {num_hidden} hidden units and shared weights.")
        wavefxn = OneD_RNN_wavefxn(Lx,Ly,V,Omega,delta,num_hidden,learning_rate,weight_sharing,trunc,seed)
    elif config['RNN'] =='TwoD':
        if config['Print'] ==True:
            print(f"Training a two-D RNN wave function with {num_hidden} hidden units and shared weights = {weight_sharing}.")
        if config['MDGRU']:
            print("Using GRU cell.")
            wavefxn = MDRNNWavefunction(Lx,Ly,V,Omega,delta,num_hidden,learning_rate,weight_sharing,trunc,seed,cell=MDRNNGRUcell)
        else:
            wavefxn = MDRNNWavefunction(Lx,Ly,V,Omega,delta,num_hidden,learning_rate,weight_sharing,trunc,seed,cell=MDTensorizedRNNCell)
    else:
        raise ValueError(f"{config['RNN']} is not a valid option for the RNN wave function. Please choose OneD or TwoD.")

    if config['Print'] ==True:
        print(f"The experimental parameters are: V = {V}, delta = {delta}, Omega = {Omega}.")
        print(f"The system is an array of {Lx} by {Ly} Rydberg Atoms")

    @tf.function
    def train_step(input_batch):
        print("Tracing!")
        with tf.GradientTape() as tape:
            logpsi = wavefxn.logpsi(input_batch,initial_pass=False)
            loss = - 2.0 * tf.reduce_mean(logpsi)
        # Compute the gradients either with qmc_loss
        gradients = tape.gradient(loss, wavefxn.trainable_variables)
        clipped_gradients = [tf.clip_by_value(g, -10., 10.) for g in gradients]
        # Update the parameters
        wavefxn.optimizer.apply_gradients(zip(clipped_gradients, wavefxn.trainable_variables))

    # Training Parameters
    ns = config['ns']
    batch_size = config['batch_size']
    data_step = config['data_step']
    epochs = config['Data_epochs']
    exact_e = load_exact_Es(Lx)
    data = load_QMC_data(Lx)
    tf_dataset = create_tf_dataset(data,data_step)

    for n in range(1, epochs+1):
        #use data to update RNN weights
        dset = tf_dataset.shuffle(len(tf_dataset))
        dset = dset.batch(batch_size)
        
        for i, batch in enumerate(dset):
            # Evaluate the loss function in AD mode
            train_step(batch)

        #append the energy to see convergence
        samples, _ = wavefxn.sample(ns,initial_pass=False)
        sample_logpsi = wavefxn.logpsi(samples,initial_pass=False)
        sample_eloc = wavefxn.localenergy(samples, sample_logpsi)
        energies = sample_eloc.numpy()
        avg_E = np.mean(energies)/float(wavefxn.N)
        var_E = np.var(energies)/float(wavefxn.N)
        energy.append(avg_E)
        variance.append(var_E)

        if (config['Print'] ==True):
            print(f"Step #{n}")
            print(f"Energy = {avg_E}")
            print(f"Variance = {var_E}")
            print(" ")
    
    if config['Write_Data']==True:
        samples_final,_ = wavefxn.sample(10000,initial_pass=False)
        path = config['save_path']
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+'/config.txt', 'w') as file:
            for k,v in config.items():
                file.write(k+f'={v}\n')
        np.save(path+'/Energy',energy)
        np.save(path+'/Variance',variance)
        np.save(path+'/Samples',samples)
    
    if config['Plot']:
        plt.title("Energy")
        plt.plot(np.arange(len(energy)),energy,label="RNN")
        plt.hlines(exact_e,0,len(energy),label="exact E")
        plt.legend()
        plt.xlabel("steps")
        plt.ylabel("RNN energy")
        plt.show()
        plt.title("Variance")
        plt.plot(np.arange(len(variance)),variance)
        plt.xlabel("steps")
        plt.ylabel("Variance")
        plt.show()
            
    return wavefxn, energy, variance

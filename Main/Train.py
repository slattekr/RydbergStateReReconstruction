import os
import tensorflow as tf
import numpy as np
from OneD_RNN import OneD_RNN_wavefxn
from TwoD_RNN import MDRNNWavefunction
from helpers import save_path
from dset_helpers import load_QMC_data,create_tf_dataset
from plots import plot_E,plot_var,plot_loss

def train_w_data(config,energy,variance,cost):

    '''
    Train an RNN to represent a quantum ground state wave function using data.
    '''

    # Set the parameters of the System 
    Lx = config['Lx']
    Ly = config['Ly']
    V = config['V']
    delta = config['delta']
    Omega = config['Omega']
    trunc = config['trunc']
    if config['Print'] ==True:
        print(f"The system is an array of {Lx} by {Ly} Rydberg Atoms.")
        print(f"The experimental parameters are: V = {V}, delta = {delta}, Omega = {Omega}.")

    # Set the parameters of the RNN 
    num_hidden = config['nh']
    learning_rate = config['lr']
    weight_sharing = config['weight_sharing']
    seed = config['seed']

    # Construct the RNN wave function, one-D or two-d
    if config['RNN'] == 'OneD':
        if config['Print'] ==True:
            print(f"Training a one-D RNN wave function with {num_hidden} hidden units and shared weights.")
        wavefxn = OneD_RNN_wavefxn(Lx,Ly,V,Omega,delta,num_hidden,learning_rate,weight_sharing,trunc,seed)
    elif config['RNN'] =='TwoD':
        if config['Print'] ==True:
            print(f"Training a two-D RNN wave function with {num_hidden} hidden units and shared weights = {weight_sharing}.")
        wavefxn = MDRNNWavefunction(Lx,Ly,V,Omega,delta,num_hidden,learning_rate,weight_sharing,trunc,seed)
    else:
        raise ValueError(f"{config['RNN']} is not a valid option for the RNN wave function. Please choose OneD or TwoD.")

    # Set training parameters
    ns = config['ns']
    batch_size = config['batch_size']
    epochs = config['epochs']
    exact_E, full_data = load_QMC_data(Lx)
    data = create_tf_dataset(full_data,100) # Check...!


    for n in range(1, epochs+1):

        dset = data.shuffle(len(data))
        dset = dset.batch(batch_size)
        
        for i, batch in enumerate(dset):
            # print(f"batch #{i}")
            # print(f"Shape of data batches: {tf.shape(batch)}")
            # Evaluate the loss function in AD mode
            with tf.GradientTape() as tape:
                logpsi = wavefxn.logpsi(batch)
                loss = - 2.0 * tf.reduce_mean(logpsi)

            gradients = tape.gradient(loss, wavefxn.trainable_variables)
            wavefxn.optimizer.apply_gradients(zip(gradients, wavefxn.trainable_variables))
            
        #append the energy to see convergence
        samples, _ = wavefxn.sample(ns)
        # print("Calling logpsi on samples drawn from RNN")
        # print(f"Shape of samples: {tf.shape(samples)}")
        sample_logpsi = wavefxn.logpsi(samples)
        # print("localenergy calls logpsi")
        sample_eloc = wavefxn.localenergy(samples, sample_logpsi)
        energies = sample_eloc.numpy()
        avg_E = np.mean(energies)/float(wavefxn.N)
        var_E = np.var(energies)/float(wavefxn.N)
        energy.append(avg_E)
        variance.append(var_E)
        cost.append(loss)

        if (config['Print'] ==True):
            print(f"Step #{n}")
            print(f"Energy = {avg_E}")
            print(f"Variance = {var_E}")
            print(" ")
    
    if config['Write_Data']:
        samples_final,_ = wavefxn.sample(10000)
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
        plot_E(energy,exact_E,wavefxn.N,epochs)
        plot_var(variance,wavefxn.N,epochs)
        plot_loss(cost,wavefxn.N,epochs)

    return wavefxn, energy, variance

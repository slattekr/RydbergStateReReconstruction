import os
import tensorflow as tf
import numpy as np
from dset_helpers import create_KZ_tf_dataset, data_given_param
from OneD_RNN import OneD_RNN_wavefxn
from TwoD_RNN import MDRNNWavefunction,MDTensorizedRNNCell,MDRNNGRUcell
from helpers import save_path

def Train_w_Data(config,energy,variance,cost):

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
    sweep_rate = config['sweep_rate']

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
            logpsi = wavefxn.logpsi(input_batch)
            loss = - 2.0 * tf.reduce_mean(logpsi)
        # Compute the gradients either with qmc_loss
        gradients = tape.gradient(loss, wavefxn.trainable_variables)
        clipped_gradients = [tf.clip_by_value(g, -10., 10.) for g in gradients]
        # Update the parameters
        wavefxn.optimizer.apply_gradients(zip(clipped_gradients, wavefxn.trainable_variables))
        return loss

    # Training Parameters
    ns = config['ns']
    batch_size = config['batch_size']
    epochs = config['Data_epochs']
    data = data_given_param(sweep_rate,delta)
    tf_dataset = create_KZ_tf_dataset(data)

    for n in range(1, epochs+1):
        #use data to update RNN weights
        dset = tf_dataset.shuffle(len(tf_dataset))
        dset = dset.batch(batch_size)
        loss = []

        for i, batch in enumerate(dset):
            # Evaluate the loss function in AD mode
            batch_loss = train_step(batch)
            loss.append(batch_loss)

        #append the energy to see convergence
        avg_loss = np.mean(loss)
        samples, _ = wavefxn.sample(ns)
        sample_logpsi = wavefxn.logpsi(samples)
        sample_eloc = wavefxn.localenergy(samples, sample_logpsi)
        energies = sample_eloc.numpy()
        avg_E = np.mean(energies)/float(wavefxn.N)
        var_E = np.var(energies)/float(wavefxn.N)
        energy.append(avg_E)
        variance.append(var_E)
        cost.append(avg_loss)

        if (config['Print'] ==True):
            print(f"Step #{n}")
            print(f"Energy = {avg_E}")
            print(f"Variance = {var_E}")
            print(" ")
    
    if config['Write_Data']==True:
        samples_final,_ = wavefxn.sample(10000)
        exp_name = config['name']
        path = f'./data/N_{Lx*Ly}/delta_{delta}/{exp_name}'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+'/config.txt', 'w') as file:
            for k,v in config.items():
                file.write(k+f'={v}\n')
        np.save(path+'/Energy',energy)
        np.save(path+'/Variance',variance)
        np.save(path+'/Cost',cost)
        np.save(path+'/Samples',samples)
            
    return wavefxn, energy, variance,cost

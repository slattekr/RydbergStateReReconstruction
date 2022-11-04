import os
import tensorflow as tf
import numpy as np
from dset_helpers import load_exact_Es,load_QMC_data,create_tf_dataset
from OneD_RNN import OneD_RNN_wavefxn
from helpers import save_path
from plots import plot_E,plot_var,plot_loss
import matplotlib.pyplot as plt

def run_DataPlusVMC(config,energy,variance):

    '''
    Run RNN using vmc sampling or qmc data. If qmc_data is None, uses vmc sampling. 
    Otherwise uses qmc data loaded in qmc_data
    '''

    # Set the parameters of the RNN 
    Lx = config['Lx']
    Ly = config['Ly']
    V = config['V']
    delta = config['delta']
    Omega = config['Omega']
    num_hidden = config['nh']
    learning_rate = config['lr']
    weight_sharing = config['weight_sharing']
    trunc = config['trunc']
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

    if config['Print'] ==True:
        print(f"The experimental parameters are: V = {V}, delta = {delta}, Omega = {Omega}.")
        print(f"The system is an array of {Lx} by {Ly} Rydberg Atoms.")

    # Training Parameters
    ns = config['ns']
    batch_size = config['batch_size']
    data_step = config['data_step']
    exact_e = load_exact_Es(Lx)
    data = load_QMC_data(Lx)
    tf_dataset = create_tf_dataset(data,data_step)
    VMC_epochs = config['VMC_epochs']
    Data_epochs = config['Data_epochs']

    # Training Methods
    if (config['VMC_epochs'] == 0) & (config['Data_epochs'] == 0):
        raise ValueError("Total epochs = 0.")

    elif config['Data_epochs'] == 0:
        config['Train_Method'] = 'VMC_Only'
        if config['Print'] ==True:
            print("Training with VMC.")
            print(" ")
        Total_epochs = VMC_epochs
        data = None

    elif config['VMC_epochs'] == 0:
        config['Train_Method'] = 'Data_Only'
        if config['Print'] ==True:  
            print("Training with data.")
            print(" ")
        Total_epochs = Data_epochs
        data = tf_dataset

    else:
        config['Train_Method'] = 'Hybrid'
        if config['Print'] ==True:
            print("Training with data then VMC.")
            print(" ")
        Total_epochs = VMC_epochs+Data_epochs
        data = tf_dataset

    for n in range(1, Total_epochs+1):

        #use data to update RNN weights
        if (data != None) & (n<=Data_epochs):
            dset = data.shuffle(len(data))
            dset = dset.batch(batch_size)
        
            for i, batch in enumerate(dset):
                # Evaluate the loss function in AD mode
                with tf.GradientTape() as tape:
                    logpsi = wavefxn.logpsi(batch)
                    
                    loss = - 2.0 * tf.reduce_mean(logpsi)

                # Compute the gradients either with qmc_loss
                gradients = tape.gradient(loss, wavefxn.trainable_variables)
              
                # Update the parameters
                wavefxn.optimizer.apply_gradients(zip(gradients, wavefxn.trainable_variables))

        else:
            samples, _ = wavefxn.sample(ns)
      
            # Evaluate the loss function in AD mode
            with tf.GradientTape() as tape:
                sample_logpsi = wavefxn.logpsi(samples)
                with tape.stop_recording():
                    sample_eloc = tf.stop_gradient(wavefxn.localenergy(samples, sample_logpsi))
                    sample_Eo = tf.stop_gradient(tf.reduce_mean(sample_eloc))
                  
                sample_loss = tf.reduce_mean(2.0*tf.multiply(sample_logpsi, tf.stop_gradient(sample_eloc)) - 2.0*sample_Eo*sample_logpsi)
          
                # Compute the gradients either with sample_loss
                gradients = tape.gradient(sample_loss, wavefxn.trainable_variables)
        
                # Update the parameters
                wavefxn.optimizer.apply_gradients(zip(gradients, wavefxn.trainable_variables))
           
        #append the energy to see convergence
        samples, _ = wavefxn.sample(ns)
        sample_logpsi = wavefxn.logpsi(samples)
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
        samples_final,_ = wavefxn.sample(10000,False)
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
        plot_E(energy, exact_e, wavefxn.N, Total_epochs)
        plot_var(variance, wavefxn.N, Total_epochs)
            
    return wavefxn, energy, variance

#This is the same as train_data.py but without the print statements.

import os
import tensorflow as tf
import numpy as np
from dset_helpers import load_essiqurke_data,create_tf_essiqurke_dataset
from OneD_RNN import OneD_RNN_wavefxn
from plots import plot_E,plot_var,plot_loss
import matplotlib.pyplot as plt
from Learning_Criterion import learning_criterion, plot_learning

def Train_w_Data(config,energy,variance,cost, learning_criterion_vec):

    '''
    Run RNN using vmc sampling or qmc data. If qmc_data is None, uses vmc sampling. 
    Otherwise uses qmc data loaded in qmc_data
    '''

    # System Parameters 
    L = config['L']
    V = config['V']
    delta = config['delta']
    Omega = config['Omega']
    
    #Confidence interval for calculating learning rate
    C = 2.576

    # RNN Parameters
    num_hidden = config['nh']
    learning_rate = config['lr']
    trunc = config['trunc']
    seed = config['seed']

    # Initiate RNN wave function
    wavefxn = OneD_RNN_wavefxn(L,1,V,Omega,delta,num_hidden,learning_rate,trunc,seed)

    if config['Print'] ==True:
        print(f"The experimental parameters are: V = {V}, delta = {delta}, Omega = {Omega}.")
        print(f"The system is a one-D chain of {L} Rydberg Atoms")

    @tf.function
    def train_step(input_batch):
        print("Tracing!")
        with tf.GradientTape() as tape:
            logpsi = wavefxn.logpsi(input_batch)
            loss = - 2.0 * tf.reduce_mean(logpsi)
        # Compute the gradients
        gradients = tape.gradient(loss, wavefxn.trainable_variables)
        clipped_gradients = [tf.clip_by_value(g, -10., 10.) for g in gradients]
        # Update the parameters
        wavefxn.optimizer.apply_gradients(zip(clipped_gradients, wavefxn.trainable_variables))
        return loss

    # Training Parameters
    ns = config['ns']
    batch_size = config['batch_size']
    data_step = config['data_step']
    epochs = config['Data_epochs']
    data = load_essiqurke_data(L,delta,data_step)
    tf_dataset = create_tf_essiqurke_dataset(data)

    for n in range(1, epochs+1):
        dset = tf_dataset.shuffle(len(tf_dataset))
        dset = dset.batch(batch_size)
        loss = []
        for i, batch in enumerate(dset):
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
        
        #calculate learning criterion
        eps = learning_criterion(avg_E, L, var_E, C)
        learning_criterion_vec.append(eps)
        learning_criterion_min = np.amin(learning_criterion_vec)

#        if (config['Print'] ==True):
#            print(f"Step #{n}")
#            print(f"Energy = {avg_E}")
#            print(f"Variance = {var_E}")
#            print(f"Learning Criterion = {eps}")
#            print(" ")
    
    if config['Write_Data']==True:
        samples_final,_ = wavefxn.sample(10000)
        exp_name = config['name']
        path = f'./results/L_{L}/delta_{delta}/'+exp_name
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+'/config.txt', 'w') as file:
            for k,v in config.items():
                file.write(k+f'={v}\n')
        np.save(path+'/Energy',energy)
        np.save(path+'/Variance',variance)
        np.save(path+'/Loss',cost)
        np.save(path+'/Samples',samples)
    
    if config['Plot']:
        plot_E(energy, None, wavefxn.N, epochs)
        plot_var(variance, wavefxn.N, epochs)
        plot_loss(cost, wavefxn.N, epochs, loss_type='KL')
        plot_learning(learning_criterion_vec, epochs)
        
            
    return wavefxn, energy, variance, cost, learning_criterion_vec, learning_criterion_min

import os
import tensorflow as tf
import numpy as np
import glob

Exact_Es = {'4':-0.4534132086591546,'8':-0.40518005298872917,'12':-0.3884864748124427,'16':-0.380514770608724}

def load_exact_Es(dim):
    path = "../QMC_data"
    dim_path = f"Dim={dim}_M=1000000_V=7_omega=1.0_delta=1.0" # Can change this to look at Dim = 4, 8, 12, 16
    exact_e = Exact_Es[f'{dim}']
    exact_e = Exact_Es[f'{dim}']
    return exact_e

def load_QMC_data(dim):
    path = "../../QMC_data"
    dim_path = f"Dim={dim}_M=1000000_V=7_omega=1.0_delta=1.0" # Can change this to look at Dim = 4, 8, 12, 16
    files_we_want = glob.glob(os.path.join(path,dim_path,'samples*'))
    uploaded = {}
    for file in files_we_want:
        data = np.loadtxt(file)
        uploaded[file] = data
    return uploaded

def create_tf_dataset_from_QMCdata(uploaded_files, data_step_size=100):
    '''
    create tensor flow data set from uploaded files
    data_step_size (int): determines step size when loading data (for QMC need to have this to overcome autocorrelations)
    '''
    data = []
    for file in uploaded_files:
        new_data = uploaded_files[file]
        new_data = new_data.astype(int)
        new_data = new_data[::data_step_size]
        data.extend(new_data)

    #convert to tf.data.Dataset
    data = np.array(data)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    return dataset

def create_KZ_tf_dataset(data):
    # The data comes in shape ((lattice,shape),1,shots) where (lattice,shape) = square lattice (EX: 16x16)
    Lx = np.shape(data)[0]
    shots = np.shape(data)[-1]
    data = np.reshape(data,(Lx**2,shots))
    data = data.T
    tf_data = tf.data.Dataset.from_tensor_slices(data)
    return tf_data

def bool_to_bin(rydberg_dataset):
    return rydberg_dataset.astype(int)

def data_given_param(sweep_rate:int,delta_value):
    data_files = np.load(f'../../KZ_Data/KZ_data_16x16_{sweep_rate}_MHz_per_us.npz')
    param_value = delta_value/data_files['rabi_freq']
    index = np.where(np.isclose(data_files['params'],param_value))[0]
    if len(index) < 1:
        return print("This is not a valid value of delta!")
    data = data_files['rydberg_data'][:,:,index,:]
    data = bool_to_bin(data)
    return data
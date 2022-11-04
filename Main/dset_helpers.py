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
    path = "../QMC_data"
    dim_path = f"Dim={dim}_M=1000000_V=7_omega=1.0_delta=1.0" # Can change this to look at Dim = 4, 8, 12, 16
    files_we_want = glob.glob(os.path.join(path,dim_path,'samples*'))
    uploaded = {}
    for file in files_we_want:
        data = np.loadtxt(file)
        uploaded[file] = data
    return uploaded

def create_tf_dataset(uploaded_files, data_step_size=100):
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
import os
import tensorflow as tf

def save_path(config,Nspins):#rnn_wf,rnn_type:str,model:str):
    N = Nspins
    delta = config['delta']
    Omega = config['Omega']
    V = config['V']
    rnn = config['RNN']
    model = config['Train_Method']
    name = config['name']
    datapath = f'../data/Experiments_1/{model}'
    rnn_type = f'/{rnn}'
    experiment = f'/N={N}/V={V}/delta={delta}/Omega={Omega}/{name}'
    save_path = datapath + rnn_type + experiment 
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path+'/config.txt', 'w') as file:
        for k,v in config.items():
            file.write(k+f'={v}\n')

    return save_path

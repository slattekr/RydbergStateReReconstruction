import tensorflow as tf
import numpy as np

def site(Ly,x,y,num_dropped_so_far):
        return Ly*x + y - num_dropped_so_far

def create_tf_dataset(data,atom_coords=np.array([None])):

    num_shots = np.shape(data)[0]

    if atom_coords.any() != None:
        #add zeros to dropped atom places
        max_x = np.amax(atom_coords[:,0])
        max_y = np.amax(atom_coords[:,1])
        syst_size = max_y+1
        data_grid = np.zeros((max_x+1,max_y+1,num_shots))
        num_zeros = 0
        for x_ in range(max_x+1):
            for y_ in range(max_y+1):
                if np.any(np.all(atom_coords==[x_,y_],axis=1)):
                    index = site(syst_size,x_,y_,num_zeros)
                    data_grid[x_,y_,:] = data[:,index]
                else:
                    num_zeros += 1
                    data_grid[x_,y_] = 0
        
        data = np.reshape(data_grid,((max_x+1)*(max_y+1),num_shots))
        data = data.T
        data = data.astype(int)

    tf_data = tf.data.Dataset.from_tensor_slices(data)

    return tf_data

def read_in_data(path:str):
    data = np.load(path)
    xs = data['xs']
    xs = xs-min(xs)
    xs = np.reshape(xs,(len(xs),1))

    ys = data['ys']
    ys = ys-min(ys)
    ys = np.reshape(ys,(len(ys),1))

    atom_coords = np.append(xs,ys,axis=1)   

    rawdata = data['rydberg_populations']
    rawdata = rawdata.T
    rawdata = rawdata.astype(int)

    processed_data = data['rydberg_populations_processed']
    processed_data = processed_data.T
    processed_data = processed_data.astype(int)

    reducted_data = data['rydberg_populations_reducted']
    reducted_data = reducted_data.T
    reducted_data = reducted_data.astype(int)

    added_data = data['rydberg_populations_added']
    added_data = added_data.T
    added_data = added_data.astype(int)

    return atom_coords,rawdata,processed_data,reducted_data,added_data
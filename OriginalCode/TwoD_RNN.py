import numpy as np
import tensorflow as tf
from Tensordot2 import tensordot
import random
from bloqade import unit_disk_graph,mis_postprocessing

class MDTensorizedRNNCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """The 2D Tensorized RNN cell.
    """
    def __init__(self, units = None, output_size=None, activation = None, name=None, dtype = None):
        super(MDTensorizedRNNCell, self).__init__(name=name)
        # save class variables
        self._num_in = output_size
        self._num_units = units
        self._state_size = units
        self._output_size = output_size
        self.activation = activation

        # set up input -> hidden connection
        if name != None:
            self.W = tf.Variable(name="W_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([units, 2 * units, 2 * self._num_in]),
                             dtype = dtype)

            self.b = tf.Variable(name="b_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([units]),
                             dtype = dtype)
        else:
            self.W = tf.Variable(name="W",
                             initial_value=tf.keras.initializers.GlorotNormal()([units, 2 * units, 2 * self._num_in]),
                             dtype = dtype)

            self.b = tf.Variable(name="b",
                             initial_value=tf.keras.initializers.GlorotNormal()([units]),
                             dtype = dtype)

    # needed properties

    @property
    def input_size(self):
        return self._num_in # real

    @property
    def state_size(self):
        return self._state_size # real

    @property
    def output_size(self):
        return self._output_size # real

    @property
    def trainable_variables(self):
        return [self.W, self.b]   

    def call(self, inputs, states):

        inputstate_mul = tf.einsum('ij,ik->ijk', tf.concat(states, 1), tf.concat(inputs,1))
        # prepare input linear combination

        state_mul = tensordot(tf, inputstate_mul, self.W, axes=[[1,2],[1,2]]) # [batch_sz, units]

        preact = state_mul + self.b

        output = self.activation(preact) # [batch_sz, units] C

        new_state = output

        return output, new_state

class MDRNNWavefunction(object):
    def __init__(self, Lx:int, Ly:int,                #system size parameters
                 V, Omega, delta,                     #experiment parameters
                 num_hidden:int,                      #num_hidden = units!!!! = number of hidden units between RNN Cells
                 learning_rate,                       #does not get used here
                 weight_sharing = True,               #indicates whether RNN cells' weights are shared (same w dense layer)
                 trunc=2,                             #defines the order of NN interactions to include
                 seed=1234,
                 cell=MDTensorizedRNNCell,            #defined above
                 Num_Dropped=0,Atom_Coords=None,      #specific to MIS problem
                 post_process = False):
    
        super(MDRNNWavefunction, self).__init__()
        
        """ PARAMETERS """
        self.Lx       = Lx              # Size along x
        self.Ly       = Ly              # Size along y
        self.V        = V               # Van der Waals potential
        self.Omega    = Omega           # Rabi frequency
        self.delta    = delta           # Detuning
        self.trunc    = trunc           # Truncation, set to Lx+Ly for none, default is 2
        self.nh       = num_hidden      # Number of hidden units in the RNN
        self.seed     = seed            # Seed of random number generator 
        self.K        = 2               # Dimension of the local Hilbert space
        self.weight_sharing = weight_sharing # Option to share weights between RNN cells or not (default = True)
        self.optimizer = tf.optimizers.Adam(learning_rate, epsilon=1e-8)


        """ MIS SPECIFIC PARAMETERS """
        self.Num_Dropped = Num_Dropped
        self.Atom_Coords = Atom_Coords
        if self.Atom_Coords.all() != None:
            self.Atom_Sites = self.coord_to_site(self.Atom_Coords[:,0],self.Atom_Coords[:,1])
        # if self.Atom_Coords != None:
        #     self.Atom_Sites = self.coord_to_site(self.Atom_Coords[:,0],self.Atom_Coords[:,1])
        self.N = self.Lx * self.Ly - self.Num_Dropped

        self.post_process = post_process
        if self.post_process:
            self.g = unit_disk_graph(self.Atom_Coords,radius=1) # radius can also be changed

        # Set the seed of the rng
        tf.random.set_seed(self.seed)
        
        if self.weight_sharing == True:
            self.rnn = cell(units=self.nh, output_size=self.K,activation=tf.nn.relu, dtype=tf.float32)
            self.dense = tf.keras.layers.Dense(self.K, activation=tf.nn.softmax, dtype=tf.float32)
        
        else:
            self.rnn = [cell(units=self.nh, output_size=self.K,activation=tf.nn.relu, name=f"RNN_{0}_{i}", dtype=tf.float32) for i in
                            range(self.Lx * self.Ly)]
            self.dense = [tf.keras.layers.Dense(self.K, activation=tf.nn.softmax, name=f'RNNWF_dense_{i}', dtype=tf.float32) for
                            i in range(self.Lx * self.Ly)]
        

        # Generate the list of interactions
        self.buildlattice()
        if self.Num_Dropped != 0:
            self.map_interactions() #generate list of MAPPED interactions

        # Generate trainable variables
        self.sample(1,sample_post_process = False)
        self.trainable_variables = []
        if self.weight_sharing == True:
            self.trainable_variables.extend(self.rnn.trainable_variables)
            self.trainable_variables.extend(self.dense.trainable_variables)
            # Check that there are the correct amount of trainable variables
            self.variables_names = [v.name for v in self.trainable_variables]
            sum = 0
            for k, v in zip(self.variables_names, self.trainable_variables):
                v1 = tf.reshape(v, [-1])
                print(k, v1.shape)
                sum += v1.shape[0]
            print(f'The sum of params is {sum}')
        else:
            for cell in self.rnn:
                self.trainable_variables.extend(cell.trainable_variables)
            for node_dense in self.dense:
                self.trainable_variables.extend(node_dense.trainable_variables)
            # Check that there are the correct amount of trainable variables
            self.variables_names = [v.name for v in self.trainable_variables]
            sum = 0
            for k, v in zip(self.variables_names, self.trainable_variables):
                v1 = tf.reshape(v, [-1])
                print(k, v1.shape)
                sum += v1.shape[0]
            print(f'The sum of params is {sum}')


    def sample(self, numsamples,sample_post_process):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:
            numsamples:      int
                             number of samples to be produced
            ------------------------------------------------------------------------
            Returns:         a tuple (samples,log-probs)
            samples:         tf.Tensor of shape (numsamples,systemsize_x, systemsize_y)
                             the samples in integer encoding
            log-probs        tf.Tensor of shape (numsamples,)
                             the log-probability of each sample
        """

        # Initial input to feed to the lstm                                    *****THINK WE CHANGED x AND y HERE
        samples = [[[] for nx in range(self.Lx)] for ny in range(self.Ly)]
        probs = [[[] for nx in range(self.Lx)] for ny in range(self.Ly)]
        rnn_states = {}
        inputs = {}

        for ny in range(self.Ly):  # Loop over the boundaries for initialization
            if ny % 2 == 0:
                nx = -1
                inputs[f"{nx}{ny}"] = tf.zeros((numsamples, self.K),
                                               dtype=tf.float32)  # Feed the table b in tf.
                if self.weight_sharing == True:
                    rnn_states[f"{nx}{ny}"] = self.rnn.get_initial_state(inputs[f"{nx}{ny}"], dtype=tf.float32)
                else:
                    rnn_states[f"{nx}{ny}"] = self.rnn[0].get_initial_state(inputs[f"{nx}{ny}"], dtype=tf.float32)


            if ny % 2 == 1:
                nx = self.Lx
                inputs[f"{nx}{ny}"] = tf.zeros((numsamples, self.K),
                                               dtype=tf.float32)  # Feed the table b in tf.
                if self.weight_sharing == True:
                    rnn_states[f"{nx}{ny}"] = self.rnn.get_initial_state(inputs[f"{nx}{ny}"], dtype=tf.float32)
                else:
                    rnn_states[f"{nx}{ny}"] = self.rnn[0].get_initial_state(inputs[f"{nx}{ny}"], dtype=tf.float32)
                
        for nx in range(self.Lx):  # Loop over the boundaries for initialization
            ny = -1
            inputs[f"{nx}{ny}"] = tf.zeros((numsamples, self.K),
                                           dtype=tf.float32)  # Feed the table b in tf.
            if self.weight_sharing == True:
                rnn_states[f"{nx}{ny}"] = self.rnn.get_initial_state(inputs[f"{nx}{ny}"], dtype=tf.float32)
            else:
                rnn_states[f"{nx}{ny}"] = self.rnn[0].get_initial_state(inputs[f"{nx}{ny}"], dtype=tf.float32)
        
        # Making a loop over the sites with the 2DRNN
        for ny in range(self.Ly):

            if ny % 2 == 0:

                for nx in range(self.Lx):  # left to right
                    
                    if self.weight_sharing == True:
                        rnn_output,rnn_states[f"{nx}{ny}"] = self.rnn((inputs[f"{nx - 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx - 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense(rnn_output)
                        
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[ny * self.Lx + nx]((inputs[f"{nx - 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx - 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense[ny * self.Lx + nx](rnn_output)

                    sample_temp = tf.reshape(tf.random.categorical(tf.math.log(output), num_samples=1), [-1, ])
                    samples[nx][ny] = sample_temp
                    probs[nx][ny] = output
                    inputs[f"{nx}{ny}"] = tf.one_hot(sample_temp, depth=self.K, dtype=tf.float32)

            if ny % 2 == 1:

                for nx in range(self.Lx - 1, -1, -1):  # right to left

                    if self.weight_sharing == True:
                        rnn_output,rnn_states[f"{nx}{ny}"] = self.rnn((inputs[f"{nx + 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx + 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense(rnn_output)
                        
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[ny * self.Lx + nx]((inputs[f"{nx + 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx + 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense[ny * self.Lx + nx](rnn_output)
                    
                    sample_temp = tf.reshape(tf.random.categorical(tf.math.log(output), num_samples=1), [-1, ])
                    samples[nx][ny] = sample_temp
                    probs[nx][ny] = output
                    inputs[f"{nx}{ny}"] = tf.one_hot(sample_temp, depth=self.K, dtype=tf.float32)

        samples = tf.transpose(tf.stack(values=samples, axis=0), perm=[2, 0, 1])
        probs = tf.transpose(tf.stack(values=probs, axis=0), perm=[2, 0, 1, 3])
        one_hot_samples = tf.one_hot(samples, depth=self.K, dtype=tf.float32)
        log_probs = tf.reduce_sum(
            tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs, one_hot_samples), axis=3)), axis=2), axis=1)
        full_samples = tf.reshape(samples,(numsamples,self.Lx*self.Ly)) 
        print(tf.shape(full_samples))

        # # Check that log_probs are calculated correctly
        # log_probs_fxn = self.logpsi(full_samples)
        # if np.sum(abs(log_probs-log_probs_fxn)) != 0:
        #     print("SOS!")

        if sample_post_process:
            print("post processing samples!")
            full_samples = mis_postprocessing(full_samples,self.g)
            print(tf.shape(full_samples))
            log_probs = self.logpsi(full_samples)

        return full_samples, log_probs

    def logpsi(self, samples):
        """
        calculate the log-probabilities of ```samples``
        ------------------------------------------------------------------------
        Parameters:
        samples:         tf.Tensor
                         a tf.placeholder of shape (number of samples,systemsize_x, systemsize_y)
                         containing the input samples in integer encoding
        ------------------------------------------------------------------------
        Returns:
        log-probs        tf.Tensor of shape (number of samples,)
                         the log-probability of each sample
        """
        
        numsamples = samples.shape[0]
        samples_ = tf.reshape(samples, (numsamples,self.Lx,self.Ly))
        samples_ = tf.transpose(samples_, perm=[1, 2, 0])
        rnn_states = {}
        inputs = {}

        for ny in range(self.Ly):  # Loop over the boundaries for initialization
            if ny % 2 == 0:
                
                nx = -1
                if self.weight_sharing == True:
                    rnn_states[f"{nx}{ny}"] = self.rnn.zero_state(numsamples, dtype=tf.float32)
                else:
                    rnn_states[f"{nx}{ny}"] = self.rnn[0].zero_state(numsamples, dtype=tf.float32)
                inputs[f"{nx}{ny}"] = tf.zeros((numsamples, self.K),
                                               dtype=tf.float32)  # Feed the table b in tf.

            if ny % 2 == 1:
                
                nx = self.Lx
                if self.weight_sharing == True:
                    rnn_states[f"{nx}{ny}"] = self.rnn.zero_state(numsamples, dtype=tf.float32)
                else:
                    rnn_states[f"{nx}{ny}"] = self.rnn[0].zero_state(numsamples, dtype=tf.float32)
                inputs[f"{nx}{ny}"] = tf.zeros((numsamples, self.K),
                                               dtype=tf.float32)  # Feed the table b in tf.

        for nx in range(self.Lx):  # Loop over the boundaries for initialization
            ny = -1
            if self.weight_sharing == True:
                rnn_states[f"{nx}{ny}"] = self.rnn.zero_state(numsamples, dtype=tf.float32)
            else:
                rnn_states[f"{nx}{ny}"] = self.rnn[0].zero_state(numsamples, dtype=tf.float32)
            inputs[f"{nx}{ny}"] = tf.zeros((numsamples, self.K),
                                           dtype=tf.float32)  # Feed the table b in tf.

        probs = [[[] for nx in range(self.Lx)] for ny in range(self.Ly)]

        # Making a loop over the sites with the 2DRNN
        for ny in range(self.Ly):

            if ny % 2 == 0:

                for nx in range(self.Lx):  # left to right

                    if self.weight_sharing == True:
                        rnn_output,rnn_states[f"{nx}{ny}"] = self.rnn((inputs[f"{nx - 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx - 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense(rnn_output)
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[ny * self.Lx + nx]((inputs[f"{nx - 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx - 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense[ny * self.Lx + nx](rnn_output)

                    probs[nx][ny] = output
                    inputs[f"{nx}{ny}"] = tf.one_hot(samples_[nx, ny], depth=self.K, dtype=tf.float32)

            if ny % 2 == 1:

                for nx in range(self.Lx - 1, -1, -1):  # right to left

                    if self.weight_sharing == True:
                        rnn_output,rnn_states[f"{nx}{ny}"] = self.rnn((inputs[f"{nx + 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx + 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense(rnn_output)
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[ny * self.Lx + nx]((inputs[f"{nx + 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx + 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense[ny * self.Lx + nx](rnn_output)
                        
                    probs[nx][ny] = output
                    inputs[f"{nx}{ny}"] = tf.one_hot(samples_[nx, ny], depth=self.K, dtype=tf.float32)

        probs = tf.transpose(tf.stack(values=probs, axis=0), perm=[2, 0, 1, 3])
        samples_ = tf.transpose(samples_, perm=[2, 0, 1])
        one_hot_samples = tf.one_hot(samples_, depth=self.K, dtype=tf.float32)
        log_probs = tf.reduce_sum(
            tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs, one_hot_samples), axis=3)), axis=2), axis=1)

        return log_probs
    
    # THE BELOW SHOULD NOT CHANGE FROM ONE D RNN WF
    
    # TODO: vectorize this function, make tf.function to speed up calculations significantly
    #@tf.function
    def localenergy(self,samples,logpsi):
        eloc = tf.zeros(shape=[tf.shape(samples)[0]],dtype=tf.float32)

        # Chemical potential
        for j in range(self.N):
            eloc += - self.delta * tf.cast(samples[:,j],tf.float32)
     
        if self.Num_Dropped != 0:
            for n in range(len(self.mapped_interactions)):
                eloc += (self.V/self.mapped_interactions[n][0]) * tf.cast(samples[:,self.mapped_interactions[n][1]]*samples[:,self.mapped_interactions[n][2]],tf.float32)
        else:
            for n in range(len(self.interactions)):
                eloc += (self.V/self.interactions[n][0]) * tf.cast(samples[:,self.interactions[n][1]]*samples[:,self.interactions[n][2]],tf.float32)

        flip_logpsi = tf.zeros(shape=[tf.shape(samples)[0]])

        # Off-diagonal part
        for j in range(self.N):
            flip_samples = np.copy(samples)
            flip_samples[:,j] = 1 - flip_samples[:,j]
            flip_logpsi = self.logpsi(flip_samples)
            eloc += -0.5*self.Omega * tf.math.exp(flip_logpsi-logpsi)
            
        return eloc

    # Square Lattice
    def coord_to_site(self,x,y):
        return self.Ly*x+y
    
    # All-to-all interactions unless truncation specified
    def buildlattice(self):
        self.interactions = []
        
        for n in range(1,self.Lx):
            for n_ in range(n+1):
                
                if n+n_ > self.trunc:
                    continue
        
                else:
                    for x in range(self.Lx-n_):
                        for y in range(self.Ly-n):
                            coeff = np.sqrt(n**2+n_**2)**6
                            if n_ == 0 :
                                self.interactions.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x,y+n)])
                            elif n == n_: 
                                self.interactions.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x+n,y+n)])
                                self.interactions.append([coeff,self.coord_to_site(x+n,y),self.coord_to_site(x,y+n)])
                            else:
                                self.interactions.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x+n_,y+n)])
                                self.interactions.append([coeff,self.coord_to_site(x+n_,y),self.coord_to_site(x,y+n)])
                            
                    for y in range(self.Ly-n_):
                        for x in range(self.Lx-n):
                            coeff = np.sqrt(n**2+n_**2)**6
                            if n_ == 0 :
                                self.interactions.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x+n,y)])
                            elif n == n_: 
                                continue #already counted above
                            else:
                                self.interactions.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x+n,y+n_)])
                                self.interactions.append([coeff,self.coord_to_site(x,y+n_),self.coord_to_site(x+n,y)])
                                
    def map_interactions(self):
        self.mapped_interactions = []
        num_dropped = 0
        coord_mapping = []
        
        for i in range(self.Lx*self.Ly):
            if np.all(i!=self.Atom_Sites):
                num_dropped += 1
            else:
                coord_mapping.append((i,i-num_dropped))
        
        coord_mapping_array = np.array(coord_mapping)
        
        for n in range(len(self.interactions)):
            interaction = self.interactions[n][:]
            if np.any(coord_mapping_array[:,0]==interaction[1])&np.any(coord_mapping_array[:,0]==interaction[2]):
                coeff = interaction[0]
                spin1map = np.where(interaction[1]==coord_mapping_array[:,0])
                spin2map = np.where(interaction[2]==coord_mapping_array[:,0])
                self.mapped_interactions.append([coeff,coord_mapping_array[spin1map,1][0][0],coord_mapping_array[spin2map,1][0][0]])  
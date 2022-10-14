import numpy as np 
import tensorflow as tf 
import random

class OneD_RNN_wavefxn(tf.keras.Model):
    
    # Constructor
    def __init__(self, Lx, Ly, 
                 V, Omega, delta,
                 num_hidden, learning_rate,
                 weight_sharing = True,
                 trunc=2, seed=1234):
        
        super(OneD_RNN_wavefxn, self).__init__()

        """ PARAMETERS """
        self.Lx       = Lx              # Size along x
        self.Ly       = Ly              # Size along y
        self.N        = self.Lx * self.Ly
        self.V        = V               # Van der Waals potential
        self.Omega    = Omega           # Rabi frequency
        self.delta    = delta           # Detuning
        self.trunc    = trunc           # Truncation, set to Lx+Ly for none, default is 2
        self.nh       = num_hidden      # Number of hidden units in the RNN
        self.seed     = seed            # Seed of random number generator 
        self.K        = 2               # Dimension of the local Hilbert space
        self.weight_sharing = weight_sharing # Option to share weights between RNN cells or not (default = True)
         
        # Set the seed of the rng
        tf.random.set_seed(self.seed)

        # Optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate, epsilon=1e-8)

        # Build the model RNN
        # RNN layer: N -> nh
        self.rnn = tf.keras.layers.GRU(self.nh, kernel_initializer='glorot_uniform',
                                       kernel_regularizer = tf.keras.regularizers.l2(0.001),
                                       return_sequences = True,
                                       return_state = True,
                                       stateful = False)

        # Dense layer: nh - > K
        self.dense = tf.keras.layers.Dense(self.K, activation = tf.nn.softmax,
                                           kernel_regularizer = tf.keras.regularizers.l2(0.001))

        self.sample(1)
        self.trainable_variables_ = []
        self.trainable_variables_.extend(self.rnn.trainable_variables)
        self.trainable_variables_.extend(self.dense.trainable_variables)
        # Check that there are the correct amount of trainable variables
        self.variables_names = [v.name for v in self.trainable_variables_]
        sum_ = 0
        for k, v in zip(self.variables_names, self.trainable_variables_):
            v1 = tf.reshape(v, [-1])
            print(k, v1.shape)
            sum_ += v1.shape[0]
        print(f'The sum of params is {sum_}')

        # Generate the list of interactions
        self.buildlattice()
        
    @tf.function
    def sample(self,nsamples):
        # Zero initialization for visible and hidden state 
        # print("sampling!")
        inputs = 0.0*tf.one_hot(tf.zeros(shape=[nsamples,1],dtype=tf.int32),depth=self.K)
        hidden_state = tf.zeros(shape=[nsamples,self.nh])

        logP = tf.zeros(shape=[nsamples,],dtype=tf.float32)

        for j in range(self.N):
            # Run a single RNN cell
            rnn_output,hidden_state = self.rnn(inputs,initial_state=hidden_state)
            # Compute log probabilities
            probs = self.dense(rnn_output)
            log_probs = tf.reshape(tf.math.log(1e-10+probs),[nsamples,self.K])
            # Sample
            sample = tf.random.categorical(log_probs,num_samples=1)
            if (j == 0):
                samples = tf.identity(sample)
            else:
                samples = tf.concat([samples,sample],axis=1)
            # Feed result to the next cell
            inputs = tf.one_hot(sample,depth=self.K)
            add = tf.reduce_sum(log_probs*tf.reshape(inputs,(nsamples,self.K)),axis=1)

            logP = logP+tf.reduce_sum(log_probs*tf.reshape(inputs,(nsamples,self.K)),axis=1)
        
        return samples,logP

    @tf.function
    def logpsi(self,samples):

        num_samples = tf.shape(samples)[0]
        data = tf.one_hot(samples[:,0:self.N-1],depth=self.K)

        x0 = 0.0*tf.one_hot(tf.zeros(shape=[num_samples,1],dtype=tf.int32),depth=self.K) #initialization
        inputs = tf.concat([x0,data],axis=1)
        
        hidden_state = tf.zeros(shape=[num_samples,self.nh])
        rnn_output,_ = self.rnn(inputs,initial_state = hidden_state)
        probs        = self.dense(rnn_output)
            
        log_probs   = tf.reduce_sum(tf.multiply(tf.math.log(1e-10+probs),tf.one_hot(samples,depth=self.K)),axis=2)
        
        return 0.5 * tf.reduce_sum(log_probs, axis=1)

    #@tf.function
    def localenergy(self,samples,logpsi):
        eloc = tf.zeros(shape=[tf.shape(samples)[0]],dtype=tf.float32)

        # Chemical potential
        for j in range(self.N):
            eloc += - self.delta * tf.cast(samples[:,j],tf.float32)
     
        
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

    """ Generate the square lattice structures """
    def coord_to_site(self,x,y):
        return self.Ly*x+y
    
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

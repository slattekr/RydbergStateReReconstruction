import scipy
import scipy.sparse.linalg as sspla
import numpy as np
import tensorflow as tf
from math import ceil

def get_Heisenberg_Energy_Vectorized(interactions, log_fxn):
    print("getting energy function!")

    @tf.function()
    def Heisenberg_Energy_Vectorized_tf_function(J, J_matrix, samples, og_amps):
        N = tf.shape(samples)[1]
        Energies_zz = tf.reduce_sum(tf.math.abs(J_matrix @ (2 * tf.transpose(samples) - 1)) - 1, axis=0)
        Energies_zz = tf.complex(Energies_zz, 0.0)
        samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], len(interactions), axis=2)
        samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped +
                                                tf.transpose(J_matrix)[tf.newaxis, :, :], 2)
        samples_tiled_sub = samples_tiled_flipped - samples_tiled_not_flipped
        signs = tf.complex(tf.math.abs(tf.reduce_sum(samples_tiled_sub, axis=1)) - 1, 0.0)
        samples_tiled_flipped = tf.transpose(samples_tiled_flipped, perm=[0, 2, 1])
        flip_logprob, flip_logamp = log_fxn(tf.reshape(samples_tiled_flipped, (-1, N)), symmetrize=symmetrize,initial_pass=False)
        amp_ratio = tf.math.exp(tf.reshape(flip_logamp, (-1, len(interactions))) - og_amps[:, tf.newaxis])
        Energies_xx = tf.reduce_sum(amp_ratio, axis=1)
        Energies_yy = -tf.reduce_sum(signs * amp_ratio, axis=1)
        Energies = 0.25 * J * (ms * Energies_xx + ms * Energies_yy + Energies_zz)
        return Energies

    return Heisenberg_Energy_Vectorized_tf_function
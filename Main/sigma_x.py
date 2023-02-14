import numpy as np
import tensorflow as tf

def calculate_sigma_x(O_mat,samples,sample_log_probs,log_fxn):
    numsamples, N = np.shape(samples.numpy())
    samples_log_probs_fxn = log_fxn(samples,initial_pass=False)
    samples = tf.cast(samples,dtype=tf.float32)
    samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], N, axis=2)
    samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped + tf.transpose(O_mat)[tf.newaxis, :, :], 2) # numsamples, n_spins, n_spins
    samples_tiled_flipped = tf.transpose(samples_tiled_flipped,perm=[0,2,1]) 
    samples_tiled_flipped = tf.cast(tf.reshape(samples_tiled_flipped,(numsamples*N,N)),dtype=tf.int64) # every one sample becomes N samples each with 1 spin flip
    log_probs_flipped = log_fxn(samples_tiled_flipped,initial_pass=False)
    log_probs_flipped = tf.reshape(log_probs_flipped,(numsamples,N))
    mean_logprob = tf.reduce_mean(samples_log_probs_fxn)
    # print(f"The avg LOG of the probs of the samples are:\n {mean_logprob}")
    # print(f"Therefore, the probabilities are:\n {tf.math.exp(sample_log_probs)}")
    mean_flip_logprob = tf.reduce_mean(log_probs_flipped)
    # print(f"The avg LOG of the probs of the samples with a flipped spind are:\n {mean_flip_logprob}")
    # print(f"Therefore, the probabilities are:\n {tf.math.exp(log_probs_flipped)}")
    log_prob_ratio = tf.math.exp(log_probs_flipped - samples_log_probs_fxn[:, tf.newaxis])
    # sigma_xs = tf.reduce_sum(log_prob_ratio,axis=1) # should now have one value for each of the samples
    sigma_xs_all_sites = log_prob_ratio
    sigma_xs = tf.reduce_mean(sigma_xs_all_sites,axis=1) # should now have one value for each of the samples
    return sigma_xs, sigma_xs_all_sites, mean_logprob, mean_flip_logprob


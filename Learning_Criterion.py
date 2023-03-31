from dset_helpers import load_exact_Es
import numpy as np
import matplotlib.pyplot as plt

def learning_criterion(rnn_E, dim, var, C):
    exact_E = load_exact_Es(dim)
    num_samples = dim * dim
    st_dev = np.sqrt(var)
    eps = np.maximum((exact_E - (rnn_E + C * st_dev * np.sqrt(num_samples)))/exact_E, (exact_E - (rnn_E - C * st_dev * np.sqrt(num_samples)))/exact_E)
    return eps

def plot_learning(learning_criterion_vec, total_epochs):
  '''
  Plots the Learning criterion over each step
  '''
  fig = plt.figure(1,figsize=(6,2.5), dpi=120, facecolor='w', edgecolor='k')
  plt.plot(learning_criterion_vec,marker='o',markersize=2,linewidth=0.0,markevery=1,label="Learning Criterion")
  plt.xlabel("Step",fontsize=15)
  plt.ylabel("\u03B5",fontsize=20)
  #plt.title("{} sites".format(N))
  plt.legend(loc="best")
  plt.show()
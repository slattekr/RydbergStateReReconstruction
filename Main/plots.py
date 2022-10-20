import numpy as np
import matplotlib.pyplot as plt

def plot_E(energy, exact_energy, N, total_epochs):
  '''
  Plots the energy over epochs. Cutoff shows the difference if combining 
  vmc sampling and qmc data training. If it is 0, nothing is plotted.
  '''
  fig = plt.figure(1,figsize=(6,2.5), dpi=120, facecolor='w', edgecolor='k')
  plt.plot(energy,marker='o',markersize=2,linewidth=0.0,markevery=1,label="RNN Energy")
  if exact_energy != None:
    plt.hlines(exact_energy,0, total_epochs,linestyle="--",label="Exact")
  plt.xlabel("Step",fontsize=15)
  plt.ylabel("$\\langle H \\rangle$",fontsize=20)
  plt.ylim(-.5,0)
  plt.title("{} sites".format(N))
  plt.legend(loc="best")
  plt.show()

  avg_final_energies = energy[-10:-1]
  final_energy = np.mean(avg_final_energies)
  print(f"Final Energy {final_energy} (Exact Energy is {exact_energy})")
  if exact_energy!=None:
      final_error = np.abs(final_energy-exact_energy)
      print("Final Error after {} epochs is {}".format(total_epochs, final_error))
  return final_energy

def plot_var(variance, N, total_epochs):
  '''
  Plots the variance over epochs.
  '''
  fig = plt.figure(1,figsize=(6,2.5), dpi=120, facecolor='w', edgecolor='k')
  plt.plot(variance,marker='o',markersize=2,linewidth=0.0,markevery=1,label="RNN variance")
  plt.xlabel("Step",fontsize=15)
  plt.ylabel("$\\sigma^{2}$",fontsize=20)
  plt.ylim(-0.1,5)
  plt.hlines(y=0,xmin=0,xmax=len(variance),color='k',label = "target variance")
  plt.title("{} sites".format(N))
  plt.legend(loc="best")
  plt.show()

  avg_final_variances = variance[-10:-1]
  final_variance = np.mean(avg_final_variances)
  print("Final Variance after {} epochs is {}".format(total_epochs,final_variance))
  return final_variance


def plot_loss(cost, N, total_epochs):
  '''
  Plots the loss over epochs.
  '''
  fig = plt.figure(1,figsize=(6,2.5), dpi=120, facecolor='w', edgecolor='k')
  plt.plot(cost,marker='o',markersize=2,linewidth=0.0,markevery=1,label="training loss")
  plt.xlabel("Step",fontsize=15)
  plt.ylabel("KL",fontsize=20)
  plt.hlines(y=0,xmin=0,xmax=len(cost),color='k',label = "target loss")
  plt.title("{} sites".format(N))
  plt.legend(loc="best")
  plt.show()

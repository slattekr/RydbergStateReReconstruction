import numpy as np
import matplotlib.pyplot as plt

def plot_E(energy, exact_energy, N, total_epochs, cutoff=0, plot_logs=False):
  '''
  Plots the energy over epochs. Cutoff shows the difference if combining 
  vmc sampling and qmc data training. If it is 0, nothing is plotted.
  '''
  fig = plt.figure(1,figsize=(6,2.5), dpi=120, facecolor='w', edgecolor='k')
  #minimum_e = np.min(energy)

  plt.plot(energy,marker='o',markersize=2,linewidth=0.0,markevery=1,label="RNN Energy")
  if plot_logs != False:
    plt.plot(np.log(energy),marker='o',markersize=2,linewidth=0.0,markevery=1,label="log(RNN Energy)")
  if exact_energy != None:
    plt.hlines(exact_energy,0, total_epochs,linestyle="--",label="Exact")
  #plt.hlines(minimum_e,0, total_epochs,linestyle="-",label="minimum")
    
  if cutoff != 0:
    plt.vlines(cutoff, energy[0], exact_energy, linestyle="--", label="QMC train cutoff")

  plt.xlabel("Step",fontsize=15)
  plt.ylabel("$\\langle H \\rangle$",fontsize=20)
  plt.title("{} sites".format(N))
  plt.legend(loc="best")
  #plt.xlim(200,500)
  #plt.ylim(-0.5,-0.2)

  plt.show()

  avg_final_energies = energy[-10:-1]
  final_energy = np.mean(avg_final_energies)
  if exact_energy!=None:
      final_error = np.abs(final_energy-exact_energy)
      print("Final Error after {} epochs is {}".format(total_epochs, final_error))
  print("Final Energy {}".format(final_energy))
  return final_energy

def plot_var(variance, N, total_epochs, cutoff=0):
  '''
  Plots the variance over epochs.
  '''
  fig = plt.figure(1,figsize=(6,2.5), dpi=120, facecolor='w', edgecolor='k')

  plt.plot(variance,marker='o',markersize=2,linewidth=0.0,markevery=1,label="RNN variance")

  plt.xlabel("Step",fontsize=15)
  plt.ylabel("$\\sigma^{2}$",fontsize=20)
  plt.title("{} sites".format(N))
  plt.legend(loc="best")

  plt.show()

  avg_final_variances = variance[-10:-1]
  final_variance = np.mean(avg_final_variances)
  print("Final Variance {}".format(final_variance))
  return final_variance

def plot_Es(energy_vmc,energy_data,energy_hybrid, exact_energy, N, total_epochs, cutoff):
  '''
  Plots the energy over epochs. Cutoff shows the difference if combining 
  vmc sampling and qmc data training. If it is 0, nothing is plotted.
  '''
  fig = plt.figure(1,figsize=(6,2.5), dpi=120, facecolor='w', edgecolor='k')
  #minimum_e = np.min(energy)

  plt.plot(energy_vmc,marker='o',markersize=2,linewidth=0.0,markevery=1,label="VMC")
  plt.plot(energy_data,marker='o',markersize=2,linewidth=0.0,markevery=1,label="Data")
  plt.plot(energy_hybrid,marker='o',markersize=2,linewidth=0.0,markevery=1,label="Hybrid")
  if exact_energy != None:
    plt.hlines(exact_energy,0, total_epochs,linestyle="--",label="Exact")
  #plt.hlines(minimum_e,0, total_epochs,linestyle="-",label="minimum")
    
  if cutoff != 0:
    plt.vlines(cutoff, energy[0], exact_energy, linestyle="--", label="QMC train cutoff")

  plt.xlabel("Step",fontsize=15)
  plt.ylabel("$\\langle H \\rangle$",fontsize=20)
  plt.title("{} sites".format(N))
  plt.legend(loc="best")
  #plt.xlim(200,500)
  #plt.ylim(-0.5,-0.2)

  plt.show()
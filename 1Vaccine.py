# -*- coding: utf-8 -*-
"""
Created on Sun May 23 16:21:53 2021

@author: Tarun Mohan
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# beta -> Virus transmission rate
beta = 1.12
# delta -> Natural Birth rate / Natural Death Rate
delta = 3.3 * 10**-5
# alpha -> Recovery Rate
alpha = 1.0 / 7.0 + 0.0366
# gamma -> rate of vaccination (.001 -> 1) (*logspace*)
gamma = 0.01
#epsilon -> vaccine efficacy
epsilon = .5
# n -> total population
n = 328.2 * 10**6
# s -> # suspectible people, iw-> # infected w. wild type virus
# ie -> # infected w. escaped virus, # v -> # vaccinated, r-> # recovered
# fixed points - set time derivatives equal to zero
def model(m, t): 
    s = m[0]
    iw = m[1]
    ie = m[2]
    v = m[3]
    r = m[4]
# dsdt -> change in susceptible category   
    dsdt = delta * n - beta*iw*s/n - beta*ie*s/n - gamma*s - delta*s
#diwdt = change in # ppl infected w/ wild type virus
    diwdt = beta*iw*s/n - (delta + alpha)*iw 
    diwdt_tot = beta*iw*s/n
#diedt = change in # ppl infected w/ escaped virus
    diedt = beta*ie*s/n + beta*ie*v/n  + (1-epsilon)*beta*iw*v/n - (delta + alpha)*ie
    diedt_tot = beta*ie*s/n + beta*ie*v/n  + (1-epsilon)*beta*iw*v/n
#dvdt = change in # ppl vaccinated
    dvdt = gamma*s - beta*ie*v/n - (1-epsilon)*beta*iw*v/n - delta*v
#drdt = change in # ppl recovered
    drdt = alpha*(iw + ie) - delta*r
    dmdt = [dsdt, diwdt, diedt, dvdt, drdt, diwdt_tot, diedt_tot]        
    return dmdt
l= 1000
m0 = [n-1,1,0,0,0,0,0]
t = np.linspace(0,365,l)
#e = np.linspace(0, 1,6)
#g = np.logspace(-3,0,6)
m = odeint(model, m0, t)
#code for graphing the model with given values for gamma and epsilon
plt.plot(t, np.log10(m[:,0]), "b-", label = "S")
plt.plot(t, np.log10(m[:,1]), "r--", label = "I$_w$")
plt.plot(t, np.log10(m[:,2]), "g-.", label = "I$_e$")
plt.plot(t, np.log10(m[:,3]), "k:", label = "V")
plt.plot(t, np.log10(m[:,4]), "y-", label = "R")
plt.xlabel("Time (d)", fontsize = "16")
plt.ylabel("Population", fontsize = "16")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim([0, 9])
plt.legend(loc = "best", fontsize = "12")
plt.savefig("1Vaccine.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()

#code for creating multiple graphs given varying values for gamma and epsilon
#figure, axes = plt.subplots(nrows = 6, ncols = 6, sharex = "col", sharey = "row")
#figure.set_figheight(20)
#figure.set_figwidth(30)
#for i in range(6):
 #   for j in range(6):
  #      epsilon = e[i]
   #     gamma = g[j]
    #    m = odeint(model, m0, t)
     #   axes[i,j].plot(t, m[:,0], "b-", label = "s")
      #  axes[i,j].plot(t, m[:,1], "r-", label = "iw")
       # axes[i,j].plot(t, m[:,2], "g-", label = "ie")
        #axes[i,j].plot(t, m[:,3], "k-", label = "v")
        #axes[i,j].plot(t, m[:,4], "y-", label = "r")
        #axes[i,j].set_title("gamma: " + str(round(gamma,4)) + "   epsilon: " + str(round(epsilon,4)) + "  %esc = " + str(round(100* (m[l-1,6]) / (m[l-1,6] + m[l-1,5]), 4)))
#lines, labels = figure.axes[-1].get_legend_handles_labels()
#figure.legend(lines, labels, bbox_to_anchor=(1.05, .5, 0.3, 0.2), loc='upper left',prop={"size":20})
#figure.tight_layout()
#figure.add_subplot(111, frame_on=False)
#plt.tick_params(labelcolor="none", bottom=False, left=False)

#plt.xlabel("Time")
#plt.ylabel("Population")
#plt.show()

# def heatmap(data, row_labels, col_labels, ax=None,
#             cbar_kw={}, cbarlabel="", **kwargs):


#     if not ax:
#         ax = plt.gca()

#     # Plot the heatmap
#     im = ax.imshow(data, **kwargs)

#     # Create colorbar
#     cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
#     cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
#     #plt.xscale("log")
#     # We want to show all ticks...
#     ax.set_xticks([0, 16.6666, 33.3333, 50, 66.6666, 83.3333, 100])
#     ax.set_yticks([2,20,40,60,80,99.2])
#     # ... and label them with the respective list entries.
#     ax.set_xticklabels(col_labels)
#     ax.set_yticklabels(row_labels)
#     # Let the horizontal axes labeling appear on top.
#     ax.tick_params(top=False, bottom=True,
#                    labeltop=False, labelbottom=True)

#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
#              rotation_mode="anchor")

#     # Turn spines off and create white grid.
#     for edge, spine in ax.spines.items():
#         spine.set_visible(False)


#     return im, cbar


# def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
#                      textcolors=("black", "white"),
#                      threshold=None, **textkw):


#     if not isinstance(data, (list, np.ndarray)):
#         data = im.get_array()

#     # Normalize the threshold to the images color range.
#     if threshold is not None:
#         threshold = im.norm(threshold)
#     else:
#         threshold = im.norm(data.max())/2.

#     # Set default alignment to center, but allow it to be
#     # overwritten by textkw.
#     kw = dict(horizontalalignment="center",
#               verticalalignment="center")
#     kw.update(textkw)

#     # Get the formatter in case a string is supplied
#     if isinstance(valfmt, str):
#         valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

#     # Loop over the data and create a `Text` for each "pixel".
#     # Change the text's color depending on the data.
#     texts = []
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            
            

#     return texts
# fig, ax = plt.subplots(figsize = (20,20))
# #code for heatmap generation
# gam = np.logspace(-6,0,100)
# eps = np.linspace(1,0,100)
# esc = (100,100)
# esc = np.zeros(esc)
# gam_r = ["$\mathregular{10^{-6}}$","$\mathregular{10^{-5}}$", "$\mathregular{10^{-4}}$", "$\mathregular{10^{-3}}$", "$\mathregular{10^{-2}}$", "$\mathregular{10^{-1}}$", "$\mathregular{10^{0}}$"]
# eps_r = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
# for i in range(100): 
#     for j in range(100):
#         epsilon = eps[i]
#         gamma = gam[j]
#         m = odeint(model, m0, t)
#         esc[i,j] = m[l-1,6] / (m[l-1,6] + m[l-1,5])
# im, cbar = heatmap(esc, eps_r, gam_r, ax=ax,
#                    cmap="PuBu", cbarlabel="Fraction Escaped Virus",)
# cbar.ax.set_ylabel("Fraction Escaped Virus",rotation=-90, va="bottom", size = 65)
# texts = annotate_heatmap(im, valfmt="{x:.1f}")
# plt.xlabel("Gamma")
# plt.ylabel("Epsilon")
# cbar.ax.tick_params(labelsize=50) 
# ax.tick_params(axis='x', labelsize=50)
# ax.tick_params(axis='y', labelsize=50)
# ax.xaxis.label.set_size(65)
# ax.yaxis.label.set_size(65)

# plt.show()

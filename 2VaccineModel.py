# -*- coding: utf-8 -*-
"""
Created on Sun May 30 16:14:46 2021

@author: Tarun Mohan
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib
import matplotlib.pyplot as plt

beta_0 = 1.12
beta_1 = beta_0
beta_2 = beta_0
beta_e = beta_0
# delta -> Natural Birth rate / Natural Death Rate
delta = 3.3 * 10**-5
# alpha -> Recovery Rate
alpha = 1.0 / 7.0
# gamma -> rate of vaccination (.001 -> 1) (*logspace*)
gamma_1 = .01
gamma_2 = gamma_1
#epsilon -> vaccine efficacy
eps_1 = .5
eps_2 = eps_1
# n -> total population
n = 328.2*10**6
#delta_v -> rate of death due to virus
delta_v = 0.00366
#omega -> rate of waning immunity
omega = 1.0 / 30.0
# s -> # suspectible people, iw-> # infected w. wild type virus
# ie -> # infected w. escaped virus, # v -> # vaccinated, r-> # recovered
# fixed points - set time derivatives equal to zero
def model(m, t): 
    s = m[0]
    i0 = m[1]
    i_1_0 = m[2]
    i_1_1 = m[3]
    i_2_0 = m[4]
    i_2_2 = m[5]
    ie_0 = m[6]
    ie_1 = m[7]
    ie_2 = m[8]
    v1 = m[9]
    v2 = m[10]
    r0 = m[11]
    r1 = m[12]
    r2 = m[13]
    d = m[14]

    dsdt = delta*(n-d) - beta_0*i0*s/n - beta_1*(i_1_0 + i_1_1)*s/n - beta_2*(i_2_0 + i_2_2)*s/n - beta_e*(ie_0 + ie_1 + ie_2)*s/n - gamma_1*s - gamma_2*s - delta*s + omega*r0
    
    di0dt = beta_0*i0*s/n - (delta + alpha + delta_v)*i0
    di0dt_tot = beta_0*i0*s/n
    
    di_1_0dt = beta_1*(i_1_0 + i_1_1)*s/n - (delta + alpha + delta_v)*i_1_0
    di_1_0dt_tot = beta_1*(i_1_0 + i_1_1)*s/n
    
    di_1_1dt = beta_1*(i_1_0 + i_1_1)*v1/n + (1-eps_1)*beta_0*i0*v1/n - (delta + alpha + delta_v)*i_1_1
    di_1_1dt_tot = beta_1*(i_1_0 + i_1_1)*v1/n + (1-eps_1)*beta_0*i0*v1/n
    
    di_2_0dt = beta_2*(i_2_0 + i_2_2)*s/n - (delta + alpha + delta_v)*i_2_0
    di_2_0dt_tot = beta_2*(i_2_0 + i_2_2)*s/n
    
    di_2_2dt = beta_2*(i_2_0 + i_2_2)*v2/n + (1-eps_2)*beta_0*i0*v2/n - (delta + alpha + delta_v)*i_2_2
    di_2_2dt_tot = beta_2*(i_2_0 + i_2_2)*v2/n + (1-eps_2)*beta_0*i0*v2/n

    die_0dt = beta_e*(ie_0 + ie_1 + ie_2)*s/n - (delta + alpha + delta_v)*ie_0
    die_0dt_tot = beta_e*(ie_0 + ie_1 + ie_2)*s/n

    die_1dt = beta_e*(ie_0 + ie_1 + ie_2)*v1/n + (1-eps_1)*beta_2*(i_2_0 + i_2_2)*v1/n - (delta + alpha + delta_v)*ie_1
    die_1dt_tot = beta_e*(ie_0 + ie_1 + ie_2)*v1/n + (1-eps_1)*beta_2*(i_2_0 + i_2_2)*v1/n
    
    die_2dt = beta_e*(ie_0 + ie_1 + ie_2)*v2/n + (1-eps_2)*beta_1*(i_1_0 + i_1_1)*v2/n - (delta + alpha + delta_v)*ie_2
    die_2dt_tot = beta_e*(ie_0 + ie_1 + ie_2)*v2/n + (1-eps_2)*beta_1*(i_1_0 + i_1_1)*v2/n
    
    dv1dt = gamma_1*(s + r0) - beta_1*(i_1_0 + i_1_1)*v1/n - (1-eps_1)*beta_0*i0*v1/n -beta_e*(ie_0 + ie_1 + ie_2)*v1/n - (1-eps_2)*beta_1*(i_1_0 + i_1_1)*v1/n - delta*v1 + omega*r1
    
    dv2dt = gamma_2*(s + r0) - beta_2*(i_2_0 + i_2_2)*v2/n - (1-eps_2)*beta_0*i0*v2/n -beta_e*(ie_0 + ie_1 + ie_2)*v2/n - (1-eps_1)*beta_2*(i_2_0 + i_2_2)*v2/n - delta*v2 + omega*r2
    
    dr0dt = alpha*(i0 + i_1_0 + i_2_0 + ie_0) - gamma_1*r0 - gamma_2*r0 - delta*r0 - omega*r0
    
    dr1dt = alpha*(i_1_1 + ie_1) - delta*r1 - omega*r1
    
    dr2dt = alpha*(i_2_2 + ie_2) - delta*r2 - omega*r2
    
    dddt = delta_v*(i0 + i_1_0 + i_1_1 + i_2_0 + i_2_2 + ie_0 + ie_1 + ie_2)
    
    dmdt = [dsdt, di0dt, di_1_0dt, di_1_1dt, di_2_0dt, di_2_2dt, die_0dt, die_1dt, die_2dt, dv1dt, dv2dt, dr0dt, dr1dt, dr2dt, dddt, di0dt_tot, di_1_0dt_tot, di_1_1dt_tot, di_2_0dt_tot, di_2_2dt_tot, die_0dt_tot, die_1dt_tot, die_2dt_tot]    
    return dmdt
l = 1000
m0 = [n-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
t = np.linspace(0,365, l)
e = np.linspace(0, 1,6)
g = np.logspace(-3,0,6)
m = odeint(model, m0, t)

#code for graphing the model with given values for gamma and epsilon
plt.plot(t, np.log10(m[:,0]), "b-", label = "S")
plt.plot(t, np.log10(m[:,1]), "r--", label = "I$_0$")
plt.plot(t, np.log10(m[:,2]+m[:,3]), "g-.", label = "I$_1$")
plt.plot(t, np.log10(m[:,4]+m[:,5]), "k:", label = "I$_2$", linewidth = "3")
plt.plot(t, np.log10(m[:,6]+m[:,7]+m[:,8]), "y-", label = "I$_e$")
plt.plot(t, np.log10(m[:,9]+m[:,10]), "m--", label = "V")
plt.plot(t, np.log10(m[:,11]+m[:,12]+m[:,13]), "c-.", label = "R")
plt.plot(t, np.log10(m[:,14]), "brown", label = "D")
plt.xlabel("Time (d)", fontsize = "16")
plt.ylabel("Population", fontsize = "16")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim([0,9])
#plt.text(-2,7,'A', fontsize = '28')
plt.legend(loc = "lower right",fontsize = "12")
plt.savefig("2Vaccine-1.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()

plt.plot(t, np.log10(m[:,7]), "g-", label = "I$_e^{(1)}$")
plt.plot(t, np.log10(m[:,8]), "b--", label = "I$_e^{(2)}$")
plt.plot(t, np.log10(m[:,9]), "r-.", label = "V$_1$")
plt.plot(t, np.log10(m[:,10]), "g:", label = "V$_2$")
plt.plot(t, np.log10(m[:,11]), "k-", label = "R$_0$")
plt.plot(t, np.log10(m[:,12]), "y--", label = "R$_1$")
plt.plot(t, np.log10(m[:,13]), "m-.", label = "R$_2$")
plt.plot(t, np.log10(m[:,14]), "c:", label = "D")
plt.xlabel("Time (d)", fontsize = "16")
plt.ylabel("Population", fontsize = "16")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim([0,9])
plt.text(-2,8,'B', fontsize = '28')
plt.legend(loc = "lower right",fontsize = "12")
plt.savefig("2Vaccine-2.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()


#code for creating multiple graphs given varying values for gamma and epsilon
#figure, axes = plt.subplots(nrows = 6, ncols = 6, sharex = "col", sharey = "row")
#figure.set_figheight(20)
#figure.set_figwidth(30)
#for i in range(6):
 #   for j in range(6):
  #      epsilon = e[i]
   #     gamma = g[j]
   #     m = odeint(model, m0, t)
   #     axes[i,j].plot(t, m[:,0], "b-", label = "s")
   #     axes[i,j].plot(t, m[:,1], "r-", label = "iw")
   #     axes[i,j].plot(t, m[:,2], "g-", label = "ieu")
   #     axes[i,j].plot(t, m[:,3], "k-", label = "ie")
   #     axes[i,j].plot(t, m[:,4], "y-", label = "v")
   #     axes[i,j].plot(t, m[:,5], "m-", label = "r")
   #    axes[i,j].plot(t, m[:,6], "c-", label = "rv")
   #     axes[i,j].plot(t, m[:,7], "g:", label = "d")
   #     axes[i,j].set_title("gamma: " + str(round(gamma,4)) + "   epsilon: " + str(round(epsilon,4)))
#lines, labels = figure.axes[-1].get_legend_handles_labels()
#figure.legend(lines, labels, bbox_to_anchor=(1.05, .5, 0.3, 0.2), loc='upper left',prop={"size":20})
#figure.tight_layout()
#figure.add_subplot(111, frame_on=False)
#plt.tick_params(labelcolor="none", bottom=False, left=False)

#plt.xlabel("Time")
#plt.ylabel("Population")
#plt.show()
#total_pop = m[999][0] + m[999][1] + m[999][2] + m[999][3] + m[999][4] + m[999][5] + m[999][6] + m[999][7]
#print(str(total_pop))
# def heatmap(data, row_labels, col_labels, ax=None,
#             cbar_kw={}, cbarlabel="", **kwargs):


#     if not ax:
#         ax = plt.gca()

#     # Plot the heatmap
#     im = ax.imshow(data, **kwargs)

#     # Create colorbar
#     cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
#     cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

#     # We want to show all ticks...
#     ax.set_xticks([0, 16.6666, 33.3333, 50, 66.6666, 83.3333, 100])
#     ax.set_yticks([0,20,40,60,80,99.2])
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
#         eps_1 = eps[i]
#         eps_2 = eps[i]
#         gamma_1 = gam[j]/2.0
#         gamma_2 = gam[j]/2.0
#         m = odeint(model, m0, t)
#         esc[i,j] = (m[l-1,22] + m[l-1, 21] + m[l-1, 20]) / (m[l-1,15] + m[l-1, 16] + m[l-1, 17] + m[l-1, 18] + m[l-1,19]+ m[l-1,22] + m[l-1, 21] + m[l-1, 20])
# im, cbar = heatmap(esc, eps_r, gam_r, ax=ax,
#                    cmap="PuBu", cbarlabel="Fraction Escaped Virus")
# cbar.ax.set_ylabel("Fraction Escaped Virus",rotation=-90, va="bottom", size = 65)
# texts = annotate_heatmap(im, valfmt="{x:.1f}")
# plt.xlabel("Gamma")
# plt.ylabel("Epsilon")
# cbar.ax.tick_params(labelsize=50) 
# ax.tick_params(axis='x', labelsize=50)
# ax.tick_params(axis='y', labelsize=50)
# ax.xaxis.label.set_size(65)
# ax.yaxis.label.set_size(65)
# fig.tight_layout()
# plt.show()

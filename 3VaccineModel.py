# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 16:22:16 2022

@author: Tarun Mohan
"""
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
beta_12 = beta_0
beta_13 = beta_0
beta_2 = beta_0
beta_23 = beta_0
beta_3 = beta_0
beta_e = beta_0

# delta -> Natural Birth rate / Natural Death Rate
delta = 3.3 * 10**-5
# alpha -> Recovery Rate
alpha = 1.0 / 7.0
# gamma -> rate of vaccination (.001 -> 1) (*logspace*)
gamma_1 = .01
gamma_2 = gamma_1
gamma_3 = gamma_1
#epsilon -> vaccine efficacy
eps_1 = .5
eps_2 = eps_1
eps_3 = eps_1
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
    i_0 = m[1]
    i_1_0 = m[2]
    i_2_0 = m[3]
    i_3_0 = m[4]
    i_12_0 = m[5]
    i_13_0 = m[6]
    i_23_0 = m[7]
    i_e_0 = m[8]
    i_1_1 = m[9]
    i_12_1 = m[10]
    i_13_1 = m[11]
    i_e_1 = m[12]
    i_2_2 = m[13]
    i_12_2 = m[14]
    i_23_2 = m[15]
    i_e_2 = m[16]
    i_3_3 = m[17]
    i_13_3 = m[18]
    i_23_3 = m[19]
    i_e_3 = m[20]
    v_1 = m[21]
    v_2 = m[22]
    v_3 = m[23]
    r_0 = m[24]
    r_1 = m[25]
    r_2 = m[26]
    r_3 = m[27]
    d = m[28]
    
    dsdt = delta*(n - d) - beta_0*i_0*s/n - beta_1*(i_1_0 + i_1_1)*s/n - beta_2*(i_2_0 + i_2_2)*s/n - beta_3*(i_3_0 + i_3_3)*s/n - beta_12*(i_12_0 + i_12_1 + i_12_2)*s/n - beta_13*(i_13_0 + i_13_1 + i_13_3)*s/n - beta_23*(i_23_0 + i_23_2 + i_23_3)*s/n - beta_e*(i_e_0 + i_e_1 + i_e_2 + i_e_3)*s/n - gamma_1*s - gamma_2*s - gamma_3*s - delta*s + omega*r_0
    
    di_0dt = beta_0*i_0*s/n - (delta+alpha+delta_v)*i_0
    di_0dt_tot = beta_0*i_0*s/n
    
    di_1_0dt = beta_1*(i_1_0 + i_1_1)*s/n - (delta+alpha+delta_v)*i_1_0
    di_1_0dt_tot = beta_1*(i_1_0 + i_1_1)*s/n
    
    di_2_0dt = beta_2*(i_2_0 + i_2_2)*s/n - (delta+alpha+delta_v)*i_2_0
    di_2_0dt_tot = beta_2*(i_2_0 + i_2_2)*s/n
    
    di_3_0dt = beta_3*(i_3_0 + i_3_3)*s/n - (delta+alpha + delta_v)*i_3_0
    di_3_0dt_tot = beta_3*(i_3_0 + i_3_3)*s/n
    
    di_12_0dt = beta_12*(i_12_0 + i_12_1 + i_12_2)*s/n - (delta+alpha+delta_v)*i_12_0
    di_12_0dt_tot = beta_12*(i_12_0 + i_12_1 + i_12_2)*s/n
    
    di_13_0dt = beta_13*(i_13_0 + i_13_1 + i_13_3)*s/n - (delta+alpha+delta_v)*i_13_0
    di_13_0dt_tot = beta_13*(i_13_0 + i_13_1 + i_13_3)*s/n 
    
    di_23_0dt = beta_23*(i_23_0 + i_23_2 + i_23_3)*s/n - (delta+alpha+delta_v)*i_23_0
    di_23_0dt_tot = beta_23*(i_23_0 + i_23_2 + i_23_3)*s/n 
    
    di_e_0dt = beta_e*(i_e_0 + i_e_1 + i_e_2 + i_e_3)*s/n - (delta+alpha+delta_v)*i_e_0
    di_e_0dt_tot = beta_e*(i_e_0 + i_e_1 + i_e_2 + i_e_3)*s/n 
    
    di_1_1dt = beta_1*(i_1_0 + i_1_1)*v_1/n + (1-eps_1)*beta_0*i_0*v_1/n - (delta+alpha+delta_v)*i_1_1
    di_1_1dt_tot = beta_1*(i_1_0 + i_1_1)*v_1/n + (1-eps_1)*beta_0*i_0*v_1/n 
    
    di_12_1dt = beta_12*(i_12_0 + i_12_1 + i_12_2)*v_1/n + (1-eps_1)*beta_2*(i_2_0 + i_2_2)*v_1/n - (delta+alpha+delta_v)*i_12_1
    di_12_1dt_tot = beta_12*(i_12_0 + i_12_1 + i_12_2)*v_1/n + (1-eps_1)*beta_2*(i_2_0 + i_2_2)*v_1/n 
    
    di_13_1dt = beta_13*(i_13_0 + i_13_1 + i_13_3)*v_1/n + (1-eps_1)*beta_3*(i_3_0 + i_3_3)*v_1/n - (delta + alpha+delta_v)*i_13_1
    di_13_1dt_tot = beta_13*(i_13_0 + i_13_1 + i_13_3)*v_1/n + (1-eps_1)*beta_3*(i_3_0 + i_3_3)*v_1/n
    
    di_e_1dt = beta_e*(i_e_0 + i_e_1 + i_e_2 + i_e_3)*v_1/n + (1-eps_1)*beta_23*(i_23_0 + i_23_2 + i_23_3)*v_1/n - (delta+alpha+delta_v)*i_e_1
    di_e_1dt_tot = beta_e*(i_e_0 + i_e_1 + i_e_2 + i_e_3)*v_1/n + (1-eps_1)*beta_23*(i_23_0 + i_23_2 + i_23_3)*v_1/n
    
    di_2_2dt = beta_2*(i_2_0 + i_2_2)*v_2/n + (1-eps_2)*beta_0*i_0*v_2/n - (delta+alpha+delta_v)*i_2_2
    di_2_2dt_tot = beta_2*(i_2_0 + i_2_2)*v_2/n + (1-eps_2)*beta_0*i_0*v_2/n 
    
    di_12_2dt = beta_12*(i_12_0 + i_12_1 + i_12_2)*v_2/n + (1-eps_2)*beta_1*(i_1_0 + i_1_1)*v_2/n - (delta+alpha+delta_v)*i_12_2
    di_12_2dt_tot = beta_12*(i_12_0 + i_12_1 + i_12_2)*v_2/n + (1-eps_2)*beta_1*(i_1_0 + i_1_1)*v_2/n 
    
    di_23_2dt = beta_23*(i_23_0 + i_23_2 + i_23_3)*v_2/n + (1-eps_2)*beta_3*(i_3_0 + i_3_3)*v_2/n - (delta + alpha+delta_v)*i_23_2
    di_23_2dt_tot = beta_23*(i_23_0 + i_23_2 + i_23_3)*v_2/n + (1-eps_2)*beta_3*(i_3_0 + i_3_3)*v_2/n 
    
    di_e_2dt = beta_e*(i_e_0 + i_e_1 + i_e_2 + i_e_3)*v_2/n + (1-eps_2)*beta_13*(i_13_0 + i_13_1 + i_13_3)*v_2/n - (delta+alpha+delta_v)*i_e_2
    di_e_2dt_tot = beta_e*(i_e_0 + i_e_1 + i_e_2 + i_e_3)*v_2/n + (1-eps_2)*beta_13*(i_13_0 + i_13_1 + i_13_3)*v_2/n 
    
    di_3_3dt = beta_3*(i_3_0 + i_3_3)*v_3/n + (1-eps_3)*beta_0*i_0*v_3/n - (delta+alpha+delta_v)*i_3_3
    di_3_3dt_tot = beta_3*(i_3_0 + i_3_3)*v_3/n + (1-eps_3)*beta_0*i_0*v_3/n 
    
    di_13_3dt = beta_13*(i_13_0 + i_13_1 + i_13_3)*v_3/n + (1-eps_3)*beta_1*(i_1_0 + i_1_1)*v_3/n - (delta+alpha+delta_v)*i_13_3
    di_13_3dt_tot = beta_13*(i_13_0 + i_13_1 + i_13_3)*v_3/n + (1-eps_3)*beta_1*(i_1_0 + i_1_1)*v_3/n 
    
    di_23_3dt = beta_23*(i_23_0 + i_23_2 + i_23_3)*v_3/n + (1-eps_3)*beta_2*(i_2_0 + i_2_2)*v_3/n - (delta + alpha+delta_v)*i_23_3
    di_23_3dt_tot = beta_23*(i_23_0 + i_23_2 + i_23_3)*v_3/n + (1-eps_3)*beta_2*(i_2_0 + i_2_2)*v_3/n 
    
    di_e_3dt = beta_e*(i_e_0 + i_e_1 + i_e_2 + i_e_3)*v_3/n + (1-eps_3)*beta_12*(i_12_0 + i_12_1 + i_12_2)*v_3/n - (delta+alpha+delta_v)*i_e_3
    di_e_3dt_tot = beta_e*(i_e_0 + i_e_1 + i_e_2 + i_e_3)*v_3/n + (1-eps_3)*beta_12*(i_12_0 + i_12_1 + i_12_2)*v_3/n
    
    dv_1dt = gamma_1*(s+r_0) - beta_1*(i_1_0 + i_1_1)*v_1/n - (1-eps_1)*beta_0*i_0*v_1/n - beta_12*(i_12_0 + i_12_1 + i_12_2)*v_1/n - (1-eps_1)*beta_2*(i_2_0 + i_2_2)*v_1/n - beta_13*(i_13_0 + i_13_1 + i_13_3)*v_1/n - (1-eps_1)*beta_3*(i_3_0 + i_3_3)*v_1/n - (1-eps_1)*beta_23*(i_23_0 + i_23_2 + i_23_3)*v_1/n - beta_e*(i_e_0 + i_e_1 + i_e_2 + i_e_3)*v_1/n - delta*v_1 + omega*r_1
    
    dv_2dt = gamma_2*(s+r_0) - beta_2*(i_2_0 + i_2_2)*v_2/n - (1-eps_2)*beta_0*i_0*v_2/n - beta_12*(i_12_0 + i_12_1 + i_12_2)*v_2/n - (1-eps_2)*beta_1*(i_1_0 + i_1_1)*v_2/n - beta_23*(i_23_0 + i_23_2 + i_23_3)*v_2/n - (1-eps_2)*beta_3*(i_3_0 + i_3_3)*v_2/n - (1-eps_2)*beta_13*(i_13_0 + i_13_1 + i_13_3)*v_2/n - beta_e*(i_e_0 + i_e_1 + i_e_2 + i_e_3)*v_2/n - delta*v_2 + omega*r_2
    
    dv_3dt = gamma_3*(s+r_0) - beta_3*(i_3_0 + i_3_3)*v_3/n - (1-eps_3)*beta_0*i_0*v_3/n - beta_13*(i_13_0 + i_13_1 + i_13_3)*v_3/n - (1-eps_3)*beta_1*(i_1_0 + i_1_1)*v_3/n - beta_23*(i_23_0 + i_23_2 + i_23_3)*v_3/n - (1-eps_3)*beta_2*(i_2_0 + i_2_2)*v_3/n - (1-eps_3)*beta_12*(i_12_0 + i_12_1 + i_12_2)*v_3/n - beta_e*(i_e_0 + i_e_1 + i_e_2 + i_e_3)*v_3/n - delta*v_3 + omega*r_3

    dr_0dt = alpha*(i_0 + i_1_0 + i_2_0 + i_3_0 + i_12_0 + i_13_0 + i_23_0 + i_e_0) - gamma_1*r_0 - gamma_2*r_0 - gamma_3*r_0 - delta*r_0 - omega*r_0
    
    dr_1dt = alpha*(i_1_1 + i_12_1 + i_13_1 + i_e_1) - delta*r_1 - omega*r_1
    
    dr_2dt = alpha*(i_2_2 + i_12_2 + i_23_2 + i_e_2) - delta*r_2 - omega*r_2
    
    dr_3dt = alpha*(i_3_3 + i_13_3 + i_23_3 + i_e_3) - delta*r_3 - omega*r_3
    
    dddt = delta_v*(i_0 + i_1_0 + i_2_0 + i_3_0 + i_12_0 + i_13_0 + i_23_0 + i_e_0 + i_1_1 + i_12_1 + i_13_1 + i_e_1 + i_2_2 + i_12_2 + i_23_2 + i_e_2 + i_3_3 + i_13_3 + i_23_3 + i_e_3)



    dmdt = [dsdt, di_0dt, di_1_0dt, di_2_0dt, di_3_0dt, di_12_0dt, di_13_0dt, di_23_0dt, di_e_0dt, di_1_1dt, di_12_1dt, di_13_1dt, di_e_1dt, di_2_2dt, di_12_2dt, di_23_2dt, di_e_2dt, di_3_3dt, di_13_3dt, di_23_3dt, di_e_3dt, dv_1dt, dv_2dt, dv_3dt, dr_0dt, dr_1dt, dr_2dt, dr_3dt, dddt, di_0dt_tot, di_1_0dt_tot, di_2_0dt_tot, di_3_0dt_tot, di_12_0dt_tot, di_13_0dt_tot, di_23_0dt_tot, di_e_0dt_tot, di_1_1dt_tot, di_12_1dt_tot, di_13_1dt_tot, di_e_1dt_tot, di_2_2dt_tot, di_12_2dt_tot, di_23_2dt_tot, di_e_2dt_tot, di_3_3dt_tot, di_13_3dt_tot, di_23_3dt_tot, di_e_3dt_tot]    
    return dmdt
l = 1000
m0 = [n-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
t = np.linspace(0,365, l)
e = np.linspace(0, 1,6)
g = np.logspace(-3,0,6)
m = odeint(model, m0, t)


# #code for graphing the model with given values for gamma and epsilon
plt.plot(t, np.log10(m[:,0]), "b-", label = "S")
plt.plot(t, np.log10(m[:,1]), "r--", label = "I (wild-type)")
plt.plot(t, np.log10(m[:,2]+m[:,9]), "g-.", label = "I (escaped 1)")
plt.plot(t, np.log10(m[:,5]+m[:,10]+m[:,14]), "k:", label = "I (escaped 2) ")
plt.plot(t, np.log10(m[:,8]+m[:,12]+m[:,16]+m[:,20]), "y-", label = "I (escaped all)")
plt.plot(t, np.log10(m[:,21]+m[:,22]+m[:,23]), "m--", label = "V (all)")
plt.plot(t, np.log10(m[:,24]+m[:,25]+m[:,26]+m[:,27]), "c-.", label = "R (all)")
plt.plot(t, np.log10(m[:,28]), "brown", label = "D")
plt.xlabel("Time (d)", fontsize = "16")
plt.ylabel("Population", fontsize = "16")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim([0, 9])
#plt.text(-2,7,'A', fontsize = '28')
plt.legend(loc = "lower right",fontsize = "12")
plt.savefig("3Vaccine-1.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()

plt.plot(t, np.log10(m[:,7]), "aqua", label = "I$_{23}^{(0)}$")
plt.plot(t, np.log10(m[:,8]), "azure", linestyle = '--', label = "I$_e^{(0)}$")
plt.plot(t, np.log10(m[:,9]), "beige", linestyle = '-.', label = "I$_1^{(1)}$")
plt.plot(t, np.log10(m[:,10]), "brown", linestyle = ':', label = "I$_{12}^{(1)}$")
plt.plot(t, np.log10(m[:,11]), "chartreuse", label = "I$_{13}^{(1)}$")
plt.plot(t, np.log10(m[:,12]), "darkblue", linestyle = '--', label = "I$_e^{(1)}$")
plt.plot(t, np.log10(m[:,13]), "darkgreen", linestyle = '-.', label = "I$_2^{(2)}$")
plt.xlabel("Time (d)", fontsize = "16")
plt.ylabel("Population", fontsize = "16")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim([0, 9])
plt.text(-2,8,'B', fontsize = '28')
plt.legend(loc = "lower right",fontsize = "12")
plt.savefig("3Vaccine-2.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()

plt.plot(t, np.log10(m[:,14]), "gold", label = "I$_{12}^{(2)}$")
plt.plot(t, np.log10(m[:,15]), "grey", linestyle = '--', label = "I$_{23}^{(2)}$")
plt.plot(t, np.log10(m[:,16]), "indigo", linestyle = '-.', label = "I$_e^{(2)}$")
plt.plot(t, np.log10(m[:,17]), "ivory", linestyle = ':', label = "I$_3^{(3)}$")
plt.plot(t, np.log10(m[:,18]), "lavender", label = "I$_{13}^{(3)}$")
plt.plot(t, np.log10(m[:,19]), "magenta", linestyle = '--', label = "I$_{23}^{(3)}$")
plt.plot(t, np.log10(m[:,20]), "olive", linestyle = '-.', label = "I$_e^{(3)}$")
plt.xlabel("Time (d)", fontsize = "16")
plt.ylabel("Population", fontsize = "16")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim([0, 9])
plt.text(-2,8,'C', fontsize = '28')
plt.legend(loc = "lower right",fontsize = "12")
plt.savefig("3Vaccine-3.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()

plt.plot(t, np.log10(m[:,21]), "orange", label = "V$_1$")
plt.plot(t, np.log10(m[:,22]), "pink", linestyle = '--', label = "V$_2$")
plt.plot(t, np.log10(m[:,23]), "plum", linestyle = '-.', label = "V$_3$")
plt.plot(t, np.log10(m[:,24]), "salmon", linestyle = ':', label = "R$_0$")
plt.plot(t, np.log10(m[:,25]), "teal", label = "R$_1$")
plt.plot(t, np.log10(m[:,26]), "turquoise", linestyle = '--', label = "R$_2$")
plt.plot(t, np.log10(m[:,27]), "sienna", linestyle = '-.', label = "R$_3$")
plt.plot(t, np.log10(m[:,28]), "tomato", linestyle = ':', label = "D")
plt.xlabel("Time (d)", fontsize = "16")
plt.ylabel("Population", fontsize = "16")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim([0, 9])
plt.text(-2,8,'D', fontsize = '28')
plt.legend(loc = "lower right",fontsize = "12")
plt.savefig("3Vaccine-4.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()

# tot_pop = 0
# for j in range(29):
#     tot_pop += m[999][j]
# print(tot_pop)


#code for creating multiple graphs given varying values for gamma and epsilon
# figure, axes = plt.subplots(nrows = 6, ncols = 6, sharex = "col", sharey = "row")
# figure.set_figheight(20)
# figure.set_figwidth(30)
# for i in range(6):
#     for j in range(6):
#         epsilon = e[i]
#         gamma = g[j]
#         m = odeint(model, m0, t)
#         axes[i,j].plot(t, m[:,0], "b-", label = "s")
#         axes[i,j].plot(t, m[:,1], "r-", label = "iw")
#         axes[i,j].plot(t, m[:,2], "g-", label = "ieu")
#         axes[i,j].plot(t, m[:,3], "k-", label = "ie")
#         axes[i,j].plot(t, m[:,4], "y-", label = "v")
#         axes[i,j].plot(t, m[:,5], "m-", label = "r")
#         axes[i,j].plot(t, m[:,6], "c-", label = "rv")
#         axes[i,j].plot(t, m[:,7], "g:", label = "d")
#         axes[i,j].set_title("gamma: " + str(round(gamma,4)) + "   epsilon: " + str(round(epsilon,4)))
# lines, labels = figure.axes[-1].get_legend_handles_labels()
# figure.legend(lines, labels, bbox_to_anchor=(1.05, .5, 0.3, 0.2), loc='upper left',prop={"size":20})
# figure.tight_layout()
# figure.add_subplot(111, frame_on=False)
# plt.tick_params(labelcolor="none", bottom=False, left=False)

# plt.xlabel("Time")
# plt.ylabel("Population")
# plt.show()
# total_pop = m[999][0] + m[999][1] + m[999][2] + m[999][3] + m[999][4] + m[999][5] + m[999][6] + m[999][7]
# print(str(total_pop))

#hetmap generator code
#def heatmap(data, row_labels, col_labels, ax=None,
#            cbar_kw={}, cbarlabel="", **kwargs):


#    if not ax:
#        ax = plt.gca()

    # Plot the heatmap
#    im = ax.imshow(data, **kwargs)

    # Create colorbar
#    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
#    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
#    ax.set_xticks([0, 16.6666, 33.3333, 50, 66.6666, 83.3333, 100])
#    ax.set_yticks([0,20,40,60,80,99.2])
    # ... and label them with the respective list entries.
#    ax.set_xticklabels(col_labels)
#    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
#    ax.tick_params(top=False, bottom=True,
#                    labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
#    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
#              rotation_mode="anchor")

    # Turn spines off and create white grid.
#    for edge, spine in ax.spines.items():
#        spine.set_visible(False)


#    return im, cbar


#def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
#                      textcolors=("black", "white"),
#                      threshold=None, **textkw):


#    if not isinstance(data, (list, np.ndarray)):
#        data = im.get_array()

#     # Normalize the threshold to the images color range.
#    if threshold is not None:
#        threshold = im.norm(threshold)
#    else:
#        threshold = im.norm(data.max())/2.

#     # Set default alignment to center, but allow it to be
#     # overwritten by textkw.
#    kw = dict(horizontalalignment="center",
#              verticalalignment="center")
#    kw.update(textkw)

#     # Get the formatter in case a string is supplied
#    if isinstance(valfmt, str):
#        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

#     # Loop over the data and create a `Text` for each "pixel".
#     # Change the text's color depending on the data.
#    texts = []
#    for i in range(data.shape[0]):
#        for j in range(data.shape[1]):
#            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            
            

#    return texts
#fig, ax = plt.subplots(figsize = (20,20))
# #code for heatmap generation
#gam = np.logspace(-6,0,100)
#eps = np.linspace(1,0,100)
#esc = (100,100)
#esc = np.zeros(esc)
#gam_r = ["$\mathregular{10^{-6}}$","$\mathregular{10^{-5}}$", "$\mathregular{10^{-4}}$", "$\mathregular{10^{-3}}$", "$\mathregular{10^{-2}}$", "$\mathregular{10^{-1}}$", "$\mathregular{10^{0}}$"]
#eps_r = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
#for i in range(100):
#    for j in range(100):
#        eps_1 = eps[i]
#        eps_2 = eps[i]
 #       eps_3 = eps[i]
#        gamma_1 = gam[j]/3.0
#        gamma_2 = gam[j]/3.0
#        gamma_3 = gam[j]/3.0
#        m = odeint(model, m0, t)
#        esc[i,j] = (m[l-1,36] + m[l-1, 40] + m[l-1, 44] + m[l-1, 48]) / (m[l-1,29] + m[l-1, 30] + m[l-1, 31] + m[l-1, 32] + m[l-1,33]+ m[l-1,34] + m[l-1, 35] + m[l-1, 37] + m[l-1, 38]+ m[l-1, 39]+ m[l-1, 41]+ m[l-1, 42]+ m[l-1, 43]+ m[l-1, 45]+ m[l-1, 46]+ m[l-1, 47] + m[l-1,36] + m[l-1, 40] + m[l-1, 44] + m[l-1, 48])
#im, cbar = heatmap(esc, eps_r, gam_r, ax=ax,
#                    cmap="PuBu", cbarlabel="Fraction Escaped Virus")
#cbar.ax.set_ylabel("Fraction Escaped Virus",rotation=-90, va="bottom", size = 65)
#texts = annotate_heatmap(im, valfmt="{x:.1f}")
#plt.xlabel("Gamma")
#plt.ylabel("Epsilon")
#cbar.ax.tick_params(labelsize=50) 
#ax.tick_params(axis='x', labelsize=50)
#ax.tick_params(axis='y', labelsize=50)
#ax.xaxis.label.set_size(65)
#ax.yaxis.label.set_size(65)
#fig.tight_layout()
#plt.show()

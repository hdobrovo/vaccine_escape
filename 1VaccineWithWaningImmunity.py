import numpy as np
from scipy.integrate import odeint
import matplotlib
import matplotlib.pyplot as plt

# beta_w -> wild type virus transmission rate
beta_w = 1.12
#beta_e -> escaped virus transmission rate
beta_e = 1.12 
# delta -> Natural Birth rate / Natural Death Rate
delta = 3.3 * 10**-5
# alpha -> Recovery Rate
alpha = 1.0 / 7.0
# gamma -> rate of vaccination (.001 -> 1) (*logspace*)
gamma = .01
#epsilon -> vaccine efficacy
epsilon = .5
# n -> total population
n = 328.2 * 10**6
#delta_v -> rate of death due to virus
delta_v = 0.00366
#omega -> rate of waning immunity
omega = 1.0 / 30.0
# s -> # suspectible people, iw-> # infected w. wild type virus
# ie -> # infected w. escaped virus, # v -> # vaccinated, r-> # recovered
# fixed points - set time derivatives equal to zero
def model(m, t): 
    s = m[0]
    iw = m[1]
    ieu = m[2]
    ie = m[3]
    v = m[4]
    r = m[5]
    rv = m[6]
    d = m[7]
# dsdt -> change in susceptible category   
    dsdt = delta*(n-d) - beta_w*iw*s/n - beta_e*(ie+ ieu)*s/n - gamma*s - delta*s + omega*r
#diwdt = change in # ppl infected w/ wild type virus
    diwdt = beta_w*iw*s/n - (delta + alpha + delta_v)*iw 
    diwdt_tot = beta_w*iw*s/n
#dieudt --> change in unvaccinated ppl with escaped virus    
    dieudt = beta_e*(ie + ieu)*s/n - (delta+alpha+delta_v)*ieu
    dieudt_tot = beta_e*(ie + ieu)*s/n
#diedt = change in # vacinnated ppl infected w/ escaped virus
    diedt = beta_e*(ie+ieu)*v/n + (1-epsilon)*beta_w*iw*v/n - (delta + alpha + delta_v)*ie
    diedt_tot = beta_e*(ie+ieu)*v/n + (1-epsilon)*beta_w*iw*v/n
#dvdt = change in # ppl vaccinated
    dvdt = gamma*s - beta_e*(ie+ieu)*v/n - (1-epsilon)*beta_w*iw*v/n - delta*v + omega*rv + gamma*r
#drdt = change in # unvaccinated ppl recovered
    drdt = alpha*(iw + ieu) - delta*r - omega*r - gamma*r
#drvdt -> change in # vaccinated ppl recovering from escaped virus
    drvdt = alpha*ie - delta*rv - omega*rv
#dddt = change in # of ppl dead due to virus
    dddt = delta_v*(iw + ie + ieu) 
    dmdt = [dsdt, diwdt, dieudt, diedt, dvdt, drdt, drvdt, dddt, diwdt_tot, dieudt_tot, diedt_tot]        
    return dmdt
m0 = [n-1,1,0,0,0,0,0,0,1,0,0]
l = 1000
t = np.linspace(0,365,l)
e = np.linspace(0, 1,6)
g = np.logspace(-3,0,6)
m = odeint(model, m0, t)

#code for graphing the model with given values for gamma and epsilon
plt.plot(t, np.log10(m[:,0]), "b-", label = "S")
plt.plot(t, np.log10(m[:,1]), "r--", label = "I$_w$")
plt.plot(t, np.log10(m[:,2]+m[:,3]), "g-.", label = "I$_e$ (all)")
plt.plot(t, np.log10(m[:,4]), "k:", label = "V")
plt.plot(t, np.log10(m[:,5]+m[:,6]), "y-", label = "R (all)")
plt.plot(t, np.log10(m[:,7]), "m--", label = "D")
plt.xlabel("Time (d)", fontsize = "16")
plt.ylabel("Population", fontsize = "16")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim([0, 9])
plt.legend(loc = "lower right", fontsize = "12")
plt.savefig("1VaccinewI.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()
#total_pop = m[999][0] + m[999][1] + m[999][2] + m[999][3] + m[999][4] + m[999][5] + m[999][6] + m[999][7]
#print(str(total_pop - n))


#code for creating multiple graphs given varying values for gamma and epsilon
#figure, axes = plt.subplots(nrows = 6, ncols = 6, sharex = "col", sharey = "row")
#figure.set_figheight(20)
#figure.set_figwidth(30)
#for i in range(6):
#    for j in range(6):
#        epsilon = e[i]
#        gamma = g[j]
#        m = odeint(model, m0, t)
#        axes[i,j].plot(t, m[:,0], "b-", label = "s")
#        axes[i,j].plot(t, m[:,1], "r-", label = "iw")
#        axes[i,j].plot(t, m[:,2], "g-", label = "ieu")
#       axes[i,j].plot(t, m[:,3], "k-", label = "ie")
#        axes[i,j].plot(t, m[:,4], "y-", label = "v")
#        axes[i,j].plot(t, m[:,5], "m-", label = "r")
#        axes[i,j].plot(t, m[:,6], "c-", label = "rv")
#        axes[i,j].plot(t, m[:,7], "g:", label = "d")
#        axes[i,j].set_title("gamma: " + str(round(gamma,4)) + "   epsilon: " + str(round(epsilon,4)))
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
#     ax.set_yticks([4,20,40,60,80,99.2])
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
#         esc[i,j] = (m[l-1,10] + m[l-1, 9]) / (m[l-1,10] + m[l-1, 9] + m[l-1, 8])
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

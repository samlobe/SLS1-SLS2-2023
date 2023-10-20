#%%
import matplotlib.pyplot as plt
import numpy as np

colors = ['black','gold','lightseagreen','#FF7F0E','#1F77B4']
markers = ['o','^','s','D','*']
files = ['bulk','P301','301L','jR2R3','jR2R3_P301L']
labels = ['Bulk','P301','L301','jR2R3','jR2R3 P301L']

fig, ax = plt.subplots(figsize=(6, 4))

for i in range(len(files)):
    data = np.loadtxt(files[i] + '.dat')
    if i == 0:
        ax.plot(data[:, 0], data[:, 1], linestyle='--', color=colors[i], marker=markers[i], label=labels[i])
    else:
        ax.plot(data[:, 0], data[:, 1], color=colors[i], marker=markers[i], label=labels[i])

# show fewer y-ticks
ax.set_yticks([0.885,0.890,0.895,0.900])
ax.set_xticks([300,325,350,375])
plt.ylabel(r'$\mathrm{S_θ}$ / $\mathrm{k_B}$',fontsize=16)
plt.xlabel('Temperature (K)',fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('shannon_entropy_3body.png', dpi=300)

plt.show()

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_labels = ['301_fullres']
# data_labels = ['299_sidechains','300_sidechains','301_sidechains']
b_tetrahedral_frac = 0.23202 # for 300K from 3body.py

confidence = []
for label in data_labels:
    # convert relative tetrahedrality back to % tetrahedral waters
    confidence.append(pd.read_csv(f'res{label}_confidence.csv',index_col=0)*b_tetrahedral_frac*100)
        
for df in confidence:
    df.insert(3,'error_down',df['50%']-df['5%'])
    df.insert(4,'error_up',df['95%']-df['50%'])

#%%
fig,ax = plt.subplots(figsize=(4,4))

barWidth = 0.2
colors = ['#FF7F0E','#1F77B4']
br = np.array([0.7,1]) * np.ones((len(data_labels),2))
for i,row in enumerate(br):
    br[i] = br[i] + i

for i,df in enumerate(confidence):
    plt.bar(br[i],df['50%'],yerr=df.iloc[:,-2:].T.to_numpy(),
            color=colors,width=barWidth,align='edge',label=['jR2R3','jR2R3_P301L'])

plt.xlim(left=0.6,right=1.3)
plt.ylim(bottom=23,top=24)
# plt.xticks([0.95],['aa301'],fontsize=15)
# delete all x-ticks
# plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.title('waters near residue 301',fontsize=15)

# plt.text(0.15, 1.04, 'waters near', transform=plt.gca().transAxes, color='black', fontsize=20, ha='left', va='center')
# plt.text(0.5, 1.08, 'residue 301', transform=plt.gca().transAxes, color='black', fontsize=15, ha='center', va='center')

plt.yticks(fontsize=12)
plt.ylabel('% tetrahedral waters',fontsize=15)

import matplotlib.patches as mpatches

label1 = 'jR2R3'
label2 = 'jR2R3 P301L'

# create patch objects for the custom colors and labels
patch1 = mpatches.Patch(color=colors[0], label=label1)
patch2 = mpatches.Patch(color=colors[1], label=label2)

# create the legend
# plt.legend(handles=[patch1, patch2],fontsize=15,loc='upper left')

plt.xticks([0.8,1.1],[label1,label2],fontsize=15)

# plot a horizontal dashed line at b_tetrahedral_frac*100
plt.axhline(y=b_tetrahedral_frac*100,linestyle='--',color='black')

# write 'bulk water' next to the dashed line centered in the middle of the plot
plt.text(0.95,b_tetrahedral_frac*100+0.03,'bulk water',fontsize=15,color='black',horizontalalignment='center')


# set the y-axis ticks on both sides
major_ticks = np.arange(22, 25.1, 1)
minor_ticks = np.arange(22, 25.1, 0.2)
ax.yaxis.set_major_locator(plt.FixedLocator(major_ticks))
ax.yaxis.set_minor_locator(plt.FixedLocator(minor_ticks))
ax2 = ax.twinx() # create a twin y-axis
ax2.yaxis.set_major_locator(plt.FixedLocator(major_ticks))
ax2.yaxis.set_minor_locator(plt.FixedLocator(minor_ticks))


plt.tight_layout()

# save figure
plt.savefig('aa301_tetrahedral_confidence.png',dpi=300)

plt.show()

# %%

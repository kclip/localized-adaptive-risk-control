import matplotlib.pyplot as plt
import numpy as np
import pickle
plt.rc('font', family='serif', serif='Computer Modern Roman', size=18)
plt.rcParams.update({
    'font.size': 18,
    'text.usetex': True,
})
plt.rcParams['figure.figsize'] = [7, 3.75]
pos_BS = np.asarray([0, -32])
K_ACI_Log,ACI_Log,MondrianACI_Log=[],[],[]

kappa=1.
la= 1e-4
with open('Logs/ARC.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
ARC=loaded_dict['Log']
plt.plot(np.arange(len(ARC[0]))*2,1-np.cumsum(ARC[0])/np.arange(1,len(ARC[0])+1),color='tab:grey',label='ARC')
ls=2.
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
L_ARC=loaded_dict['Log']
plt.plot(np.arange(len(L_ARC[0]))*2,1-np.cumsum(L_ARC[0])/np.arange(1,len(L_ARC[0])+1),color='tab:green',label=r'L-ARC $l=$'+str(ls))

ls=1.
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
L_ARC=loaded_dict['Log']
plt.plot(np.arange(len(L_ARC[0]))*2,1-np.cumsum(L_ARC[0])/np.arange(1,len(L_ARC[0])+1),color='tab:blue',label=r'L-ARC $l=$'+str(ls))

ls=0.5
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
L_ARC=loaded_dict['Log']
plt.plot(np.arange(len(L_ARC[0]))*2,1-np.cumsum(L_ARC[0])/np.arange(1,len(L_ARC[0])+1),color='tab:orange',label=r'L-ARC $l=$'+str(ls))
plt.grid()
plt.xlim([0,len(ARC[0])*2])
plt.legend(loc='upper right',ncol=2,labelspacing=0.05,handlelength=1.,columnspacing=1.)
plt.ylim([0.89,0.92])
plt.yticks(np.arange(0.89, 0.92, 0.01))
plt.xlabel('Time-step $(t)$')
plt.ylabel('Long-term Coverage')
plt.tight_layout()
plt.savefig('Images/LongRunElec_kappa_'+str(kappa)+'_ls_'+str(ls)+'.pdf')
plt.show()



marginal=[]
weekend=[]
week=[]
with open('Logs/ARC.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
marginal.append(np.mean(loaded_dict['err']))
week.append(np.mean(loaded_dict['err_week']))
weekend.append(np.mean(loaded_dict['err_weekend']))

ls=2.
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
marginal.append(np.mean(loaded_dict['err']))
week.append(np.mean(loaded_dict['err_week']))
weekend.append(np.mean(loaded_dict['err_weekend']))

ls=1.
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
marginal.append(np.mean(loaded_dict['err']))
week.append(np.mean(loaded_dict['err_week']))
weekend.append(np.mean(loaded_dict['err_weekend']))

ls=0.5
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
marginal.append(np.mean(loaded_dict['err']))
week.append(np.mean(loaded_dict['err_week']))
weekend.append(np.mean(loaded_dict['err_weekend']))

species = (r"ARC", r" $l=$"+str(2), r"$l=$"+str(1),r"$l=$"+str(0.5))
penguin_means = {
    'Marginal': np.asarray(marginal),
    'Mon-Fri': np.asarray(week),
    'Sat-Sun': np.asarray(weekend),
}

x = np.arange(len(species))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute,alpha=0.9)
    multiplier += 1
plt.axhline(y = 0.1, color = 'tab:grey', linestyle = '-.')
plt.annotate(r'$\alpha$',xy=(3.2,0.1),xytext=(2.8,0.13),arrowprops=dict(arrowstyle='->'), fontsize=22)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Miscoverage Error')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left')
ax.set_ylim(0, 0.25)
ax.set_axisbelow(True)
plt.grid(zorder=-10)

plt.tight_layout()
plt.savefig('Images/ConditionalElec_kappa_'+str(kappa)+'_ls_'+str(ls)+'.pdf')
plt.show()
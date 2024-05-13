import matplotlib.pyplot as plt
import numpy as np
import pickle
plt.rc('font', family='serif', serif='Computer Modern Roman', size=14)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.rcParams['figure.figsize'] = [5, 3.75]
pos_BS = np.asarray([0, -32])
K_ACI_Log,ACI_Log,MondrianACI_Log=[],[],[]
MC_reps=20

kappa=1.
la= 1e-4
with open('Logs/ARC.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
long_run=[x[0][0] for x in loaded_dict]
plt.plot(np.arange(len(long_run[0]))*2,np.cumsum(long_run[0])/np.arange(1,len(long_run[0])+1),color='tab:grey',label='ARC')
ls=2.
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
long_run=[x[0][0] for x in loaded_dict]
plt.plot(np.arange(len(long_run[0]))*2,np.cumsum(long_run[0])/np.arange(1,len(long_run[0])+1),color='tab:green',label=r'L-ARC $l$='+str(ls))
ls=1.
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
long_run=[x[0][0] for x in loaded_dict]
plt.plot(np.arange(len(long_run[0]))*2,np.cumsum(long_run[0])/np.arange(1,len(long_run[0])+1),color='tab:blue',label=r'L-ARC $l$='+str(ls))
ls=0.5
kappa=1.
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
long_run=[x[0][0] for x in loaded_dict]
plt.plot(np.arange(len(long_run[0]))*2,np.cumsum(long_run[0])/np.arange(1,len(long_run[0])+1),color='tab:orange',label=r'L-ARC $l$='+str(ls))
plt.grid()
plt.xlim([0,len(long_run[0])])
plt.legend(loc='upper right',ncol=2,labelspacing=0.05,handlelength=1.,columnspacing=1.)
plt.ylim([0.05,0.125])
plt.yticks(np.arange(0.05,0.15, 0.025))
plt.xlabel('Time-step $(t)$')
plt.ylabel('Long-term FNR')
plt.tight_layout()
plt.savefig('Images/LongRunPolyps.pdf')
plt.show()



means=[]
stds=[]
marginal_std=[]
with open('Logs/ARC.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
means.append(np.mean([x[1] for x in loaded_dict],axis=0))
stds.append(np.std([x[1] for x in loaded_dict],axis=0)/np.sqrt(MC_reps))
marginal_std.append(np.std(np.mean([x[2] for x in loaded_dict],axis=1))/np.sqrt(MC_reps))

kappa=1.
ls=2.
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
means.append(np.mean([x[1] for x in loaded_dict],axis=0))
stds.append(np.std([x[1] for x in loaded_dict],axis=0)/np.sqrt(MC_reps))
marginal_std.append(np.std(np.mean([x[2] for x in loaded_dict],axis=1))/np.sqrt(MC_reps))

ls=1.
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
means.append(np.mean([x[1] for x in loaded_dict],axis=0))
stds.append(np.std([x[1] for x in loaded_dict],axis=0)/np.sqrt(MC_reps))
marginal_std.append(np.std(np.mean([x[2] for x in loaded_dict],axis=1))/np.sqrt(MC_reps))

ls=0.5
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
means.append(np.mean([x[1] for x in loaded_dict],axis=0))
stds.append(np.std([x[1] for x in loaded_dict],axis=0)/np.sqrt(MC_reps))
marginal_std.append(np.std(np.mean([x[2] for x in loaded_dict],axis=1))/np.sqrt(MC_reps))

means=np.asarray(means)
stds=np.asarray(stds)
species = ("ARC", r" $l=$"+str(2), r"$l=$"+str(1),r"$l=$"+str(0.5))
data_means = {
    'Marginal': np.mean(means,axis=1),
    'Kvasir': means[:,0],
    'CVC-300': means[:,1],
    'CVC-ClinicDB': means[:,2],
    'CVC-ColonDB': means[:,3],
    'ETIS-LaribPolypDB': means[:,4],
}

data_stds = {
    'Marginal': np.asarray(marginal_std),
    'Kvasir': stds[:,0],
    'CVC-300': stds[:,1],
    'CVC-ClinicDB': stds[:,2],
    'CVC-ColonDB': stds[:,3],
    'ETIS-LaribPolypDB': stds[:,4],
}


x = np.arange(len(species))  # the label locations
width = 0.1  # the width of the bars
multiplier =-1

fig, ax = plt.subplots(layout='constrained')
for m,s in zip(data_means.items(),data_stds.items()):
    attribute=m[0]
    measurement=m[1]
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute,alpha=0.9)
    ax.errorbar(x + offset, measurement,2*s[1],linestyle='none')
    multiplier += 1
plt.axhline(y = 0.1, color = 'tab:grey', linestyle = '-.')
plt.annotate(r'$\alpha$',xy=(2.6,0.1),xytext=(2.25,0.22),arrowprops=dict(arrowstyle='->'), fontsize=20)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average FNR Error')
ax.set_xticks(x + width, species)
ax.legend(loc='upper right',ncol=2,labelspacing=0.05,handlelength=1.,columnspacing=1.)
plt.xlabel('')
ax.set_ylim(0, 0.5)
ax.set_axisbelow(True)
plt.grid(zorder=-10)
plt.tight_layout()
plt.savefig('Images/ConditionalPolyps.pdf')
plt.show()





means=[]
stds=[]
marginal_std=[]
with open('Logs/ARC.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
means.append(np.mean([x[2] for x in loaded_dict],axis=0))
stds.append(np.std([x[2] for x in loaded_dict],axis=0)/np.sqrt(MC_reps))
marginal_std.append(np.std(np.mean([x[2] for x in loaded_dict],axis=1))/np.sqrt(MC_reps))



kappa=1.
ls=2.
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
means.append(np.mean([x[2] for x in loaded_dict],axis=0))
stds.append(np.std([x[2] for x in loaded_dict],axis=0)/np.sqrt(MC_reps))
marginal_std.append(np.std(np.mean([x[2] for x in loaded_dict],axis=1))/np.sqrt(MC_reps))

ls=1.
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
means.append(np.mean([x[2] for x in loaded_dict],axis=0))
stds.append(np.std([x[2] for x in loaded_dict],axis=0)/np.sqrt(MC_reps))
marginal_std.append(np.std(np.mean([x[2] for x in loaded_dict],axis=1))/np.sqrt(MC_reps))

ls=0.5
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
means.append(np.mean([x[2] for x in loaded_dict],axis=0))
stds.append(np.std([x[2] for x in loaded_dict],axis=0)/np.sqrt(MC_reps))
marginal_std.append(np.std(np.mean([x[2] for x in loaded_dict],axis=1))/np.sqrt(MC_reps))

means=np.asarray(means)
stds=np.asarray(stds)
lbls = ("ARC", r" $l=$"+str(2), r"$l=$"+str(1),r"$l=$"+str(0.5))
data_means = {
    'Marginal': np.mean(means,axis=1),
    'Kvasir': means[:,0],
    'CVC-300': means[:,1],
    'CVC-ClinicDB': means[:,2],
    'CVC-ColonDB': means[:,3],
    'ETIS-LaribPolypDB': means[:,4],
}

data_stds = {
    'Marginal': np.asarray(marginal_std),
    'Kvasir': stds[:,0],
    'CVC-300': stds[:,1],
    'CVC-ClinicDB': stds[:,2],
    'CVC-ColonDB': stds[:,3],
    'ETIS-LaribPolypDB': stds[:,4],
}

x = np.arange(len(lbls))  # the label locations
width = 0.1  # the width of the bars
multiplier =-1

fig, ax = plt.subplots(layout='constrained')
for m,s in zip(data_means.items(),data_stds.items()):
    attribute=m[0]
    measurement=m[1]
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute,alpha=0.9)
    ax.errorbar(x + offset, measurement,2*s[1],linestyle='none')
    multiplier += 1
# Add some text for labels, title and custom x-axis tick labels, etc.x
ax.set_ylabel('Average Mask Size')
ax.set_xticks(x + width, lbls)
ax.legend(loc='upper right',ncol=2,labelspacing=0.05,handlelength=1.,columnspacing=1.)
ax.set_ylim(0, 0.5)
ax.set_axisbelow(True)
plt.grid(zorder=-10)
plt.tight_layout()
plt.savefig('Images/SizePolyps.pdf')
plt.show()
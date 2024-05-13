import matplotlib.pyplot as plt
import numpy as np
import pickle
plt.rc('font', family='serif', serif='Computer Modern Roman', size=16)
plt.rcParams.update({
    'font.size': 16,
    'text.usetex': True,
})
plt.rcParams['figure.figsize'] = [5.5, 3.75]
pos_BS = np.asarray([0, -32])
K_ACI_Log,ACI_Log,MondrianACI_Log=[],[],[]
MC_reps=20
ls=0.5
la= 1e-4
target_FNR=0.25
with open('Logs/ARC.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
long_run=[x[0][0] for x in loaded_dict]
plt.plot(np.arange(len(long_run[0]))*2,1-np.cumsum(long_run[0])/np.arange(1,len(long_run[0])+1),color='tab:grey',label='ARC  $\kappa$=0')
kappa=1
ls=1
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
long_run=[x[0][0] for x in loaded_dict]
plt.plot(np.arange(len(long_run[0]))*2,1-np.cumsum(long_run[0])/np.arange(1,len(long_run[0])+1),color='tab:green',label=r'L-ARC $l$='+str(ls))
ls=0.5
kappa=1
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
long_run=[x[0][0] for x in loaded_dict]
plt.plot(np.arange(len(long_run[0]))*2,1-np.cumsum(long_run[0])/np.arange(1,len(long_run[0])+1),color='tab:blue',label=r'L-ARC $l$='+str(ls))
ls=0.25
kappa=1
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
long_run=[x[0][0] for x in loaded_dict]
plt.plot(np.arange(len(long_run[0]))*2,1-np.cumsum(long_run[0])/np.arange(1,len(long_run[0])+1),color='tab:orange',label=r'L-ARC $l$='+str(ls))
plt.grid()
plt.xlim([0,len(long_run[0])])
plt.legend(loc='upper right',ncol=2,labelspacing=0.05,handlelength=1.,columnspacing=1.)
plt.ylim([0.72,0.77])
plt.yticks(np.arange(0.72,0.77, 0.02))
plt.xlabel('Time-step $(t)$')
plt.ylabel('Long-term Coverage')
plt.tight_layout()
plt.savefig('Images/LongRun_Image.pdf')
plt.show()
plt.rcParams['figure.figsize'] = [11, 3.75]
fig, axes = plt.subplots(nrows=1, ncols=2, )
means=[]
stds=[]
week=[]
ths=np.linspace(0,1.,21)
with open('Logs/ARC.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
dom_te=np.mean([x[1] for x in loaded_dict],axis=0)
dom_std=np.std([x[1] for x in loaded_dict],axis=0)/np.sqrt(MC_reps)
size_te=np.mean([x[2] for x in loaded_dict],axis=0)
size_te_std=np.std([x[2] for x in loaded_dict],axis=0)/np.sqrt(MC_reps)
axes[0].plot(ths[:-1], dom_te, color='tab:gray', label='ARC', marker='x')
axes[0].fill_between(ths[:-1], np.asarray(dom_te) - 2 * np.asarray(dom_std), np.asarray(dom_te) + 2 * np.asarray(dom_std), alpha=0.5, color='tab:gray')
axes[1].plot(ths[:-1], size_te, color='tab:gray', label='ARC', marker='x')
axes[1].fill_between(ths[:-1], np.asarray(size_te) - 2 * np.asarray(size_te_std), np.asarray(size_te) + 2 * np.asarray(size_te_std), alpha=0.5, color='tab:gray')


kappa=1
ls=1
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
dom_te=np.mean([x[1] for x in loaded_dict],axis=0)
dom_std=np.std([x[1] for x in loaded_dict],axis=0)/np.sqrt(MC_reps)
size_te=np.mean([x[2] for x in loaded_dict],axis=0)
size_te_std=np.std([x[2] for x in loaded_dict],axis=0)/np.sqrt(MC_reps)
axes[0].plot(ths[:-1], dom_te, color='tab:green', label='L-ARC $l$='+str(ls), marker='o')
axes[0].fill_between(ths[:-1], np.asarray(dom_te) - 2 * np.asarray(dom_std), np.asarray(dom_te) + 2 * np.asarray(dom_std), alpha=0.5, color='tab:green')
axes[1].plot(ths[:-1], size_te, color='tab:green', label='L-ARC $l$='+str(ls), marker='o')
axes[1].fill_between(ths[:-1], np.asarray(size_te) - 2 * np.asarray(size_te_std), np.asarray(size_te) + 2 * np.asarray(size_te_std), alpha=0.5, color='tab:green')


ls=0.5
kappa=1
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
dom_te=np.mean([x[1] for x in loaded_dict],axis=0)
dom_std=np.std([x[1] for x in loaded_dict],axis=0)/np.sqrt(MC_reps)
size_te=np.mean([x[2] for x in loaded_dict],axis=0)
size_te_std=np.std([x[2] for x in loaded_dict],axis=0)/np.sqrt(MC_reps)
axes[0].plot(ths[:-1], dom_te, color='tab:blue', label='L-ARC $l$='+str(ls), marker='^')
axes[0].fill_between(ths[:-1], np.asarray(dom_te) - 2 * np.asarray(dom_std), np.asarray(dom_te) + 2 * np.asarray(dom_std), alpha=0.5, color='tab:blue')
axes[1].plot(ths[:-1], size_te, color='tab:blue', label='L-ARC $l$='+str(ls), marker='^')
axes[1].fill_between(ths[:-1], np.asarray(size_te) - 2 * np.asarray(size_te_std), np.asarray(size_te) + 2 * np.asarray(size_te_std), alpha=0.5, color='tab:blue')


ls=0.25
kappa=1
with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
dom_te=np.mean([x[1] for x in loaded_dict],axis=0)
dom_std=np.std([x[1] for x in loaded_dict],axis=0)/np.sqrt(MC_reps)
size_te=np.mean([x[2] for x in loaded_dict],axis=0)
size_te_std=np.std([x[2] for x in loaded_dict],axis=0)/np.sqrt(MC_reps)
axes[0].plot(ths[:-1], dom_te, color='tab:orange', label='L-ARC $l$='+str(ls), marker='+')
axes[0].fill_between(ths[:-1], np.asarray(dom_te) - 2 * np.asarray(dom_std), np.asarray(dom_te) + 2 * np.asarray(dom_std), alpha=0.5, color='tab:orange')
axes[1].plot(ths[:-1], size_te, color='tab:orange', label='L-ARC $l$='+str(ls), marker='+')
axes[1].fill_between(ths[:-1], np.asarray(size_te) - 2 * np.asarray(size_te_std), np.asarray(size_te) + 2 * np.asarray(size_te_std), alpha=0.5, color='tab:orange')
axes[0].hlines(1 - target_FNR, 0, 1, color='k')
axes[0].text(0.4,0.76,r'Target Coverage Level')
axes[0].set_xlabel(r"Model's Confidence ($\mathrm{Conf}(X)$)")
axes[1].set_xlabel(r"Model's Confidence ($\mathrm{Conf}(X)$)")
axes[0].set_ylabel('Coverage Rate')
axes[0].set_xlim([0, 1])
axes[0].grid()
axes[0].legend()
axes[0].set_xlim([0, 0.9])
axes[1].set_xlim([0, 0.9])
axes[1].set_ylim([0, 16])
axes[1].legend()
axes[1].grid()
axes[1].set_ylabel('Set Size')
plt.tight_layout()
plt.savefig('Images/ConditionalCoverage_Image.pdf')
plt.show()



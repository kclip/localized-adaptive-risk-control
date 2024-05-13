import matplotlib.pyplot as plt
import numpy as np
import pickle
plt.rc('font', family='serif', serif='Computer Modern Roman', size=12)
plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
})
plt.rcParams['figure.figsize'] = [3.5, 3.5]
fig, axes = plt.subplots(nrows=2, ncols=1,)
pos_BS = np.asarray([0, -32])
LARC_Log,ARC_Log,MondrianARC_Log=[],[],[]
MC_reps=30
kappa=1.
ls=10.
for seed in np.arange(0,1):
    id=2
    with open('Results/ResultsSEED'+str(seed)+'kmax'+str(kappa)+'_ls_'+str(ls)+'.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    ALPHAS=loaded_dict['alphas']
    ARC=loaded_dict['ARC']
    MondrianARC=loaded_dict['MondrianARC']
    LARC=loaded_dict['LARC']
    y_hat_te=loaded_dict['y_hat_te']
    X_te=loaded_dict['X_te']
    y_te=loaded_dict['y_te']
    h_eff_te=loaded_dict['h_eff_te']
    length_scale=loaded_dict['length_scale']
    epsilon=loaded_dict['kappa']
    X_CP=loaded_dict['X_CP']
    MondrianARC_LRisk=np.zeros(len(X_CP))
    d=50
    id_1 = np.where(np.linalg.norm(np.abs(X_CP - pos_BS[np.newaxis, :]), axis=1) <= d)[0]
    id_2 = np.where(np.linalg.norm(np.abs(X_CP - pos_BS[np.newaxis, :]), axis=1) > d)[0]
    MondrianARC_LRisk[id_1]=MondrianARC[id][0]
    MondrianARC_LRisk[id_2] = MondrianARC[id][2]
    axes[0].plot(np.cumsum(np.asarray(ARC[id][0]))/np.arange(1,len(ARC[id][0])+1),label='ARC')
    axes[0].plot(np.cumsum(np.asarray(LARC[id][0]))/np.arange(1,len(LARC[id][0])+1),label='L-ARC')
    axes[0].plot(np.cumsum(np.asarray(MondrianARC_LRisk )) / np.arange(1, len(MondrianARC_LRisk) + 1), label='Mondr. ARC')
    axes[0].grid()
    axes[0].set_xlim(0,len(ARC[id][0]))
    axes[0].set_ylim([0.09 ,0.1025])
    axes[0].legend(loc='lower right')
    axes[0].set_xlabel(r'Time step ($t$)')
    axes[0].set_ylabel('Long-term Risk')

for seed in np.arange(0,MC_reps):
    with open('Logs/Logs_seed_'+str(seed)+'kmax'+str(kappa)+'_ls_'+str(ls)+'.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    LARC_Log.append(loaded_dict["LARC_Log"])
    ARC_Log.append(loaded_dict["ARC_Log"])
    MondrianARC_Log.append(loaded_dict["MondrianARC_Log"])
    ALPHAS=loaded_dict["ALPHAS"]
long_run_ARC=np.mean([[k[0] for k in LOG_seed ]for LOG_seed in ARC_Log],axis=0)
long_run_KARC=np.mean([[k[0] for k in LOG_seed ]for LOG_seed in LARC_Log],axis=0)
long_run_Mon1=np.mean([[k[0] for k in LOG_seed ]for LOG_seed in MondrianARC_Log],axis=0)
long_run_Mon2=np.mean([[k[4] for k in LOG_seed ]for LOG_seed in MondrianARC_Log],axis=0)
size_ARC=np.mean([[k[2] for k in LOG_seed ]for LOG_seed in ARC_Log],axis=0)
size_MON_1=np.mean([[k[2] for k in LOG_seed ]for LOG_seed in MondrianARC_Log],axis=0)
size_MON_2=np.mean([[k[6] for k in LOG_seed ]for LOG_seed in MondrianARC_Log],axis=0)
size_MON=np.concatenate((size_MON_2,size_MON_1),axis=1)
size_KARC=np.mean([[k[2] for k in LOG_seed ]for LOG_seed in LARC_Log],axis=0)
risLARC=np.mean([[k[1] for k in LOG_seed ]for LOG_seed in ARC_Log],axis=0)
risk_MON_1=np.mean([[k[1] for k in LOG_seed ]for LOG_seed in MondrianARC_Log],axis=0)
risk_MON_2=np.mean([[k[5] for k in LOG_seed ]for LOG_seed in MondrianARC_Log],axis=0)
risk_MON=np.concatenate((risk_MON_1,risk_MON_2),axis=1)
risk_KARC=np.mean([[k[1] for k in LOG_seed ]for LOG_seed in LARC_Log],axis=0)

axes[1].plot(ALPHAS,np.mean(size_ARC,axis=1),label='ARC',marker='o')
axes[1].fill_between(ALPHAS, np.mean(size_ARC,axis=1)-np.std(size_ARC,axis=1)/np.sqrt(MC_reps), np.mean(size_ARC,axis=1)+np.std(size_ARC,axis=1)/np.sqrt(MC_reps),alpha=0.25)
axes[1].plot(ALPHAS,np.mean(size_KARC,axis=1),label='L-ARC',marker='^')
axes[1].fill_between(ALPHAS, np.mean(size_KARC,axis=1)-np.std(size_KARC,axis=1)/np.sqrt(MC_reps), np.mean(size_KARC,axis=1)+np.std(size_KARC,axis=1)/np.sqrt(MC_reps),alpha=0.25)
axes[1].plot(ALPHAS,np.mean(size_MON,axis=1),label='Mondr. ARC',marker='X')
axes[1].fill_between(ALPHAS, np.mean(size_MON,axis=1)-np.std(size_MON,axis=1)/np.sqrt(MC_reps), np.mean(size_MON,axis=1)+np.std(size_MON,axis=1)/np.sqrt(MC_reps),alpha=0.25)

axes[1].grid()
axes[1].set_xlabel(r'Target Risk $(L^*)$')
axes[1].set_xlim([np.min(ALPHAS),0.4])
axes[1].set_ylim([0.75,2])
axes[1].set_ylabel(r'Avg. Beam Set Size')
plt.tight_layout()
plt.subplots_adjust(hspace=0.4,bottom=0.15,left=0.195,right=0.9,top=0.975)

plt.savefig('Images/BeamSweeping_ls_'+str(ls)+'.pdf')
plt.close()

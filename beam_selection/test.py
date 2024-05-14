import numpy as np
import pickle

n_tx=8
pos_BS = np.asarray([0, -32])
MC_reps=1
kappa=1.
for ls in [10.,25.,50.]:
    for seed in np.arange(0,MC_reps):
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
        kappa=loaded_dict['kappa']
        X_CP=loaded_dict['X_CP']
        id_te=0
        ARC_Log=[]
        MondrianARC_Log=[]
        L_ARC_Log=[]
        for alpha in ALPHAS:
            print(alpha)
            q_ARC=ARC[id_te][1]
            '''     TESTING     '''
            cov_te_cond=[]
            SNR_te=[]
            for i in range(0,y_hat_te.shape[0]):
                if len(np.where(y_hat_te[i,:]>q_ARC[-1])[0])>0:
                    pred_set = np.where(y_hat_te[i, :] > q_ARC[-1])[0]
                    cov_te_cond.append(-(np.max(y_te[i,pred_set])-np.max(y_te[i]))/np.max(y_te[i]))
                    SNR_te.append(np.max(h_eff_te[i, pred_set]))
                else:
                    cov_te_cond.append(1)
                    SNR_te.append(0)
            cov_te_cond=1-np.asarray(cov_te_cond)
            size_te_ARC=np.asarray([np.sum(y_hat_te[i,:]>q_ARC[-1]) for i in range(0,y_hat_te.shape[0])])
            ARC_Log.append([np.mean(ARC[id_te][0]),cov_te_cond,size_te_ARC,SNR_te])
            print('ARC Convergence Risk: '+str(round(np.mean(cov_te_cond),3))+'Avg. Set Size: '+str(round(np.mean(size_te_ARC),3)))
            d=50
            q_ARC_1 = MondrianARC[id_te][1]
            q_ARC_2 = MondrianARC[id_te][3]
            id_1_te=np.where(np.linalg.norm(np.abs(X_te-pos_BS[np.newaxis,:]),axis=1)<=d)[0]
            id_2_te=np.where(np.linalg.norm(np.abs(X_te-pos_BS[np.newaxis,:]),axis=1)>d)[0]
            cov_te_cond_1=[]
            SNR_te_1=[]
            for i in id_1_te:
                if len(np.where(y_hat_te[i,:]>q_ARC_1[-1])[0])>0:
                    pred_set=np.where(y_hat_te[i,:]>q_ARC_1[-1])[0]
                    cov_te_cond_1.append(-(np.max(y_te[i,pred_set])-np.max(y_te[i]))/np.max(y_te[i]))
                    SNR_te_1.append(np.max(h_eff_te[i,pred_set]))
                else:
                    cov_te_cond_1.append(1)
                    SNR_te_1.append(0)
            cov_te_cond_1=1-np.asarray(cov_te_cond_1)
            cov_te_cond_2=[]
            SNR_te_2=[]
            for i in id_2_te:
                if len(np.where(y_hat_te[i,:]>q_ARC_2[-1])[0])>0:
                    pred_set = np.where(y_hat_te[i, :] > q_ARC_2[-1])[0]
                    cov_te_cond_2.append(-(np.max(y_te[i,pred_set])-np.max(y_te[i]))/np.max(y_te[i]))
                    SNR_te_2.append(np.max(h_eff_te[i, pred_set]))
                else:
                    cov_te_cond_2.append(1)
                    SNR_te_2.append(0)
            cov_te_cond_2=1-np.asarray(cov_te_cond_2)
            size_te_ARC_1=np.asarray([np.sum(y_hat_te[i,:]>q_ARC_1[-1]) for i in id_1_te])
            size_te_ARC_2=np.asarray([np.sum(y_hat_te[i,:]>q_ARC_2[-1]) for i in id_2_te])
            MondrianARC_Log.append([np.mean(MondrianARC[id_te][0]), cov_te_cond_1, size_te_ARC_1,SNR_te_1,np.mean(MondrianARC[id_te][2]), cov_te_cond_2, size_te_ARC_2,SNR_te_2])
            print('Mondrian ARC Convergence Risk: ' + str(round(np.mean(np.hstack((cov_te_cond_1,cov_te_cond_2))), 3)) + 'Avg. Set Size: ' + str(round(np.mean(np.hstack((size_te_ARC_1,size_te_ARC_2))), 3)))
            cov_te_cond_LARC=[]
            size_te_LARC=[]
            SNR_te_LARC=[]
            q_LARC=LARC[id_te][3]   # for last iterate use LARC[id_te][1]
            c_LARC=LARC[id_te][4]   # for last iterate use LARC[id_te][2]
            for i in range(0, y_hat_te.shape[0]):
                kernels =  kappa*np.exp(-np.linalg.norm((X_te[i, :] - X_CP), ord=2, axis=1) ** 2 / (length_scale ** 2))
                q =np.sum(kernels * q_LARC)+c_LARC
                if len(np.where(y_hat_te[i,:]>q)[0])>0:
                    pred_set =np.where(y_hat_te[i,:]>q)[0]
                    loss_cond=-(np.max(y_te[i,pred_set])-np.max(y_te[i]))/np.max(y_te[i])
                    SNR_te_LARC.append(np.max(h_eff_te[i, pred_set]))
                else:
                    loss_cond=1
                    SNR_te_LARC.append(0)
                size= np.sum(y_hat_te[i, :] > q)
                size_te_LARC.append(size)
                cov_te_cond_LARC.append(loss_cond)
            cov_te_cond_LARC=1-np.asarray(cov_te_cond_LARC)
            print('L-ARC Convergence Risk'+str(round(np.mean(cov_te_cond_LARC),3))+'Avg. Set Size: '+str(round(np.mean(size_te_LARC),3)))
            L_ARC_Log.append([np.mean(LARC[id_te][0]), cov_te_cond_LARC, size_te_LARC,SNR_te_LARC])
            id_te=id_te+1
        dictionary = {
                "LARC_Log": L_ARC_Log,
                "ARC_Log":ARC_Log,
                "MondrianARC_Log":  MondrianARC_Log,
                "ALPHAS": ALPHAS
                }
        with open('Logs/Logs_seed_'+str(seed)+'kmax'+str(kappa)+'_ls_'+str(ls)+'.pkl', 'wb') as f:
            pickle.dump(dictionary, f)


import numpy as np
import matplotlib.pyplot as plt
import picklet
from scipy.special import softmax



def ARC_calibration(X,Y_HAT,Y,eta_init,miscoverage_target,decay):
    q=0
    errs,qs,errs_val=[],[],[]
    qs.append(q)
    for t in range(0,X.shape[0]):
        eta_t = eta_init / (t+1)**(decay)
        err=1-np.sum((np.where(Y_HAT[t,:]>q)[0]-Y[t])==0)
        q = q + eta_t * (miscoverage_target- err)
        errs.append(err)
        qs.append(q)
    print('Long-Run Risk ARC = '+ str(round(np.mean(errs),3)))
    return errs,qs

def LARC_calibration(X,Y_HAT,Y,eta_init,miscoverage_target,reg_param,length_scale,kappa,decay,Mem_max=2**32-1):
    errs, coeff_list, cs = [], [], []
    coeff_mean = np.zeros(X.shape[0])
    q = 0
    c = 0
    for  t in range(0,len(X)):
        if t==0:
            eta_t = eta_init / (t + 1) ** (decay)
            err = 1 - np.sum((np.where(Y_HAT[t, :] > q)[0] - Y[t]) == 0)
            c= c + eta_t * (miscoverage_target- err)
            errs.append(err)
            coeff_t = eta_t * (miscoverage_target- err)
            cs.append(c)
            coeff_mean[0] = coeff_t
        else:
            eta_t = eta_init / (t+1)**(decay)
            if t >Mem_max:
                kernels = np.exp(-(X[t] - X[t-Mem_max:t]) ** 2 / (length_scale ** 2))
                coeff_t=coeff_t[-Mem_max:]
            else:
                kernels=np.exp(-(X[t] - X[t-Mem_max:t])**2/(length_scale**2))
            q=kappa*float(np.sum(kernels*coeff_t))+c
            err = 1 - np.sum((np.where(Y_HAT[t, :] > q)[0] - Y[t]) == 0)
            c = c + eta_t * (miscoverage_target - err)
            cs.append(c)
            errs.append(err)
            coeff_t = np.asarray(coeff_t) * (1 - eta_t * reg_param)
            coeff_t = np.append(coeff_t, eta_t * (miscoverage_target - err))
            coeff_mean[:t + 1] = coeff_mean[:t + 1] + coeff_t
    print('Long-Run Risk L-ARC = '+ str(round(np.mean(errs),3)))
    return errs,coeff_t,c,coeff_mean/X.shape[0],np.mean(cs)

'''Loading Data'''
data=np.load('data/logits.npy')
logits=np.reshape(data,(-1,131))
logits_sfmx = softmax(logits, axis=1)
X=np.max(logits_sfmx,axis=1)
Y=np.reshape(np.load('data/labels.npy'),(-1))


Reps=10
'''Algorithm Parameters'''
eta_init = 1.
la = 1e-4  # Regularization Parameter
decay = 1. / 2.
target_miscoverage = 0.25
kappa=1
for ls  in [1,0.5,0.25]:
    LOG_ARC = []
    LOG_LARC = []
    np.random.seed(0)
    for rep in range(0,Reps):
        '''Data Split'''
        id_tr = np.random.choice(X.shape[0],8000,replace=False)
        id_te=np.setdiff1d(np.arange(0,X.shape[0]),id_tr)
        np.random.shuffle(id_tr)
        X_tr=X[id_tr]
        Y_tr=Y[id_tr]
        Y_hat_tr=logits_sfmx[id_tr]
        X_te = X[id_te]
        Y_te = Y[id_te]
        Y_hat_te = logits_sfmx[id_te]
        '''ARC'''
        ARC = ARC_calibration(X_tr,Y_hat_tr, Y_tr,eta_init,target_miscoverage,decay)
        L_ARC = LARC_calibration(X=X_tr, Y_HAT=Y_hat_tr, Y=Y_tr, eta_init=eta_init, miscoverage_target=target_miscoverage, reg_param=la, length_scale=ls, kappa=kappa, decay=decay)
        dom_te,dom_std,size_te,size_te_std=[],[],[],[]
        ths=np.linspace(0,1.,21)
        i = 0
        for start in ths[:-1]:
            end=ths[i+1]
            ids=np.where((X_te>start) & (X_te<end))[0]
            fnr_te,size = [],[]
            for y,y_gt in zip (Y_hat_te[ids],Y_te[ids]):
                fnr_te.append(np.sum((np.where(y >  ARC[1][-1])[0] - y_gt) == 0))
                size.append(np.sum(y > ARC[1][-1]))
            i=i+1
            dom_te.append(np.mean(fnr_te))
            dom_std.append(np.std(fnr_te)/np.sqrt(len(fnr_te)))
            size_te.append(np.mean(size))
            size_te_std.append(np.std(size)/np.sqrt(len(size)))
        print('ARC Cond. Cov:'+str(dom_te)+'Marginal Cov:'+str(np.mean(dom_te)))
        LOG_ARC.append((ARC, dom_te, size_te))
        dom_te, dom_std, size_te, size_te_std = [], [], [], []
        i=0
        for start in ths[:-1]:
            end = ths[i + 1]
            ids = np.where((X_te > start) & (X_te < end))[0]
            fnr_te, size = [], []
            for x,y,y_gt in zip (X_te[ids],Y_hat_te[ids],Y_te[ids]):
                kernels = np.exp(-(x - X_tr) ** 2 / (ls ** 2))
                q = float(np.sum(kappa * kernels * L_ARC[3])) + L_ARC[4]  # for last iterate use float(np.sum(kappa * kernels * L_ARC[1])) + L_ARC[2]
                fnr_te.append(np.sum((np.where(y > q)[0] - y_gt) == 0))
                size.append(np.sum(y > q ))
            i = i + 1
            dom_te.append(np.mean(fnr_te))
            dom_std.append(np.std(fnr_te)/np.sqrt(len(fnr_te)))
            size_te.append(np.mean(size))
            size_te_std.append(np.std(size) / np.sqrt(len(size)))
        print('L-ARC Cond. Cov:' + str(dom_te) + 'Marginal Cov:' + str(np.mean(dom_te)))
        LOG_LARC.append((L_ARC, dom_te,size_te))
    with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'wb') as f:
        pickle.dump(LOG_LARC, f)
    with open('Logs/ARC.pkl', 'wb') as f:
        pickle.dump(LOG_ARC, f)

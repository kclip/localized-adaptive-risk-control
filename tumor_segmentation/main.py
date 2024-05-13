import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
import pickle
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA


def fnr(mask_pred, mask_gt):
    '''Computation of the False Negative Rate'''
    intersection=np.sum((mask_pred+mask_gt)>1.1)
    size_gt=np.sum((mask_gt)>0)
    return 1-intersection/size_gt

def ARC_calibration(X,Y_HAT,Y,eta_init,fnr_target,decay):
    '''Adaptive Risk Control'''
    q=0
    errs,qs,errs_val=[],[],[]
    qs.append(q)
    for t in range(0,X.shape[0]):
        eta_t = eta_init / (t+1)**(decay)
        fnr_t=fnr(Y_HAT[t]>q,Y[t])
        q = q + eta_t * (fnr_target- fnr_t)
        errs.append(fnr_t)
        qs.append(q)
    print('Long-Run Risk ARC = '+ str(round(np.mean(errs),3)))
    return errs,qs

def LARC_calibration(X,Y_HAT,Y,eta_init,fnr_target,reg_param,length_scale,kappa,decay,Mem_max=2**32-1):
    '''Localized Adaptive Risk Control'''
    errs, coeff_list, cs = [], [], []
    coeff_mean = np.zeros(X.shape[0])
    q = 0
    c = 0
    for  t in range(0,len(X)):
        if t==0:
            eta_t = eta_init / (t + 1) ** (decay)
            fnr_t=fnr(Y_HAT[t]>q,Y[t])
            c= c + eta_t * (fnr_target- fnr_t)
            errs.append(fnr_t)
            coeff_t = eta_t * (fnr_target- fnr_t)
            cs.append(c)
            coeff_mean[0] = coeff_t
        else:
            eta_t = eta_init / (t+1)**(decay)
            if t >Mem_max:
                kernels = kappa*np.exp(-np.linalg.norm(X[t] - X[t-Mem_max:t], ord=2, axis=1) ** 2 / (length_scale ** 2))
                coeff_t=coeff_t[-Mem_max:]
            else:
                kernels=kappa*np.exp(-np.linalg.norm(X[t]-X[:t],ord=2,axis=1)**2/(length_scale**2))
            q=float(np.sum(kernels*coeff_t))+c
            fnr_t=fnr(Y_HAT[t]>q,Y[t])
            c = c + eta_t * (fnr_target - fnr_t)
            cs.append(c)
            errs.append(fnr_t)
            coeff_t = np.asarray(coeff_t) * (1 - eta_t * reg_param)
            coeff_t = np.append(coeff_t, eta_t * (fnr_target - fnr_t))
            coeff_mean[:t + 1] = coeff_mean[:t + 1] + coeff_t
    print('Long-Run Risk L-ARC = '+ str(round(np.mean(errs),3)))
    return errs,coeff_t,c,coeff_mean/X.shape[0],np.mean(cs)

'''Loading Data'''
cache_path='./cache/'
Y_hat=np.load(cache_path + 'sigmoids.npy' )
img_names=np.load(cache_path + 'img_names.npy')
Y=np.load(cache_path + 'masks.npy')
features=np.load(cache_path + 'features.npy')

'''Extracting L-ARC Features'''
X = StandardScaler().fit_transform(features)
pca = PCA(n_components=5)
X = pca.fit_transform(X)
X_n=MinMaxScaler().fit_transform(X)

'''Getting indices of images in each data set'''
id_kvasir,id_CVC300,id_Clinic,id_Colon,id_Larib,Calibration=[],[],[],[],[],[]
i=0
for s in img_names:
    if 'Kvasir' in s:
        id_kvasir.append(i)
    elif 'CVC-300' in s:
        id_CVC300.append(i)
    elif 'Clinic' in s:
        id_Clinic.append(i)
    elif 'Colon' in s:
        id_Colon.append(i)
    elif 'Larib' in s:
        id_Larib.append(i)
    i=i+1

Reps=10  #Number of Monte-Carlo runs
'''Algorithm Parameters'''
eta_init = 1.
decay = 1. / 2.
target_FNR = 0.1  #Target reliability
kappa=1. # Kernel scale
la = 1e-4  # Regularization strength
for ls in [2.,1.,0.5]:  # Length-scales to test
    LOG_ARC,LOG_LARC = [],[]
    np.random.seed(0)
    for rep in range(0,Reps):
        '''Random Data Split'''
        id_kvasir_te = np.random.choice(id_kvasir,50,replace=False)
        id_CVC300_te = np.random.choice(id_CVC300,50,replace=False)
        id_Clinic_te = np.random.choice(id_Clinic,50,replace=False)
        id_Colon_te = np.random.choice(id_Colon,50,replace=False)
        id_Larib_te = np.random.choice(id_Larib,50,replace=False)
        id_te=np.asarray([*id_kvasir_te,*id_CVC300_te,*id_Clinic_te,*id_Colon_te,*id_Larib_te])
        id_tr=np.setdiff1d(np.arange(0,len(Y)),id_te)
        np.random.shuffle(id_tr)
        X_tr=X_n[id_tr]
        Y_tr=Y[id_tr]
        Y_hat_tr=Y_hat[id_tr]
        '''ARC'''
        ARC = ARC_calibration(X_tr,Y_hat_tr, Y_tr,eta_init,target_FNR,decay)
        id_tes=[id_kvasir_te,id_CVC300_te,id_Clinic_te,id_Colon_te,id_Larib_te]
        id_string=['Kvasir/','CVC300/','Clinic/','Colon/','Larib/']
        dom_te,size_te=[],[]
        for id,dataset_name in zip(id_tes,id_string):
            fnr_te,size = [],[]
            i=0
            for y,y_gt in zip (Y_hat[id],Y[id]):
                fnr_te.append(fnr(y > ARC[1][-1], y_gt))
                size.append(np.mean(y > ARC[1][-1]))
                i=i+1
            dom_te.append(np.mean(fnr_te))
            size_te.append(np.mean(size))
        print('Dataset Conditional : '+str(np.round(dom_te,3))+' Marginal : '+str(np.round(np.mean(dom_te),3))+' Avg. Mask Size : ' + str(np.round(np.mean(size_te),3)))
        LOG_ARC.append((ARC,dom_te,size_te))
        L_ARC = LARC_calibration(X=X_tr, Y_HAT=Y_hat_tr, Y=Y_tr, eta_init=eta_init, fnr_target=target_FNR, reg_param=la, length_scale=ls, kappa=kappa, decay=decay)
        dom_te,size_te=[],[]
        for id,dataset_name in zip(id_tes,id_string):
            fnr_te,size = [],[]
            i=0
            for x,y,y_gt in zip (X_n[id],Y_hat[id],Y[id]):
                kernels = np.exp(-np.linalg.norm(x - X_tr, ord=2, axis=1) ** 2 / (ls ** 2))
                q = float(np.sum(kappa * kernels * L_ARC[3])) + L_ARC[4]  # for last iterate use float(np.sum(kappa * kernels * L_ARC[1])) + L_ARC[2]
                i = i + 1
                fnr_te.append(fnr(y > q, y_gt))
                size.append(np.mean(y > q))
            dom_te.append(np.mean(fnr_te))
            size_te.append(np.mean(size))
        print('Dataset Conditional : ' + str(np.round(dom_te,3)) + ' Marginal : ' + str(np.round(np.mean(dom_te),3)) + ' Avg. Mask Size : ' + str(np.round(np.mean(size_te),3)))
        LOG_LARC.append((L_ARC, dom_te,size_te))
    with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'wb') as f:
        pickle.dump(LOG_LARC, f)
    with open('Logs/ARC.pkl', 'wb') as f:
        pickle.dump(LOG_ARC, f)
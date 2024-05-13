from frouros.datasets.real import Elec2
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

def smooth_array(arr, window_size):
    smoothed = np.convolve(arr, np.ones(window_size) / window_size, mode='same')
    return smoothed

def ARC_calibration(X,Y_HAT,Y,eta_init,alpha_target,decay):
    '''Adapative Risk Control'''
    q=0 #Initial Threshold
    errs,qs,errs_val=[],[],[]
    qs.append(q)
    for t in range(0,X.shape[0]):
        eta_t = eta_init / (t+1)**(decay)
        score_t= np.abs(Y[t] - Y_HAT[t])
        err_t=(score_t>q).astype(int)
        q = q - eta_t * (alpha_target- err_t)
        errs.append(err_t)
        qs.append(q)
    print('ARC Long-Term Risk = '+ str(round(np.mean(errs),3)))
    return errs,qs

def LARC_calibration(X,Y_HAT,Y,eta_init,alpha_target,reg_param,length_scale,kappa,decay):
    '''Localized Adaptive Risk Control'''
    errs, cs,te_subpop = [], [], []
    coeff_mean=np.zeros(X.shape[0])
    q,c = 0,0 #Initial Threshold
    cs.append(c)
    for  t in range(0,len(X)):
        if t==0:
            eta_t = eta_init / (t + 1) ** (decay)
            score_t = np.abs(Y[t] - Y_HAT[t])
            err_t = (score_t > q).astype(int)
            c= c - eta_t * (alpha_target- err_t)
            cs.append(c)
            errs.append(err_t)
            coeff_t = - eta_t * (alpha_target - err_t)
            coeff_mean[0]=coeff_t
        else:
            eta_t = eta_init / (t+1)**(decay)
            kernels=np.exp(-np.linalg.norm(X[t]-X[:t],ord=2,axis=1)**2/(length_scale**2))
            q=kappa*float(np.sum(kernels*coeff_t))+c
            score_t = np.abs(Y[t] - Y_HAT[t])
            err_t = (score_t > q).astype(int)
            c = c - eta_t * (alpha_target - err_t)
            cs.append(c)
            errs.append(err_t)
            coeff_t = np.asarray(coeff_t) * (1 - eta_t * reg_param)
            coeff_t = np.append(coeff_t, -eta_t * (alpha_target - err_t))
            coeff_mean[:t+1] = coeff_mean[:t+1]+coeff_t
    print('L-ARC Long-Run Risk = '+ str(round(np.mean(errs),3)))
    return errs,coeff_t,c,coeff_mean/X.shape[0],np.mean(cs)

'''Loading Data'''
elec2 = Elec2()
elec2.download()
data = elec2.load()
Y = data[["nswdemand"]]
day = data[["day"]].astype(int)
'''Getting Weekday and Weekend indices'''
weekend_id=np.where(day[::2]>5)
week_id=np.where(day[::2]<=5)
Y = np.asarray([y.astype(float) for y in Y])
'''Predictions'''
Yhat = smooth_array(np.roll(Y,-48), 12)
'''Feature Vector for L-ARC'''
days=7
X=np.pad(np.roll(Y,-48*days), (48*days, 0), 'constant')
Xs= np.asarray([ X[i:i+48*days][0:-1:48] for i in range(0,len(X)-48*days)])
Xs=MinMaxScaler().fit_transform(Xs)
Yhat[0] = 0
'''Computing NC scores'''
scores = np.abs(Y - Yhat)
'''Validation Data'''
Y_val=Y[::2]
Yhat_val=Yhat[::2]
Xs_val=Xs[::2]
scores_val = scores[::2]
'''Training Data'''
Y_tr=Y[1::2]
Yhat_tr=Yhat[1::2]
Xs_tr=Xs[1::2]
scores = scores[1::2]
'''ARC/L-ARC Parameters'''
eta_0=1.
decay=1./2.
alpha = 0.1  #Target Reliabilityt
'''ARC'''
ARC = ARC_calibration(Xs_tr,Yhat_tr, Y_tr,eta_0,alpha,decay)
err_val_ARC=(scores_val>ARC[1][-1]).astype(int)
cov_marginalized=np.mean(err_val_ARC)
cov_week=np.mean(err_val_ARC[week_id])
cov_weekend=np.mean(err_val_ARC[weekend_id])
dictionary = {
    "Log": ARC,
    "err": err_val_ARC,
    "err_weekend": err_val_ARC[weekend_id],
    "err_week": err_val_ARC[week_id]
}
with open('Logs/ARC.pkl', 'wb') as f:
    pickle.dump(dictionary, f)
print('Average Coverage: '+str(np.round(cov_marginalized,3))+' Week Coverage: '+str(np.round(cov_week,3))+' Week-end Coverage: '+str(np.round(cov_weekend,3)))

'''L-ARC'''
kappa=1. # Scale of the RBF kernel
la = 1e-4  # Regularization strength
pre_computed_d = [-np.linalg.norm(x - Xs_tr, ord=2, axis=1) ** 2 for x in Xs_val]
for ls in [2.,1.,0.5]: # Length-scales to test
    print('Length scale : '+str(ls))
    L_ARC = LARC_calibration(X=Xs_tr,Y_HAT=Yhat_tr,Y=Y_tr,eta_init=eta_0,alpha_target=alpha,reg_param=la,length_scale=ls,kappa=kappa,decay=decay)
    err_LARC=[]
    i=0
    '''Testing Post-Calibration Performance'''
    for x in Xs_val:
        q = kappa*float(np.sum(np.exp(pre_computed_d[i]/ (ls ** 2))  * L_ARC[3])) + L_ARC[4]  #for last iterate use  kappa*float(np.sum(pre_computed_ke[i] * L_ARC[1])) + L_ARC[2]
        err_LARC.append((scores_val[i]>q).astype(int))
        i=i+1
    err_LARC=np.asarray(err_LARC)
    cov_marginalized_LARC=np.mean(err_LARC)
    cov_week_LARC=np.mean(err_LARC[week_id])
    cov_weekend_LARC=np.mean(err_LARC[weekend_id])
    print('Average Coverage: ' + str(np.round(cov_marginalized_LARC, 3)) + ' Week Coverage: ' + str(np.round(cov_week_LARC, 3)) + ' Week-end Coverage: ' + str(np.round(cov_weekend_LARC, 3)))
    dictionary = {
        "Log": L_ARC,
        "kappa":kappa,
        "err": err_LARC,
        "err_weekend": err_LARC[weekend_id],
        "err_week": err_LARC[week_id]
    }
    with open('Logs/LARC_kappa_'+str(kappa)+'_ls_'+str(ls)+'_la_'+str(la)+'.pkl', 'wb') as f:
        pickle.dump(dictionary, f)


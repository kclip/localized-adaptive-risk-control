import matplotlib.pyplot as plt
import numpy as np
# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
# For link-level simulations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


# Define the function to calculate mean of neighbors
def mean_of_neighbors(arr):
    return np.mean(arr)

def coverage_map(X,y):
    # Extract x and y coordinates
    x_coords, y_coords = X[:, 0], X[:, 1]
    # Set up the grid
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    grid_resolution = 1.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_resolution), np.arange(y_min, y_max, grid_resolution))
    # Create an array to hold the mean values
    mean_values = np.zeros_like(xx)
    # Iterate over each point and calculate the mean of its neighbors
    for i in range(len(xx)):
        for j in range(len(xx[i])):
            x_point, y_point = xx[i, j], yy[i, j]
            distances = np.sqrt((x_coords - x_point) ** 2 + (y_coords - y_point) ** 2)
            indices = np.where(distances <= 2)[0]  # consider points within a radius of 1.5 units as neighbors
            mean_values[i, j] = mean_of_neighbors(y[indices])
    return mean_values,x_min, x_max, y_min, y_max

def ARC_calibration(X,Y_HAT,Y,eta_init,alpha_target,decay):
    q=0
    errs,qs,errs_val=[],[],[]
    for t in range(0,X.shape[0]):
        eta_t = eta_init / (t+1)**(decay)
        if len(np.where(Y_HAT[t,:]>q)[0])>0:
            loss_t=(np.max(Y[t])- np.max(Y[t, np.where(Y_HAT[t, :] > q)[0]])) / np.max(Y[t])
        else:
            loss_t=1
        q = q + eta_t * (alpha_target- loss_t)
        errs.append(loss_t)
        qs.append(q)
    print('Long-Run Risk ARC = '+ str(round(np.mean(errs),3)))
    return errs,qs

def LARC_calibration(X,Y_HAT,Y,eta_init,alpha_target,reg_param,length_scale,kappa,decay):
    errs, coeff_list, cs,te_subpop = [], [], [],[]
    coeff_mean = np.zeros(X.shape[0])
    q = 0
    c=0
    for  t in range(0,X.shape[0]):
        if t==0:
            eta_t = eta_init / (t + 1) ** (decay)
            if len(np.where(Y_HAT[t, :] > q)[0]) > 0:
                loss_t = (np.max(Y[t])- np.max(Y[t, np.where(Y_HAT[t, :] > q)[0]])) / np.max(Y[t])
            else:
                loss_t = 1
            errs.append(loss_t)
            c = c + eta_t * (alpha_target - loss_t)
            coeff_t = eta_t *(alpha_target - loss_t)
            cs.append(c)
            coeff_mean[0] = coeff_t
        else:
            eta_t = eta_init / (t+1)**(decay)
            kernels=kappa*np.exp(-np.linalg.norm(X[t,:]-X[:t,:],ord=2,axis=1)**2/(length_scale**2))
            q=float(np.sum(kernels*coeff_t))+c
            if len(np.where(Y_HAT[t, :] > q)[0]) > 0:
                loss_t = (np.max(Y[t])- np.max(Y[t, np.where(Y_HAT[t, :] > q)[0]])) / np.max(Y[t])
            else:
                loss_t = 1.
            c = c + eta_t * (alpha_target - loss_t)
            errs.append(loss_t)
            cs.append(c)
            coeff_list.append(eta_t * (alpha_target - loss_t))
            coeff_t = np.asarray(coeff_t) * (1 - eta_t * reg_param)
            coeff_t = np.append(coeff_t, eta_t * (alpha_target - loss_t))
            coeff_mean[:t + 1] = coeff_mean[:t + 1] + coeff_t
    print('Long-Run Risk LARC= '+ str(round(np.mean(errs),3)))
    return errs,coeff_t,c,coeff_mean/X.shape[0],np.mean(cs)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 11)

    def forward(self, x):
        x=x/70.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def sample_channel_gain(scenario):
    with open( scenario + '.pkl', 'rb') as handle:
        data = pickle.load(handle)[0]
    trajectories = []
    for d in data:
        trajectories.append(d[0])
    return trajectories

def generate_ray_tracing_data(positions):
    # Load Scene
    scene = load_scene('.ray_tracing_files/scenario.xml')
    # Create new camera with different configuration
    my_cam = Camera("my_cam", position=[0, -8, 250], look_at=[0, -8, 0])
    scene.add(my_cam)
    # Configure antenna array for all transmitters
    n_tx = 8
    scene.tx_array = PlanarArray(num_rows=1, num_cols=n_tx, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="tr38901", polarization="V")
    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization='V')
    # Create transmitter
    tx = Transmitter(name="tx", position=[0, -32, 30])
    tx.look_at([0, 30, 10])
    # Add transmitter instance to scene
    scene.add(tx)
    scene.render(camera="my_cam")
    plt.savefig('Deployment')
    plt.close()
    for i in range(positions.shape[1]):
        # Create a receiver
        rx = Receiver(name="rx" + str(i), position=positions[:, i], orientation=[0, 0, 0])
        # Add receiver instance to scene
        scene.add(rx)
    # Transmitter points towards receiver
    scene.frequency = 2.14e9  # in Hz; implicitly updates RadioMaterials
    scene.synthetic_array = True  # If set to False, ray tracing will be done per antenna element (slower for large arrays)
    # Default parameters in the PUSCHConfig
    # Compute propagation paths
    paths = scene.compute_paths(max_depth=5, num_samples=1e6)  # Number of rays shot into directions defined
    a, tau = paths.cir()
    return np.sum(a[0, :, 0, 0, :, :, 0], axis=-1)


AlPHAS = np.hstack([0.025,np.linspace(0.05, 0.5, 10)])
MC_reps=30  #Number of Monte Carlo runs
kappa=1. #RBF kernel magnitude
for length_scale in [10.]:  #RBF kernel length scale
    for seed in np.arange(0,MC_reps):
        np.random.seed(seed)
        torch.manual_seed(seed)
        n=50000  #Number of training data points
        '''Sampling random locations'''
        positions=np.random.rand(3,n)
        positions[0,:]=(positions[0,:]-0.5)*140
        positions[1,:]=(positions[1,:]-0.2)*100
        positions[2,:]=1.
        n_tx=8  #Number of Antennas at the Tx
        pos_BS = np.asarray([0, -32])
        GEN=False  #Set to True if you wish to generate ray-tracing data, otherwise previously generated data will be loaded.
        if GEN:
            h=generate_ray_tracing_data(positions)
            data=[positions,h]
            with open("data", "wb") as fp:   #Pickling
                 pickle.dump(data, fp)
        else:
            with open("data", "rb") as fp:  # Pickling
                data= pickle.load(fp)
        h=data[1]
        positions=data[0]
        # DFT codebook Generation
        n_beams=11
        ang=np.linspace(-np.pi/2,np.pi/2,n_beams)
        DFT_codebook=np.asarray([np.exp(1j*np.arange(0,n_tx)*a) for a in ang])
        # Effective Channel with Rayleigh fading
        sigma_noise=0.0001
        h_eff=np.abs(DFT_codebook@np.transpose(h))
        h_best=np.max(h_eff,axis=0)
        valid_id=np.where(h_best>1e-6)[0]
        h_best=h_best[valid_id]
        h_eff=h_eff[:,valid_id]
        positions=positions[:,valid_id]
        h_eff=h_eff+np.abs(np.random.randn(h_eff.shape[0],h_eff.shape[1])*sigma_noise+1j*np.random.randn(h_eff.shape[0],h_eff.shape[1])*sigma_noise)
        # Normalized Channel Gain
        h_eff_norm=h_eff/np.sum(h_eff,axis=0)
        best_beam_id=np.argmax(h_eff,axis=0)
        frac_tr=0.2
        frac_CP=0.4
        frac_te=0.4
        n=positions.shape[1]
        n_tr=2500
        n_CP =20000
        n_te=n-n_CP-n_tr
        print('Training Set Size: '+str(n_tr)+' CP Set Size: '+str(n_CP)+'  Testing Set Size: '+str(n_te))
        '''Data Split'''
        pos_tr=positions[:,:n_tr]
        pos_te=positions[:,n_tr:n_te+n_tr]
        pos_CP=positions[:,n_te+n_tr:n_te+n_tr+n_CP]
        X_tr=np.transpose(pos_tr[0:2,:])
        X_te=np.transpose(pos_te[0:2,:])
        X_CP=np.transpose(pos_CP[0:2,:])
        h_eff_tr=np.transpose(h_eff[:,:n_tr])
        h_eff_te=np.transpose(h_eff[:,n_tr:n_te+n_tr])
        h_eff_CP=np.transpose(h_eff[:,n_te+n_tr:n_te+n_tr+n_CP])
        y_tr=np.transpose(h_eff_norm[:,:n_tr])
        y_te=np.transpose(h_eff_norm[:,n_tr:n_te+n_tr])
        y_CP=np.transpose(h_eff_norm[:,n_te+n_tr:n_te+n_tr+n_CP])
        best_beam_id_tr=best_beam_id[:n_tr]
        best_beam_id_te=best_beam_id[n_tr:n_te+n_tr]
        best_beam_id_CP=best_beam_id[n_te+n_tr:n_te+n_tr+n_CP]

        '''Training a NN predictor'''
        TRAIN=True  #Set True if you want to train from scratch
        if TRAIN:
            net = Net()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.01,weight_decay=0.001)
            n_epochs=2000
            loss_log=[]
            i=1
            for epoch in range(n_epochs):  # loop over the dataset multiple times
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(torch.from_numpy(X_tr).float())
                loss = criterion(outputs, torch.from_numpy(y_tr).float())
                loss.backward()
                optimizer.step()
                loss_log.append(loss.item())
                beam_hat=torch.argmax(outputs,axis=1).numpy()
                acc=np.mean(np.abs(beam_hat-best_beam_id_tr)==0)
                if(epoch%20==0):
                    print(f'[{epoch + 1}, {i + 1:6d}] loss: {loss.item():.3f} acc: {acc:.3f}')
            torch.save(net, 'NN_beam_predictor')
            print('Finished Training')
        else:
            print('Loading Model')
            net=torch.load('NN_beam_predictor')

        '''Model Predictions'''
        y_hat_CP=F.softmax(net(torch.from_numpy(X_CP).float()),dim=1).detach().numpy()
        y_hat_te=F.softmax(net(torch.from_numpy(X_te).float()),dim=1).detach().numpy()
        ARC=[]
        MondrianARC=[]
        LARC=[]
        for alpha in AlPHAS:
            '''Adaptive Online Conformal Prediction'''
            lr=1.
            decay_lr=0.5
            ARC.append(ARC_calibration(X_CP,y_hat_CP,y_CP,lr,alpha,decay_lr))

            '''Mondrian Online Conformal Prediction'''
            d=50
            id_1=np.where(np.linalg.norm(np.abs(X_CP-pos_BS[np.newaxis,:]),axis=1)<=d)[0]
            id_2=np.where(np.linalg.norm(np.abs(X_CP-pos_BS[np.newaxis,:]),axis=1)>d)[0]
            X_CP_1=X_CP[id_1]
            y_hat_CP_1=y_hat_CP[id_1]
            y_CP_1=y_CP[id_1]
            X_CP_2=X_CP[id_2]
            y_hat_CP_2=y_hat_CP[id_2]
            y_CP_2=y_CP[id_2]
            errs_ARC_1,q_ARC_1=ARC_calibration(X_CP_1,y_hat_CP_1,y_CP_1,lr,alpha,decay_lr)
            errs_ARC_2,q_ARC_2=ARC_calibration(X_CP_2,y_hat_CP_2,y_CP_2,lr,alpha,decay_lr)
            MondrianARC.append([errs_ARC_1,q_ARC_1,errs_ARC_2,q_ARC_2])

            '''Local Adaptive Online Conformal Prediction'''
            reg_param=1e-4 # strength of regularizer
            LARC.append(LARC_calibration(X_CP,y_hat_CP,y_CP,lr,alpha,reg_param,length_scale,kappa,decay_lr))

        dictionary = {
            "alphas": AlPHAS,
            "ARC":ARC,
            "MondrianARC":  MondrianARC,
            "LARC": LARC,
            "y_hat_te":y_hat_te,
            "X_te":X_te,
            "y_te":y_te,
            "h_eff_te":h_eff_te,
            "length_scale": length_scale,
            "kappa":kappa,
            "X_CP":X_CP,
            }
        with open('Results/ResultsSEED'+str(seed)+'kmax'+str(kappa)+'_ls_'+str(length_scale)+'.pkl', 'wb') as f:
            pickle.dump(dictionary, f)



import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from matplotlib.patches import Circle
import pickle

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



# Define the function to calculate mean of neighbors
def mean_of_neighbors(arr):
    return np.mean(arr)

def coverage_map(X,y):
    # Extract x and y coordinates
    x_coords, y_coords = X[:, 0], X[:, 1]

    # Set up the grid
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    grid_resolution = 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_resolution), np.arange(y_min, y_max, grid_resolution))

    # Create an array to hold the mean values
    mean_values = np.zeros_like(xx)

    # Iterate over each point and calculate the mean of its neighbors
    for i in range(len(xx)):
        for j in range(len(xx[i])):
            x_point, y_point = xx[i, j], yy[i, j]
            distances = np.sqrt((x_coords - x_point) ** 2 + (y_coords - y_point) ** 2)
            indices = np.where(distances <= 2)[0]
            mean_values[i, j] = mean_of_neighbors(y[indices])

    return mean_values,x_min, x_max, y_min, y_max



plt.rc('font', family='serif', serif='Computer Modern Roman', size=22)
plt.rcParams.update({
    'font.size': 22,
    'text.usetex': True,
})

building_mask = np.zeros((100, 140)) * np.nan
build_c = 0.5
building_mask[16:24, 27:46] = build_c
building_mask[37:57, 27:35] = build_c
building_mask[23:43, 9:17] = build_c
building_mask[26:34, 88:96] = build_c
building_mask[38:46, 104:112] = build_c
building_mask[17:25, 107:115] = build_c
building_mask[61:69, 64:123] = build_c
building_mask[80:88, 85:106] = build_c
building_mask[80:88, 113:134] = build_c
fig, ax = plt.subplots()
plt.tight_layout()
ax.set_axisbelow(True)
plt.grid()
ax.imshow(building_mask, extent=(-70, 70, -30, 70), cmap='binary', vmin=0, vmax=1,zorder=1)
circle1 = plt.Circle((0, -29), 2, fill=True, color='tab:green')
plt.text(0, -23, 'Transmitter', color='tab:green', ha='center', va='center')
ax.add_patch(circle1)
plt.subplots_adjust(left=0.15)
plt.xlabel(r'$p_\mathrm{x}$')
plt.ylabel(r'$p_\mathrm{y}$')
plt.savefig('Images/Deployment.pdf')
plt.clf()
plt.close()

n_tx=8
pos_BS = np.asarray([0, -32])
MC_reps=1
kappa=1.
ls=10.
for seed in np.arange(0,MC_reps):
    with open('Results/ResultsSEED'+str(seed)+'kmax'+str(kappa)+'_ls_'+str(ls)+'.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    ALPHAS=loaded_dict['alphas']
    alpha = ALPHAS[0]
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
    LARC_Log=[]
    id_ARC=4
    id_MON=2
    id_LARC=1
    q_ARC=ARC[id_ARC][1]
    '''     TESTING     '''
    cov_te_cond=[]
    SNR_te=[]
    for i in range(0,y_hat_te.shape[0]):
        if len(np.where(y_hat_te[i,:]>q_ARC[-1])[0])>0:
            pred_set = np.where(y_hat_te[i, :] > q_ARC[-1])[0]
            cov_te_cond.append(q_ARC[-1])
            SNR_te.append(np.max(h_eff_te[i, pred_set]))
        else:
            cov_te_cond.append(q_ARC[-1])
            SNR_te.append(0)
    cov_te_cond=1-np.asarray(cov_te_cond)
    size_te_ARC=np.asarray([np.sum(y_hat_te[i,:]>q_ARC[-1]) for i in range(0,y_hat_te.shape[0])])
    ARC_Log.append([np.mean(ARC[id_ARC][0]),cov_te_cond,size_te_ARC,SNR_te])
    print('ARC Convergence Risk: '+str(round(np.mean(cov_te_cond),3))+'Avg. Set Size: '+str(round(np.mean(size_te_ARC),3)))
    d=50
    q_ARC_1 = MondrianARC[id_MON][1]
    q_ARC_2 = MondrianARC[id_MON][3]
    id_1_te=np.where(np.linalg.norm(np.abs(X_te-pos_BS[np.newaxis,:]),axis=1)<=d)[0]
    id_2_te=np.where(np.linalg.norm(np.abs(X_te-pos_BS[np.newaxis,:]),axis=1)>d)[0]
    cov_te_cond_1=[]
    SNR_te_1=[]
    for i in id_1_te:
        if len(np.where(y_hat_te[i,:]>q_ARC_1[-1])[0])>0:
            pred_set=np.where(y_hat_te[i,:]>q_ARC_1[-1])[0]
            cov_te_cond_1.append(q_ARC_1[-1])
            SNR_te_1.append(np.max(h_eff_te[i,pred_set]))
        else:
            cov_te_cond_1.append(q_ARC_1[-1])
            SNR_te_1.append(0)
    cov_te_cond_1=1-np.asarray(cov_te_cond_1)
    cov_te_cond_2=[]
    SNR_te_2=[]
    for i in id_2_te:
        if len(np.where(y_hat_te[i,:]>q_ARC_2[-1])[0])>0:
            pred_set = np.where(y_hat_te[i, :] > q_ARC_2[-1])[0]
            cov_te_cond_2.append(q_ARC_2[-1])
            SNR_te_2.append(np.max(h_eff_te[i, pred_set]))
        else:
            cov_te_cond_2.append(q_ARC_2[-1])
            SNR_te_2.append(0)
    cov_te_cond_2=1-np.asarray(cov_te_cond_2)
    size_te_ARC_1=np.asarray([np.sum(y_hat_te[i,:]>q_ARC_1[-1]) for i in id_1_te])
    size_te_ARC_2=np.asarray([np.sum(y_hat_te[i,:]>q_ARC_2[-1]) for i in id_2_te])
    print('Mondrian Convergence Risk 1 : '+str(round(np.mean(cov_te_cond_1),3))+'Avg. Set Size : '+str(round(np.mean(size_te_ARC_1),3)))
    print('Mondrian Convergence Risk 2 : '+str(round(np.mean(cov_te_cond_2),3))+'Avg. Set Size : '+str(round(np.mean(size_te_ARC_2),3)))
    print('Mondrian ARC Convergence Risk: ' + str(round(np.mean(np.hstack((cov_te_cond_1, cov_te_cond_2))), 3)) + 'Avg. Set Size: ' + str(round(np.mean(np.hstack((size_te_ARC_1, size_te_ARC_2))), 3)))
    MondrianARC_Log.append([np.mean(MondrianARC[id_MON][0]), cov_te_cond_1, size_te_ARC_1,SNR_te_1,np.mean(MondrianARC[id_MON][2]), cov_te_cond_2, size_te_ARC_2,SNR_te_2])
    cov_te_cond_LARC=[]
    size_te_LARC=[]
    SNR_te_LARC=[]
    q_LARC=LARC[id_LARC][3]
    for i in range(0, y_hat_te.shape[0]):
        kernels = np.exp(-np.linalg.norm(X_te[i, :] - X_CP, ord=2, axis=1) ** 2 / (length_scale ** 2))
        q = float(np.sum(kernels * q_LARC))+LARC[id_LARC][4]
        if len(np.where(y_hat_te[i,:]>q)[0])>0:
            pred_set =np.where(y_hat_te[i,:]>q)[0]
            loss_cond=-(np.max(y_te[i,pred_set])-np.max(y_te[i]))/np.max(y_te[i])
            SNR_te_LARC.append(np.max(h_eff_te[i, pred_set]))
        else:
            loss_cond=1
            SNR_te_LARC.append(0)
        size= np.sum(y_hat_te[i, :] > q)
        size_te_LARC.append(size)
        cov_te_cond_LARC.append(q)
    cov_te_cond_LARC=1-np.asarray(cov_te_cond_LARC)
    print('K-ARC Convergence Risk'+str(round(np.mean(cov_te_cond_LARC),3))+'Avg. Set Size: '+str(round(np.mean(size_te_LARC),3)))
    LARC_Log.append([np.mean(LARC[id_LARC][0]), cov_te_cond_LARC, size_te_LARC,SNR_te_LARC])

    plt.rcParams['axes.facecolor'] = 'black'
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(19, 5), gridspec_kw={'width_ratios': [1, 1, 1, 0.03]})
    cond_cov_ARC, x_min, x_max, y_min, y_max = coverage_map(X_te, np.asarray(SNR_te))
    cond_cov_LARC, x_min, x_max, y_min, y_max = coverage_map(X_te, np.asarray(SNR_te_LARC))
    cond_cov_MonARC, x_min, x_max, y_min, y_max = coverage_map(np.vstack((X_te[id_1_te], X_te[id_2_te])), np.asarray(np.hstack((SNR_te_1, SNR_te_2))))
    cmap = plt.get_cmap('hot')
    #cond_cov_LARC[np.isnan(cond_cov_LARC)==False] = 10*np.log10(cond_cov_LARC[np.isnan(cond_cov_LARC)==False]+10e-7)
    #cond_cov_MonARC[np.isnan(cond_cov_MonARC)==False] =  10*np.log10(cond_cov_MonARC[np.isnan(cond_cov_MonARC)==False]+10e-7)
    #cond_cov_ARC[np.isnan(cond_cov_ARC)==False] =  10*np.log10(cond_cov_ARC[np.isnan(cond_cov_ARC)==False] +10e-7)
    cond_cov_LARC[cond_cov_LARC==0] = np.nan
    cond_cov_MonARC[cond_cov_MonARC==0] =  np.nan
    cond_cov_ARC[cond_cov_ARC==0]=  np.nan


    max = np.max([np.max(cond_cov_ARC[np.isnan(cond_cov_ARC)==False]), np.max(cond_cov_LARC[np.isnan(cond_cov_LARC)==False]), np.max(cond_cov_MonARC[np.isnan(cond_cov_MonARC)==False])])
    min = np.min([np.min(cond_cov_ARC[np.isnan(cond_cov_ARC)==False]), np.min(cond_cov_LARC[np.isnan(cond_cov_LARC)==False]), np.min(cond_cov_MonARC[np.isnan(cond_cov_MonARC)==False])])
    max = 10 * np.log10(max)
    min = 10 * np.log10(min)
    aspect_ratio=1
    im0 = axes[0].imshow(10*np.log10(cond_cov_ARC), extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=cmap, alpha=1., vmin=min, vmax=max,aspect=aspect_ratio)
    im1 = axes[1].imshow(10*np.log10(cond_cov_MonARC), extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=cmap, alpha=1., vmin=min, vmax=max,aspect=aspect_ratio)
    im2 = axes[2].imshow(10*np.log10(cond_cov_LARC), extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=cmap, alpha=1., vmin=min, vmax=max,aspect=aspect_ratio)
    res=1
    building_mask=np.zeros((100,140))*np.nan
    build_c=0.25
    building_mask[16:24,27:46]=build_c
    building_mask[37:57, 27:35] = build_c
    building_mask[23:43, 9:17] = build_c
    building_mask[26:34, 88:96] = build_c
    building_mask[38:46, 104:112] = build_c
    building_mask[17:25, 107:115] = build_c
    building_mask[61:69, 64:123] = build_c
    building_mask[80:88, 85:106] = build_c
    building_mask[80:88, 113:134] = build_c
    axes[0].imshow(building_mask, extent=(x_min, x_max, y_min, y_max), cmap='binary', vmin=0, vmax=1,aspect=aspect_ratio)
    axes[1].imshow(building_mask, extent=(x_min, x_max, y_min, y_max), cmap='binary', vmin=0, vmax=1,aspect=aspect_ratio)
    axes[2].imshow(building_mask, extent=(x_min, x_max, y_min, y_max),cmap='binary',vmin=0,vmax=1,aspect=aspect_ratio)
    cax = axes[3]
    fig.colorbar(im2, label=r'SNR', shrink=0.4, cax=cax, aspect=20)
    circ = Circle((0, -32), d, fill=False, color='tab:green', linestyle='--', linewidth=2)

    axes[1].add_patch(circ)
    axes[0].add_patch(Circle((0, -29), 2, fill=True, color='tab:green'))
    axes[1].add_patch(Circle((0, -29), 2, fill=True, color='tab:green'))
    axes[2].add_patch(Circle((0, -29), 2, fill=True, color='tab:green'))
    axes[0].text(0, -23, r'\textbf{Transmitter}',color='tab:green',ha='center', va='center')
    axes[1].text(0, -23, r'\textbf{Transmitter}',color='tab:green',ha='center', va='center')
    axes[2].text(0, -23, r'\textbf{Transmitter}',color='tab:green',ha='center', va='center')
    axes[0].set_title(r'ARC')
    axes[1].set_title(r'Mondrian ARC')
    axes[2].set_title(r'L-ARC')
    axes[0].set_xlabel(r'$p_\mathrm{x}$')
    axes[1].set_xlabel(r'$p_\mathrm{x}$')
    axes[2].set_xlabel(r'$p_\mathrm{x}$')
    axes[0].set_ylabel(r'$p_\mathrm{y}$')
    axes[2].set_yticks([])
    axes[1].set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.04 ,left=0.05 ,right=0.94 )
    plt.savefig('Images/SNR_Map_ls_' + str(ls) + '.pdf')
    plt.clf()
    plt.close()

    plt.rcParams['axes.facecolor'] = 'dimgrey'
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(19, 5), gridspec_kw={'width_ratios': [1, 1, 1, 0.03]})
    cond_cov_ARC, x_min, x_max, y_min, y_max = coverage_map(X_te, cov_te_cond)
    cond_cov_LARC, x_min, x_max, y_min, y_max = coverage_map(X_te, np.asarray(cov_te_cond_LARC))
    cond_cov_MonARC, x_min, x_max, y_min, y_max = coverage_map(np.vstack((X_te[id_1_te], X_te[id_2_te])), np.asarray(np.hstack((cov_te_cond_1, cov_te_cond_2))))
    cmap = plt.get_cmap('Reds')
    new_cmap = np.flip(truncate_colormap(cmap, 0, 0 + 0.5 / (1 - alpha)))
    min=np.min(cov_te_cond_LARC)
    im0 = axes[0].imshow(cond_cov_ARC, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=cmap, alpha=1., vmin=min, vmax=1.0)
    im1 = axes[1].imshow(cond_cov_MonARC, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=cmap, alpha=1., vmin=min, vmax=1.0)
    im2 = axes[2].imshow(cond_cov_LARC, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=cmap, alpha=1., vmin=min, vmax=1.0)
    cax = axes[3]
    fig.colorbar(im2, label='Normalized Communication Rate', shrink=0.4, cax=cax, aspect=20)

    circ = Circle((0, -32), d, fill=False, color='tab:green', linestyle='--', linewidth=2)
    axes[1].add_patch(circ)
    axes[0].add_patch(Circle((0, -29), 2, fill=True, color='tab:green'))
    axes[1].add_patch(Circle((0, -29), 2, fill=True, color='tab:green'))
    axes[2].add_patch(Circle((0, -29), 2, fill=True, color='tab:green'))
    axes[0].text(0, -23, 'Transmitter', color='tab:green', ha='center', va='center')
    axes[1].text(0, -23, 'Transmitter', color='tab:green', ha='center', va='center')
    axes[2].text(0, -23, 'Transmitter', color='tab:green', ha='center', va='center')
    axes[0].set_title(r'ARC')
    axes[1].set_title(r'Mondrian ARC')
    axes[2].set_title(r'L-ARC')
    axes[0].set_xlabel(r'$p_\mathrm{x}$')
    axes[1].set_xlabel(r'$p_\mathrm{x}$')
    axes[2].set_xlabel(r'$p_\mathrm{x}$')
    axes[0].set_ylabel(r'$p_\mathrm{y}$')
    axes[2].set_yticks([])
    axes[1].set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.04, left=0.05, right=0.94)
    plt.savefig('Images/ConditionalCoverage_ls_' + str(ls) + '.pdf')
    plt.clf()
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(19, 5), gridspec_kw={'width_ratios': [1, 1, 1, 0.03]})
    cond_cov_ARC, x_min, x_max, y_min, y_max = coverage_map(X_te, size_te_ARC)
    cond_cov_LARC, x_min, x_max, y_min, y_max = coverage_map(X_te, np.asarray(size_te_LARC))
    cond_cov_MonARC, x_min, x_max, y_min, y_max = coverage_map(np.vstack((X_te[id_1_te], X_te[id_2_te])), np.asarray(np.hstack((size_te_ARC_1, size_te_ARC_2))))
    max = np.max([np.max(cond_cov_ARC[np.isnan(cond_cov_ARC) == False]), np.max(cond_cov_LARC[np.isnan(cond_cov_LARC) == False]), np.max(cond_cov_MonARC[np.isnan(cond_cov_MonARC) == False])])
    min = np.min([np.min(cond_cov_ARC[np.isnan(cond_cov_ARC) == False]), np.min(cond_cov_LARC[np.isnan(cond_cov_LARC) == False]), np.min(cond_cov_MonARC[np.isnan(cond_cov_MonARC) == False])])
    np.max(cond_cov_MonARC)
    im0 = axes[0].imshow(cond_cov_ARC, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='jet', alpha=1., vmin=0, vmax=max)
    im1 = axes[1].imshow(cond_cov_MonARC, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='jet', alpha=1., vmin=0, vmax=max)
    im2 = axes[2].imshow(cond_cov_LARC, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='jet', alpha=1., vmin=0, vmax=max )
    cax = axes[3]
    fig.colorbar(im2, label='Beam Training Length', shrink=0.4, cax=cax, aspect=20)
    circ = Circle((0, -32), d, fill=False, color='tab:green', linestyle='--', linewidth=2)
    axes[1].add_patch(circ)
    axes[0].add_patch(Circle((0, -29), 2, fill=True, color='tab:green'))
    axes[1].add_patch(Circle((0, -29), 2, fill=True, color='tab:green'))
    axes[2].add_patch(Circle((0, -29), 2, fill=True, color='tab:green'))
    axes[0].text(0, -23, 'Transmitter', color='tab:green', ha='center', va='center')
    axes[1].text(0, -23, 'Transmitter', color='tab:green', ha='center', va='center')
    axes[2].text(0, -23, 'Transmitter', color='tab:green', ha='center', va='center')
    axes[0].set_title(r'ARC')
    axes[1].set_title(r'Mondrian ARC')
    axes[2].set_title(r'L-ARC')
    axes[0].set_xlabel(r'$p_\mathrm{x}$')
    axes[1].set_xlabel(r'$p_\mathrm{x}$')
    axes[2].set_xlabel(r'$p_\mathrm{x}$')
    axes[0].set_ylabel(r'$p_\mathrm{y}$')
    axes[2].set_yticks([])
    axes[1].set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.04, left=0.05, right=0.94)
    plt.savefig('Images/ConditionalSetSize_ls_' + str(ls) + '.pdf')
    plt.clf()
    plt.close()
    plt.tight_layout()
    plt.rcParams['axes.facecolor'] = 'white'
    plt.grid()




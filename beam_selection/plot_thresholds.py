import matplotlib.pyplot as plt
import numpy as np
# Import Sionna RT components
import pickle
# For link-level simulations
import matplotlib.cm as cm
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
id_plot=0
seed=0
d=50
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(19, 5.25), gridspec_kw={'width_ratios': [1, 1, 1, 0.03]})
plt.rcParams['axes.facecolor'] = 'black'
for ls in [50.,25.,10.]:
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

    LARC_Log=[]

    id_LARC=1
    '''     TESTING     '''

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
    print('K-ARC Convergence Risk'+str(round(np.mean(cov_te_cond_LARC),3))+'Avg. Set Size: '+str(round(np.mean(size_te_LARC),3)))
    LARC_Log.append([np.mean(LARC[id_LARC][0]), cov_te_cond_LARC, size_te_LARC,SNR_te_LARC])




    cond_cov_LARC, x_min, x_max, y_min, y_max = coverage_map(X_te, np.asarray(SNR_te_LARC))
    cmap = plt.get_cmap('Reds_r')
    cond_cov_LARC[cond_cov_LARC==0] = np.nan
    plt.rcParams['axes.facecolor'] = 'black'
    cond_cov_LARC, x_min, x_max, y_min, y_max = coverage_map(X_te, np.asarray(cov_te_cond_LARC))
    max = np.max([np.max(cond_cov_LARC[np.isnan(cond_cov_LARC)==False])])
    min = np.min([np.min(cond_cov_LARC[np.isnan(cond_cov_LARC)==False])])
    max = 10 * np.log10(max)
    min = 10 * np.log10(min)
    max=-22
    min=-44
    aspect_ratio=1
    min=np.min(cov_te_cond_LARC)
    im0 = axes[id_plot].imshow(cond_cov_LARC, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=np.flip(cmap), alpha=1., vmin=0, vmax=0.3)
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
    fig.colorbar(im0, label=r'$\bar{g}_T(p_\mathrm{x},p_\mathrm{x})$', shrink=0.4, cax=cax, aspect=20)
    id_plot=id_plot+1
circ = Circle((0, -32), d, fill=False, color='tab:green', linestyle='--', linewidth=2)
axes[0].add_patch(Circle((0, -29), 2, fill=True, color='tab:green'))
axes[1].add_patch(Circle((0, -29), 2, fill=True, color='tab:green'))
axes[2].add_patch(Circle((0, -29), 2, fill=True, color='tab:green'))
axes[0].text(0, -23, r'\textbf{Transmitter}',color='tab:green',ha='center', va='center')
axes[1].text(0, -23, r'\textbf{Transmitter}',color='tab:green',ha='center', va='center')
axes[2].text(0, -23, r'\textbf{Transmitter}',color='tab:green',ha='center', va='center')
axes[0].set_title(r'L-ARC $l=50$')
axes[1].set_title(r'L-ARC $l=25$')
axes[2].set_title(r'L-ARC $l=10$')
axes[0].set_xlabel(r'$p_\mathrm{x}$')
axes[1].set_xlabel(r'$p_\mathrm{x}$')
axes[2].set_xlabel(r'$p_\mathrm{x}$')
axes[0].set_ylabel(r'$p_\mathrm{y}$')
axes[2].set_yticks([])
axes[1].set_yticks([])
plt.tight_layout()
plt.subplots_adjust(wspace=0.04 ,left=0.05 ,right=0.94 )
plt.savefig('Images/Threhsolds.pdf')
plt.show()
plt.clf()
plt.close()

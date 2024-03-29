import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from ks_util import *
from time import time

big_start = time()
p = param()
r = res()

init_shocks(p, r)
get_prices(p, r)

while r.rsq < 0.98 or r.error > 1e-4 and r.it < 8:
    start = time()

    vfi(p, r)
    pseudo_panel(p, r)
    calc_errors(p, r)
    end = time()
    print("Elapse time is {} s".format(end-start))
    r.it += 1

end = time()
print("Elapse time is {} s".format(end-big_start))

################################################################
# simulation
dpath = "../BKM/data_output"
var_list = ["prod", "ak", "i", "c", "r", "wage", "y"]
# Productivity Shocks
T = 1500
drop = 100
rng = np.random.default_rng(seed=1234)
sigma_eta = 0.012
rho = 0.9
mu_eta = -sigma_eta**2/2
mu_a = mu_eta/(1-rho)
eta_t = np.random.normal(mu_eta, sigma_eta, T)
agg_shocks = np.zeros(T)
Ass = 1
agg_shocks[0] = Ass**rho*np.exp(eta_t[0])
for tidx in range(1, T):
    agg_shocks[tidx] = agg_shocks[tidx-1]**rho*np.exp(eta_t[tidx])

################################################################
# Get BKM results which is also IRF
dist_path_bkm = np.genfromtxt(dpath+"/"+"dist_path"+".txt")  # NA x T
var_path_bkm = np.zeros((len(var_list), dist_path_bkm.shape[1]))
for vi, var in enumerate(var_list):
    var_path_bkm[vi, :] = np.genfromtxt(dpath+"/"+var+"_path"+".txt")

################################################################
# initiate steady states results
# TODO: change to not BKM dependent
Kss = 2.5**(1/(1-p.alpha))
Css = var_path_bkm[3, -1]  # last period of BKM IRF
ss_list = np.array([1, Kss, Kss**p.alpha-Css, Css, p.alpha*Kss**(p.alpha-1)+1. - p.delta, (1-p.alpha)*Kss**(p.alpha), Kss**p.alpha])
log_ss = np.log(ss_list)

# init KS simulation results arrays
dist_path_ks = np.zeros((p.NK*p.NY, T))
dist_path_ks[:, 0] = dist_path_bkm[:, 0]
agg_k_ks = np.zeros(T)
agg_k_ks[0] = Kss  # from steady state aiyagari
agg_i_ks = np.zeros(T)

# KS prediction using policy function
small_k = np.zeros((p.NK, p.NY, T))
small_k_decision = p.k_grid[r.pfunc]
# TODO: change to interp2d
interp_small_k = interpolate.RegularGridInterpolator((p.k_grid, p.ind_states, p.ak_grid, p.agg_states), small_k_decision, bounds_error=False, fill_value=None)

for tidx in range(1,T):
    tmp_coor =np.meshgrid(p.k_grid, p.ind_states, agg_k_ks[tidx - 1], agg_shocks[tidx - 1])
    tmp_coor = np.meshgrid(tmp_coor, indexing="ij")
    coors =  np.reshape(tmp_coor, (4, -1), order="C").T  # dimension is  (NA*NZ) x 4
    chosen_this_small_k = interp_small_k(coors).T
    chosen_this_small_k[chosen_this_small_k<0] = 0
    dist_path_ks[:, tidx] = calc_dist(p.NK, p.NY, dist_path_ks[:, tidx-1], p.ymarkov, p.k_grid, chosen_this_small_k)
    agg_k_ks[tidx] = np.sum(dist_path_ks[:, tidx-1]*chosen_this_small_k)
    agg_i_ks[tidx] = (agg_k_ks[tidx] - agg_k_ks[tidx - 1]) + p.delta * agg_k_ks[tidx - 1]

agg_i_ks = agg_i_ks[drop:]
agg_k_ks = agg_k_ks[drop:]
agg_shocks_ks = agg_shocks[drop:]
agg_y_ks = agg_shocks_ks*agg_k_ks**p.alpha
agg_c_ks = agg_y_ks - agg_i_ks
interest_ks = 1+ p.alpha*agg_shocks_ks*agg_k_ks**(p.alpha - 1) - p.delta  # r
wage_ks = (1-p.alpha)*agg_shocks_ks*agg_k_ks**p.alpha

################################################################
# KS variance & cov
KS_list = np.zeros((len(var_list), T-drop))
KS_list[:, :] = [agg_shocks_ks, agg_k_ks, agg_i_ks, agg_c_ks, interest_ks, wage_ks, agg_y_ks]
KS_logged =np.subtract(np.log(KS_list).T,  log_ss).T
KS_var = np.var(KS_list, axis=1)
KS_corr = np.corrcoef(KS_list)
################################################################
# BKM

shock = .012
bkm_agg_list = np.zeros((len(var_list), T))

bkm_normalized = np.subtract(np.log(var_path_bkm).T,  log_ss).T/shock  # NVar x IRF_len
effect_len = bkm_normalized.shape[1]  # length of IRF

for tidx in range(0, T):
    s = eta_t[tidx]
    if T > tidx+effect_len: # "prod", "ak", "i", "c", "r", "wage", "y"
        for i in range(len(var_list)):
            bkm_agg_list[i, tidx:tidx + effect_len] += bkm_normalized[i, :] * s
    else:  #
        remain = T - tidx
        for i in range(len(var_list)):
            bkm_agg_list[i, tidx:] += bkm_normalized[i, :remain] * s
################################################################
# Draw Simulation Graph

for idx, it in enumerate(var_list):
    fig = plt.figure(idx)
    # fig.canvas.set_window_title(int(idx))
    plt.plot(range(drop, T), KS_logged[idx, :], color='blue', label='KS')
    plt.plot(range(drop, T), bkm_agg_list[idx, drop:], color='orange', label='BKM')
    plt.legend()

    fname = 'compare_{}'.format(var_list[idx])
    plt.suptitle(fname)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.legend()
    plt.savefig('data_output/{}.png'.format(fname))
################################################################
# IRF - is completely wrong

# SetUp
T= 250
shock = sigma_eta
agg_shocks = np.zeros(T)
agg_shocks[0] = Ass**rho*np.exp(sigma_eta)
for tidx in range(1, T):
    agg_shocks[tidx] = agg_shocks[tidx-1]**rho

# KS
dist_path_ks = np.zeros((p.NK*p.NY, T))
dist_path_ks[:, 0] = dist_path_bkm[:, 0]
agg_k_ks = np.zeros(T)
agg_k_ks[0] = Kss  # from steady state aiyagari
agg_i_ks = np.zeros(T)

# KS prediction using policy function
small_k = np.zeros((p.NK, p.NY, T))
small_k_decision = p.k_grid[r.pfunc]
# TODO: change to interp2d
interp_small_k = interpolate.RegularGridInterpolator((p.k_grid, p.ind_states, p.ak_grid, p.agg_states), small_k_decision, bounds_error=False, fill_value=None)
#########################################################################################################
# TODO: Hot fix, don't know why KS at start is not going correct, maybe the steady state values and distribution
# from BKM is slightly different, so I just pre-loop so distribution get to steady state
for not_used_i in range(1,200):
    tmp_coor =np.meshgrid(p.k_grid, p.ind_states, agg_k_ks[0], agg_shocks[-1])
    tmp_coor = np.meshgrid(tmp_coor, indexing="ij")
    coors =  np.reshape(tmp_coor, (4, -1), order="C").T  # dimension is  (NA*NZ) x 4
    chosen_this_small_k = interp_small_k(coors).T
    chosen_this_small_k[chosen_this_small_k<0] = 0
    dist_path_ks[:, 0] = calc_dist(p.NK, p.NY, dist_path_ks[:, 0], p.ymarkov, p.k_grid, chosen_this_small_k)
    agg_k_ks[0] = np.sum(dist_path_ks[:, 0]*chosen_this_small_k)

# ["prod", "ak", "i", "c", "r", "wage", "y"]
Kss = agg_k_ks[0]
Iss = Kss*p.delta  # TODO: from law of motion of capital
Css = Kss**p.alpha-Iss
ss_list_ks = np.array([1, Kss, Kss**p.alpha-Css, Css, p.alpha*Kss**(p.alpha-1)+1. - p.delta, (1-p.alpha)*Kss**(p.alpha), Kss**p.alpha])
log_ss_ks = np.log(ss_list_ks)
################################################################################

for tidx in range(1,T):
    tmp_coor =np.meshgrid(p.k_grid, p.ind_states, agg_k_ks[tidx - 1], agg_shocks[tidx - 1])
    tmp_coor = np.meshgrid(tmp_coor, indexing="ij")
    coors =  np.reshape(tmp_coor, (4, -1), order="C").T  # dimension is  (NA*NZ) x 4
    chosen_this_small_k = interp_small_k(coors).T
    chosen_this_small_k[chosen_this_small_k<0] = 0
    dist_path_ks[:, tidx] = calc_dist(p.NK, p.NY, dist_path_ks[:, tidx-1], p.ymarkov, p.k_grid, chosen_this_small_k)
    agg_k_ks[tidx] = np.sum(dist_path_ks[:, tidx-1]*chosen_this_small_k)
    agg_i_ks[tidx] = (agg_k_ks[tidx] - agg_k_ks[tidx - 1]) + p.delta * agg_k_ks[tidx - 1]

agg_i_ks[0] = Iss
agg_shocks_ks = agg_shocks
agg_y_ks = agg_shocks_ks*agg_k_ks**p.alpha
agg_c_ks = agg_y_ks - agg_i_ks
interest_ks = 1+ p.alpha*agg_shocks_ks*agg_k_ks**(p.alpha - 1) - p.delta  # r
wage_ks = (1-p.alpha)*agg_shocks_ks*agg_k_ks**p.alpha
KS_list = np.zeros((len(var_list), T))
KS_list[:, :] = [agg_shocks_ks, agg_k_ks, agg_i_ks, agg_c_ks, interest_ks, wage_ks, agg_y_ks]
KS_logged =np.subtract(np.log(KS_list).T,  log_ss_ks).T

# BKM is already IRF
# Draw IRF
for idx, it in enumerate(var_list):
    fig = plt.figure(idx)
    # fig.canvas.set_window_title(int(idx))
    plt.plot(range(0, T), KS_logged[idx, :], color='blue', label='KS')
    plt.plot(range(0, T), bkm_normalized[idx, :-50]*shock, color='orange', label='BKM')
    plt.legend()

    fname = 'IRF_{}'.format(var_list[idx])
    plt.suptitle(fname)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.legend()
    plt.savefig('data_output/{}.png'.format(fname))

plt.show()




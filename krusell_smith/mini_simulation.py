import numpy as np
import matplotlib.pyplot as plt
from .ks_util import *

# simulation and get variance
T = 2000
NZ = 2
delta = .06
alpha = .36
rng = np.random.default_rng(1000)
amarkov = np.genfromtxt("../krusell_smith/input/amarkov.txt")
states = np.array([.9832, 1.0157])
agg_c = np.zeros(T)
agg_i = np.zeros(T)
agg_k = np.zeros(T)
agg_y = np.zeros(T)
agg_a = np.zeros(T)


def log_normalize(ss, res_arr):
    lss = np.log(ss)  # normalize all to starting point
    res_arr = np.log(res_arr) - lss
    return res_arr

kss = 4.1858
iss = 0.254863
yss = kss**0.36
css = yss- iss
agg_k[0] = kss

# assume markov matrix
# agg_shocks = np.zeros(T).astype(np.int8)
# def ks_forcast(good, current_k):
#     slope_g =0.91772906
#     slope_b =0.93611402
#     inter_g = 0.11780762
#     inter_b = 0.09135248
#     if good:
#         return np.exp(np.log(current_k) * slope_g + inter_g)  # log normalized
#     else:
#         return np.exp(np.log(current_k) * slope_b + inter_b)
# agg_shocks[0] = rng.choice(range(NZ)).astype(np.int8)
# for tidx in range(1, T):
#     agg_shocks[tidx] = rng.choice(range(NZ), p=amarkov[agg_shocks[tidx - 1], :])
# assume AR1 process
def ks_forcast(productivity, current_k):
    slope =np.array([0.92064057, 0.95586539])
    intercept = np.array([0.10843067, 0.06036801])
    ks_states = np.array([.9832, 1.0157])
    # interpolate
    interp_slope = np.interp(productivity, ks_states, slope)
    interp_intercept = np.interp(productivity, ks_states, intercept)
    return np.exp(np.log(current_k) * interp_slope + interp_intercept)  # log normalized
agg_shocks = np.ones(T)
sigma = .012
rho = .9
mu = -sigma**2/2
agg_shocks[0] = 1.
for tidx in range(1, T):
    agg_shocks[tidx] = np.exp(rho*np.log(agg_shocks[tidx - 1]) + rng.normal(mu, sigma))

print(agg_shocks[:50], "agg")
print("============================================")
# markov
# agg_a = states[agg_shocks]
agg_a = agg_shocks  # ar1
p = param()
r = res()
init_shocks(p, r)
get_prices(p, r)
r.sim_ak = 
for tidx in range(1, T):
    
    agg_k[tidx] = ks_forcast(agg_shocks[tidx - 1], agg_k[tidx - 1])
    agg_i[tidx] = (agg_k[tidx] - agg_k[tidx - 1]) + delta * agg_k[tidx - 1]
    agg_y[tidx] = agg_a[tidx]*(agg_k[tidx]**alpha)
    agg_c[tidx] = agg_y[tidx] - agg_i[tidx]

agg_k = log_normalize(np.mean(agg_k), agg_k)
agg_c = log_normalize(np.mean(agg_c), agg_c)
agg_i = log_normalize(np.mean(agg_i), agg_i)
agg_y = log_normalize(np.mean(agg_y), agg_y)

drop = 1
variances = np.zeros(3)
variances[0] = np.var(agg_c[drop:])
variances[1] = np.var(agg_y[drop:])
variances[2] = np.var(agg_i[drop:])

covar = {0:None, 1:None, 2:None}
Xc = np.stack((agg_c[drop:], agg_a[drop:]), axis=0)
covar[0] = np.cov(Xc)
Xy = np.stack((agg_y[drop:], agg_a[drop:]), axis=0)
covar[1] = np.cov(Xc)
Xi = np.stack((agg_i[drop:], agg_a[drop:]), axis=0)
covar[2] = np.cov(Xc)
corr = {0:None, 1:None, 2:None}
corr[0] = np.corrcoef(agg_c[drop:], agg_a[drop:])
corr[1] = np.corrcoef(agg_y[drop:], agg_a[drop:])
corr[2] = np.corrcoef(agg_i[drop:], agg_a[drop:])
print("=============Krusell Smith================")

print("var", *variances)
print("=============================")
print(*covar.items(), "cov")
print("=============================")
print("corr", *corr.items())
print("=============================")

# BKM - positive 1 as numeraire
drop = 1   # drop the last periods weird behaviour
dir = "../bkm/data_output/"
k_res = np.genfromtxt(dir+"ak_path.txt")[:-drop]
c_res = np.genfromtxt(dir+"c_path.txt")[:-drop]
i_res = np.genfromtxt(dir+"i_path.txt")[:-drop]
y_res = np.genfromtxt(dir+"y_path.txt")[:-drop]
effect_len = int(len(k_res))
state_resd = np.log(states)   # we care about deviate from norm
bkm_agg_c = np.zeros(T)
bkm_agg_i = np.zeros(T)
bkm_agg_k = np.zeros(T)
bkm_agg_y = np.zeros(T)
bkm_li = [bkm_agg_c, bkm_agg_i, bkm_agg_k, bkm_agg_y]
kss = k_res[-1] # should be 4.1858, but there are errors
iss = i_res[-1] # 0.254863
yss = kss**alpha
css = yss-iss
shock = .15

k_res = log_normalize(kss, k_res)/shock
i_res = log_normalize(iss, i_res)/shock
y_res = log_normalize(yss, y_res)/shock
c_res = log_normalize(css, c_res)/shock

for tidx in range(0, T):
    # markov
    # s = np.log(state_resd[agg_shocks[tidx]])
    # AR 1
    s = np.log(agg_shocks[tidx])
    if T > tidx+effect_len:
        bkm_agg_k[tidx:tidx + effect_len] += k_res * s
        bkm_agg_c[tidx:tidx + effect_len] += c_res * s
        bkm_agg_i[tidx:tidx + effect_len] += i_res * s
        bkm_agg_y[tidx:tidx + effect_len] += y_res * s
    else:  #
        remain = T - tidx
        bkm_agg_k[tidx:] += k_res[:remain]*s
        bkm_agg_c[tidx:] += c_res[:remain]*s
        bkm_agg_i[tidx:] += i_res[:remain]*s
        bkm_agg_y[tidx:] += y_res[:remain]*s

drop = 0
variances = np.zeros(3)
variances[0] = np.var(bkm_agg_c[drop:])
variances[1] = np.var(bkm_agg_y[drop:])
variances[2] = np.var(bkm_agg_i[drop:])

covar = {0: None, 1: None, 2: None}
Xc = np.stack((bkm_agg_c[drop:], agg_a[drop:]), axis=0)
covar[0] = np.cov(Xc)
Xy = np.stack((bkm_agg_y[drop:], agg_a[drop:]), axis=0)
covar[1] = np.cov(Xc)
Xi = np.stack((bkm_agg_i[drop:], agg_a[drop:]), axis=0)
covar[2] = np.cov(Xc)
corr = {0: None, 1: None, 2: None}
corr[0] = np.corrcoef(bkm_agg_c[drop:], agg_a[drop:])
corr[1] = np.corrcoef(bkm_agg_y[drop:], agg_a[drop:])
corr[2] = np.corrcoef(bkm_agg_i[drop:], agg_a[drop:])
print("============= BKM ================")

print("var", *variances)
print("=============================")
print(*covar.items(), "cov")
print("=============================")
print("corr", *corr.items())
print("=============================")

name = ["c", "i", "k", "y"]
drop = 500
for idx, it in enumerate([agg_c, agg_i, agg_k, agg_y]):
    fig = plt.figure(idx)
    # fig.canvas.set_window_title(int(idx))
    plt.plot(range(T-drop), it[drop:], color='blue', label='KS')
    plt.plot(range(T-drop), bkm_li[idx][drop:], color='orange', label='BKM')
    plt.legend()

    fname = 'compare_{}'.format(name[idx])
    plt.suptitle(fname)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.legend()
    # plt.savefig('data_output/{}.png'.format(fname))

plt.show()

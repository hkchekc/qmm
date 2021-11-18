import numpy as np
import matplotlib.pyplot as plt

# simulation and get variance
T = 5000
NZ = 2
delta = .06
rng = np.random.default_rng()
amarkov = np.genfromtxt("../krusell_smith/input/amarkov.txt")
states = np.array([.984569568887006, 1.01713588246477])
agg_c = np.zeros(T)
agg_i = np.zeros(T)
agg_k = np.zeros(T)
agg_y = np.zeros(T)
agg_a = np.zeros(T)

agg_k[0] = 7.350


def ks_forcast(good, current_k):
    slope_g =0.90282904
    slope_b = - 0.78135636
    inter_g = - 0.41778431
    inter_b =  - 0.97349898
    if good:
        return np.exp(np.log(current_k) * slope_g + inter_g)
    else:
        return np.exp(np.log(current_k) * slope_b + inter_b)


agg_shocks = np.zeros(T).astype(np.int8)
agg_shocks[0] = rng.choice(range(NZ)).astype(np.int8)
for tidx in range(1, T):
    agg_shocks[tidx] = rng.choice(range(NZ), p=amarkov[agg_shocks[tidx - 1], :])
agg_a = states[agg_shocks]
for tidx in range(1, T):
    agg_k[tidx] = ks_forcast(agg_a[tidx - 1], agg_k[tidx - 1])
    print(agg_k[tidx])
    agg_i[tidx] = (agg_k[tidx] - agg_k[tidx - 1]) + delta * agg_k[tidx - 1]
    agg_y[tidx] = agg_a[tidx]*agg_k[tidx]
    agg_c[tidx] = agg_y[tidx] - agg_i[tidx]

# agg_y = agg_a*agg_k
# agg_c = agg_y - agg_i
drop = 100
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
drop = 50   # drop the last periods weird behaviour
dir = "../bkm/data_output/"
k_res = np.genfromtxt(dir+"ak_path.txt")[:-drop]
c_res = np.genfromtxt(dir+"c_path.txt")[:-drop]
i_res = np.genfromtxt(dir+"i_path.txt")[:-drop]
y_res = np.genfromtxt(dir+"y_path.txt")[:-drop]
effect_len = int(len(k_res))
state_resd = states - 1  # we care about deviate from norm
bkm_agg_c = np.zeros(T)
bkm_agg_i = np.zeros(T)
bkm_agg_k = np.zeros(T)
bkm_agg_y = np.zeros(T)
bkm_li = [bkm_agg_c, bkm_agg_i, bkm_agg_k, bkm_agg_y]

bkm_agg_k[:] = 7.350  # normalize all to starting point
for tidx in range(1, T):
    s = agg_shocks[tidx]
    if T > tidx+effect_len:
        bkm_agg_k[tidx:tidx + effect_len] += k_res * state_resd[s]
        bkm_agg_c[tidx:tidx + effect_len] += c_res * state_resd[s]
        bkm_agg_i[tidx:tidx + effect_len] += i_res * state_resd[s]
        bkm_agg_y[tidx:tidx + effect_len] += y_res * state_resd[s]
    else:  #
        remain = T- tidx -1
        bkm_agg_k[tidx:] += k_res[remain]*state_resd[s]
        bkm_agg_c[tidx:] += c_res[remain]*state_resd[s]
        bkm_agg_i[tidx:] += i_res[remain]*state_resd[s]
        bkm_agg_y[tidx:] += y_res[remain]*state_resd[s]

drop = 100
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

for idx, it in enumerate([agg_c, agg_i, agg_k, agg_y]):
    fig = plt.figure(idx)
    # fig.canvas.set_window_title(int(idx))
    plt.plot(range(T), it, color='blue', label='KS')
    plt.plot(range(T), bkm_li[idx], color='orange', label='BKM')
    plt.legend()

    fname = 'compare_{}'.format(name[idx])
    plt.suptitle(fname)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.legend()
    plt.savefig('data_output/{}.png'.format(fname))

plt.show()

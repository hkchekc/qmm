import numpy as np
from numpy import random
from numba import njit
from scipy import interpolate
from statsmodels import api

class param:

    def __init__(self):
        self.rng = random.default_rng(seed=5000)
        self.beta = .917493
        self.gamma = 2.
        self.alpha = 0.36
        self.delta = .06
        self.k_max = 15.01
        self.k_min = 0.01
        self.ak_max = 15.01
        self.ak_min = 0.01
        self.ind_states = np.array([0.715494756670886, 1. , 1.39763428128095])
        self.agg_states = np.array([.984569568887006, 1.01713588246477])
        self.NH = 400
        self.NT = 5000
        self.DROP = 20
        self.NZ = self.agg_states.size
        self.NY = self.ind_states.size
        self.NK = 200
        self.NAK = 20
        self.k_grid = np.linspace(self.k_min, self.k_max, self.NK)
        self.ak_grid = np.linspace(self.ak_min, self.ak_max, self.NAK)
        self.markov = np.genfromtxt("input/markov.txt")
        self.amarkov = np.genfromtxt("input/amarkov.txt")
        self.ymarkov = np.genfromtxt("input/ymarkov.txt")
        for i in range(self.NY):
            self.ymarkov[i, :] /= sum(self.ymarkov[i, :])
        # len = t- drop
        self.agg_shocks = np.zeros([self.NT]).astype(np.int8)
        self.ind_shocks = np.zeros([self.NT, self.NH]).astype(np.int8)
        # crit
        self.crit = 1e-3
        self.vfi_crit = 1e-2


class res:

    def __init__(self):
        # initial guess
        p = param()
        self.intercept = np.array([-0.005, 0.15])
        self.slope = np.array([.84, 0.95])
        self.interst =  np.zeros((p.NAK, p.NZ))
        self.wage =  np.zeros((p.NAK, p.NZ))
        self.vfunc = None
        self.pfunc = None
        self.error = 0.
        self.sim_ak = np.zeros(p.NT)
        self.sim_small_k = np.zeros([p.NT, p.NH])
        self.count_good = 0
        self.count_bad = 0


# void
def init_shocks(p, r):
    p.agg_shocks[0] = p.rng.choice(range(p.NZ))
    p.ind_shocks[0, :] = p.rng.choice(range(p.NY), p.NH)
    for tidx in range(1, p. NT):
        p.agg_shocks[tidx] = p.rng.choice(range(p.NZ), p=p.amarkov[p.agg_shocks[tidx-1], :])
        for hidx in range(p.NH):
            p.ind_shocks[tidx, hidx] = p.rng.choice(range(p.NY), p=p.ymarkov[p.ind_shocks[tidx-1, hidx], :])
    p.agg_shocks = p.agg_shocks[p.DROP:]
    p.ind_shocks = p.ind_shocks[p.DROP:, :]
    # randomize initial period capital endowment
    r.sim_small_k[0, :] = p.rng.choice(p.k_grid, size=p.NH)
    r.sim_ak[0] = np.mean(r.sim_small_k[0, :])
    print(r.sim_ak[0], "===========================")

def get_prices(p, r):
    r.interest, r.wage = _get_prices(p.ak_grid, p.agg_states, p.NAK, p.NZ, p.alpha, p.delta)

@njit
def _get_prices(ak_grid, zgrid, NAK, NZ, alpha, delta):
    interest = np.zeros((NAK, NZ))
    wage = np.zeros((NAK, NZ))
    for aki in range(NAK):
        for zi in range(NZ):
            interest[aki, zi] = alpha*zgrid[zi]/(ak_grid[aki]**(1.-alpha)) + 1. - delta
            wage[aki, zi] = (1.-alpha) * zgrid[zi] / (ak_grid[aki] ** (alpha))
    return interest, wage

def vfi(p, r):
    ak_est = _est_agg_cap(p.ak_grid, p.NK, p.NY, p.NAK, p.NZ, r.intercept, r.slope)
    coh_arr = _get_coh(p.NK, p.NY, p.NAK, p.NZ, r.interest, p.ind_states, p.k_grid, r.wage)
    consum_arr, util_arr = _get_consum_arr(p.NK, p.NY, p.NAK, p.NZ, coh_arr, p.k_grid, p.gamma)

    err_vfi = 100.
    r.vfunc = np.amax(util_arr, axis=4)  # initial guess
    r.pfunc = np.zeros((p.NK, p.NY, p.NAK, p.NZ)).astype(np.int8)
    while err_vfi > p.vfi_crit:
        err_vfi, r.pfunc, r.vfunc = _vfi(r.vfunc, ak_est, util_arr, p.beta, p.ak_grid, p.NK, p.NY, p.NAK, p.NZ, p.markov)
        print(err_vfi)

def pseudo_panel(p, r):
    net_time = p.NT - p.DROP
    k_decision = p.k_grid[r.pfunc]
    r.sim_ak[1:] = 0
    r.sim_small_k[1:, :] = 0
    r.sim_ak, r.sim_small_k = _pseudo_panel(net_time, p.NY, p.agg_shocks, p.ind_shocks,
                                            p.k_grid, p.ak_grid, k_decision, r.sim_small_k, r.sim_ak)

def calc_errors(p, r):
    # sort by good =1 and bad=0
    tmp_idx = np.arange(p.NT-p.DROP)[p.agg_shocks == 1]
    tmp_idx = tmp_idx[:-1]  # drop to avoid out of bound, dropping one observation is ok I think
    good_ak  = r.sim_ak[tmp_idx]
    tmp_idx += 1
    good_next_ak = r.sim_ak[tmp_idx]
    # same thing for bad states
    tmp_idx = np.arange(p.NT-p.DROP)[p.agg_shocks == 0]
    tmp_idx = tmp_idx[:-1]  # drop to avoid out of bound, dropping one observation is ok I think
    bad_ak = r.sim_ak[tmp_idx]
    tmp_idx += 1
    bad_next_ak = r.sim_ak[tmp_idx]
    r.error = 0
    good_x = api.add_constant(np.log(good_ak))
    model_good = api.OLS(np.log(good_next_ak), good_x)
    results = model_good.fit()
    r.error = results.rsquared
    ratio = 1
    r.slope[1] = ratio*results.params[1]+(1-ratio)*r.slope[1]
    r.intercept[1] = ratio*results.params[0]+(1-ratio)*r.intercept[0]
    r.error = results.rsquared
    bad_x = api.add_constant(np.log(bad_ak))
    model_bad = api.OLS(np.log(bad_next_ak), bad_x)
    results = model_bad.fit()
    bad_rsq = results.rsquared

    r.slope[0] = ratio*results.params[1]+(1-ratio)*r.slope[0]
    r.intercept[0] = ratio*results.params[0]+(1-ratio)*r.intercept[0]
    r.error = max(bad_rsq, r.error)
    print("bad/good intercepts -> ",r.intercept)
    print("bad/good slopes -> ",r.slope)
    print("r_squares", r.error)


def _est_agg_cap(ak_grid, NK, NY, NAK, NZ, intercept, slope):
    next_ak = np.zeros((NK, NY, NAK, NZ))
    for aki in range(NAK):
        for zi in range(NZ):
            next_ak[:, :, aki, zi] = np.exp(intercept[zi]+slope[zi]*np.log(ak_grid[aki]))
    mx = next_ak < ak_grid[0]
    next_ak[mx] = ak_grid[0]
    mx = next_ak > ak_grid[-1]
    next_ak[mx] = ak_grid[-1]
    return next_ak

@njit
def _get_coh(NK, NY, NAK, NZ, interest_arr, ind_states, k_grid, wage_arr):
    coh_arr = np.zeros((NK, NY, NAK, NZ))
    for ki in range(NK):
        tmp_cap_gain = interest_arr*k_grid[ki]
        for yi in range(NY):
            coh_arr[ki, yi, :, :] = tmp_cap_gain + wage_arr*ind_states[yi]
    return coh_arr

def  _get_consum_arr(NK, NY, NAK, NZ, coh_arr, k_grid, gamma):
    consum_arr = np.zeros((NK, NY, NAK, NZ, NK))
    for ki, k in enumerate(k_grid):
        consum_arr[:, :, :, :, ki] = coh_arr - k
    util_arr = np.ones((NK, NY, NAK, NZ, NK)) * -1e-3
    mx = consum_arr > 0
    util_arr[mx] = (np.power(consum_arr[mx], 1-gamma) -1.)/(1-gamma)
    return consum_arr, util_arr

def _vfi(vfunc, ak_est, util_arr, beta, ak_grid, NK, NY, NAK, NZ, markov):
    vfunc_tmp = _construct_next_vfunc_tmp(vfunc, ak_est, util_arr, beta, ak_grid, NK, NY, NAK, NZ, markov)
    pfunc = np.argmax(vfunc_tmp, axis=4)
    vfunc_new = np.amax(vfunc_tmp, axis=4)  # improve efficiency bt using pfunc
    error = np.max(np.abs(vfunc_new - vfunc))
    return error, pfunc, vfunc_new

@njit
def _construct_next_vfunc_tmp(vfunc, ak_est, util_arr, beta, ak_grid, NK, NY, NAK, NZ, markov):
    exp_next_u = np.zeros((NK, NY, NAK, NZ, NK))
    for zi in range(NZ):
        for yi in range(NY):
            this_idx = zi*NY +yi
            for kpi in range(NK):
                for aki in range(NAK):
                    for nzi in range(NZ):
                        for nyi in range(NY):
                            next_idx = nzi*NY +nyi
                            exp_next_u[:, yi, aki, zi, kpi] += markov[this_idx,
                                        next_idx]*np.interp(ak_est[0, yi, aki, zi], ak_grid, vfunc[kpi, nyi, :, nzi])

    return util_arr + beta*exp_next_u  # 5 dimensional object

def _pseudo_panel(net_time, NY, agg_shock, ind_shock,k_grid, ak_grid, k_decision, sim_small_k, sim_ak):
    for tidx in range(1, net_time):
        zi = agg_shock[tidx-1]
        last_ak = sim_ak[tidx-1]
        for yi in range(NY):
            interpolate_f = interpolate.interp2d(ak_grid, k_grid, k_decision[:, yi, :, zi], kind="linear")
            mx = ind_shock[tidx-1, :] == yi
            last_small_k = sim_small_k[tidx - 1, :][mx]
            sim_small_k[tidx, :][mx] = interpolate_f(last_small_k, last_ak)
        sim_ak[tidx] = np.mean(sim_small_k[tidx, :])
    return sim_ak, sim_small_k
import numpy as np
from numpy import random
from numba import njit, jit
from scipy import interpolate
from statsmodels import api

class param:

    def __init__(self):
        self.rng = random.default_rng(seed=5000)
        self.beta = .917473
        self.gamma = 2.
        self.alpha = 0.36
        self.delta = .06
        self.k_max = 22.
        self.k_min = 0.
        self.ak_max = 8.3716
        self.ak_min = 2.0929
        self.ind_states = np.array([0.690935359826633, 0.965674945042889 , 1.349660407766035])
        self.agg_states = np.array([.9832, 1.0157])
        self.NH = 3000
        self.NT = 1500
        self.DROP = 20
        self.NZ = self.agg_states.size
        self.NY = self.ind_states.size
        self.NK = 1000
        self.NAK = 25
        self.k_grid = np.exp(np.linspace(self.k_min, np.log(self.k_max-self.k_min+1.), self.NK)) - 1. + self.k_min
        self.k_grid = np.linspace(self.k_min, self.k_max, self.NK)  # regular grid
        self.ak_grid = np.linspace(self.ak_min, self.ak_max, self.NAK)
        self.markov = np.genfromtxt("input/markov.txt")
        self.amarkov = np.genfromtxt("input/amarkov.txt")
        self.ymarkov = np.genfromtxt("input/ymarkov.txt")
        for i in range(self.NY):
            self.ymarkov[i, :] /= sum(self.ymarkov[i, :])
        self.agg_shocks = np.zeros([self.NT]).astype(np.int8)
        self.ind_shocks = np.zeros([self.NT, self.NH]).astype(np.int8)


class res:

    def __init__(self):
        # initial guess
        p = param()
        self.it = 0
        self.intercept = np.array([0.01, 0.01])
        self.slope = np.array([.99, 0.99])
        self.interest = np.zeros((p.NAK, p.NZ))
        self.wage = np.zeros((p.NAK, p.NZ))
        self.vfunc = None
        self.pfunc = None
        self.error = 100.
        self.rsq = 0.
        self.sim_ak = np.zeros(p.NT)
        self.sim_small_k = np.zeros([p.NT, p.NH])


# void
def init_shocks(p, r):
    p.agg_shocks[0] = p.rng.choice(range(p.NZ))
    p.ind_shocks[0, :] = p.rng.choice(range(p.NY), p.NH)
    for tidx in range(1, p. NT):
        p.agg_shocks[tidx] = p.rng.choice(range(p.NZ), p=p.amarkov[p.agg_shocks[tidx-1], :])
        for yidx in range(p.NY):
            mx = p.ind_shocks[tidx-1, :] == yidx
            count = np.sum(mx)
            p.ind_shocks[tidx, :][mx] = p.rng.choice(range(p.NY), count,p=p.ymarkov[yidx, :])
        # for hidx in range(p.NH):
        #     p.ind_shocks[tidx, hidx] = p.rng.choice(range(p.NY), p=p.ymarkov[p.ind_shocks[tidx-1, hidx], :])
    p.agg_shocks = p.agg_shocks[p.DROP:]
    p.ind_shocks = p.ind_shocks[p.DROP:, :]
    # randomize initial period capital endowment
    r.sim_small_k[0, :] = p.rng.choice(p.k_grid, size=p.NH)
    r.sim_ak[0] = np.mean(r.sim_small_k[0, :])
    # print(r.sim_ak[0], "===========================")
    # print(p.ind_shocks[-1, :] == 0)

def get_prices(p, r):
    r.interest, r.wage = _get_prices(p.ak_grid, p.agg_states, p.NAK, p.NZ, p.alpha, p.delta)

@njit
def _get_prices(ak_grid, zgrid, NAK, NZ, alpha, delta):
    interest = np.zeros((NAK, NZ))
    wage = np.zeros((NAK, NZ))
    for aki in range(NAK):
        for zi in range(NZ):
            interest[aki, zi] = alpha*zgrid[zi]/(ak_grid[aki]**(1.-alpha)) + 1. - delta
            wage[aki, zi] = (1.-alpha) * zgrid[zi]* (ak_grid[aki] ** (alpha))
    return interest, wage

def vfi(p, r):
    ak_est = _est_agg_cap(p.ak_grid, p.NAK, p.NZ, r.intercept, r.slope)
    mx = ak_est < p.ak_grid[0]
    ak_est[mx] = p.ak_grid[0]
    mx = ak_est > p.ak_grid[-1]
    ak_est[mx] = p.ak_grid[-1]
    coh_arr = _get_coh(p.NK, p.NY, p.NAK, p.NZ, r.interest, p.ind_states, p.k_grid, r.wage)
    consum_arr, util_arr = _get_consum_arr(p.NK, p.NY, p.NAK, p.NZ, coh_arr, p.k_grid)
    mx = consum_arr > 0
    util_arr[mx] = (np.power(consum_arr[mx], 1. - p.gamma))/(1-p.gamma)

    err_vfi = 100.
    if r.it == 0:
        r.vfunc = np.amax(util_arr, axis=4)  # initial guess
        r.pfunc = np.zeros((p.NK, p.NY, p.NAK, p.NZ)).astype(np.int16)
    while err_vfi > 1e-6:
        err_vfi, r.pfunc, r.vfunc = _vfi(r.vfunc, ak_est, util_arr, p.beta, p.ak_grid, p.NK, p.NY, p.NAK, p.NZ, p.markov, p.amarkov, p.ymarkov)
        # print(err_vfi)
        # print(p.k_grid[r.pfunc[0,0,0,0]])
        # print(p.k_grid[r.pfunc[-1,-1,-1,-1]])

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
    # ols
    good_x = api.add_constant(np.log(good_ak))
    model_good = api.OLS(np.log(good_next_ak), good_x)
    results = model_good.fit()
    r.rsq = results.rsquared
    r.error = abs(results.params[1] - r.slope[1])
    ratio = 1
    r.slope[1] = ratio*results.params[1]+(1-ratio)*r.slope[1]
    r.intercept[1] = ratio*results.params[0]+(1-ratio)*r.intercept[0]
    # same thing for bad
    bad_x = api.add_constant(np.log(bad_ak))
    model_bad = api.OLS(np.log(bad_next_ak), bad_x)
    results = model_bad.fit()
    r.rsq = max(results.rsquared, r.rsq)
    r.error = max(r.error, abs(results.params[1] - r.slope[0]))

    r.slope[0] = ratio*results.params[1]+(1-ratio)*r.slope[0]
    r.intercept[0] = ratio*results.params[0]+(1-ratio)*r.intercept[0]
    print("bad/good intercepts -> ",r.intercept)
    print("bad/good slopes -> ",r.slope)
    print("error_param, r_squares", r.error, r.rsq)

# not masked
@njit
def _est_agg_cap(ak_grid, NAK, NZ, intercept, slope):
    next_ak = np.zeros((NAK, NZ))
    for aki in range(NAK):
        for zi in range(NZ):
            next_ak[aki, zi] = np.exp(intercept[zi]+slope[zi]*np.log(ak_grid[aki]))
    return next_ak

@njit
def _get_coh(NK, NY, NAK, NZ, interest_arr, ind_states, k_grid, wage_arr):
    coh_arr = np.zeros((NK, NY, NAK, NZ))
    # for ki in range(NK):
    #     tmp_cap_gain = interest_arr*k_grid[ki]
    #     for yi in range(NY):
    #         coh_arr[ki, yi, :, :] = tmp_cap_gain + wage_arr*ind_states[yi]
    for ki in range(NK):
        tmp_cap_gain = interest_arr*k_grid[ki]  # interest - (NAK x NZ)
        for yi in range(NY):
            coh_arr[ki, yi, :, :] = tmp_cap_gain + wage_arr*ind_states[yi]
    return coh_arr

@njit
def  _get_consum_arr(NK, NY, NAK, NZ, coh_arr, k_grid):
    consum_arr = np.zeros((NK, NY, NAK, NZ, NK))
    for ki, k in enumerate(k_grid):
        consum_arr[:, :, :, :, ki] = coh_arr - k
    util_arr = np.ones((NK, NY, NAK, NZ, NK)) * -1e3
    return consum_arr, util_arr

def _vfi(vfunc, ak_est, util_arr, beta, ak_grid, NK, NY, NAK, NZ, markov, zmarkov, ymarkov):
    exp_next_u_no_current = _construct_next_vfunc_tmp(vfunc, ak_est, ak_grid, NK, NY, NAK, NZ, markov, zmarkov, ymarkov)
    exp_next_u =np.tile(exp_next_u_no_current, (NK, 1, 1, 1, 1))
    vfunc_tmp = util_arr + beta*exp_next_u
    pfunc = np.argmax(vfunc_tmp, axis=4)
    vfunc_new = np.amax(vfunc_tmp, axis=4)  # improve efficiency by using pfunc
    error = np.max(np.abs(vfunc_new - vfunc))
    return error, pfunc, vfunc_new

@njit
def _construct_next_vfunc_tmp(vfunc, ak_est, ak_grid, NK, NY, NAK, NZ, markov, zmarkov, ymarkov):
    exp_next_u_no_current = np.zeros((NY, NAK, NZ, NK))
    # tmp_arr = _contstruct_tmr_vfunc(NK, NAK, NZ, NY, ak_est, ak_grid, vfunc)
    tmp_arr = _contstruct_tmr_vfunc2(NK, NAK, NZ, NY, ak_est, ak_grid, vfunc, zmarkov).reshape(NY, NAK*NZ*NK)
    exp_next_u_no_current = np.dot(ymarkov, tmp_arr).reshape(NY, NAK, NZ, NK)
    # exp_next_u_no_current = np.tensordot(markov, tmp_arr, axes=1)
    # exp_next_u_no_current = exp_next_u_no_current[:NY, :, :, :]
    #
    # for nzi in range(NZ):
    #     for nyi in range(NY):
    #         next_idx = nzi * NY + nyi
    #         for kpi in range(NK):
    #             for aki in range(NAK):
    #                 for zi in range(NZ):
    #                     tmp = tmp_arr[next_idx, aki, zi, kpi]
    #                     for yi in range(NY):
    #                         this_idx = zi * NY + yi
    #                         exp_next_u_no_current[yi, aki, zi, kpi] += markov[this_idx, next_idx]*tmp
    return exp_next_u_no_current # 4 dimensional object (current k doesn't matter )

@njit
def _contstruct_tmr_vfunc(NK, NAK, NZ, NY, ak_est, ak_grid, vfunc):
    tmr_vfunc = np.zeros((NY*NZ, NAK, NZ, NK))
    for nzi in range(NZ):
        for nyi in range(NY):
            next_idx = nzi * NY + nyi
            for kpi in range(NK):
                tmr_vfunc[next_idx, :, :, kpi] = np.interp(ak_est, ak_grid, vfunc[kpi, nyi, :, nzi])
    return tmr_vfunc

@njit
def _contstruct_tmr_vfunc2(NK, NAK, NZ, NY, ak_est, ak_grid, vfunc, zmarkov):
    tmr_vfunc = np.zeros((NY, NAK, NZ, NK))
    for nzi in range(NZ):
        for nyi in range(NY):
            for kpi in range(NK):
                this_val = np.interp(ak_est, ak_grid, vfunc[kpi, nyi, :, nzi])  # NAK x NZ (current z) wrong
                tmr_vfunc[nyi, :, :, kpi] += this_val[:, :]*zmarkov[:, nzi]
    return tmr_vfunc

def _pseudo_panel(net_time, NY, agg_shock, ind_shock,k_grid, ak_grid, k_decision, sim_small_k, sim_ak):
    for tidx in range(1, net_time):
        zi = agg_shock[tidx-1]
        last_ak = sim_ak[tidx-1]
        for yi in range(NY):
            # flipped arond
            interpolate_f = interpolate.interp2d(ak_grid, k_grid, k_decision[:, yi, :, zi], kind="linear")
            mx = ind_shock[tidx-1, :] == yi
            last_small_k = sim_small_k[tidx - 1, :][mx]
            try:
                sim_small_k[tidx, :][mx] = interpolate_f(last_small_k, last_ak)
            except ValueError:
                print(sim_small_k[tidx, :][mx], tidx)
        sim_ak[tidx] = np.mean(sim_small_k[tidx, :])
    return sim_ak, sim_small_k

def calc_dist(NA, NY, dist, ymarkov, k_grid, kdecision):
    # Young's method
    # pos of last element that is smaller than kdecision
    # TODO: maybe wrong with corner cases, but shouldn't be a problem as steps are small anyway
    ceil_arr = np.searchsorted(k_grid, kdecision, side='right')
    floor_arr = ceil_arr - 1
    ceil_arr[ceil_arr == NA] = NA -1  
    floor_arr[floor_arr < 0] = 0
    # weight put on floor !
    weight_arr = (k_grid[ceil_arr] - kdecision)/(k_grid[ceil_arr] - k_grid[floor_arr])
    weight_arr[floor_arr == NA - 1] = 1
    # print(weight_arr)
    # print(ceil_arr)
    # print(floor_arr)
    # print(kdecision)
    # print(k_grid[ceil_arr])
    new_dist = np.zeros((NA, NY)).astype(np.float32)
    for ai in range(NA):  # next choice is ai
        rel_floor = floor_arr == ai
        rel_ceil = ceil_arr == ai
        floor_dist = np.matmul((dist*rel_floor.astype(np.float32)*weight_arr).reshape((NY, NA)).T, ymarkov)
        ceil_dist = np.matmul((dist*rel_ceil.astype(np.float32)*(1-weight_arr)).reshape((NY, NA)).T, ymarkov)
        new_dist[ai, :] += np.sum(floor_dist, axis=0)
        new_dist[ai, :] += np.sum(ceil_dist, axis=0)
        # print("floor is non-zero: {}".format(np.sum(np.count_nonzero(rel_floor))))
    return new_dist.T.reshape(NA*NY)
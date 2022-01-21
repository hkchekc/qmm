import numpy as np
from numba import jit, njit
from scipy import interpolate

class Param:

    def __init__(self):
        self.alpha = 1.  # EV shock
        self.beta = .8  # discount
        self.interest = 1.1  # gross interest
        self.euler_gamma = .5772  # euler's constant for EV shock
        self.tau = 0.  # default cost
        self.states = np.array([.2, 1.2])
        self.markov = np.array([[.2, .8],[.2, .8]])
        self.NB = 150
        self.NZ = len(self.states)
        self.b_grid = np.linspace(0., 2., num=self.NB)
        # Functions Parameters
        self.egm = True
        self.EV_shock = True
        self.crit_q = 1e-3
        self.crit_vfi = 1e-3
        self.update_ratio = 1


class Res:

    def __init__(self):
        param = Param()
        self.vfunc_clean = np.zeros((param.NB, param.NZ))
        self.vfunc_o = np.zeros((param.NB, param.NZ))
        self.q = np.ones((param.NB, param.NZ))*.9
        self.dfunc = np.zeros((param.NB, param.NZ)).astype(np.int8)
        self.vfunc_def = np.zeros(param.NZ)
        self.err_q = 100.
        if param.egm:
            self.consum_arr = np.zeros((param.NB, param.NZ))


def vfi(r, p):
    error_vfi = 100.
    it = 0
    while error_vfi > p.crit_vfi:
        vfunc_clean_new = _bellman_clean(p.NZ, p.NB, p.states, r.q, p.b_grid, p.markov, r.vfunc_o, p.beta, p.EV_shock, r.vfunc_def, r.vfunc_clean)
        vfunc_def_new = _bellman_default(p.NZ, p.states, p.tau, p.beta, p.markov, r.vfunc_def)
        if p.EV_shock:
            r.vfunc_o, r.dfunc = _pop_vfunc_o_shock(vfunc_def_new, vfunc_clean_new, p.NB, p.NZ, p.euler_gamma, p.alpha)
        else:
            r.vfunc_o, r.dfunc = _pop_vfunc_o_no_shock(vfunc_def_new, vfunc_clean_new, p.NB, p.NZ)
        error_clean = np.max(np.abs(vfunc_clean_new-r.vfunc_clean))
        error_def = np.max(np.abs(vfunc_def_new-r.vfunc_def))
        error_vfi = max(error_clean, error_def)
        r.vfunc_def = vfunc_def_new
        r.vfunc_clean = vfunc_clean_new
        it += 1
        if it % 100 == 0:
            print("Error of VFI at loop {} is {}".format(it, error_vfi))

def calc_q(r, p):
    q_zero_arr = np.zeros((p.NB, p.NZ))
    for nbi in range(p.NB):
        for zi in range(p.NZ):
            risk = 0.
            if p.EV_shock:
                for nzi in range(p.NZ):
                    risk += p.markov[zi, nzi]/(1. + np.exp(p.alpha*(r.vfunc_def[nzi]-r.vfunc_clean[nbi,nzi])))
                q_zero_arr[nbi, zi] = risk / p.interest * p.b_grid[nbi]
            else:
                for nzi in range(p.NZ):
                    risk += p.markov[zi, nzi]*r.dfunc[nbi, nzi]
                q_zero_arr[nbi, zi] = (1. - risk)/p.interest * p.b_grid[nbi]
    r.err_q = np.max(np.abs(r.q - q_zero_arr))
    r.q[:, :] = q_zero_arr


def egm(r, p):
    error_vfi = 100.
    it = 0
    # populate v_d
    for zi in range(p.NZ):
        tmp = -p.markov[0,0]/(p.states[0]-p.tau) - p.markov[0,1]/(p.states[1]-p.tau)
        r.vfunc_def[zi] = -1./(p.states[zi]-p.tau) + p.beta*tmp/(1-p.beta)
    r.vfunc_o = np.zeros((p.NB, p.NZ))
    # initial guess consumption
    for zi in range(p.NZ):
        r.consum_arr[:, zi] = p.states[zi] + p.b_grid[:]*(1/p.interest - 1.)
    # r.consum_arr[r.consum_arr <= 0.] = 0.01
    # start iteration
    while error_vfi > p.crit_vfi:
        calc_q(r, p)
        vfunc_clean_new, consum_arr_new = _find_next_c(p.NB, p.NZ, p.alpha, p.markov,
                    p.interest, p.beta, p.states, p.b_grid, r.consum_arr, r.vfunc_def, r.vfunc_clean, r.q, r.vfunc_o)
        vfunc_o_new, r.dfunc = _pop_vfunc_o_shock(r.vfunc_def, vfunc_clean_new, p.NB, p.NZ, p.euler_gamma, p.alpha)
        # Error and Updating
        error_vfi = np.max(np.abs(vfunc_o_new-r.vfunc_o))
        r.vfunc_o = p.update_ratio*vfunc_o_new+(1-p.update_ratio)*r.vfunc_o
        r.vfunc_clean = p.update_ratio*vfunc_clean_new+(1-p.update_ratio)*r.vfunc_clean
        r.consum_arr = p.update_ratio*consum_arr_new+(1-p.update_ratio)*r.consum_arr
        it += 1
        if it % 100 == 0:
            print("Error of VFI at loop {} is {}".format(it, error_vfi))


def u_prime(consum):
    return 1/consum**2 + (consum<=0)*100000

def util_func(consum):
    return -1/consum - (consum<=0)*100000

def _find_next_c(NB, NZ, alpha, markov, interest, beta, states, b_grid, consum_arr, vfunc_def, vfunc_clean, q, vfunc_o):
    exo_future_debt = np.zeros((NB, NZ))  # b'(b, y) policy function
    exo_consum = np.zeros((NB, NZ))  # c(b, y)  consumption policy
    prob_arr = _get_prob_arr(NB, NZ, alpha, vfunc_def, vfunc_clean)
    v_prime = markov[0, 0]*prob_arr[:, 0]*u_prime(consum_arr[:, 0]) + markov[0, 1] *prob_arr[:, 1]*u_prime(consum_arr[:, 1])  # EV'

    q_deriv_arr = _get_q_derivative(NB, NZ, alpha, markov, interest, b_grid, consum_arr, prob_arr)

    qb_mask = q_deriv_arr > 0.
    # populate consumption arr
    consum_arr_new = _get_consumption(NB, NZ, beta, markov, prob_arr, consum_arr,  q_deriv_arr)

    consum_arr_new = np.where(qb_mask, consum_arr_new, np.nan)  # keep shape while trimming
    q_masked = np.where(qb_mask, q, np.nan)
    vfunc_o_masked = np.where(qb_mask, vfunc_o, np.nan)
    # not all b' grid points are optimal, some are never optimal - check and drop those
    consum_arr_new, drop_mask = _non_concavity(NB, NZ, markov, beta, b_grid, states, vfunc_o_masked, q_masked, v_prime, consum_arr_new)

    implied_current_debt = np.zeros((NB, NZ))
    for zi in range(NZ):
        implied_current_debt[:, zi] = states[zi] + q_masked[:, zi] - consum_arr_new[:, zi]

    consum_mask = ~np.isnan(consum_arr_new[:, :])

    q_masked = np.where(consum_mask, q, np.nan)
    vfunc_o_masked = np.where(consum_mask, vfunc_o, np.nan)
    for zi in range(NZ):
        consum_arr_new[:, zi] = np.where(consum_arr_new[:, zi] > 0., consum_arr_new[:, zi], np.nan)  # consumption undefined or =0 etc.
        mask = ~np.isnan(consum_arr_new[:, zi])
        # map consumption from b' to b
        try:  # if only 1 element in the masked array (all nans)
            debt_f = interpolate.interp1d(implied_current_debt[:, zi][mask], b_grid[mask]) # no , fill_value="extrapolate"
            for bi in range(NB):  # perform extrapolation manually, falling back to VFI
                if b_grid[bi] < implied_current_debt[:, zi][mask][0] or b_grid[bi] > implied_current_debt[:, zi][mask][-1]:
                    future_bidx = np.argmax(util_func(states[zi]-b_grid[bi]+q_masked[:, zi])+beta*(markov[zi, 0]
                                                                    *vfunc_o_masked[:, 0]+markov[zi, 1]*vfunc_o_masked[:, 1]))
                    exo_future_debt[bi, zi] = b_grid[future_bidx]
                elif b_grid[bi] < implied_current_debt[0, zi]:
                    exo_future_debt[bi, zi] = b_grid[0]
                else:
                    exo_future_debt[bi, zi] = debt_f(b_grid[bi])
            # consum_f = interpolate.interp1d(implied_current_debt[:, zi][mask], consum_arr_new[:, zi][mask], fill_value="extrapolate")
            # print("Trues in mask after suboptimal drops {}".format(sum(mask)))
        except ValueError:
            pass
        ###############################################################################################################
        ##################################exogenous part - b' based to b ##############################################
        ###############################################################################################################
        # catch over-grid
        # exo_consum[:, zi] = consum_f(b_grid)
    vfunc_clean_new = np.zeros((NB, NZ)) # value function
    for zi in range(NZ):
        nb_idx_li = []
        for bi in range(NB):
            nb_idx_li.append(_find_nearest(b_grid, exo_future_debt[bi, zi]))
        exo_consum[:, zi] = states[zi] + q[nb_idx_li, zi] - b_grid
        another_mask = exo_consum[:, zi] > 0
        # exo_consum[:, zi][~another_mask] = .000001
        vfunc_clean_new[:, zi] = -100.
        # if best consumption not over 0, default
        vfunc_clean_new[:, zi][another_mask] = util_func(exo_consum[:, zi])[another_mask]+beta*(markov[zi, 0]*vfunc_o
        [nb_idx_li, 0]+markov[zi, 1]*vfunc_o[nb_idx_li, 1])[another_mask]

    return vfunc_clean_new, exo_consum


def _find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def _non_concavity(NB, NZ, markov, beta, b_grid, states, vfunc_o, q, v_prime, exo_consum):
    # find v max v_min
    # change method
    drop = np.ones((NB, NZ)).astype(np.int8)
    # populate concave_li
    concave_li = []
    non_concave_li = []
    if v_prime[0] < np.min(v_prime[1:]):
        concave_li.append(0)
    else:
        non_concave_li.append(0)
    for bi in range(1, NB-1):
        if np.max(v_prime[:bi]) < v_prime[bi] and v_prime[bi] < np.min(v_prime[bi+1:]):
            concave_li.append(bi)
        else:
            non_concave_li.append(bi)
    if v_prime[-1] > np.max(v_prime[:-1]):
        concave_li.append(NB-1)
    else:
        non_concave_li.append(NB-1)

    for zi in range(NZ):
        for nbi, future_debt in enumerate(b_grid):
            # if in concave region, skip
            if nbi in concave_li:
                continue
            # bellman
            this_max_util = beta*(markov[zi, 0]*vfunc_o[nbi, 0]+markov[zi, 1]*vfunc_o[nbi, 1]) +util_func(exo_consum[nbi, zi])
            this_implied_current_b = states[zi] + q[nbi, zi] - exo_consum[nbi, zi]
            # if suboptimal at the implied b, then we drop the b' and activate the b'
            other_consum = states[zi] + q[non_concave_li, zi] - this_implied_current_b
            other_util = util_func(other_consum) + beta*(markov[zi, 0]*vfunc_o[non_concave_li, 0]+markov[zi, 1]*vfunc_o[non_concave_li, 1])
            compare_util = np.sum(other_util > this_max_util)
            if compare_util == 0:
                continue
            else:
                drop[nbi, zi] = 0
    drop = drop.astype(np.bool8)  # False get dropped
    return np.where(drop, exo_consum, np.nan), drop

def _get_consumption(NB, NZ, beta, markov, prob_arr, consum_arr,  q_deriv_arr):
    consum_arr_new = np.zeros((NB, NZ))
    for zi in range(NZ):
        for nbi in range(NB):
            for nzi in range(NZ):
                consum_arr_new[nbi, zi] += markov[zi, nzi] * prob_arr[nbi, nzi] *u_prime(consum_arr[nbi, nzi])
    consum_arr_new *= beta
    consum_arr_new /= q_deriv_arr
    consum_arr_new = consum_arr_new**(-1./2.)
    # print("In function, consumption check {}".format(sum(consum_arr_new > 0)))
    return consum_arr_new

def _get_q_derivative(NB, NZ, alpha, markov, interest, b_grid, consum_arr, prob_arr):
    q_deriv_arr = np.zeros((NB, NZ))
    for nbi in range(NB):
        # find derivative of Q wrt b'
        for zi in range(NZ):  # zi in fact drop out for q, because of iid
            for nzi in range(NZ):
                q_deriv_arr[nbi, zi] += markov[zi, nzi]*prob_arr[nbi, nzi]*(1. -
                                        alpha*b_grid[nbi]*(1-prob_arr[nbi, nzi])*u_prime(consum_arr[nbi, nzi]))
    q_deriv_arr /= interest
    return q_deriv_arr


def _get_prob_arr(NB, NZ, alpha, vfunc_def, vfunc_clean):
    prob_arr = np.zeros((NB, NZ))
    for nbi in range(NB):
        # calculate the probabilities
        for nzi in range(NZ):
            prob_arr[nbi, nzi] = 1. / (1. + np.exp(alpha * (vfunc_def[nzi] - vfunc_clean[nbi, nzi])))
    return prob_arr

@njit
def _bellman_clean(NZ, NB, states, q, b_grid, markov, vfunc_o, beta, EV_shock, vfunc_def, vfunc_clean):
    vfunc_clean_new = np.ones((NB, NZ))*-100
    for zi in range(NZ):
        for bi in range(NB):
            this_max_util = -100.
            for choice in range(NB):
                consum = states[zi] + q[choice, zi] - b_grid[bi]
                if consum > 0.:
                    nu = 0.
                    if EV_shock:
                        for nzi in range(NZ):
                            nu += markov[zi, nzi]*vfunc_o[choice, nzi]
                    else:
                        for nzi in range(NZ):
                            nu += markov[zi, nzi]*np.maximum(vfunc_clean[choice, nzi], vfunc_def[nzi])
                    nu *= beta
                    util = nu - 1./consum
                    if util > this_max_util:
                        this_max_util = util
                        vfunc_clean_new[bi, zi] = util
    return vfunc_clean_new


def _bellman_default(NZ, states, tau, beta, markov, vfunc_def):
    vfunc_def_new = np.zeros(NZ)
    for zi, y in enumerate(states):
        vfunc_def_new[zi] += -1. / (y - tau)
        for nzi in range(NZ):
            vfunc_def_new[zi] += beta*markov[zi, nzi]*vfunc_def[nzi]
    return vfunc_def_new

def _pop_vfunc_o_no_shock(vfunc_def_new, vfunc_clean_new, NB, NZ):
    vfunc_o = np.zeros((NB, NZ))
    dfunc = np.zeros((NB, NZ)).astype(np.int8)
    for zi in range(NZ):
        this_def_val = vfunc_def_new[zi]
        mask = vfunc_clean_new[:, zi] < this_def_val
        vfunc_o[:, zi][mask] = this_def_val
        dfunc[:, zi][mask] = True
        vfunc_o[:, zi][~mask] = vfunc_clean_new[:, zi][~mask]
    return vfunc_o, dfunc


def _pop_vfunc_o_shock(vfunc_def_new, vfunc_clean_new, NB, NZ, euler_gamma, alpha):
    vfunc_o = np.zeros((NB, NZ))
    dfunc = np.zeros((NB, NZ)).astype(np.int8)
    for zi in range(NZ):
        for bi in range(NB):
            vfunc_o[bi, zi] = vfunc_clean_new[bi, zi] + euler_gamma / alpha + 1/alpha * np.log(1. +
                                                        np.exp(alpha*(vfunc_def_new[zi] - vfunc_clean_new[bi, zi])))
    return vfunc_o, dfunc



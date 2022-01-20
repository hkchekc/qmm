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
    r.q[:] = q_zero_arr


def egm(r, p):
    error_vfi = 100.
    it = 0
    # populate v_d
    for zi in range(p.NZ):
        tmp = -p.markov[0,0]/(p.states[0]-p.tau) - p.markov[0,1]/(p.states[1]-p.tau)
        r.vfunc_def[zi] = -1./(p.states[zi]-p.tau) + p.beta*tmp/(1-p.beta)
    r.vfunc_o = np.ones((p.NB, p.NZ)) * -5.
    # initial guess consumption
    for zi in range(p.NZ):
        r.consum_arr[:, zi] = p.states[zi] - p.b_grid[:]
    r.consum_arr[r.consum_arr <= 0.] = 0.01
    # start iteration
    while error_vfi > p.crit_vfi:
        vfunc_clean_new, r.consum_arr = _find_next_c(p.NB, p.NZ, p.alpha, p.markov,
                    p.interest, p.beta, p.states, p.b_grid, r.consum_arr, r.vfunc_def, r.vfunc_clean, r.q, r.vfunc_o)
        vfunc_o_new, r.dfunc = _pop_vfunc_o_shock(r.vfunc_def, vfunc_clean_new, p.NB, p.NZ, p.euler_gamma, p.alpha)
        error_vfi = np.max(np.abs(vfunc_o_new-r.vfunc_o))
        r.vfunc_o = vfunc_o_new
        r.vfunc_clean = vfunc_clean_new
        it += 1
        if it % 200 == 0:
            print("Error of VFI at loop {} is {}".format(it, error_vfi))
            break





def _find_next_c(NB, NZ, alpha, markov, interest, beta, states, b_grid, consum_arr, vfunc_def, vfunc_clean, q, vfunc_o):
    exo_future_debt = np.zeros((NB, NZ))  # b'(b, y) policy function
    exo_consum = np.zeros((NB, NZ))  # c(b, y)  consumption policy
    vfunc_clean_new = np.zeros((NB, NZ))  # value function

    prob_arr = _get_prob_arr(NB, NZ, alpha, vfunc_def, vfunc_clean)
    q_deriv_arr = _get_q_derivative(NB, NZ, alpha, markov, interest, b_grid, consum_arr, prob_arr)
    # populate consumption arr
    consum_arr_new, implied_current_debt = _get_consumption(NB, NZ, markov, beta, states, consum_arr, q, q_deriv_arr, prob_arr)
    for zi in range(NZ):
        mask = q_deriv_arr[:, zi] > 0.  # consumption undefined
        # map consumption from b' to b
        try:
            debt_f = interpolate.interp1d(implied_current_debt[:, zi][mask], b_grid[mask], fill_value="extrapolate")
            consum_f = interpolate.interp1d(implied_current_debt[:, zi][mask], consum_arr_new[:, zi][mask], fill_value="extrapolate")
        except ValueError:
            print(q_deriv_arr[:, zi])
            print(consum_arr)
            raise ValueError
        exo_future_debt[:, zi] = debt_f(b_grid)
        exo_future_debt[:, zi][exo_future_debt[:, zi] > b_grid[-1]] = b_grid[-1]  # assume monotonic
        exo_consum[:, zi] = consum_f(b_grid)
        # catch overgrid
        # no negative consumption
        another_mask = exo_consum[:, zi] > 0
        exo_consum[:, zi][~another_mask] = .1
        vfunc_clean_new[:, zi] = -100.
        # if best consumption not over 0, default
        vfunc_clean_new[:, zi][another_mask] = -1./exo_consum[:, zi][another_mask]
        for bi in range(NB):
            if not another_mask[bi]:
                continue
            nb_idx = _find_nearest(b_grid, exo_future_debt[bi, zi])
            for nzi in range(NZ):
                vfunc_clean_new[bi, zi] += markov[zi, nzi]*vfunc_o[nb_idx, nzi]

    # integrity check - if not concave
    # form v' array
    v_prime = -markov[0, 0] / (consum_arr_new[:, 0] ** 2) - markov[0,1]/ (consum_arr_new[:, 1] ** 2)
    # _non_concavity(NB, NZ, markov, beta, b_grid, states, vfunc_o, q, v_prime, exo_future_debt, exo_consum,
    #                vfunc_clean_new)
    return vfunc_clean_new, exo_consum


def _find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def _non_concavity(NB, NZ, markov, beta, b_grid, states, vfunc_o, q, v_prime, exo_future_debt, exo_consum, vfunc_clean_new):
    # find v max v_min
    this_max = 0
    while this_max < NB-1:
        v_max = 0.
        gap_max = 0
        gap_min = 0
        v_min = 0
        a_max = NB - 1
        a_min = 0
        for bi in range(this_max+1, NB):
            if v_prime[bi - 1] < v_prime[bi]:
                v_max = v_prime[bi - 1]
                gap_min = bi - 1
                gap_max = bi
                v_min = v_prime[bi]
                break
            if bi == NB - 1:
                a_max = NB  # break outer loop
                break
        if a_max > NB -1:
            break
        if np.isnan(v_max) or np.isnan(v_min):
            break
        for bi in range(gap_max, NB):  # find b max
            if v_max > v_prime[bi]:
                a_max = bi
                break
        for bi in range(1, gap_min):  # find b min
            if v_min < v_prime[bi]:
                a_min = bi - 1
                break
            # check for max b' on [bmin, bmax]
            # not concave
        for zi in range(NZ):
            for bi, future_debt in enumerate(exo_future_debt[:, zi]):
                if not (b_grid[a_min] <= future_debt <= b_grid[a_max]):
                    continue
                # bellman
                this_max_util = -100.
                for nbi in range(a_min, a_max):
                    consum = states[zi] + q[nbi, zi] - b_grid[bi]
                    if consum > 0.:
                        nu = 0.
                        for nzi in range(NZ):
                            nu += markov[zi, nzi]*vfunc_o[nbi, nzi]
                        nu *= beta
                        util = nu - 1./consum
                        if util > this_max_util:
                            this_max_util = util
                            exo_consum[bi, zi] = consum
                            vfunc_clean_new[bi, zi] = util
        this_max = a_max

def _get_consumption(NB, NZ, markov, beta, states, consum_arr, q, q_deriv_arr, prob_arr):
    consum_arr_new = np.zeros((NB, NZ))
    implied_current_debt = np.zeros((NB, NZ))
    for zi in range(NZ):
        for nbi in range(NB):
            for nzi in range(NZ):
                consum_arr_new[nbi, zi] += markov[zi, nzi]*prob_arr[nbi, nzi]/(consum_arr[nbi, nzi]**2)
            consum_arr_new[nbi, zi] *= beta
            consum_arr_new[nbi, zi] /= q_deriv_arr[nbi, zi]
            consum_arr_new[nbi, zi] = consum_arr_new[nbi, zi]**(-1./2.)
            implied_current_debt[nbi, zi] = states[zi] + q[nbi, zi] - consum_arr_new[nbi, zi]
    return consum_arr_new, implied_current_debt

def _get_q_derivative(NB, NZ, alpha, markov, interest, b_grid, consum_arr, prob_arr):
    q_deriv_arr = np.zeros((NB, NZ))
    for nbi in range(NB):
        # find derivative of Q wrt b'
        for zi in range(NZ):  # zi in fact drop out for q, because of iid
            for nzi in range(NZ):
                q_deriv_arr[nbi, zi] += markov[zi, nzi]*prob_arr[nbi, nzi]*(1. -
                                        alpha*b_grid[nbi]*(1-prob_arr[nbi, nzi])/(consum_arr[nbi, nzi]**2))
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



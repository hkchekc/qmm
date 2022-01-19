import numpy as np
from numba import jit, njit

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
        self.b_grid = np.linspace(0., 2., num=150)
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
            self.consum_arr = np.ones((param.NB, param.NZ))


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
        if it % 50 == 0:
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
        # start iteration
    while error_vfi > p.crit_vfi:
        vfunc_clean_new, consum_arr_new = _find_next_c(p.NB, p.NZ, p.alpha, p.markov,
                                    p.interest, p.beta, p.states, p.b_grid, r.consum_arr, r.vfunc_def, r.vfunc_clean, r.q, r.vfunc_o)

        vfunc_o_new, r.dfunc = _pop_vfunc_o_shock(r.vfunc_def, vfunc_clean_new, p.NB, p.NZ, p.euler_gamma, p.alpha)
        error_vfi = np.max(np.abs(vfunc_o_new-r.vfunc_o))
        r.vfunc_o = vfunc_o_new
        r.vfunc_clean = vfunc_clean_new
        it += 1
        if it % 50 == 0:
            print("Error of VFI at loop {} is {}".format(it, error_vfi))


def _find_next_c(NB, NZ, alpha, markov, interest, beta, states, b_grid, consum_arr, vfunc_def, vfunc_clean, q, vfunc_o):
    consum_arr_new = np.zeros((NB, NZ))
    prob_arr = np.zeros(NZ)
    q_deriv_arr = np.zeros((NB, NZ))
    implied_current_debt = np.zeros((NB, NZ))
    vfunc_clean_new = np.zeros((NB, NZ))
    for nbi in range(NB):
        # calculate the probabilities
        for nzi in range(NZ):
            prob_arr[nzi] = 1/ (1. + np.exp(alpha * (vfunc_def[nzi] - vfunc_clean[nbi, nzi])))
        # find derivative of Q wrt b'
        for zi in range(NZ):  # zi in fact drop out for q, because of iid
            for nzi in range(NZ):
                q_deriv_arr[nbi, zi] += markov[zi, nzi]*prob_arr[nzi]*(1. -
                                        alpha*(1-prob_arr[nzi])/(consum_arr[nbi, nzi]**2))
            q_deriv_arr[nbi, zi] /= interest
            for nzi in range(NZ):
                consum_arr_new[nbi, zi] += markov[zi, nzi]*prob_arr[nzi]/(consum_arr[nbi, nzi]**2)
            consum_arr_new[nbi, zi] *= beta
            consum_arr_new[nbi, zi] /= q_deriv_arr[nbi, zi]
            consum_arr_new[nbi, zi] = 1/np.power(consum_arr_new[nbi, zi], .5)
            implied_current_debt[nbi, zi] = states[zi] + q[nbi, zi] - consum_arr_new[nbi, zi]
    exo_future_debt = np.zeros((NB, NZ))  # b'(b, y)
    exo_consum = np.zeros((NB, NZ)) # c(b, y)
    for zi in range(NZ):
        exo_future_debt[:, zi] = np.interp(b_grid, implied_current_debt[:, zi], b_grid)  # flipped around
    for zi in range(NZ):
        exo_consum[:, zi] = np.interp(exo_future_debt[:, zi], consum_arr_new[:, zi], b_grid)  # flipped around
    mask = 
    vfunc_clean_new = -1./exo_consum + beta*vfunc_o  # increment to new vf
    return vfunc_clean_new, exo_consum


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



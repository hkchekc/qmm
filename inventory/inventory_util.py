#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
from scipy import interpolate, stats, optimize
from numba import njit, jit

def golden_search(func, bmin, bmax):
    max_it = 10000
    precision = 1e-10
    
    r = (3. - math.sqrt(5.))/2.
    
    c = bmin + r*(bmax - bmin)
    x = c
    
    fc = func(x)
    fc = -1.*fc
    
    d = bmin + (1. - r)*(bmax -bmin)
    x = d
    fd = func(x)
    fd = -1.*fd
    
    for i in range(max_it):
        if fc >= fd:
            z = bmin + (1-r)*(d-bmin)
            bmin = c
            c = d
            fc = fd
            d = z
            x = d
            fd = -1.*func(x)
        else:
            z = bmin + r*(d-bmin)
            bmax = d
            d = c
            fd = fc
            c = z
            x =c
            fc = -1.*func(x)
    return x

class param(object):
    
    def __init__(self):
        self.beta = .984
        self.eta = 2.128
        self.alpha = .3739
        self.theta_m = .4991
        self.theta_n = .3275
        self.delta = .0173
        self.del_cost_max = .2198
        self.del_cost_min = 0
        self.cost_dist = stats.uniform(loc =self.del_cost_min , scale=self.del_cost_max)
        self.z_bar = 1.0032  # no uncertainty
        self.store_cost = .012
        self.NS = 25
        s_min = 0
        s_max = 2.5
        phi0 = 10
        phi1 = 25
        lin_min = np.log(.1042/phi1)/np.log(phi0)
        lin_max = np.log(s_max)/np.log(10)
        step  = self.NS-1
        self.s_grid = np.zeros(self.NS)
        self.s_grid[1:] = 10**np.linspace(lin_min, lin_max, step)
        self.NJ = 10  # number of epriods w/o adjustment
        self.crit_vf = 1e-6 # should be -6
        self.crit_clear = 1e-4
        self.crit_used_up = 1e-8
        self.crit_gs = 1e-10
        self.p_max = 3.3
        self.p_min = 3.2


class result(object):
    
    def __init__(self):
        self.m_star = 0
        self.n_star = 1
        P = param()
        self.p_star = (P.p_max+P.p_min)/2
        # self.p_star = 3.2375
        self.wage, self.q = get_price(P, self.p_star)
        self.vals_s1 = np.zeros(P.NS)
        self.vals_s1_new = np.zeros(P.NS)
        self.val_s_star = 10
        self.val_s_star_new = 0
        self.vfunc0 = np.zeros(P.NS)  # the expected values
        self.vfunc0_new = np.zeros(P.NS)
        self.val_adj = 0  # scalar
        self.s_star = 0
        self.m_arr = np.zeros(P.NS)
        self.n_arr = np.zeros(P.NS)
        self.m_arr_j = np.zeros(P.NJ)
        self.n_arr_j = np.zeros(P.NJ)
        self.output_arr_j = np.zeros(P.NJ)
        self.s_arr = np.zeros(P.NJ)
        self.adj_pmf = np.zeros(P.NJ)
        self.stat_dist = np.zeros(P.NJ)
        self.output_dist = np.zeros(P.NJ)
        self.h = np.zeros(P.NJ)
        self.threshold_arr = np.zeros(P.NS)

def get_price(param, price):
    w = param.eta/price
    ugly = (1-param.beta*(1-param.delta))/param.beta/param.alpha
    inv_alpha = 1- param.alpha
    q = (price**(-inv_alpha))*(ugly**param.alpha)*((param.eta/inv_alpha)**inv_alpha)/param.z_bar
    return w, q  # wage and intermediate good price

def vfi(r, p):
    r.vfunc0 = _init_guess_expected_v0(p.s_grid, r.p_star, p.theta_m, p.theta_n, p.eta)
    vf_err = 100
    it = 0
    while vf_err > p.crit_vf:
        # set v_s_star_new and adjustment value/ v0 in paper
        r.val_s_star_new, r.s_star, r.val_adj = _optimal_s(p.s_grid, r.p_star, r.q, r.vals_s1)
        _optimal_m(r, p) # find the value for all s1
        # r.threshold_arr = _del_cost_threhold(r, p)
        r.threshold_arr = _del_cost_threhold(r.vals_s1_new, r.p_star, r.q, p.s_grid, r.val_adj, r.wage, p.del_cost_max)
        r.vfunc0_new = _find_expected_v0(r, p)
        vf_err = max(np.max(np.abs(r.vals_s1_new- r.vals_s1)), np.max(np.abs(r.vfunc0_new- r.vfunc0)))
        r.val_s_star = r.val_s_star_new
        r.vfunc0 = r.vfunc0_new
        r.vals_s1 = r.vals_s1_new
        it += 1
        if it % 50 == 0:
            print("done with 50 vfi {} ,{}".format(vf_err, it))
        if it == 5000:
            raise ValueError
            

def inventory_sequence(r, p):
    not_used_up = 1
    # highly doubtful if this is faster than np.interp with njit
    pol_spline = interpolate.CubicSpline(p.s_grid, r.m_arr)
    threshold_spline = interpolate.CubicSpline(p.s_grid, r.threshold_arr)
    while (not_used_up > 0):
        this_s1 = float(r.s_star)
        r.m_arr_j = np.zeros(p.NJ)
        r.s_arr = np.zeros(p.NJ)
        r.adj_pmf = np.zeros(p.NJ)
        for ti in range(p.NJ):
            if (this_s1 < p.crit_used_up):
                if this_s1 > 0:
                    r.s_arr[ti] = this_s1
                    r.m_arr_j[ti] = this_s1

                    if ti != p.NJ:
                        r.adj_pmf[ti + 1] = 1 - np.sum(r.adj_pmf)
                    else:
                        r.adj_pmf[ti] += 1 - np.sum(r.adj_pmf)
                else:
                    r.s_arr[ti] = 0
                    r.m_arr_j[ti-1] -= this_s1
                    r.adj_pmf[ti-1] += 1 - np.sum(r.adj_pmf)
                threshold = threshold_spline(0)  # must adjust
                r.adj_pmf[ti] = (1 - np.sum(r.adj_pmf[:ti])) * p.cost_dist.cdf(threshold)
                r.h[ti:] = p.cost_dist.cdf(threshold)
                break
                # pmf remains zero for later period when all firms adjusted (as all inventory are done)
            else:
                r.s_arr[ti] = this_s1
                r.m_arr_j[ti] = pol_spline(this_s1)
                # find the share of firms adjusting at time t
                threshold = threshold_spline(this_s1 - r.m_arr_j[ti])
                if ti == 0:
                    r.h[ti] = p.cost_dist.cdf(threshold)
                    r.adj_pmf[ti] = p.cost_dist.cdf(threshold)
                else:
                    r.h[ti] = p.cost_dist.cdf(threshold)
                    r.adj_pmf[ti] = (1-np.sum(r.adj_pmf[:ti]))*p.cost_dist.cdf(threshold)
                this_s1 -= r.m_arr_j[ti]
        r.n_arr_j = _find_n(r.m_arr_j, p.theta_m, p.theta_n, p.eta, r.p_star)
        r.output_arr_j = _final_production(r.m_arr_j, r.n_arr_j, p.theta_m, p.theta_n)
        # if there is next loop (not all used up), increase jmax by 1
        # not_used_up = 1- np.sum(r.adj_pmf)
        not_used_up = 0

def final_good_dist(r, p):
    # use the method where firm differs only in adjustment periods
    stat_dist = np.zeros(p.NJ)
    stat_dist[0] = 1.  # just adjusted (s = s* - m*)
    # the grids correspond to the this_s1_arr in inventory_sequence
    for ti in range(1, p.NJ):
        stat_dist[ti] = stat_dist[ti-1]*(1-r.h[ti-1])
    stat_dist[-1] /= (1-p.cost_dist.cdf(r.threshold_arr[-1])) # let's assume last period everyone transit for now
    stat_dist /= np.sum(stat_dist)
    return stat_dist
    
        
def market_clear(r, p):
    # find demand for m, n
    interm_demand = 0.
    for ji in range(p.NJ-1):
        # demand is how much is restocked
        interm_demand += r.h[ji]*r.stat_dist[ji]*(r.s_star - r.s_arr[ji+1])
    # this is wrong, because restock have nothing to do with production
    n_demand = np.sum(r.n_arr_j*r.stat_dist)
    # find agg capital and labor
    agg_cap, interm_good_labor = _find_agg_cap_lab(r.q, p.z_bar, p.alpha, r.wage, interm_demand)
    agg_lab = interm_good_labor+ n_demand
    # find consumption by the aggregate C=Y-dK
    consum = 0
    for ji in range(p.NJ):
        consum += r.output_arr_j[ji] *r.stat_dist[ji] #
        consum -= p.store_cost*(r.s_arr[ji]-r.m_arr_j[ji])*r.stat_dist[ji]
    # update p
    consum -= agg_cap*p.delta
    new_p = 1/ consum
    if new_p < r.p_star:
        p.p_max = float(r.p_star)
    else:
        p.p_min = float(r.p_star)
    p_star_new = (p.p_max +p.p_min)/2
    err = np.abs(r.p_star - new_p)
    return p_star_new, err

@njit
def _find_agg_cap_lab(q, z, alpha, w, demand):
    # agg_labor is ugly term 
    ugly_term = (w/q/z/(1-alpha))**(1/alpha)
    agg_cap = demand/(ugly_term**(1-alpha)) 
    interm_lab = ugly_term*agg_cap
    return agg_cap, interm_lab
        
def _optimal_s(s_grid, p_star, q, vals_s1):
    vals_s1_with_cost = p_star*q*s_grid - vals_s1
    v1_wc_spline = interpolate.CubicSpline(s_grid, vals_s1_with_cost)
    # s_star = optimize.minimize(v1_wc_spline, np.array([1.8]), bounds=[(s_grid[0], s_grid[-1])])
    s_star = optimize.minimize_scalar(v1_wc_spline, bracket=(s_grid[0], s_grid[-1]),
                                      method="bounded", bounds=(s_grid[0], s_grid[-1]), tol=1e-10)
    # s_star = optimize.golden(v1_wc_spline, brack=(s_grid[0], s_grid[-1]), tol=1e-10, full_output=True)
    s_star = s_star.x
    # if s_star > s_grid[-1]:
    #     s_star = s_grid[-1]
    # s_star = golden_search(v1_wc_spline, s_grid[0], s_grid[-1])
    # print(s_star)
    val_s_star_new = -1.*v1_wc_spline(s_star)
    val_adj = val_s_star_new
    return val_s_star_new, s_star, val_adj
    

def _optimal_m(r, p):  # void function
    v0_spline = interpolate.CubicSpline(p.s_grid, r.vfunc0)
    def find_m(m, si, p, r):
        n = _find_n(m, p.theta_m, p.theta_n, p.eta, r.p_star)
        this_val = r.p_star*_final_net_prod(m, n, p.theta_m, p.theta_n, p.s_grid[si], p.store_cost, r.wage)
        next_u = p.beta*v0_spline(p.s_grid[si]-m)
        return -1.*(this_val+next_u)  # because the function is minimized
    for si, s1 in enumerate(p.s_grid):
        # m0 = s1/1.2
        # res = optimize.minimize(find_m, m0, args=(si, p, r), method='Nelder-Mead', tol=p.crit_vf, bounds=[(0, s1)])
        res = optimize.minimize_scalar(find_m, args=(si, p, r), bracket=(0, s1), method="golden",
                                          bounds=(0, s1), tol=1e-10)

        r.m_arr[si] = res.x
    mx = r.m_arr > p.s_grid
    r.m_arr[mx] = p.s_grid[mx]
    r.n_arr = _find_n(r.m_arr, p.theta_n, p.theta_m, p.eta, r.p_star)
    r.vals_s1_new =r.p_star*_final_net_prod(r.m_arr, r.n_arr, p.theta_m,
                             p.theta_n, p.s_grid, p.store_cost, r.wage) + p.beta*v0_spline(p.s_grid-r.m_arr)
    # TODO: find a way to store q (net_prod) and s

@njit
def _init_guess_expected_v0(s_grid, p_star, theta_m, theta_n, eta):
    # returns arr of length NS 
    inv_the_n = 1-theta_n
    first = p_star**(1/(1-theta_m))
    second  = (theta_n/eta)**(theta_n/inv_the_n)
    third = np.power(s_grid, theta_m/inv_the_n)
    return first*inv_the_n*second*third

@njit
def _find_n(m, theta_n, theta_m, eta, price):
    pow_i = 1/(1-theta_n)
    return np.power(theta_n*price*(np.power(m, theta_m)/eta), pow_i)

@njit
def _final_production(m, n, theta_m, theta_n):
    return (m**theta_m)*(n**theta_n)

@njit
def _final_net_prod(m, n, theta_m, theta_n, s1, store_cost, wage):  # q
    cost = store_cost*(s1-m) + wage*n
    return _final_production(m, n, theta_m, theta_n) - cost

@njit
def _interm_prod(z, k, l, alpha):
    return z*(k**alpha)*(l**(1-alpha))

def _find_expected_v0(r, p):
    percent_adj = p.cost_dist.cdf(r.threshold_arr)
    return _find_expected_v0_helper(percent_adj, r.val_adj, r.p_star, p.s_grid, r.q, r.threshold_arr, r.wage, r.vals_s1_new)

@njit
def _find_expected_v0_helper(percent_adj, val_adj, p_star, s_grid, q, threshold, wage, vals_s1):
    first = percent_adj*(val_adj+p_star*s_grid*q)
    area = percent_adj*threshold/2  # prob of adj * conditional expected cost
    second = p_star*wage* area
    third = (1 - percent_adj)*vals_s1
    return first - second + third

@njit
def _del_cost_threhold(vals_s1, p_star, q, s_grid, val_adj, wage, del_cost_max):
    # return arr of length NS
    thres_arr = -1. * (vals_s1 - p_star * q * s_grid - val_adj) / p_star / wage
    thres_arr[thres_arr < 0] = 0
    thres_arr[thres_arr > del_cost_max] = del_cost_max
    return thres_arr

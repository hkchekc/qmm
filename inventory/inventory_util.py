#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
from scipy import interpolate, stats, optimize
from numba import njit, jit

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
        # self.s_grid = np.linspace(s_min, s_max, self.NS)
        self.NJ = 12  # number of epriods w/o adjustment
        self.crit_vf = 1e-6 # should be -6
        self.crit_clear = 1e-8
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
        self.threshold_arr = np.zeros(P.NS)
        self.h_arr = np.zeros(P.NS)
        self.m_star = 0  # values related to s_star values
        self.n_star = 0
        self.output_star = 0
        self.s_arr = np.zeros(P.NJ)  # starting with s1 = s* - m(s*)
        self.m_arr_j = np.zeros(P.NJ)
        self.n_arr_j = np.zeros(P.NJ)
        self.output_arr_j = np.zeros(P.NJ)
        self.stat_dist = np.zeros(P.NJ)
        self.output_dist = np.zeros(P.NJ)
        self.output_star_percentage = 0
        self.h_j = np.zeros(P.NJ)

def get_price(param, price):
    w = param.eta/price
    ugly = (1-param.beta*(1-param.delta))/param.beta/param.alpha
    inv_alpha = 1- param.alpha
    q = (price**(-inv_alpha))*(ugly**param.alpha)*((param.eta/inv_alpha)**inv_alpha)/param.z_bar
    return w, q  # wage and intermediate good price

def vfi(r, p):
    vf_err = 100
    it = 0
    while vf_err > p.crit_vf:
        # set v_s_star_new and adjustment value/ v0 in paper
        r.val_s_star_new, r.s_star, r.val_adj = _optimal_s(p.s_grid, r.p_star, r.q, r.vals_s1)
        _optimal_m(r, p) # find the value for all s1
        r.threshold_arr = _del_cost_threshold(r.vals_s1_new, r.p_star, r.q, p.s_grid, r.val_adj, r.wage, p.del_cost_max)
        r.h_arr = p.cost_dist.cdf(r.threshold_arr)
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
    m_pol_spline = interpolate.CubicSpline(p.s_grid, r.m_arr, bc_type="not-a-knot")
    threshold_spline = interpolate.CubicSpline(p.s_grid, r.threshold_arr, bc_type="not-a-knot")
    h_spline = interpolate.CubicSpline(p.s_grid, r.h_arr, bc_type="not-a-knot")
    r.m_star = m_pol_spline(r.s_star)
    r.n_star = _find_n(r.m_star, p.theta_m, p.theta_n, p.eta, r.p_star)
    r.output_star = _final_production(r.m_star, r.n_star, p.theta_m, p.theta_n)
    while (not_used_up > 0):
        this_s1 = float(r.s_star - r.m_star)  # start with s1 = s* - m*
        r.m_arr_j = np.zeros(p.NJ)
        r.s_arr = np.zeros(p.NJ)
        for ti in range(p.NJ):
            if (this_s1 < p.crit_used_up):
                if this_s1 > 0:
                    r.s_arr[ti] = this_s1
                    r.m_arr_j[ti] = this_s1
                else: # some interpolation mistake, otherwise shouldn't end up here
                    r.s_arr[ti] = 0
                    r.m_arr_j[ti-1] -= this_s1
                threshold = threshold_spline(0)  # must adjust
                r.h_j[ti:] = p.cost_dist.cdf(threshold)
                r.h_j[ti:] = h_spline(this_s1)
                break
                # pmf remains zero for later period when all firms adjusted (as all inventory are done)
            else:
                r.s_arr[ti] = this_s1
                r.m_arr_j[ti] = m_pol_spline(this_s1)
                # find the share of firms adjusting at time t
                threshold = threshold_spline(this_s1)
                r.h_j[ti] = p.cost_dist.cdf(threshold)
                r.h_j[ti] = h_spline(this_s1)
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
        stat_dist[ti] = stat_dist[ti-1]*(1-r.h_j[ti-1])
    stat_dist[-1] /= (1-p.cost_dist.cdf(r.threshold_arr[-1])) # let's assume last period everyone transit for now
    stat_dist /= np.sum(stat_dist)
    return stat_dist
    
        
def market_clear(r, p):
    # find output dist
    r.output_star_percentage, r.output_dist = _get_output_dist(r, p)
    interm_demand = 0. # find demand for m, n
    for ji in range(p.NJ):  # demand is how much is restocked
        interm_demand += r.h_j[ji]*r.stat_dist[ji]*(r.s_star - r.s_arr[ji])
    n_demand = np.sum(r.n_arr_j*r.stat_dist)
    # find agg capital and labor
    agg_cap, interm_good_labor = _find_agg_cap_lab(r.q, p.z_bar, p.alpha, r.wage, interm_demand)
    agg_lab = interm_good_labor+ n_demand
    # find consumption by the aggregate C=Y-dK
    consum = 0
    for ji in range(p.NJ):
        consum += r.output_arr_j[ji] *r.stat_dist[ji]*(1- r.h_j[ji])
        consum += r.output_star *r.stat_dist[ji]*(r.h_j[ji])
        consum -= p.store_cost*(r.s_arr[ji]-r.m_arr_j[ji])*r.stat_dist[ji]*(1- r.h_j[ji])
        consum -= p.store_cost*(r.s_star-r.m_star)*r.stat_dist[ji]*r.h_j[ji]
    # update p
    consum -= agg_cap*p.delta
    new_p = 1 / consum
    if np.abs(new_p - r.p_star) < 0.0001:
        ratio = 0.9
    else:
        ratio = 0
    if new_p < r.p_star:
        p.p_max = (1-ratio)*float(r.p_star)+ratio*p.p_max
    else:
        p.p_min = (1-ratio)*float(r.p_star)+ratio*p.p_min
    p_star_new = (p.p_max +p.p_min)/2
    err = np.abs(r.p_star - new_p)
    return p_star_new, err

def _get_output_dist(r, p):
    output_star_percentage = 0
    output_arr_j = np.zeros(p.NJ)
    for ji in range(p.NJ):
        output_star_percentage += r.output_star*r.stat_dist[ji]*r.h_j[ji]
        output_arr_j[ji] = r.output_arr_j[ji]*r.stat_dist[ji]*(1-r.h_j[ji])
    out_sum = output_star_percentage + np.sum(output_arr_j)
    return output_star_percentage/out_sum, output_arr_j/out_sum

@njit
def _find_agg_cap_lab(q, z, alpha, w, demand):
    ugly_term = (w/q/z/(1-alpha))**(1/alpha)
    interm_lab = demand/(ugly_term**(alpha))
    agg_cap = ugly_term*interm_lab
    return agg_cap, interm_lab

@jit(nopython=False, forceobj=True)
def _optimal_s(s_grid, p_star, q, vals_s1):
    vals_s1_with_cost = p_star*q*s_grid - vals_s1
    v1_wc_spline = interpolate.CubicSpline(s_grid, vals_s1_with_cost, bc_type="not-a-knot")
    s_star = optimize.minimize_scalar(v1_wc_spline, bracket=(s_grid[0], s_grid[-1]),
                                      method="bounded", bounds=(s_grid[0], s_grid[-1]))
    # s_star = optimize.golden(v1_wc_spline, brack=(s_grid[0], s_grid[-1]), tol=1e-10, full_output=True)
    s_star = s_star.x
    val_adj = -1.*v1_wc_spline(s_star)
    val_s_star_new = val_adj +p_star*q*s_star
    return val_s_star_new, s_star, val_adj

def _optimal_m(r, p):  # void function
    v0_spline_out = interpolate.CubicSpline(p.s_grid, r.vfunc0, bc_type="not-a-knot")
    def _find_m_minimizer(m, si, p, r, v0_spline):
        n = _find_n(m, p.theta_m, p.theta_n, p.eta, r.p_star)
        this_val = r.p_star*_final_net_prod(m, n, p.theta_m, p.theta_n, p.s_grid[si], p.store_cost, r.wage)
        next_u = p.beta*v0_spline(p.s_grid[si]-m)
        return -1.*(this_val+next_u)  # because the function is minimized
    for si, s1 in enumerate(p.s_grid):
        res = optimize.minimize_scalar(_find_m_minimizer, args=(si, p, r, v0_spline_out), bracket=(0, s1), method="bounded",
                                          bounds=(0, s1))
        # res = optimize.minimize_scalar(_find_m, args=(si, p.beta, p.theta_m, p.theta_n, p.eta, p.s_grid,
        #                                               p.store_cost,  r.p_star, r.vfunc0, r.wage), bracket=(0, s1), method="bounded",
        #                                   bounds=(0, s1), tol=1e-10)

        r.m_arr[si] = res.x
    r.vals_s1_new, r.n_arr = _mask_v0(r.m_arr, p.s_grid, p.theta_m, p.theta_n, p.eta, r.p_star, p.store_cost, r.wage, v0_spline_out(0), v0_spline_out(p.s_grid-r.m_arr), p.beta)

@njit
def _mask_v0(m_arr, s_grid, theta_m, theta_n, eta, p_star, store_cost, wage, corner, next_val, beta):
    mx = m_arr > s_grid  # check m* < s1
    m_arr[mx] = s_grid[mx]
    n_arr = _find_n(m_arr, theta_m, theta_n, eta, p_star)
    vals_s1_new =p_star*_final_net_prod(m_arr, n_arr, theta_m,
                             theta_n, s_grid, store_cost, wage) + beta*next_val
    # check corner solution
    vals_exhaust = p_star*_final_net_prod(s_grid, _find_n(s_grid, theta_m, theta_n, eta, p_star), theta_m,
                             theta_n, s_grid, store_cost, wage) + beta*corner
    exhaust_mx = vals_exhaust > vals_s1_new
    vals_s1_new[exhaust_mx] = vals_exhaust[exhaust_mx]
    return vals_s1_new, n_arr

@njit
def _find_m(m, si, beta, theta_m,theta_n, eta, s_grid, store_cost,  p_star, vfunc0, wage):
    n = _find_n(m, theta_m, theta_n, eta, p_star)
    this_val = p_star*_final_net_prod(m, n, theta_m, theta_n, s_grid[si], store_cost, wage)
    next_u = beta*np.interp(s_grid[si]-m, s_grid, vfunc0)
    return -1.*(this_val+next_u)  # because the function is minimized

@njit
def init_guess_expected_v0(s_grid, p_star, theta_m, theta_n, eta):
    # returns arr of length NS 
    inv_the_n = 1-theta_n
    first = p_star**(1/(1-theta_m))
    second  = (theta_n/eta)**(theta_n/inv_the_n)
    third = np.power(s_grid, theta_m/inv_the_n)
    return first*inv_the_n*second*third

@njit
def _find_n(m, theta_m, theta_n, eta, price):
    pow_i = 1/(1-theta_n)
    return np.power(theta_n*price*(np.power(m, theta_m)/eta), pow_i)

@njit
def _final_production(m, n, theta_m, theta_n):
    return (m**theta_m)*(n**theta_n)

@njit
def _final_net_prod(m, n, theta_m, theta_n, s1, store_cost, wage):  # q
    cost = store_cost*(s1-m) + wage*n
    return _final_production(m, n, theta_m, theta_n) - cost


def _find_expected_v0(r, p):
    percent_adj = p.cost_dist.cdf(r.threshold_arr)
    return _find_expected_v0_helper(percent_adj, r.val_adj, r.p_star, p.s_grid, r.q, r.threshold_arr, r.wage, r.vals_s1_new)

@njit
def _find_expected_v0_helper(percent_adj, val_adj, p_star, s_grid, q, threshold, wage, vals_s1):
    first = percent_adj*(val_adj+p_star*s_grid*q)
    area = percent_adj*threshold/2  # prob of adj * conditional expected cost, assumed min = 0, uniform dist
    second = p_star*wage* area
    third = (1 - percent_adj)*vals_s1
    return first - second + third

@njit
def _del_cost_threshold(vals_s1, p_star, q, s_grid, val_adj, wage, del_cost_max):
    thres_arr = -1. * (vals_s1 - p_star * q * s_grid - val_adj) / p_star / wage
    thres_arr[thres_arr < 0] = 0
    thres_arr[thres_arr > del_cost_max] = del_cost_max
    return thres_arr  # return arr of length NS


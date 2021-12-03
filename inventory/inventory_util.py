#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
from scipy import interpolate, stats, optimize
from numba import njit

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
    fd = -1.*fd  # is it a typo?
    
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
        self.cost_dist = stats.uniform(self.del_cost_min , self.del_cost_max)
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
        self.crit_clear = 1e-8
        self.crit_used_up = 1e-8
        self.p_max = 3.3
        self.p_min = 3.2


class result(object):
    
    def __init__(self):
        self.m = 1
        self.n = 1
        P = param()
        self.p_star = (P.p_max+P.p_min)/2
        # self.p_star = 3.25
        self.p_star_new = 0
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
        self.s_arr = np.zeros(P.NJ)
        self.adj_pmf = np.zeros(P.NJ)
        self.stat_dist = np.zeros(P.NJ)
        self.h = np.zeros(P.NJ)
        self.threshold_arr = np.zeros(P.NS)

def get_price(param, price):
    w = param.eta/price
    ugly = (1-param.beta*(1-param.delta))/param.beta/param.alpha
    inv_alpha = 1- param.alpha
    q = (price**(-inv_alpha))*(ugly**param.alpha)*((param.eta/inv_alpha)**inv_alpha)
    return w, q  # wage and intermediate good price

def vfi(r, p):
    r.vfunc0 = _init_guess_expected_v0(p.s_grid, r.p_star, p.theta_m, p.theta_n, p.eta)
    #TODO: set init v star new
    vf_err = 100
    it = 0
    while vf_err > 1e-5:
        _optimal_m(r, p) # find the value for all s1
        # set v_s_star_new and adjustment value/ v0 in paper
        r.vals_s1_new, r.val_s_star_new, r.s_star, r.val_adj = _optimal_s(p.NS, p.s_grid, r.m_arr, r.n_arr,
                p.theta_m, p.theta_n, p.store_cost, r.wage, p.beta, r.vfunc0, r.p_star, r.q) 
        r.threshold_arr = _del_cost_threhold(r, p)
        for si  in range(p.NS):
            r.vfunc0_new[si] = _find_expected_v0(si, r, p)  # set new v0 (should be of length NS)
        #TODO: check corner
        vf_err = max(np.abs(r.val_s_star_new- r.val_s_star), np.max(np.abs(r.vfunc0_new- r.vfunc0)))
        r.val_s_star = r.val_s_star_new
        r.vfunc0 = r.vfunc0_new
        r.vals_s1 = r.vals_s1_new
        it += 1
        # vf_err = 0
        print("done with vfi {} ,{}".format(vf_err, it))
        if it == 5000:
            raise ValueError
            

def inventory_sequence(r, p):
    not_used_up = 1
    # highly doubtful if this is faster than np.interp with njit
    pol_spline = interpolate.UnivariateSpline(p.s_grid, r.m_arr)
    threshold_spline = interpolate.UnivariateSpline(p.s_grid, r.threshold_arr)
    while (not_used_up > 0):
        #TODO: populate this s1 arr
        this_s1 = r.s_star
        r.m_arr_j = np.zeros(p.NJ)
        r.s_arr = np.zeros(p.NJ)
        r.adj_pmf = np.zeros(p.NJ)
        for ti in range(p.NJ):
            if (this_s1 < p.crit_used_up):
                if this_s1 > 0:
                    r.s_arr[ti] = this_s1
                    r.m_arr_j[ti] = this_s1
                    threshold = threshold_spline(0)  # must adjust
                    r.adj_pmf[ti] = np.sum(r.adj_pmf[:ti]) * p.cost_dist.cdf(threshold)
                    if ti != p.NJ:
                        r.adj_pmf[ti + 1] = 1 - np.sum(r.adj_pmf)
                    else:
                        r.adj_pmf[ti] += 1 - np.sum(r.adj_pmf)
                else:
                    r.s_arr[ti] = 0
                    r.m_arr_j[ti-1] -= this_s1
                    r.adj_pmf[ti-1] += 1 - np.sum(r.adj_pmf)
                break
                # pmf remains zero for later period when all firms adjusted (as all inventory are done)
            else:
                r.s_arr[ti] = this_s1
                # no this_s_star have to be solved I think, maybe interp valuefunction
                # m_arr[ti] =  optimal_m(r, p, this_s1) # find the value for all s1
                r.m_arr_j[ti] = pol_spline(this_s1)
                # find the share of firms adjusting at time t
                threshold = threshold_spline(this_s1 - r.m_arr_j[ti])
                if ti == 0:
                    r.adj_pmf[ti] = p.cost_dist.cdf(threshold)
                else:
                    r.adj_pmf[ti] = np.sum(r.adj_pmf[:ti])*p.cost_dist.cdf(threshold)
                this_s1 -= r.m_arr_j[ti]
        r.n_arr_j = _find_n(r.m_arr_j, p.theta_m, p.theta_n, p.eta, r.p_star)
        # if there is next loop (not all used up), increase jmax by 1
        not_used_up = 1- np.sum(r.adj_pmf)
    
        
    
def final_good_dist(r, p):
    # use the method where firm differs only in adjustment periods
    stat_dist = np.ones(p.NJ)/p.NJ  # uniform distributed
    stat_dist[0] = 1.
    adj_cdf = np.cumsum(r.adj_pmf)
    not_adj = 1 - adj_cdf
    #TODO: use kierans method
    # the grids correspond to the this_s1_arr in inventory_sequence
    # for ti in range(p.NJ):
    #     pass
    # use transition matrix
    h = np.zeros(p.NJ)
    h[not_adj>0] = r.adj_pmf[not_adj>0]/not_adj[not_adj>0]  # chance of adjusting at period t
    trans_mat = np.zeros((p.NJ, p.NJ))
    for ji in range(p.NJ):
        trans_mat[ji, 0] = h[ji]
        trans_mat[ji, ji] = 1-h[ji]
    
    almost_stat = np.linalg.matrix_power(trans_mat, 500)
    stat_dist = almost_stat.dot(stat_dist)
    return stat_dist, h
    
        
def market_clear(r, p):
    # find demand for m, n
    interm_demand = 0.
    for ji in range(p.NJ):
        # demand is how much is restocked
        interm_demand += r.h[ji]*r.stat_dist[ji]*(r.s_star - r.s_arr[ji])
    # this is wrong, because restock have nothing to do with production
    n_demand = np.sum(r.n_arr_j*r.stat_dist)
    # find agg capital and labor
    agg_cap, interm_good_labor = _find_agg_cap_lab(r.q, p.z_bar, p.alpha, r.wage, interm_demand)
    agg_lab = interm_good_labor+ n_demand
    # find consumption by the aggregate C=Y-dK
    consum = 0
    for ji in range(p.NJ):
        consum += _final_production(r.m_arr_j[ji], r.n_arr_j[ji], p.theta_m, p.theta_n)  #
        consum -= p.store_cost*(r.s_arr[ji]-r.m_arr_j[ji])
        print(consum, "consum")
        consum *= r.stat_dist[ji]

    # update p
    consum -= agg_cap*p.delta
    new_p = 1/ consum
    if new_p < r.p_star:
        p.p_max = r.p_star
    else:
        p.p_min = r.p_star
    p_star_new = (p.p_max +p.p_min)/2
    err = np.abs(r.p_star - p_star_new)
    return p_star_new, err

@njit
def _find_agg_cap_lab(q, z, alpha, w, demand):
    # agg_labor is ugly term 
    ugly_term = (w/q/z/(1-alpha))**(1/alpha)
    agg_cap = demand/(ugly_term**(1-alpha)) 
    interm_lab = ugly_term*agg_cap
    return agg_cap, interm_lab
        
def _optimal_s(NS, s_grid, m_arr, n_arr, theta_m, theta_n, store_cost, wage, beta, vfunc0, p_star, q):
    vals_s1 = np.zeros(NS)
    v0_spline = interpolate.UnivariateSpline(s_grid, vfunc0)
    for si, s1 in enumerate(s_grid):
        vals_s1[si] =p_star*_final_net_prod(m_arr[si], n_arr[si], theta_m, 
                        theta_n, s1, store_cost, wage) + beta*v0_spline(s_grid[si]-m_arr[si])
    vals_s1_with_cost = -vals_s1+p_star*q*s_grid  # as minimize, *-1
    v1_spline = interpolate.UnivariateSpline(s_grid, vals_s1_with_cost)
    # TODO: use spline
    s_star = optimize.minimize(v1_spline, 1.7, bounds=[(s_grid[0], s_grid[-1])])
    s_star = s_star.x
    s_star = s_grid[np.argmin(vals_s1_with_cost)]
    val_s_star_new = -1.*v1_spline(s_star)
    val_adj = val_s_star_new
    return vals_s1, val_s_star_new, s_star, val_adj
    

def _optimal_m(r, p):  # void function
    v0_spline = interpolate.UnivariateSpline(p.s_grid, r.vfunc0)
    def find_m(m, si, p, r):
        n = _find_n(m, p.theta_m, p.theta_n, p.eta, r.p_star)
        this_val = r.p_star*_final_net_prod(m, n, p.theta_m, p.theta_n, s1, p.store_cost, r.wage)
        next_u = p.beta*v0_spline(p.s_grid[si]-m)
        return -1.*(this_val+next_u)  # because the function is minimized
    for si, s1 in enumerate(p.s_grid):
        m0 = s1/2
        res = optimize.minimize(find_m, m0, args=(si, p, r), method='Nelder-Mead', tol=p.crit_vf, bounds=[(0, s1)])
        r.m_arr[si] = res.x
        if r.m_arr[si] > s1:
            r.m_arr[si] = s1
        r.n_arr[si] = _find_n(r.m_arr[si], p.theta_n, p.theta_m, p.eta, r.p_star)
        # TODO: find a way to store q (net_prod) and s

@njit
def _init_guess_expected_v0(s_grid, p_star, theta_m, theta_n, eta):
    # returns arr of length NS 
    inv_the_n = 1-theta_n
    first = p_star**(1/(1-theta_n))
    second  = (theta_n/eta)**(theta_n/inv_the_n)
    third = np.power(s_grid, theta_m/inv_the_n)
    return first*inv_the_n*second*third

@njit
def _find_n(m, theta_n, theta_m, eta, price):
    pow_i = 1/(1-theta_n)
    return (theta_n*price*(m**theta_m)/eta)**pow_i

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

def _find_expected_v0(stock_idx, r, p):
    threshold = r.threshold_arr[stock_idx]
    percent_adj  =p.cost_dist.cdf(threshold)
    first =percent_adj*(r.val_adj+r.p_star*p.s_grid[stock_idx]*r.q)
    area = percent_adj*threshold/2  # prob of adj * conditional expected cost
    second = r.p_star*r.wage* area
    third = (1 - percent_adj)*r.vals_s1[stock_idx]
    return first - second + third

def _del_cost_threhold(r, p):
    # return arr of length NS
    thres_arr = -1. * (r.vals_s1 - r.p_star * r.q * p.s_grid - r.val_adj) / r.p_star / r.wage
    thres_arr[thres_arr < 0] = 0
    thres_arr[thres_arr > p.del_cost_max] = p.del_cost_max
    return thres_arr


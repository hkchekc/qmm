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
    fd = -1.*fc  # is it a typo?
    
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
        self.z_bar = 1.032  # no uncertainty
        self.store_cost = .0287
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
        self.crit_vf = 1e-6
        self.crit_clear = 1e-8
        self.crit_used_up = 1e-8
        self.p_max = 3.3
        self.p_min = 3.2
        
        
        @njit
        def _get_price(self, price):
            w = self.eta/price
            ugly = (1-self.beta*(1-self.delta))/self.beta/self.alpha
            inv_alpha = 1- self.alpha
            q = (price**(-inv_alpha))*(ugly**self.alpha)*((self.eta/inv_alpha)**inv_alpha)
            return w, q  # wage and intermediate good price
        

class result(object):
    
    def __init__(self):
        self.m = 1
        self.n = 1
        self.P = param()
        self.p_star = (self.P.p_max+self.P.p_min)/2
        self.w, self.q = self.P._get_price()
        self.mu = None
        self.vals_s1 = np.zeros(self.P.NS)
        self.val_s_star = 0
        self.val_adj = 0  # scalar
        self.m_arr = np.zeros(self.P.NS)


def vfi(r, p):
    r.vfunc0 = _init_guess_expected_v0(s1, m)
    #TODO: set init v star new
    while max(np.max(np.abs(r.v_s_star_new- r.v_s_star)), np.max(np.abs(v0_new- v0))) > p.vf_crit:
        r.v_s_star = r.v_s_star_new
        _optimal_m(r, p) # find the value for all s1
        _optimal_s(r, p) # set v_s_star_new and adjustment value/ v0 in paper
        for si  in range(p.NS):
            _find_expected_v0(si)  # set new v0 (should be of length NS)
    #TODO: check corner
            

def inventory_sequence(r, p):
    not_used_up = 1
    # highly doubtful if this is faster than np.interp with njit
    pol_spline = interpolate.UnivariateSpline(p.s_grid, r.m_arr)
    while (not_used_up > 0):
        #TODO: populate this s1 arr
        this_s1_arr = np.zeros(p.j_max)
        this_s1 = r.s_star
        m_arr = np.zeros(p.j_max)
        s_arr = np.zeros(p.j_max)
        # adj_cdf = np.ones(p.j_max)  # if not changed (loop breaked )
        adj_pmf = np.zeros(p.j_max)
        for ti, time in range(p.j_max):
            if (this_s1 < p.epsilon):
                m_arr[ti] = this_s1
                threshold = _del_cost_threhold(this_s1)
                adj_pmf[ti]  = p.cost_dist.cdf(threshold)      
                if ti != p.j_max:
                    adj_pmf[ti+1]  = 1-np.sum(adj_pmf) 
                else:
                    adj_pmf[ti]  += 1-np.sum(adj_pmf) 
                this_s1_arr[ti] = this_s1
                this_s1 = 0
                break
                # pmf remains zero for later period when all firms adjusted (as all inventory are done)
            else:
                # no this_s_star have to be solved I think, maybe interp valuefunction
                # m_arr[ti] =  optimal_m(r, p, this_s1) # find the value for all s1
                m_arr[ti] = pol_spline(this_s1)
                # find the share of firms adjusting at time t
                threshold = _del_cost_threhold(this_s1)
                adj_pmf[ti]  = p.cost_dist.cdf(threshold)
                this_s1_arr[ti] = this_s1
                this_s1 -= m_arr[ti]
                
        # if there is next loop (not all used up), increase jmax by 1
        not_used_up = 1- np.sum(adj_pmf)
    
        
    
def final_good_dist(r, p):
    # use the method where firm differs only in adjustment periods
    stat_dist = np.zeros(p.NJ)
    stat_dist[0] = 1.
    adj_cdf = np.cumsum(r.adj_pmf)
    not_adj = 1 - adj_cdf
    # the grids correspond to the this_s1_arr in inventory_sequence
    for ti in range(p.NJ):
        pass
    #TODO: use kierans method
    # use transition matrix
    h = adj_pmf/not_adj  # chance of adjusting at period t
    trans_mat = np.zeros(p.NJ, p.NJ)
    
    return stat_dist
    
        
def market_clear(p, r):
    # find demand for m, n
    agg_cap = 0
    m_demand = 0.
    for ji in range(p.NJ):
        # demand is how much is restocked
        m_demand += r.h[ji]*r.stat_dist[ji]*(r.s_star - r.s_arr[ji])
    n_demand = _find_n(m_demand)
    # find agg capital and labor
    labor_supply = 1/p.eta
    final_good_labor = labor_supply - n_demand# given by HH problem
    agg_cap = _find_agg_cap(final_good_labor, m_demand, p.zbar, p.alpha)
    # find consumption
    consum = 0
    for ji in range(p.NJ):
        pass
    # update p
    new_p = 1/ consum
    
def _find_agg_cap(lab, demand, z, alpha):
    lab_con = lab**(1-alpha)
    return (demand/z/lab_con)**(1/alpha)
        
# TODO: change to return
def _optimal_s(r, p):
    for si, s1 in enumerate(p.s_grid):
        r.vals_s1[si] =
    r.val_s_star_new = np.max(r.vals_s1)
    r.s_star = p.s_grid[np.argmax(r.vals_s1)]
    r.v_a = r.val_s_star_new
    

def _optimal_m(r, p):  # void function
    
    def find_m(m, s1):
        this_val = r.p_star*_final_net_prod(m, n, theta_m, theta_n, s1, store_cost, wage)
        next_u = p.beta*_find_expected_v0(s1, m)
        return -(this_val+next_u)  # because the function is minimized
    for si, s1 in enumerate(s_grid):
        m0 = s1
        res = optimize.minimize(find_m, m0, args=(s1), method='Nelder-Mead', tol=p.crit_vf)
        r.m_arr[si] = res.x
        # find a way to store q (net_prod)

def _init_guess_expected_v0(s1, m, p_star, theta_m, theta_n, eta):
    inv_the_n = 1-theta_n
    first = p_star**(1/(1-theta_n))
    second  = (theta_n/eta)**(theta_n/inv_the_n)
    third = (s1-m)**(theta_m/inv_the_n)
    return first*inv_the_n*second*third
    
def _find_n(m, theta_n, theta_m, eta, price):
    pow_i = 1/(1-theta_n)
    return (theta_n*price*(m**theta_m)/eta)**pow_i


def _final_production(m, n, theta_m, theta_n):
    return (m**theta_m)*(n**theta_n)

def _final_net_prod(m, n, theta_m, theta_n, s1, store_cost, wage):  # q
    cost = store_cost*(s1-m) + wage*n
    return _final_production(m, n, theta_m, theta_n) - cost

def _final_prod(z, k, l, alpha):
    return z*(k**alpha)*(l**(1-alpha))

def _find_expected_v0(stock_idx, r, p):
    r.threshold = _del_cost_threhold(p.s_grid[stock_idx])
    r.percent_adj  =p.cost_dist.cdf(r.threshold)
    first =r.percent_adj*(r.val_adj+r.p_star*p.s_grid[stock_idx]*r.q)
    area = r.percent_adj*r.threshold
    second = r.p_star*r.wage* area
    third = (1 - r.percent_adj)*r.val_s_star[stock_idx]
    return first - second + third
    

def _del_cost_threhold(r, p):
    def find_threshold(threshold, r):
        lhs = r.v_a - r.p_star*r.wage*threshold
        rhs = r.val_s_star - r.p_star*r.q*r.s_star
        return rhs- lhs
    res = optimize.brentq(find_threshold, a=p.del_cost_min, b=p.del_cost_max, args=(r, p))
    #TODO:
    return res.x

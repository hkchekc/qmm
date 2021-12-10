#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from inventory_util import *
from time import time


p  = param()
r = result()
big_loop_time = time()
p_err_li = [100, ]
big_it = 0
r.vfunc0 = init_guess_expected_v0(p.s_grid, r.p_star, p.theta_m, p.theta_n, p.eta)
r.vfunc_s1 = r.vfunc0 + r.p_star*0.3
while (p_err_li[-1] > p.crit_clear):
    r.wage, r.q = get_price(p, r.p_star)
    vf_start = time()
    vfi(r, p)
    print("done with VFI, time is {}".format(time()-vf_start))
    print("============================")
    print("s_star is {}".format(r.s_star))
    inventory_sequence(r, p)
    r.stat_dist = final_good_dist(r, p)
    r.p_star , p_err = market_clear(r, p)
    
    p_err_li.append(p_err)
    big_it += 1
    if big_it %10 == 0:
        print("no convergence", big_it)
        # break
print("done with all, time is {}".format(time()-big_loop_time))

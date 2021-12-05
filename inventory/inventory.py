#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from inventory_util import *
from time import time


p  = param()
r = result()
big_loop_time = time()
p_err_li = [100, ]
big_it = 0
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
    if big_it == 10:
        print("no convergence")
        break
print("done with all, time is {}".format(time()-big_loop_time))

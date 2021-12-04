#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:21:07 2021

@author: chek_choi
"""
from inventory_util import *
from time import time


p  = param()
r = result()
big_loop_time = time()
p_err = 100
big_it = 0
while (p_err > p.crit_clear):
    vf_start = time()
    vfi(r, p)
    print("done with VFI, time is {}".format(time()-vf_start))
    print("============================")
    print("s_star is {}".format(r.s_star))
    inventory_sequence(r, p)
    r.stat_dist, r.h = final_good_dist(r, p)
    r.p_star , p_err = market_clear(r, p)
    w, q = get_price(p, r.p_star)
    
    p_err = 0
    big_it += 1
    if big_it == 20:
        print("no convergence")
        break
print("done with all, time is {}".format(time()-big_loop_time))

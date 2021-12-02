#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:21:07 2021

@author: chek_choi
"""
from inventory_util import *

p  = param()
r = result()

p_err = 100
while (p_err > p.crit_clear):
    vfi(r, p)
    inventory_sequence(r, p)
    r.stat_dist = final_good_dist(r, p)
    market_clear(r, p)
    p_err = p_star - p_star_new 
    p_err = 0
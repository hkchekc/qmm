#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:21:34 2021

@author: chek_choi
"""
import numpy as np
import math

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
        if fc >= cd:
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


            
    

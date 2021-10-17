import numpy as np
import math
import matplotlib.pyplot as plt
from numba import jit

# parameters
profile = np.genfromtxt("input/lc_profile.csv", delimiter=",")
profile = profile[1:, 1]
beta = .94
interest = 1.04
work_time = 40
retire_time = 20
last_surv_rate = 0.92
life_time = work_time+retire_time
discount_arr = np.ones(life_time)
survive_reudction = (1-last_surv_rate)/retire_time
for t in range(work_time, life_time):
    discount_arr[t] = discount_arr[t-1]- survive_reudction
discount_arr *= beta
gamma = 1.5
inv_gamma = 1/gamma
pension_rate = 0.6
NA = 1000
NZ = 10
a_min = 0.
a_max = 80
tmr_a_grid = np.linspace(a_min, a_max, NA)
markov = np.array([[0.9586, 0.0314, 0.0000, 0.0000, 0.0000, 0.0097, 0.0003, 0.0000, 0.0000, 0.0000],
                    [0.0232, 0.9137, 0.0531, 0.0000, 0.0000, 0.0002, 0.0092, 0.0005, 0.0000, 0.000],
                    [0.0000, 0.0474, 0.8952, 0.0474, 0.0000, 0.0000, 0.0005, 0.0090, 0.0005, 0.000],
                    [0.0000, 0.0000, 0.0531, 0.9137, 0.0232, 0.0000, 0.0000, 0.0005, 0.0092, 0.000],
                    [0.0000, 0.0000, 0.0000, 0.0314, 0.9586, 0.0000, 0.0000, 0.0000, 0.0003, 0.009],
                    [0.9586, 0.0314, 0.0000, 0.0000, 0.0000, 0.0097, 0.0003, 0.0000, 0.0000, 0.000],
                    [0.0232, 0.9137, 0.0531, 0.0000, 0.0000, 0.0002, 0.0092, 0.0005, 0.0000, 0.0000],
                    [0.0000, 0.0474, 0.8952, 0.0474, 0.0000, 0.0000, 0.0005, 0.0090, 0.0005, 0.000],
                    [0.0000, 0.0000, 0.0531, 0.9137, 0.0232, 0.0000, 0.0000, 0.0005, 0.0092, 0.0002],
                    [0.0000, 0.0000, 0.0000, 0.0314, 0.9586, 0.0000, 0.0000, 0.0000, 0.0003, 0.0097]])
markov_t = markov.T
states = np.array([0.5324, 0.7415, 1., 1.3487, 1.8784, 0.5324, 0.7415, 1., 1.3487, 1.8784])
trans_shock = .4
states[5:] *= trans_shock

#result arrays
exo_consum_arr = np.zeros((NA, NZ, life_time))
implied_cash_on_hand = np.zeros((NA, NZ,life_time))
greatest_constrainted_cash_on_hand = np.zeros(life_time)
# tmp array
implied_consum_arr = np.zeros((NA, NZ))
next_value = np.zeros((NA, NZ))

# start with last period
pension_income = profile[work_time-1]*pension_rate

# Populate last period
for aidx in range(NA):
    implied_consum_arr[aidx, :] = interest*tmr_a_grid[aidx]+pension_income
    exo_consum_arr[aidx, :, life_time-1] = implied_consum_arr[aidx, :]
    next_value[aidx, :] = interest/(exo_consum_arr[aidx, 0, life_time-1]**gamma)

# precalculate retired cash on hand, as there is no time dependent element
exo_pension_cash_on_hand = interest*tmr_a_grid+pension_income
# leave it out and calc in loop
exo_working_cash_on_hand = np.zeros((NA,NZ,life_time))

# Backward induction
for tidx in range(life_time-2, 0, -1):
    if tidx >= work_time:
        for aidx in range(NA):
            implied_consum_arr[aidx, :] = 1/((discount_arr[tidx+1]*next_value[aidx, 0])**inv_gamma)
            implied_cash_on_hand[aidx, :, tidx] = implied_consum_arr[aidx, :]+tmr_a_grid[aidx]
        greatest_constrainted_cash_on_hand[tidx] = implied_cash_on_hand[0, 0, tidx]
        # first populate the first state and copy it to other state just for completeness
        exo_consum_arr[:, 0, tidx] = np.interp(exo_pension_cash_on_hand,
                                               implied_cash_on_hand[:, 0, tidx], implied_consum_arr[:, 0])

        # Make sure no negative consumption or over asset constraint
        mx = np.ma.masked_less_equal(exo_pension_cash_on_hand , greatest_constrainted_cash_on_hand[tidx]).mask
        exo_consum_arr[:, 0, tidx][mx] = exo_pension_cash_on_hand[mx]
        for i in range(1, NZ):
            exo_consum_arr[:, i, tidx] = exo_consum_arr[:, 0, tidx]
        # not using next period value after getting this.consumption, therefore I can flush it here.
        next_value[:, 0] = interest/(exo_consum_arr[:,0,tidx]**gamma)
        for i in range(1, NZ):
            next_value[:, i] = next_value[:, 0]
    else:
        for zidx in range(NZ):
            for aidx in range(NA):
                implied_consum_arr[aidx, zidx] = 1 / ((discount_arr[tidx + 1] * next_value[aidx, zidx]) ** inv_gamma)
                implied_cash_on_hand[aidx, zidx, tidx] = implied_consum_arr[aidx, zidx] + tmr_a_grid[aidx]
            # see where it binds
            greatest_constrainted_cash_on_hand[tidx] = implied_cash_on_hand[0, zidx, tidx]
            # first populate the first state and copy it to other state just for completeness
            exo_working_cash_on_hand[:, zidx, tidx] = interest*tmr_a_grid+states[zidx]*profile[tidx]
            exo_consum_arr[:, zidx, tidx] = np.interp(exo_working_cash_on_hand[:, zidx, tidx],
                                                   implied_cash_on_hand[:, zidx, tidx], implied_consum_arr[:, zidx])

            # Make sure no negative consumption or over asset constraint
            mx = np.ma.masked_less_equal(exo_working_cash_on_hand[:, zidx, tidx], greatest_constrainted_cash_on_hand[tidx]).mask
            exo_consum_arr[:, zidx, tidx][mx] = exo_working_cash_on_hand[:, zidx, tidx][mx]
        # not using next period value after getting this.consumption, therefore I can flush it here.
        for zidx in range(NZ):
            for aidx in range(NA):
                this_val = interest /exo_consum_arr[aidx, zidx, tidx] ** gamma
                for last_zidx in range(NZ):
                    next_value[aidx, last_zidx] += markov_t[zidx, last_zidx] * this_val

# simulation
rng = np.random.default_rng()
init_income = 1
NHOUSE = 5000
init_cap_mean = -2.5
init_cap_se = 2
initial_capital = np.exp(rng.normal(init_cap_mean, init_cap_se))
init_cap_cap = math.exp(3*init_cap_se+init_cap_mean)
initial_capital[initial_capital>init_cap_cap] = init_cap_cap
perm_shock_mean = 0
perm_shock_se = 0.015
income_process = np.ones((NHOUSE, life_time))  # initial period is one, don't change
# worktime
for tidx in range(1, work_time):
    this_income = np.ones(NHOUSE)*profile[tidx]
# retired
for tidx in range(work_time, life_time):
    # randomize death
    pass



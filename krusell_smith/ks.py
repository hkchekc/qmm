import numpy as np
import scipy as sp
from ks_util import *
from time import time

start = time()
p = param()
r = res()

init_shocks(p, r)
get_prices(p, r)

while r.error < 0.98:
    start = time()

    vfi(p, r)
    # print(r.pfunc[])
    # print(r.pfunc[])
    pseudo_panel(p, r)
    calc_errors(p, r)
    end = time()
    print("Elapse time is {} s".format(end-start))

end = time()
print("Elapse time is {} s".format(end-start))



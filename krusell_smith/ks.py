from ks_util import *
from time import time

big_start = time()
p = param()
r = res()

init_shocks(p, r)
get_prices(p, r)

while r.rsq < 0.98 or r.error > 1e-5:
    start = time()

    vfi(p, r)
    # print(r.pfunc[])
    # print(r.pfunc[])
    pseudo_panel(p, r)
    calc_errors(p, r)
    end = time()
    print("Elapse time is {} s".format(end-start))

end = time()
print("Elapse time is {} s".format(end-big_start))



from arellano_func import *
import matplotlib.pyplot as plt

p = Param()
r = Res()

r.err_q = 100.
while r.err_q > p.crit_q:
    if p.egm and p.EV_shock:
        pass
    else:
        vfi(r, p)
    calc_q(r, p)
    # cal_moments(r, p)
    print("Q Error is {}".format(r.err_q))

# Plotting
# plt.plot( p.b_grid ,r.vfunc_o[:,0], color='blue', label="clean, low state")
# plt.plot( p.b_grid,r.vfunc_o[:,1], color='green', label="clean, high state")
plt.plot( p.b_grid, r.q[:, 1], color='BLACK', label="Q")
plt.legend()
plt.suptitle("Value Functions - VFI, EV Taste Shock")
plt.suptitle("Q Laffer, EV Taste Shock")
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

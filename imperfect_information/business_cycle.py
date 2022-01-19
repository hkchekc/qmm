import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1000)
beta = .99
k = .25
phi = 1.5

sigma_u = np.sqrt(.01)
sigma_eta = np.sqrt(1)

T = 20
a0 = 0.
at = np.zeros(T)
ut = rng.normal(0, sigma_u, size=T)
shock = np.zeros(T)
shock[0] = 1.

for ti in range(T):
    if ti != 0:
        last_a = at[ti -1]
    else:
        last_a = a0
    at[ti] = last_a + ut[ti]

# perfect information
y = at

# common information
st = at + shock
# One step ahead variance
p10 = sigma_u**2
one_step_var = np.zeros(T)
one_step_var[0] = p10
for ti in range(T):
    last_var = one_step_var[ti -1]
    one_step_var[ti] = last_var - (last_var**2)/(last_var+sigma_eta**2)+ sigma_u**2
#kalman filters
K = one_step_var/(one_step_var+sigma_eta**2)
# expected at condition on t
eat = np.zeros(T)
for ti in range(T):
    if ti != 0:
        last_eat = at[ti - 1]
    else:
        last_eat = a0
    eat[ti] = last_eat + K[ti]*(st[ti]-last_eat)

plt.plot( np.arange(T),y, color='blue', label="perfect information y")
plt.plot( np.arange(T), eat , color='green', label="common imperfect y")
plt.legend()
plt.suptitle("y IRF - sigma_u={0:.2f}, sigma_eta={1:.2f}".format(sigma_u, sigma_eta))
plt.tight_layout()
plt.subplots_adjust(top=0.88)
# plt.savefig('data_output/{}.png'.format(fname))
plt.show()



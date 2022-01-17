from scipy import optimize
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

M = np.array([[.5, .2], [.0, .9]])
N = np.array([[1., .0], [.0, .5]])
D = np.array([1., 1.]).reshape((1, 2))
sigma = 2.  # = 2.  (private signal) / 0. (average signal)

def ricatti(P, A=M, C=N, D=D, sigma=sigma):
    DP = np.matmul(D, P)
    tmp5 = gain(P, D=D, sigma=sigma)
    tmp6 = np.matmul(tmp5, DP)
    tmp7 = P - tmp6
    A_sandwich = np.matmul(np.matmul(A, tmp7), A.T)
    CC_prime = np.matmul(C, C.T)
    return A_sandwich + CC_prime - P

def gain(P, D=D, sigma=sigma):
    PD_inv = np.matmul(P, D.T) # PD'
    DP_sandwich = np.matmul(D, PD_inv)  # DPD'
    tmp3 = linalg.inv(DP_sandwich+sigma) # [DPD'+S]^-1
    return np.matmul(PD_inv, tmp3)

P0 = optimize.broyden1(ricatti, xin=np.ones((2,2)))
K0 = gain(P0)
print("P is {} and K is {}".format(P0, K0))

## Part D - IRF ########
#################################################
rng = np.random.default_rng(1000)
NH = 10000  # agents to simulate
T = 20
time_range = np.arange(1, T+1).astype(np.int8)
# Simulation
init_shock = np.array([1., 0.]).reshape((2, 1))  # happens at t=1
init_X = np.array([0., 0.]).reshape((2, 1))  # period 0 values
X_path = np.zeros((T, 2))  # from period 1
X_path[0, :] = (init_X+np.matmul(N, init_shock)).reshape(2)
for ti in range(1, T):
    X_path[ti, :] = np.matmul(M, X_path[ti-1, :])   # u_t = 0 for all t > 0
eps = rng.normal(0, 2, (NH, T))
z_it = eps
for ti in range(T):
    z_it[:, ti] = eps[:, ti] + np.sum(X_path[ti, :])


# individual signals
expected_x = np.zeros((NH, 2, T))
expected_x[:, :, 0] = np.matmul(z_it[:, 0].reshape((NH, 1)), K0.T)  # first period people get full update from signal
for ti in range(1, T):
    last_x = expected_x[:, :, ti-1]
    one_step_ahead_x = np.matmul(M, last_x.T)  # 2 x NH
    one_step_ahead_z = np.sum(one_step_ahead_x, axis=0)  # NH
    z_diff = z_it[:, ti] - one_step_ahead_z  # NH
    adjustment = np.matmul(K0, z_diff.reshape((1, NH)))  # 2 x NH
    expected_x[:, :, ti] = adjustment.T + one_step_ahead_x.T  # NH x 2

avg_x1_belief = np.mean(expected_x[:, 0, :], axis=0)
avg_x2_belief = np.mean(expected_x[:, 1, :], axis=0)
plt.plot(time_range, avg_x1_belief, label="x1 beliefs")
plt.plot(time_range, avg_x2_belief, label="x2 beliefs")
plt.plot(time_range, X_path[:, 0], label="true x1")
plt.plot(time_range, X_path[:, 1], label="true x2")
plt.suptitle("Only private signals observed")
plt.legend()
plt.savefig('{}.png'.format("q3_private"))

# average signal (change sigma to 0 first)
# z_t = np.mean(z_it, axis=0)
# expected_x_mean = np.zeros((2, T))
# expected_x_mean[:, 0] = K0.T*z_t[0]
# for ti in range(1, T):
#     last_x = expected_x_mean[:, ti-1]
#     one_step_ahead_x = np.matmul(M, last_x.T) # 2 x 1
#     one_step_ahead_z = np.sum(one_step_ahead_x)  # scalar
#     z_diff = z_t[ti] - one_step_ahead_z  # scalar
#     adjustment = K0.T*z_diff  # 2 x 1
#     expected_x_mean[:, ti] = adjustment + one_step_ahead_x  # 2 x 1
#
# plt.plot(time_range, expected_x_mean[0], label="x1 beliefs")
# plt.plot(time_range, expected_x_mean[1], label="x2 beliefs")
# plt.plot(time_range, X_path[:, 0], label="true x1")
# plt.plot(time_range, X_path[:, 1], label="true x2")
# plt.suptitle("Average signals observed")
# plt.legend()
# plt.savefig('{}.png'.format("q3_average_signal"))
#
plt.show()
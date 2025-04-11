import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle as pk
from scipy.stats import t
from joblib import Parallel, delayed

from vape_algorithm_adversarial import *
from fan_algorithm_adversarial import *
from antitonic_algorithm_adversarial import *

# ------------
# HYPERPARAMS
# ------------
d = 3
N_context = 2
B_x = 1
B_theta = 1
B_noise = 1
L_xi = 1
B_y = B_x * B_theta + B_noise

# Initialize parameter theta
theta_star = np.ones(3)
theta_star /= np.linalg.norm(theta_star, 2)

# Set time horizons and number of runs
T_list = [500, 1000, 2000, 4000, 8000]
num_runs = 36

# Prepare arrays to store cumulative regrets
reg_AD     = np.zeros((len(T_list), num_runs))  # Antitonic algorithm
reg_fanAD  = np.zeros((len(T_list), num_runs))  # Fan algorithm
reg_vapeAD = np.zeros((len(T_list), num_runs))  # Vape algorithm


def single_run(j, T_list, d, N_context, B_x, B_theta, B_noise, L_xi, B_y, theta_star):
    reg_ant_arr  = np.zeros(len(T_list))
    reg_fan_arr  = np.zeros(len(T_list))
    reg_vape_arr = np.zeros(len(T_list))

    X_context = np.zeros((2, d))
    X_context[0, 0] = np.random.rand()
    X_context[0, 2] = np.random.rand()
    X_context[1, 1] = 1.0
    # Normalize row-wise
    X_context /= np.linalg.norm(X_context, axis=1)[:, None]
    

    for i, T in enumerate(T_list):
        # -----------------------
        # 1) Run Antitonic
        # -----------------------
        my_rew_ant, opt_rew_ant = antitonic_alg_adv(
            T=T, d=d, contexts=X_context, N_contexts=N_context,
            theta_star=theta_star, bound_x=B_x, bound_noise=B_noise, 
            bound_theta=B_theta
        )
        reg_ant_arr[i] = opt_rew_ant.sum() - my_rew_ant.sum()
        
        # -----------------------
        # 2) Run Fan
        # -----------------------
        my_rew_fan, opt_rew_fan = fan_alg_adv(
            T=T, d=d, contexts=X_context, N_contexts=N_context,
            theta_star=theta_star, bound_x=B_x, bound_noise=B_noise, 
            bound_theta=B_theta
        )
        reg_fan_arr[i] = opt_rew_fan.sum() - my_rew_fan.sum()
        
        # -----------------------
        # 3) Run Vape
        # -----------------------
        epsilon_T = get_epsilon(d, T)
        alpha_T   = get_alpha(T, B_noise)
        mu_T      = get_mu(d, epsilon_T, B_theta, B_y, B_x, T, alpha_T)
        
        my_rew_vape, opt_rew_vape = vape_adv(
            T=T, theta_star=theta_star, L_xi=L_xi, mu=mu_T, alpha=alpha_T, 
            epsilon=epsilon_T, X_context=X_context, N_contexts=N_context, 
            d=d, bound_theta=B_theta, bound_x=B_x, bound_noise=B_noise
        )
        reg_vape_arr[i] = opt_rew_vape.sum() - my_rew_vape.sum()

    return reg_ant_arr, reg_fan_arr, reg_vape_arr


def main():
    # Use joblib to distribute runs across available CPU cores
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(single_run)(
            j, T_list, d, N_context, B_x, B_theta, B_noise, 
            L_xi, B_y, theta_star
        ) for j in range(num_runs)
    )

    for j, (reg_ant_arr, reg_fan_arr, reg_vape_arr) in enumerate(results):
        reg_AD[:, j]     = reg_ant_arr
        reg_fanAD[:, j]  = reg_fan_arr
        reg_vapeAD[:, j] = reg_vape_arr
    
    print("\n----- RESULTS -----")
    print("Antitonic regrets:\n", reg_AD)
    print("Fan regrets:\n", reg_fanAD)
    print("Vape regrets:\n", reg_vapeAD)
    
    mean_reg_AD     = reg_AD.mean(axis=1)
    mean_reg_fanAD  = reg_fanAD.mean(axis=1)
    mean_reg_vapeAD = reg_vapeAD.mean(axis=1)
    
    print("\nMean regrets across runs for each T:")
    for i, T in enumerate(T_list):
        print(f"T = {T}: Antitonic = {mean_reg_AD[i]:.2f}, "
              f"Fan = {mean_reg_fanAD[i]:.2f}, "
              f"Vape = {mean_reg_vapeAD[i]:.2f}")

if __name__ == "__main__":
    main()

def compute_statistics(reg_matrix):
    mean_reg = np.mean(reg_matrix, axis=1)
    std_error = np.std(reg_matrix, axis=1, ddof=1) / np.sqrt(num_runs)
    t_critical = t.ppf(0.975, df=num_runs - 1)
    ci_lower = mean_reg - t_critical * std_error
    ci_upper = mean_reg + t_critical * std_error
    return mean_reg, ci_lower, ci_upper

mean_reg_AD, ci_lower_AD, ci_upper_AD = compute_statistics(reg_AD)
mean_reg_fanAD, ci_lower_fanAD, ci_upper_fanAD = compute_statistics(reg_fanAD)
mean_reg_vapeAD, ci_lower_vapeAD, ci_upper_vapeAD = compute_statistics(reg_vapeAD)


plt.figure(figsize=(10,6))
plt.plot(T_list, mean_reg_AD, label='Antitonic Algorithm', color='blue', marker='o')
plt.fill_between(T_list, ci_lower_AD, ci_upper_AD, color='blue', alpha=0.2)
plt.plot(T_list, mean_reg_fanAD, label='Fan Algorithm', color='red', marker='o')
plt.fill_between(T_list, ci_lower_fanAD, ci_upper_fanAD, color='red', alpha=0.2)
plt.plot(T_list, mean_reg_vapeAD, label='Vape Algorithm', color='green', marker='o')
plt.fill_between(T_list, ci_lower_vapeAD, ci_upper_vapeAD, color='green', alpha=0.2)

plt.xlabel('Time Horizon T',fontsize = 20)
plt.ylabel('Cumulative Regret',fontsize = 20)
plt.legend(fontsize = 18)
plt.savefig('Regret_comparison_vape', dpi=300)
plt.show()

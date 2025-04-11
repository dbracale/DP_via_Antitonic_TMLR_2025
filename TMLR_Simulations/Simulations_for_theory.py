from tqdm import tqdm
from joblib import Parallel, delayed
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, NonlinearConstraint
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy.linalg import LinAlgError
import pandas as pd
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
import scipy.stats as stats
from tqdm import tqdm
from scipy.stats import gamma
from scipy.optimize import newton, bisect
from joblib import Parallel, delayed
import time
from sklearn.isotonic import IsotonicRegression
plt.rcParams['text.usetex'] = True
import random

#################################
####### Pricing Function ########
#################################

def g_opt(q, F, H):
    x0 = 0 
    result = minimize(lambda p: -p * (1-F(p - q)), x0=x0, method='Nelder-Mead')
    optimal_p = result.x[0]
    return 0 if optimal_p < 0 else H if optimal_p > H else optimal_p

def g_opt_isotonic(q, F, H):
    import numpy as np
    num_points = 2000 
    p_values = np.linspace(1e-4, H, num_points)
    objective_values = [p * (1 - F(p - q)) for p in p_values]
    max_index = np.argmax(objective_values)
    optimal_p = p_values[max_index]
    return optimal_p

def nu(alpha):
    return 2/(2+alpha) if alpha < 0.5 else (2*alpha+1)/(3*alpha+1)


############################################
############## EXPLORATION #################
############################################

def exploration_and_estimation(k, Sample_x, Sample_z, H, tau_0, d, theta_0, cdf_0, domain_z, alpha):
    tau_k = tau_0 * (2**(k-1))
    tau_kplus1 = tau_0 * (2**k)

    a_k = np.ceil((d**(alpha/(2+alpha)))*((tau_k)**(nu(alpha))))
    I_k = np.arange(tau_k - tau_0, tau_k - tau_0 + a_k)
    I_k_prime = I_k
    E_k = np.arange(tau_k - tau_0 + a_k, tau_kplus1 - tau_0)


    X_tilde = Sample_x(len(I_k), d)
    Z_t = Sample_z(len(I_k))
    c_t = np.random.uniform(0, H, len(I_k))
    V_t = Z_t + np.dot(X_tilde, theta_0)
    Y_t = V_t > c_t
    c_explor_opt = [g_opt(a, cdf_0, H) for a in np.dot(X_tilde, theta_0)]
    regret_explor = c_explor_opt * (V_t > c_explor_opt).astype(int) - c_t * (V_t > c_t).astype(int)
    mod = OLS(H * Y_t, X_tilde).fit()
    theta_hat_k = mod.params
    
    w_t = np.random.uniform(-domain_z, domain_z, len(I_k_prime))
    x_tilde = Sample_x(len(I_k_prime), d)
    p_t = w_t + np.dot(x_tilde, theta_0)
    z_t = Sample_z(len(I_k_prime))
    v_t = np.dot(x_tilde, theta_hat_k)+z_t
    y_t = (p_t <= v_t).astype(int)

    ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
    ir.fit(w_t, y_t)

    def G(u):
        return 1-ir.predict(np.array([u]))[0]

    F_k = G
    
    return {"E_k":E_k, "theta_hat_k":theta_hat_k, "F_k":F_k, "c_explor_opt":c_explor_opt, "regret_explor":regret_explor}


############################################
############## EXPLOITATION ################
############################################

def exploitation(E_k, theta_hat_k, F_k, Sample_x, Sample_z, H, theta_0, cdf_0, d):
    # NPMLE
    c_hat = []
    W_exploit = []
    Y_exploit = []
    c_exploit_opt = []
    regret_exploit = []

    for t in E_k:
        x_exploit = Sample_x(1, d)        
        g_k = g_opt_isotonic(x_exploit @ theta_hat_k, F_k, H)
        c_exploit = np.clip(g_k, 0, H)
        Z_t = Sample_z(1)
        V_t = Z_t + x_exploit @ theta_0
        y_exploit = V_t > c_exploit
        w_exploit = c_exploit - x_exploit @ theta_hat_k
        opt_price = g_opt(x_exploit @ theta_0, cdf_0, H)
        reg_value = opt_price * (V_t > opt_price) - c_exploit * (V_t > c_exploit)
        regret_exploit.append(reg_value)
        c_exploit_opt.append(opt_price)
    
        c_hat.append(c_exploit)
        W_exploit.append(w_exploit)
        Y_exploit.append(y_exploit)

    return {"c_exploit_opt":c_exploit_opt, "regret_exploit":regret_exploit}


##################################################
############## Updates & Collect #################
##################################################

def updates_and_collenct_info(c_exploit_opt, regret_exploit, c_explor_opt, regret_explor, Regret):   
  c_opt = np.concatenate((c_explor_opt, c_exploit_opt))
  regret_exploit = [float(arr[0]) for arr in regret_exploit]
  reg_k = np.concatenate((regret_explor, regret_exploit)).tolist()
  Regret = np.atleast_1d(Regret)
  Regret = np.concatenate((Regret, reg_k))

  return {"Regret": Regret}
    
###############################################
############ Sample from Z ####################
###############################################

def Sample_z_cdf(cdf_func, cdf_func_inv, lower, upper, size):
    lower_cdf = cdf_func(lower)
    upper_cdf = cdf_func(upper)
    uniform_samples = np.random.uniform(lower_cdf, upper_cdf, size=size)
    truncated_samples = cdf_func_inv(uniform_samples)
    return truncated_samples

###############################################
############ Sample from X ####################
###############################################

const_x = 128 * np.sqrt(6) / 2835
def f_x(x):
    return 0 if np.abs(x)>=np.sqrt(2 / 3) else (-x**6 + 2 * x**4 - 4 * x**2/3 + 8/27) / (2 * const_x)
    
def Sample_x(N, d):
    samples = []
    max_x = np.sqrt(2 / 3)
    min_x = -max_x
    c = max([f_x(x) for x in np.linspace(min_x, max_x, 2000)])

    while len(samples) < N * d:
        x_proposal = np.random.uniform(min_x, max_x)
        u = np.random.uniform(0, c)
        if u <= f_x(x_proposal):
            samples.append(x_proposal)

    X = np.array(samples).reshape(N, d)
    X_tilde = np.column_stack((np.ones(N), X))

    return X_tilde

######################################
############ MAIN ####################
######################################

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def single_evaluation(evalu, d, H, tau_0, theta_0, num_episodes, cdf_0, Sample_x, Sample_z, domain_z, alpha, seed):
    np.random.seed(seed + evalu)  # Different seed for each worker
    random.seed(seed + evalu)
     
    Regret = []
    for k in range(1, num_episodes):
        explor = exploration_and_estimation(k, Sample_x, Sample_z, H, tau_0, d, theta_0, cdf_0, domain_z, alpha)
        E_k, theta_hat_k, F_k, c_explor_opt, regret_explor = explor.values()
        exploit = exploitation(E_k, theta_hat_k, F_k, Sample_x, Sample_z, H, theta_0, cdf_0, d)
        c_exploit_opt, regret_exploit = exploit.values()
        update = updates_and_collenct_info(c_exploit_opt, regret_exploit, c_explor_opt, regret_explor, Regret)
        Regret = update["Regret"]

    return np.array(Regret)


def main_function(d, H, tau_0, theta_0, num_episodes, max_eval, cdf_0, Sample_x, Sample_z, domain_z, alpha, SEED):
    parallel_args = [
        (evalu, d, H, tau_0, theta_0, num_episodes, cdf_0, Sample_x, Sample_z, domain_z, alpha, SEED)
        for evalu in range(max_eval)
    ]

    # Parallel evaluation loop using all cores
    results = Parallel(n_jobs=-1)(
        delayed(single_evaluation)(*args) for args in tqdm(parallel_args, desc="Hello")
    )

    # Stack the results from parallel computation
    Regret_eval = np.column_stack(results)
    return Regret_eval



################################################
############ Choose the cdf_0 ##################
################################################

def get_cdf(name):
    def cdf2(x):
        return stats.norm.cdf(x, loc=0, scale= 0.2)
        
    def cdf2_inv(y):
        return stats.norm.ppf(y, loc=0, scale= 0.2)

    def cdf3(x):
        return stats.laplace.cdf(x, loc=0, scale= 0.2)

    def cdf3_inv(y):
        return stats.laplace.ppf(y, loc=0, scale= 0.2)

    def cdf4(x):
        val = stats.cauchy.cdf(x, loc=0, scale= 0.1)
        return val

    def cdf4_inv(y):
        val = stats.cauchy.ppf(y, loc=0, scale= 0.1)
        return val

    def cdf6(x):
        return 0 if x <= -0.5 else 1 if x >= 0.5 else 6*((x / 4) - (x**3 / 3) + (1 / 12))

    def cdf6_inv(y):
        def root_func(u, target):
            return cdf6(u) - target
        if isinstance(y, np.ndarray):
            return np.array([bisect(root_func, -0.5, 0.5, args=(target,), xtol=1e-8) for target in y])
        else:
            return bisect(root_func, -0.5, 0.5, args=(y,), xtol=1e-8)

    import numpy as np

    def cdf8_1_3(x):
        s = 0.5 / (0.5 ** (1/3))
        if x <= -0.5:
            return 0.0
        elif x >= 0.5:
            return 1.0
        elif x < 0.0:
            return 0.5 - s * ((-x)**(1/3))
        else:
            return 0.5 + s * (x**(1/3))

    def cdf8_inv_1_3(y):
        s = 0.5 / (0.5 ** (1/3))
    
        y_array = np.asarray(y, dtype=float)
        out = np.empty_like(y_array)
    
        mask1 = (y_array <= 0)
        out[mask1] = -0.5

        mask2 = (y_array >= 1)
        out[mask2] = 0.5
    
        mask3 = (y_array > 0) & (y_array < 0.5)
        out[mask3] = -((0.5 - y_array[mask3]) / s) ** 3
    
        mask4 = (y_array >= 0.5) & (y_array < 1)
        out[mask4] = ((y_array[mask4] - 0.5) / s) ** 3
    
        return out if out.shape != () else out.item()

    def cdf8_sqrt(x):
        s = np.sqrt(0.5)
    
        if x <= -0.5:
            return 0.0
        elif x >= 0.5:
            return 1.0
        elif x < 0.0:
            return 0.5 - s * np.sqrt(-x)
        else:
            return 0.5 + s * np.sqrt(x)

    def cdf8_sqrt_inv(y):
        s = np.sqrt(0.5)
        y_array = np.asarray(y, dtype=float)
        out = np.empty_like(y_array)
        mask1 = (y_array <= 0)
        out[mask1] = -0.5
        mask2 = (y_array >= 1)
        out[mask2] = 0.5
        mask3 = (y_array > 0) & (y_array < 0.5)
        out[mask3] = -((0.5 - y_array[mask3]) / s) ** 2
        mask4 = (y_array >= 0.5) & (y_array < 1)
        out[mask4] = ((y_array[mask4] - 0.5) / s) ** 2
        return out if out.shape != () else out.item()

    import numpy as np

    def cdf8_3_4(x):
        s = 0.5 / (0.5 ** (3/4))
    
        if x <= -0.5:
            return 0.0
        elif x >= 0.5:
            return 1.0
        elif x < 0.0:
            return 0.5 - s * ((-x) ** (3/4))
        else:
            return 0.5 + s * (x ** (3/4))

    def cdf8_3_4_inv(y):
        s = 0.5 / (0.5 ** (3/4))
        y_array = np.asarray(y, dtype=float)
        out = np.empty_like(y_array)
        mask_left = (y_array <= 0)
        out[mask_left] = -0.5
        mask_right = (y_array >= 1)
        out[mask_right] = 0.5
        mask_mid_neg = (y_array > 0) & (y_array < 0.5)
        out[mask_mid_neg] = -((0.5 - y_array[mask_mid_neg]) / s) ** (4/3)
        mask_mid_pos = (y_array >= 0.5) & (y_array < 1)
        out[mask_mid_pos] = ((y_array[mask_mid_pos] - 0.5) / s) ** (4/3)
        return out if out.shape != () else out.item()

    cdf_mapping = {
        'Gaussian': (cdf2, cdf2_inv, 1),
        'Laplace': (cdf3, cdf3_inv, 1),
        'Cauchy': (cdf4, cdf4_inv, 1),
        'Fan': (cdf6, cdf6_inv, 1),
        '1_over_3_holder': (cdf8_1_3, cdf8_inv_1_3, 1/3),
        '1_over_2_holder': (cdf8_sqrt, cdf8_sqrt_inv, 1/2),
        '3_over_4_holder': (cdf8_3_4, cdf8_3_4_inv, 3/4),
    }

    if name in cdf_mapping:
        return cdf_mapping[name]
    else:
        print(f"Invalid name: {name}. Please provide a valid name.")
        return None

################################################
######### Interaction with Reader ##############
################################################
valid_options = ['Gaussian', 'Laplace', 'Cauchy', 'Fan', '1_over_3_holder', '1_over_2_holder', '3_over_4_holder']
user_input = input(f"Select the true CDF between: {', '.join(valid_options)}: ")
while user_input not in valid_options:
    print("Invalid choice. Select a valid CDF.")
    user_input = input(f"Select the true CDF ({', '.join(valid_options)}): ")
cdf_0, cdf_0_inv, alpha = get_cdf(user_input)
print(f"Selected true {user_input}")

def Sample_z(size):
    return Sample_z_cdf(cdf_0, cdf_0_inv, -0.5, 0.5, size)


################################################
######### PARAMETER INITIALIZATION #############
################################################
d = 3 # Dimension of features
H = 5 # Maximum price
tau_0 = 100 # Lenght of the first episode
interc_0 = 3 # True intercept of theta
slope_0 = np.repeat(np.sqrt(2 / 3), d) # True slope of theta
theta_0 = np.concatenate(([interc_0], slope_0))
num_episodes = 8
max_eval = 36

################################################
######### SEEDS FOR REPRODUCIBILITY ############
################################################
SEED = 50  # Choose any fixed number
np.random.seed(SEED)
random.seed(SEED)


################################################
############# SIMULATIONS START ################
################################################
domain_z = 0.5
output = main_function(d, H, tau_0, theta_0, num_episodes, max_eval, cdf_0, Sample_x, Sample_z, domain_z, alpha, SEED)


##################################################
################ Plot Results ####################
##################################################
def plot_results_2_with_regression_holder(Regret_eval, tau_0, alpha):
    cumulative_regret = np.cumsum(Regret_eval, axis=0)
    cumulative_regret = cumulative_regret[tau_0:, :]
    T_NPLS = np.arange(tau_0 + 1, tau_0 + cumulative_regret.shape[0] + 1)
    q_5_NPLS = np.percentile(cumulative_regret, 5, axis=1)
    q_95_NPLS = np.percentile(cumulative_regret, 95, axis=1)
    log_T_NPLS = np.log2(T_NPLS).reshape(-1, 1)
    log_regret_NPLS = np.log2(np.mean(cumulative_regret, axis=1))
    reg_NPLS = LinearRegression().fit(log_T_NPLS, log_regret_NPLS)
    reg_line_NPLS = reg_NPLS.predict(log_T_NPLS)

    lightblue = "#6699FF"  # Darker light blue
    lightred = "#FF6666"   # Darker light red
    
    plt.plot(log_T_NPLS, log_regret_NPLS, color=lightblue, linewidth=2, label=f'Empirical slope: {reg_NPLS.coef_[0]:.2f}')
    intercept_adjusted = (reg_line_NPLS[0] - nu(alpha) * log_T_NPLS[0])
    plt.plot(log_T_NPLS, nu(alpha) * log_T_NPLS + intercept_adjusted, color='black', linestyle='--', linewidth=1, label=f'Theoretical slope: {nu(alpha):.2f}')
    plt.fill_between(log_T_NPLS.flatten(), np.log2(q_5_NPLS), np.log2(q_95_NPLS), color=lightblue, alpha=0.4)
    plt.ylabel("log(Reg(t))", fontsize = 20)
    plt.xlabel("log(t)", fontsize = 20)
    plt.legend(fontsize = 14)

tau_delete = 6
delete_first = np.ceil(tau_delete).astype(int)*tau_0
plot_results_2_with_regression_holder(output, delete_first, alpha)
save_path = f'Regret_simulation_{user_input}.png'
plt.savefig(save_path, dpi=100, bbox_inches='tight')
print(f"Plot saved as: {save_path}") 
plt.show()
plt.close()

##################################################
################ SAVE RESULTS ####################
##################################################
np.save(f'arr_{user_input}.npy', output)
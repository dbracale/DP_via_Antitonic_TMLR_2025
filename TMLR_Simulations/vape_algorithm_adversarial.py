import numpy as np
import scipy.stats as stats
import random
import math 
import matplotlib.pyplot as plt
from numpy.linalg import inv


# AUXILIARY FUNCTIONS

# Functions for the VA phase

def norm_invA(x, A):
    return np.sqrt(np.dot(x, np.linalg.solve(A, x)))
    
 
def update_theta_V(x, o, theta, V, sum_ox, bound_y):
    V_new = V + np.outer(x, x)
    sum_ox_new = sum_ox + (o - 0.5) * x
    theta_new = 2 * bound_y * np.linalg.solve(V_new, sum_ox_new)
    return theta_new, V_new, sum_ox_new

# Functions for the PE phase

def precheck_actions(g_hat, K, epsilon, bound_y):
    prechecked_arms = []
    for k in range(2 * K + 1):
        est = g_hat + (k - K) * epsilon
        if est > 0 and est < bound_y:
            prechecked_arms.append(k)
    return prechecked_arms

def compute_ULC_bounds(K, g_hat, D_hat, N, alpha, L_xi, epsilon):
    ks = np.arange(-K, K + 1)
    est = g_hat + ks * epsilon
    width = np.sqrt(2 * np.log(1 / alpha) / N) + (2 * L_xi * epsilon * 1)
    UCB = est * (D_hat + width)
    LCB = est * (D_hat - width)
    return UCB, LCB

def select_arm(UCB, LCB, N, prechecked_arms):
    # compute max LCB over prechecked arms
    max_LCB = np.max(LCB[prechecked_arms])
    # select valid arms with UCB larger than max LCB
    valid_arms = []
    for k in prechecked_arms:
        if UCB[k] > max_LCB:
            valid_arms.append(k)
    # select least played valid arm
    k_select = valid_arms[0]
    for k in valid_arms:
        if N[k] < N[k_select]:
            k_select = k
    return k_select
    
# Functions for hyperparameters

def get_epsilon(d, T):
    return ((d * np.log(T)) ** 2 / T) ** (1/3)

def get_alpha(T, B_noise):
    return 1 / (T + 2 * (T**2) * (3 + (B_noise + 1) * (T**(1/3))))

def get_mu(d, epsilon, B_theta, B_y, B_x, T, alpha):
    sqrt_tmp = np.sqrt(d * np.log((1 + T * (B_x ** 2)) / alpha))
    return epsilon / (B_theta + B_y * sqrt_tmp)    


# VAPE ALGORITHM ADVERSARIAL

def vape_adv(T, theta_star, L_xi, mu, alpha, epsilon, X_context, N_contexts, d=2,
         bound_theta=1, bound_x=1, bound_noise=1):
        
    # init for noise
    bound_y = bound_theta * bound_x + bound_noise
    noise = stats.truncnorm(a=-bound_noise*(10/3), b=bound_noise*(10/3), scale=0.3) 

    # init for estimation of theta_star
    theta_hat = np.zeros(d)
    V = np.eye(d)
    sum_ox = np.zeros(d)
    
    # init for price elimination
    K = math.ceil((bound_noise + 1) / epsilon)
    N = np.ones(2 * K + 1)
    D_hat = np.zeros(2 * K + 1)
    
    # init for regret computation
    my_exp_rewards = np.zeros(T)
    opt_rewards = np.zeros(T)
    grid_optim = bound_noise * (np.array(range(-20*K, 20*K))/(20.0*K) - 0.5)
    
    exp_phase = np.ceil((T * d )**(3/4))
#     print(exp_phase)
    for t in range(T):
        # load current context and generate evaluation
        if t<exp_phase:
            i_context = np.random.randint(low=0, high=N_contexts/2)
        else:
            i_context = np.random.randint(low=N_contexts/2, high=N_contexts)
#         print(i_context)
        x_t = X_context[i_context, :]
        xi_t = noise.rvs(1)
        u_t = np.dot(x_t, theta_star)
        y_t = u_t + xi_t
        
        # compute norm to enter VA or PE
        norm = norm_invA(x_t, V)
        
        # Exploration Phase
        if norm > mu:
            # posts price uniformly at random 
            p_t = np.random.uniform(-bound_y, bound_y)  
            # obtains feedback
            o_t = int(p_t <= y_t)
            # updates estimates
            theta_hat, V, sum_ox = update_theta_V(x_t, o_t, theta_hat, V, sum_ox, bound_y)
            
        # Price Elimination phase
        else:
            # compute UCLBs
            g_hat = np.dot(x_t, theta_hat)
            UCB, LCB = compute_ULC_bounds(K, g_hat, D_hat, N, alpha, L_xi, epsilon)
            # precheck arms
            prechecked_arms = precheck_actions(g_hat, K, epsilon, bound_y)
            # select arm, post price and get feedback
            k_t = select_arm(UCB, LCB, N, prechecked_arms)
            increment = (k_t - K) * epsilon
            p_t = g_hat + increment
            o_t = int(p_t <= y_t)
            # updates quantities
            D_hat[k_t] = (N[k_t] * D_hat[k_t] + o_t) / (N[k_t] + 1)
            N[k_t] += 1
        
        # Compute regret
        def pi(p):
            return p * (1 - noise.cdf(p - u_t))
        
        my_exp_rewards[t] = pi(p_t)        
        largest_expected_revenue = np.max(pi(u_t + grid_optim))
        opt_rewards[t] = largest_expected_revenue

    return my_exp_rewards, opt_rewards
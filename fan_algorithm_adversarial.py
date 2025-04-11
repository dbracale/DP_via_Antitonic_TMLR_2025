
import numpy as np
import scipy.stats as stats
import random
import math 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# AUXILIARY FUNCTIONS

# FUNCTION TO BUILD ESTIMATE OF PARAMETER THETA

def estimate_theta(X, o):
    lr = LinearRegression(fit_intercept=False).fit(X, o)
    theta = lr.coef_
    return theta


# FUNCTIONS TO BUILD THE NW ESTIMATOR


def Kernel(x):
    lb = x > -1
    ub = x < 1
    k = (35 / 12) * (1 - x**2)**3 * lb * ub
    return k


def kernel_num(u, w, o, exp_phase, b):
    val = (w - u) / b
    ker = Kernel(val)
    num = np.dot(ker, o) / (exp_phase * b)
    return num


def kernel_den(u, w, exp_phase, b):
    val = [(x - u) / b for x in w]
    ker = [Kernel(np.array(x)) for x in val]
    ker = np.array(ker)
    den = ker.sum() / (exp_phase * b) + 1e-8
    return den


def estimate_F(u, p, X, theta, o, exp_phase, b):
    w = p - np.dot(X, theta)
    h = kernel_num(u, w, o, exp_phase, b) 
    f = kernel_den(u, w, exp_phase, b) 
    est_F = 1 - h / f                 
    return est_F


# FUNCTIONS TO BUILD THE DERIVATIVE OF THE NW ESTIMATOR


def deriv_Kernel(x):
    lb = x > -1
    ub = x < 1
    k = - 16 * x * (1 - x**2)**2 * lb * ub
    return k


def deriv_num(u, w, o, exp_phase, b):
    val = (w - u) / b
    der_ker = deriv_Kernel(val)
    num = - np.dot(der_ker, o) / (exp_phase * b**2)
    return num


def deriv_den(u, w, exp_phase, b):
    val = [(x - u) / b for x in w]
    der_ker = [deriv_Kernel(np.array(x)) for x in val]
    der_ker = np.array(der_ker)
    den = - der_ker.sum() / (exp_phase * b**2) + 1e-8
    return den


def deriv_estimate_F(u, p, X, theta, o, exp_phase, b):
    w = p - np.dot(X, theta)
    der_h = deriv_num(u, w, o, exp_phase, b)
    der_f = deriv_den(u, w, exp_phase, b)
    h = kernel_num(u, w, o, exp_phase, b)
    f = kernel_den(u, w, exp_phase, b)
    val = - (der_h * f - h * der_f) / f**2
    return val


def deriv_av_reward(u, x_theta, p, X, theta, o, exp_phase, b):
    der_est_F = deriv_estimate_F(u-x_theta, p, X, theta, o, exp_phase, b)
    est_F = estimate_F(u-x_theta, p, X, theta, o, exp_phase, b)
    val = 1 - est_F - u * der_est_F
    return val


# FUNCTION TO FIND THE MAX OF THE AVERAGE REWARD

def maximise_avg_regret(x, low, high, p, X, theta, o, exp_phase, b):
    x_theta = np.dot(x, theta)
    lo = low
    hi = high
    while hi-lo>1/100: 
        mid = (lo + hi) / 2
        der = deriv_av_reward(mid, x_theta, p, X, theta, o, exp_phase, b)
        if der > 0:
            lo = mid
        else:
            hi = mid
    return mid 


# FUNCTION TO BUILD THE GRID TO COMPUTE OPTIMAL REGRET
def get_epsilon(d, T):
    return ((d * np.log(T)) ** 2 / T) ** (1/3)
      


# FAN ALGORITHM ADVERSARIAL

def fan_alg_adv(T, d, contexts, N_contexts, theta_star, bound_x, bound_noise, bound_theta):
    
    # init noise
    B = bound_x * bound_theta + bound_noise
    noise = stats.truncnorm(a=-bound_noise*(10/3), b=bound_noise*(10/3), scale=0.3)
    
    # init regret
    epsilon = get_epsilon(d, T) 
    K = K = math.ceil((bound_noise + 1) / epsilon) 
    grid_optim = bound_noise * (np.array(range(-20*K, 20*K))/(20.0*K) - 0.5) 
    my_exp_rewards = np.zeros(T)
    opt_rewards = np.zeros(T)

    # init auxiliary vectors
    X = []
    o_t_coll = []
    p_t_coll = []

    # init lenght phases 
    exp_phase = np.ceil((T * d )**(3/4))
    bandwidth = 1.06*0.3*(exp_phase)**(-1/5)                           
    t=0
    while t<T:     
        # obtains context 
        if t<exp_phase:
            i = np.random.randint(low=0, high=N_contexts/2)
        else:
            i = np.random.randint(low=N_contexts/2, high=N_contexts)
        x =  contexts[i, :]

        # generates valuations
        u_t = np.dot(theta_star,x)
        xi_t = noise.rvs(1)
        v = u_t + xi_t 
        
        # Exploration phase
        if t<exp_phase:
            #posts price
            p_t= np.random.uniform(-B, B) 
           
            # obtains feedback and stores quantities 
            o_t_coll.append(int(p_t<=v))
            p_t_coll.append(p_t)
            X.append(x)

            if t==exp_phase-1:
                # updates estimates 
                o_coll = np.array(o_t_coll)
                p_coll = np.array(p_t_coll)
                X_coll = np.array(X)
                theta_est = estimate_theta(X_coll, B*o_coll)
          
        # Exploitation phase    
        else:
            p_t = maximise_avg_regret(x, 0, B, p_coll, X_coll, theta_est, o_coll, exp_phase, bandwidth)
                                 
        # accumulates regret
        def pi(p):
            return p * (1 - noise.cdf(p - u_t))
        
        my_exp_rewards[t] = pi(p_t)
        largest_expected_revenue = np.max(pi(u_t + grid_optim))
        opt_rewards[t] = largest_expected_revenue
        
        t=t+1
        
    return my_exp_rewards, opt_rewards    
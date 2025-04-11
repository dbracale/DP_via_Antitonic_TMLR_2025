import numpy as np
import scipy.stats as stats
import random
import math 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize

# AUXILIARY FUNCTIONS

# FUNCTION TO BUILD ESTIMATE OF PARAMETER THETA

def estimate_theta(X, o):
    lr = LinearRegression(fit_intercept=False).fit(X, o)
    theta = lr.coef_
    return theta

# FUNCTION TO FIND THE MAX OF THE AVERAGE REWARD

def maximise_avg_regret(x, low, high, p, X, theta, o):
    q = np.dot(x, theta)
    H = high-low

    w_t = p-np.dot(X, theta)
    ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
    ir.fit(w_t, o)

    # Define the isotonic regression function G(u)
    def G(u):
        return 1-ir.predict(np.array([u]))[0]

    F = G

    x0 = 0  # Adjust the initial guess as needed
    result = minimize(lambda p: -p * (1-F(p - q)), x0=x0, method='Nelder-Mead')#, method='L-BFGS-B', bounds=[(0, H)])
    optimal_p = result.x[0]
    return 0 if optimal_p < 0 else H if optimal_p > H else optimal_p


# FUNCTION TO BUILD THE GRID TO COMPUTE OPTIMAL REGRET
def get_epsilon(d, T):
    return ((d * np.log(T)) ** 2 / T) ** (1/3)
    

# FAN ALGORITHM ADVERSARIAL

def antitonic_alg_adv(T, d, contexts, N_contexts, theta_star, bound_x, bound_noise, bound_theta):
    
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
    exp_phase = np.ceil((T * np.sqrt(d) )**(3/4))
                    
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
            p_t = maximise_avg_regret(x, 0, B, p_coll, X_coll, theta_est, o_coll)
                                 
        # accumulates regret
        def pi(p):
            return p * (1 - noise.cdf(p - u_t))
        
        my_exp_rewards[t] = pi(p_t)
        largest_expected_revenue = np.max(pi(u_t + grid_optim))
        opt_rewards[t] = largest_expected_revenue
        
        t=t+1
        
    return my_exp_rewards, opt_rewards    
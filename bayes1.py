
# coded snarfed from https://towardsdatascience.com/estimating-probabilities-with-bayesian-modeling-in-python-7144be007815

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    alphas = np.array([1, 1, 1])
    c = np.array([3, 2, 1])
    
    # Create model
    with pm.Model() as model:
        # Parameters of the Multinomial are from a Dirichlet
        parameters = pm.Dirichlet('parameters', a=alphas, shape=3)
        # Observed data is from a Multinomial distribution
        observed_data = pm.Multinomial(
            'observed_data', n=6, p=parameters, shape=3, observed=c)  
    
        with model:
            # Sample from the posterior
            trace = pm.sample(draws=1000, chains=2, tune=500, 
                              discard_tuned_samples=True)
            
            pm.plots.traceplot(trace)
            plt.show()
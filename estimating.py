import numpy as np
species = ['lions', 'tigers', 'bears']
# Observations
c = np.array([3, 2, 1])
#Pseudocounts
alphas = np.array([1, 1, 1])

expected = (alphas + c) / (c.sum() + alphas.sum())

print (expected)
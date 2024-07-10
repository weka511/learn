#!/usr/bin/env python

'''example of preparing and making a prediction with a naive bayes model'''

#
# snarfed from https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/

from sklearn.datasets import make_blobs
from scipy.stats import norm
from numpy import mean, std
import matplotlib.pyplot as plt

# fit_distribution
#
# fit a probability distribution to a univariate data sample
def fit_distribution(data):
    # estimate parameters
    mu    = mean(data)
    sigma = std(data)
    print(f'mu={mu}, sigma{sigma}')
    # fit distribution
    return norm(mu, sigma)

# probability
#
# probabilitycalculate the independent conditional probability
def probability(X, prior, dist1, dist2):
    return prior * dist1.pdf(X[0]) * dist2.pdf(X[1])


X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1) # generate 2d classification dataset

Xy0 = X[y == 0]
Xy1 = X[y == 1]
plt.scatter([x for (x,y) in Xy0],[y for (x,y) in Xy0],label='y==0')
plt.scatter([x for (x,y) in Xy1],[y for (x,y) in Xy1],label='y==1')
plt.legend()

priory0 = len(Xy0) / len(X) # calculate priors
priory1 = len(Xy1) / len(X)
# create PDFs for y==0
distX1y0 = fit_distribution(Xy0[:, 0])
distX2y0 = fit_distribution(Xy0[:, 1])
# create PDFs for y==1
distX1y1 = fit_distribution(Xy1[:, 0])
distX2y1 = fit_distribution(Xy1[:, 1])
# classify one example
Xsample, ysample = X[0], y[0]
py0 = probability(Xsample, priory0, distX1y0, distX2y0)
py1 = probability(Xsample, priory1, distX1y1, distX2y1)
print('P(y=0 | %s) = %.3f' % (Xsample, py0*100))
print('P(y=1 | %s) = %.3f' % (Xsample, py1*100))
print('Truth: y=%d' % ysample)

plt.figure()

for cluster in [0,1]:
    xs=[]
    ys=[]
    zs=[]
    m = '+' if cluster==0 else 'x'
    for i in range(len(y)):
        Xsample,ysample = X[i],y[i]
        if ysample==cluster:
            py0 = probability(Xsample, priory0, distX1y0, distX2y0)
            py1 = probability(Xsample, priory1, distX1y1, distX2y1)
            xs.append(Xsample[0])
            ys.append(Xsample[1])
            zs.append(py0-py1 if ysample ==0 else py1-py0)

    plt.scatter(xs,ys,c=zs,marker=m)
plt.show()

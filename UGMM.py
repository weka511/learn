#!/usr/bin/env python
'''
    CAVI Gaussian Mixture Model
    snarfed from https://zhiyzuo.github.io/VI/#python-implementation
'''

from matplotlib.pyplot import figure, show
import numpy as np
import seaborn as sns

class UGMM(object):
    '''Univariate GMM with CAVI'''

    def __init__(self, X, K=2, sigma=1):
        self.X = X
        self.K = K
        self.N = self.X.shape[0]
        self.sigma2 = sigma**2

    def _init(self):
        self.phi = np.random.dirichlet([np.random.random() * np.random.randint(1, 10)] * self.K, self.N)
        self.m = np.random.randint(int(self.X.min()), high=int(self.X.max()), size=self.K).astype(float)
        self.m += self.X.max() * np.random.random(self.K)
        self.s2 = np.ones(self.K) * np.random.random(self.K)
        print(f'Init mean {self.m}')
        print(f'Init s2 {self.s2}')

    def get_elbo(self):
        t1 = np.log(self.s2) - self.m / self.sigma2
        t1 = t1.sum()
        t2 = -0.5 * np.add.outer(self.X**2, self.s2 + self.m**2)
        t2 += np.outer(self.X, self.m)
        t2 -= np.log(self.phi)
        t2 *= self.phi
        t2 = t2.sum()
        return t1 + t2

    def fit(self, max_iter=100, tol=1e-10):
        self._init()
        self.elbo_values = [self.get_elbo()]
        self.m_history = [self.m]
        self.s2_history = [self.s2]
        for iter_ in range(1, max_iter + 1):
            self._cavi()
            self.m_history.append(self.m)
            self.s2_history.append(self.s2)
            self.elbo_values.append(self.get_elbo())
            if iter_ % 5 == 0:
                print(iter_, self.m_history[iter_])
                print(iter_, self.s2_history[iter_])
            if np.abs(self.elbo_values[-2] - self.elbo_values[-1]) <= tol:
                print('ELBO converged with ll %.3f at iteration %d' % (self.elbo_values[-1],
                                                                       iter_))
                break

        if iter_ == max_iter:
            print('ELBO ended with ll %.3f' % (self.elbo_values[-1]))

    def _cavi(self):
        self._update_phi()
        self._update_mu()

    def _update_phi(self):
        t1 = np.outer(self.X, self.m)
        t2 = -(0.5 * self.m**2 + 0.5 * self.s2)
        exponent = t1 + t2[np.newaxis, :]
        self.phi = np.exp(exponent)
        self.phi = self.phi / self.phi.sum(1)[:, np.newaxis]

    def _update_mu(self):
        self.m = (self.phi * self.X[:, np.newaxis]).sum(0) * (1 / self.sigma2 + self.phi.sum(0))**(-1)
        assert self.m.size == self.K
        self.s2 = (1 / self.sigma2 + self.phi.sum(0))**(-1)
        assert self.s2.size == self.K


num_components = 3
SAMPLE = 1000

mu_arr = np.random.choice(np.arange(-10, 10, 2),
                          num_components) + np.random.random(num_components)

X = np.random.normal(loc=mu_arr[0], scale=1, size=SAMPLE)
for i, mu in enumerate(mu_arr[1:]):
    X = np.append(X, np.random.normal(loc=mu, scale=1, size=SAMPLE))

fig = figure(figsize=(15, 8))
ax = fig.add_subplot(2, 1, 1)
sns.histplot(X[:SAMPLE], ax=ax, kde=True, stat='density',label=f'First {SAMPLE} points')
sns.rugplot(X[:SAMPLE])
sns.histplot(X[SAMPLE:SAMPLE * 2], ax=ax, kde=True, stat='density',label=f'Middle {SAMPLE} points')
sns.rugplot(X[SAMPLE:SAMPLE * 2])
sns.histplot(X[SAMPLE * 2:], ax=ax, kde=True, stat='density',label=f'Last {SAMPLE} points')
sns.rugplot(X[SAMPLE * 2:])

ax.set_title('Data and fitted model')
ax.legend()

ugmm = UGMM(X, 3)
ugmm.fit()

ax1 = fig.add_subplot(2, 1, 2)
ax1.plot(ugmm.elbo_values)
ax1.set_ylabel('ELBO')
ax1.set_title('Convergence')
fig.tight_layout(pad=3, h_pad=9, w_pad=3)

show()

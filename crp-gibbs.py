#!/usr/bin/env python

#   Copyright (C) 2026 Simon Crase

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''Attempt to cluster data using simple Gibbs sampling'''

from argparse import ArgumentParser
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import euclidean
from shared.utils import Logger
from crp import Cluster, Process, plot_generated


class InferredCluster(Cluster):

    def __init__(self, d=2, rng=np.random.default_rng()):
        super().__init__(rng=rng)
        self.sum_points = np.zeros((d))
        
    def get_centroid(self):
        return self.sum_points / len(self)

    def add_point(self, z_index, point):
        super().append(z_index)
        self._adjust(point)

    def remove_point(self, z_index, point):
        self.indices.remove(z_index)
        self._adjust(-point)

    def _adjust(self, point):
        self.sum_points += point


class Clusterer(Process):
    def __init__(self, z, n, d, rng=np.random.default_rng(), alpha=4.0,sigma = 0.5):
        super().__init__(alpha=alpha, rng=rng)
        self.z = z
        self.n = n
        self.d = d
        self.clusters = []
        self.sigma = sigma              #FIXME
        self.zindices = -1 * np.ones((n), dtype=int)

    def get_clusters(self):
        return self.clusters

    def start(self):
        index = self.rng.choice(self.n)
        cluster = InferredCluster(d=self.d, rng=self.rng)
        self.clusters.append(cluster)
        cluster.add_point(index, self.z[index])
        self.zindices[index] = 0  # The cluster

    def step(self):
        z_index = self.rng.choice(self.n)
        z_chosen = self.z[z_index]
        priors = self.create_probabilities(self.clusters)
        likelihoods = self.get_likelihoods(z_chosen, len(priors))
        p = self.get_posteriors(priors, likelihoods)
        cluster_index = self.rng.choice(len(self.clusters) + 1, p=p)
        Logger.get_instance().log(f'zindex={z_index},cluster={cluster_index}')
        # Have we seen this point before?

        index_of_existing_cluster = self.zindices[z_index]

        # Easiest case is where the point is already in the correct cluster: we have nothing to do
        if index_of_existing_cluster == cluster_index:
            return

        # Now deal with the case where the point is in some other cluster
        if index_of_existing_cluster >= 0:
            existing_cluster = self.clusters[index_of_existing_cluster]
            try:
                existing_cluster.remove_point(z_index,z_chosen)
            except ValueError as error:                          #FIXME
                Logger.get_instance().log(error)
            Logger.get_instance().log(f'Cluster {index_of_existing_cluster}, len={len(existing_cluster)}')

        if cluster_index == len(self.clusters):
            self.clusters.append(InferredCluster(d=self.d, rng=self.rng))
            
        self.clusters[cluster_index].add_point(z_index,z_chosen)

    def get_posteriors(self, priors, likelihoods):
        product = priors * likelihoods
        return product / sum(product)

    def get_likelihoods(self, z_chosen, m):
        likelihoods = np.ones(m)
        for i, cluster in enumerate(self.clusters):
            try:
                distance = euclidean(z_chosen, cluster.get_centroid())
                likelihoods[i] = norm.pdf(distance, scale=self.sigma)
            except ValueError as error:                                              # FIXME
                Logger.get_instance().log(error)
                likelihoods[i] = 0
        return likelihoods


def parse_args():
    parser = ArgumentParser(description=__doc__)
    plotfile = Path(__file__).stem
    alpha = 5.0
    figs = './figs'
    data = './data'
    N = 1000
    freq = 100
    
    parser.add_argument('input', help='Data file')
    parser.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default=figs, help=f'Location for storing plot files [{figs}]')
    parser.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
    parser.add_argument('--data', default=data, help=f'Location of data files [{data}]')
    parser.add_argument('--N', type=int, default=N, help=f'Number of iterations [{N}]')
    parser.add_argument('--freq', type=int, default=100, help=f'Used to report progress [{freq}]')
    parser.add_argument('--alpha', default=alpha, type=float, help=f'Controls probabilty of creating a new cluster [{alpha}]')
    parser.add_argument('--plotfile', default=plotfile, help=f'Name of plotfile [{plotfile}]')
    return parser.parse_args()


def main():
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)

    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    with Logger(Path(__file__).stem, path=args.logfiles, level=Logger.WARNING) as logger:
        input_file = (Path(args.data) / args.input).with_suffix('.npz')
        data = np.load(input_file)
        z = data['z']
        n, d = z.shape
        print(f'Loaded {n} Points, dimension={d} from {input_file}')
        clusterer = Clusterer(z, n, d, rng=rng, alpha=args.alpha)
        clusterer.start()
        for i in range(1, args.N):
            clusterer.step()
            if i % args.freq == 0:
                print (f'Step {i} of {args.N}')

        fig = figure(figsize=(8, 8))
        plot_generated(z, clusterer.get_clusters(), 
                       dimensionality=d, ax=fig.add_subplot(1, 1, 1),title='Clustered Data')
        
        fig.tight_layout(pad=3, h_pad=4)
        fig.savefig((Path(args.figs) / args.plotfile).with_suffix('.png'))        

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()


if __name__ == '__main__':
    main()

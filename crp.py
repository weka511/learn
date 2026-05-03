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

'''
Generate Gaussian data using Chinese Restaurant Process. Plot size of clusters,
and time to create new clusters. Also plot
data if dimenion is 1, 2, or 3.
'''

from argparse import ArgumentParser
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc
import numpy as np
from shared.utils import generate_xkcd_colours

class Table:
    '''
    This class represents a Table/Cluster, with a positive number of points
    '''
    mu0 = 0
    sigma0 = 1
    colours = generate_xkcd_colours()

    def __init__(self, sigma=0.125, rng=np.random.default_rng()):
        self.rng = rng
        self.mu = rng.normal(loc=Table.mu0, scale=Table.sigma0)
        self.sigma = sigma
        self.indices = []
        self.colour = next(Table.colours)

    def __len__(self):
        return len(self.indices)

    def get_sample(self):
        return self.rng.normal(loc=self.mu, scale=self.sigma)

    def append(self, index):
        self.indices.append(index)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    figs = './figs'
    data = './data'
    N = 1000
    dimensionality = 2
    alpha = 5.0
    sigma0 = 2
    sigma1 = 0.5
    plotfile=Path(__file__).stem
    
    parser.add_argument('--out', '-o', required=True, help='Name of output file')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default=figs, help=f'Location for storing plot files [{figs}]')
    parser.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
    parser.add_argument('--data', default=data, help=f'Location of data files [{data}]')
    parser.add_argument('--N', type=int, default=N, help=f'Number of samples [{N}]')
    parser.add_argument('--dimensionality', '-d', type=int, default=dimensionality,
                        help=f'Dimensionality of data[{dimensionality}]')
    parser.add_argument('--alpha', default=alpha, type=float, help=f'Controls probabilty of creating a new cluster [{alpha}]')
    parser.add_argument('--mu', default=None, type=float, nargs='+',
                        help='Mean. New clusters are allocated relative to this location [origin]')
    parser.add_argument('--sigma0', default=sigma0, type=float, help=f'Inter-cluster standard deviation [{sigma0}]')
    parser.add_argument('--sigma1', default=sigma1, type=float, help=f'Intra-cluster standard deviation [{sigma1}]')
    parser.add_argument('--plotfile', default=plotfile, help=f'Name of plotfile [{plotfile}]')

    return parser.parse_args()


def create_weights(tables, alpha):
    '''
    Used to choose a Table
    
    Parameters:
        tables    List of existing tables
        alpha     Controls proababilty of creating a new cluster
        
    Returns:
        An array of proabilities, one for each Table plus one for a new table
    '''
    p = np.empty((len(tables) + 1))
    for j in range(len(tables)):
        p[j] = len(tables[j]) - 1 + alpha
    p[-1] = alpha
    return p / p.sum()


def get_projection(dimensionality=1):
    '''
    Determines whether to use a 3d projection
    '''
    return '3d' if dimensionality == 3 else None


def plot_cluster_sizes(tables, ax):
    '''
    Show the number of points in each cluster. This plot also serves as a key
    for the other plots, as it relates colours to tables.
    '''
    ax.bar(range(len(tables)), [len(table) for table in tables],
           label=[f'{i}' for i in range(len(tables))],
           color=[table.colour for table in tables])
    ax.set_xlabel('Tables')
    ax.set_ylabel('Number')
    ax.set_title(f'Cluster sizes')
    ax.legend(title='Clusters', ncols=int(np.sqrt(len(tables))))


def plot_cluster_formation(steps, step_colours, ax):
    '''
    Show evolution of clusters by displaying the points where new clusters emerge
    '''
    ax.scatter(range(len(steps)), steps, c=step_colours, s=5)
    ax.set_xlabel('Number of points')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Formation of new Clusters')


def plot_generated(z, tables, dimensionality, ax):
    '''
    Plot the actual data points within their clusters
    '''
    ax.set_title('Generated data')

    match dimensionality:
        case 1:
            for i, table in enumerate(tables):
                indices = table.indices
                ax.hist(z[indices, 0], color=table.colour, label=f'{i}', density=True)
                ax.set_xlabel('X')
                ax.set_ylabel('Frequency')

        case 2:
            for i, table in enumerate(tables):
                indices = table.indices
                ax.scatter(z[indices, 0], z[indices, 1],
                           c=table.colour, label=f'{i}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')

        case 3:
            for i, table in enumerate(tables):
                indices = table.indices
                ax.scatter(z[indices, 0], z[indices, 1], z[indices, 2],
                           c=table.colour, label=f'{i}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')


def main():
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)

    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    z = np.zeros((args.N, args.dimensionality))
    Table.mu0 = [0] * args.dimensionality if args.mu == None else args.mu
    Table.sigma0 = args.sigma0
    tables = []
    steps = np.zeros((args.N))
    step_colours = []
    for i in range(args.N):
        index = rng.choice(len(tables) + 1,
                           p=create_weights(tables, args.alpha))
        if index == len(tables):
            tables.append(Table(sigma=args.sigma1, rng=rng))
        z[i] = tables[index].get_sample()
        tables[index].append(i)
        steps[i] = len(tables)
        step_colours.append(tables[-1].colour)

    output_file = (Path(args.data) / args.out).with_suffix('.npz')
    np.savez(output_file, z=z)
    print(f'Saved {args.N} points in {len(tables)} clusters to {output_file}')

    fig = figure(figsize=(16, 8))
    fig.suptitle(
        r'$\alpha=$' + f'{args.alpha}, ' +
        r'$\sigma_0=$' + f'{args.sigma0}, ' +
        r'$\sigma_1=$' + f'{args.sigma1} '
    )

    plot_cluster_sizes(tables, ax=fig.add_subplot(2, 2, 1))
    plot_cluster_formation(steps, step_colours, ax=fig.add_subplot(2, 2, 2))
    plot_generated(z, tables, args.dimensionality,
                   ax=fig.add_subplot(2, 2, 3,
                                      projection=get_projection(dimensionality=args.dimensionality)))

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

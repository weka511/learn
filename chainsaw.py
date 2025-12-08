#!/usr/bin/env python

#   Copyright (C) 2025 Simon Crase

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
    Cut log files up and process them
'''

from argparse import ArgumentParser
from glob import glob
from os.path import join
from pathlib import Path
from re import compile
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from matplotlib import rc
from xkcd import generate_xkcd_colours


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+')
    parser.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--glob', default=False, action='store_true', help='If set, program performs it own globbing')
    parser.add_argument('--skip', default=0, type=int, help='Skip points at start')
    return parser.parse_args()


def generate_logfile_names(files, logfiles='./logfiles', mustglob=False):
    '''
    Used to iterate through all logfiles

    Parameters:
        files       List of files/patterns specified bu caller
        logfiles    Location of logfiles
        mustglob    Set to true if running under WingIDE to force globbing
    '''
    if mustglob:
        file_names = [glob(pathname, root_dir=logfiles) for pathname in files]
        logfile_names = [f for file_name_list in file_names for f in file_name_list]
        for name in set(logfile_names):
            yield join(logfiles, name), Path(name).stem
    else:
        for name in files:
            yield name, Path(name).stem


if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    start = time()
    args = parse_args()
    fig = figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    input_pattern = compile(r'Step: ([0-9]+), val_loss: ([\.0-9]+), val_acc: ([\.0-9]+)')

    colour = generate_xkcd_colours()
    for name, short_name in generate_logfile_names(args.files, logfiles=args.logfiles, mustglob=args.glob):
        steps = []
        losses = []
        accuracy = []
        with open(name) as input:
            for line in input:
                matched = input_pattern.match(line.strip())
                step = int(matched.group(1))
                if step < args.skip: continue
                steps.append(step)
                losses.append(float(matched.group(2)))
                accuracy.append(float(matched.group(3)))

        c = next(colour)
        ax1.scatter(steps, losses, label=short_name, s=5, c=c)
        ax2.scatter(steps, accuracy, label=short_name, s=5, c=c)

    ax1.legend()
    ax1.set_title('Losses')
    ax2.legend()
    ax2.set_title('Accuracy')

    skipping = '' if args.skip==0 else f', skipping first {args.skip} entries'
    fig.suptitle(f'{Path(__file__).stem.title()}{skipping}')
    fig.tight_layout(pad=3, h_pad=4, w_pad=3)
    fig.savefig(join(args.figs, Path(__file__).stem))

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()

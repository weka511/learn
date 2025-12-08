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
    Cut logs up and process them
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
    return parser.parse_args()

def generate_logfile_names(args):
    file_names = [glob(pathname,root_dir=args.logfiles) for pathname in args.files]
    logfile_names= [f for file_name_list in file_names for f in file_name_list]
    for name in set(logfile_names):
        yield(join(args.logfiles,name)),name.split('.')[0]



if __name__=='__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    start  = time()
    args = parse_args()
    fig = figure(figsize=(12,12))
    ax = fig.add_subplot(1,1,1)
    pattern = compile(r'Step: ([0-9]+), val_loss: ([\.0-9]+), val_acc: ([\.0-9]+)')

    colours = generate_xkcd_colours()
    for name,short_name in generate_logfile_names(args):
        steps = []
        losses = []
        accuracy = []
        with open(name) as input:
            for line in input:
                matched = pattern.match(line.strip())
                steps.append(int(matched.group(1)))
                losses.append(float(matched.group(2)))
                accuracy.append(float(matched.group(3)))
        ax.scatter(steps,losses,label=short_name,s=5,c=next(colours))

    ax.legend()
    ax.set_title('Losses')
    fig.tight_layout(pad=3, h_pad=4, w_pad=3)
    fig.savefig(join(args.figs, Path(__file__).stem))

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()

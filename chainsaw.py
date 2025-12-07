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

'''Cut logs up and process them'''

# https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/

from argparse import ArgumentParser
from glob import glob
from os.path import join
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+')
    parser.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    return parser.parse_args()

def generate_logfile_names(args):
    file_names = [glob(pathname,root_dir=args.logfiles) for pathname in args.files]
    logfile_names= [f for file_name_list in file_names for f in file_name_list]
    for name in set(logfile_names):
        yield(join(args.logfiles,name))

if __name__=='__main__':
    start  = time()
    args = parse_args()
    for name in generate_logfile_names(args):
        print (name)
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

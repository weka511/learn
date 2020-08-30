# Copyright (C) 2020 Greenweaves Software Limited

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>

import re
import matplotlib.pyplot as plt
import argparse
from matplotlib import rc
rc('text', usetex=True)
parser = argparse.ArgumentParser()
parser.add_argument('file')
args = parser.parse_args()

with open(args.file) as f:
    r_sqs = []
    for line in f:
        m = re.search(r'r_value=([01]\.[0-9]+)',line.strip())
        if m:
            r_sqs.append(float(m.group(1)))
            
    n,bins,_=plt.hist(r_sqs,bins='fd',color='blue')
    plt.title(rf'File {args.file}: {100*n[-1]/sum(n):.1f}\% of wells have $r^2>${bins[-2]:.4f}')
    plt.xlabel(r'$r^2$')
    plt.ylabel('N')
    plt.show()
        
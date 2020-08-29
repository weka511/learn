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

import re, matplotlib.pyplot as plt

with open('rsq.txt') as f:
    r_sqs = []
    for line in f:
        m = re.search(r'r_value=([01]\.[0-9]+)',line.strip())
        if m:
            r_sqs.append(float(m.group(1)))
    plt.hist(r_sqs,bins='fd')
    plt.show()
        
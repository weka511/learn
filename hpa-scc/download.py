#   Copyright (C) 2020-2021 Greenweaves Software Limited

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

import kaggle
from os import system 

def download(
    source      = 'train',
    file        = '000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png',
    competition = 'hpa-single-cell-image-classification',
    path        = 'c:\data\hpa-scc',
    colours     = ['green','blue','red','yellow']):
    for colour in colours:
        f = f'{source}/{file}_{colour}.png'
        print(f)
        system(f'kaggle competitions download -f {f}  -c {competition} -p {path}')

if __name__=='__main__':
    kaggle.api.authenticate()
    
    i = 0
    with open(r'C:\data\hpa-scc\train.csv') as files:
        for line in files:
            if i==0: continue
            download(file=line.split(',')[0])
            i += 1
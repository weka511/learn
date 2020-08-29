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

import os, re

def create_standards():
    def extract(line,key):
        if line.startswith(key):
            pos = len(key)
            while line[pos] in [' ','=']:
                pos+=1
            return line[pos:]
        return None
    
    product = []
    for file in os.listdir('.'):
        if file.endswith('.properties'):
            with open(file) as f:
                beadset = file[:3]
                barcode = ''
                for line in f:
                    bc = extract(line.strip(),'constants.Beadset.Barcodes')
                    if bc!=None:
                        barcode = beadset + bc 
                    standards = extract(line.strip(),'constants.Beadset.Standards')
                    if standards!=None:
                        values = [float(c) for c in standards.split(',')]
                        product.append((barcode,values))
    return product

def lookup(key,standards):
    candidates = sorted([(k,v) for (k,v) in standards if key[:3]==k[:3] and k>=key])
    if len(candidates)>0:
        _,value = candidates[0]
        return value
    else:
        return None

if __name__=='__main__':
    standards = create_standards()
    print (standards)
    print (lookup('PAP15100001',standards))
# Copyright (C) 2022 Greenweaves Software Limited

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

'''
    Use compressed data to recognize digits
'''

from argparse          import ArgumentParser
from os.path           import join
from sklearn.neighbors import KNeighborsClassifier
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
def parse_args():
    '''
        Extract command line arguments
    '''
    parser = ArgumentParser(__doc__)
    parser.add_argument('--data',
                        default = './data',
                        help    = 'Path for storing intermediate data, such as training and validation and saved weights')
    parser.add_argument('--figs',
                        default = './figs',
                        help    = 'Path for storing plots')
    parser.add_argument('--n_neighbours',
                        default = 5,
                        type    = int,
                        help    = 'Number of neighbours')
    return parser.parse_args()

def read_csv(file_name):
    with open(file_name) as f:
        for line in f:
            fields = line.strip().split(',')
            yield [float(f) for f in fields[:-1]],int(fields[-1])

class Classifier:
    def __init__(self,X,y,n_neighbors=3):
        self.neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.neigh.fit(X, y)
    def classify(self,x):
        return self.neigh.predict([x])[0],max(self.neigh.predict_proba([x])[0])


if __name__=='__main__':
    args = parse_args()
    X    = []
    y    = []
    for x,target in read_csv(join(args.data,'train.csv')):
        X.append(x)
        y.append(target)
    classifier = Classifier(X,y,n_neighbors=args.n_neighbours)

    n_correct = 0
    n_errors  = 0
    for xs,target in read_csv(join(args.data,'validation.csv')):
        classification,p = classifier.classify(xs)
        # print (p)
        if target==classification:
            n_correct += 1
        else:
            n_errors +=1
    print (f'{args.n_neighbours} neighbours: {n_correct} correct, {n_errors} errors, accuracy={100*n_correct/(n_errors+n_correct):.1f} %')

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
from matplotlib.pyplot import figure, hist, legend, savefig, show, title, xlabel,ylabel
from numpy             import argsort
from sklearn.neighbors import KNeighborsClassifier

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

#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
class Classifier:
    def __init__(self,X,y,n_neighbors=3):
        self.neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.neigh.fit(X, y)
    def classify(self,x):
        y       = self.neigh.predict([x])[0]
        p       = self.neigh.predict_proba([x])[0]
        indices = argsort(p)
        return y,p[indices[-1]]-p[indices[-2]]

def create_bins(epsilon = 0.0001,n=10):
    multiplier = 1/n
    return [multiplier*x+epsilon for x in range(-1,n+1)]

if __name__=='__main__':
    args = parse_args()
    X    = []
    y    = []
    for x,target in read_csv(join(args.data,'train.csv')):
        X.append(x)
        y.append(target)
    classifier = Classifier(X,y,n_neighbors=args.n_neighbours)

    correct = []
    errors  = []
    for xs,target in read_csv(join(args.data,'validation.csv')):
        classification,confidence = classifier.classify(xs)
        if target==classification:
            correct.append(confidence)
        else:
            errors.append(confidence)

    bins    = create_bins()
    fig     = figure(figsize=(10,10))
    ns,_,_  = hist(correct,
                   bins  = bins,
                   color = 'xkcd:blue',
                   alpha = 0.5,
                   label = 'Correct')
    n1s,_,_ = hist(errors,
                   bins  = bins,
                   color = 'xkcd:red',
                   alpha = 0.5,
                   label = 'Errors')
    xlabel('Confidence')
    ylabel('Count')
    legend()
    title (f'{args.n_neighbours} neighbours: {len(correct)} correct, {len(errors)} errors, accuracy={100*len(correct)/(len(correct)+len(errors)):.1f}%'
           f'\nBin ({bins[-2]:.2f},{bins[-1]:.1f}] contains {100*(ns[-1]+n1s[-1])/(len(correct)+len(errors)):.0f}% of the data and is {100*ns[-1]/(ns[-1]+n1s[-1]):.0f} % accurate')
    savefig(join(args.figs,'classify.png'))
    show()

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

from abc               import ABC, abstractmethod
from argparse          import ArgumentParser
from os.path           import join
from matplotlib.pyplot import figure, hist, legend, savefig, show, title, xlabel, xticks, ylabel
from numpy             import argsort
from sklearn.neighbors import KNeighborsClassifier


class Classifier(ABC):
    '''
    Abstract class used as parent for classifers
    '''
    @classmethod
    def create(cls,X,y,args):
        return NeighboursClassifier(X,y,n_neighbors=args.n_neighbours)

    @abstractmethod
    def classify(self,x):
        ...
    @abstractmethod
    def get_file_name(self):
        ...

class NeighboursClassifier(Classifier):

    def __init__(self,X,y,
                 n_neighbors = 3):
        self.n_neighbors = n_neighbors
        self.classifier  = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.classifier.fit(X, y)


    def classify(self,x):
        y       = self.classifier.predict([x])[0]
        p       = self.classifier.predict_proba([x])[0]
        indices = argsort(p)

        return y,p[indices[-1]]-p[indices[-2]]

    def get_file_name(self):
        return f'{self.n_neighbors}-neighbours.png'



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
    parser.add_argument('--show',
                        default = False,
                        action  = 'store_true',
                        help    = 'Controls whether plot shown (default is just to save)')
    parser.add_argument('--train',
                        default = 'train',
                        help    = 'File containing encoded Training Data')
    parser.add_argument('--validation',
                        default = 'validation',
                        help    = 'File containing encoded Validation Data')
    return parser.parse_args()

def read_training_data(file,path):
    '''
    Read file containing encoded Training Data
    '''
    X    = []
    y    = []
    for x,target in read_csv(join(path,'train')):
        X.append(x)
        y.append(target)
    return X,y

def read_csv(file_name,
             ext = 'csv',
             sep = ','):
    '''
    Read encoded training or validation data from csv file
    '''
    with open(f'{file_name}.{ext}') as f:
        for line in f:
            fields = line.strip().split(sep)
            yield [float(f) for f in fields[:-1]],int(fields[-1])

def create_bins(epsilon = 0.0001,
                n       = 10):
    '''
    Used to create bins for confidence values for histogram
    '''
    multiplier = 1/n
    return [multiplier*x+epsilon for x in range(-1,n+1)]

class Plotter:
    '''
    A Context Manager to encapsulate plotting
    '''
    def __init__(self,figs,classifier,show):
        self.figs       = figs
        self.classifier = classifier
        self.show       = show

    def __enter__(self):
        self.fig     = figure(figsize=(10,10))
        return self

    def __exit__(self, type, value, traceback):
        savefig(join(self.figs,self.classifier.get_file_name()))
        if self.show:
            show()

if __name__=='__main__':
    args       = parse_args()
    X,y        = read_training_data(args.train,args.data)
    classifier = Classifier.create(X,y,args)

    correct = []
    errors  = []
    for xs,target in read_csv(join(args.data,args.validation)):
        classification,confidence = classifier.classify(xs)
        if target==classification:
            correct.append(confidence)
        else:
            errors.append(confidence)

    with Plotter(args.figs,classifier,args.show):
        bins    = create_bins()
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
        xticks(bins[1:])
        legend()
        title (f'{args.n_neighbours} neighbours: {len(correct)} correct, {len(errors)} errors, accuracy={100*len(correct)/(len(correct)+len(errors)):.1f}%'
               f'\nBin ({bins[-2]:.1f},{bins[-1]:.1f}] contains {100*(ns[-1]+n1s[-1])/(len(correct)+len(errors)):.0f}% of the data and is {100*ns[-1]/(ns[-1]+n1s[-1]):.0f} % accurate')

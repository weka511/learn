# Copyright (C) 2021 Greenweaves Software Limited

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

# Based on https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1

from argparse               import ArgumentParser
from matplotlib.pyplot      import close, figure, imshow, savefig, show, title
from matplotlib.lines       import Line2D
from os.path                import join
from random                 import sample
from re                     import split
from sklearn.metrics        import silhouette_score
from time                   import time
from torch                  import device, no_grad
from torch.cuda             import is_available
from torch.nn               import Linear, Module, MSELoss, ReLU, Sequential, Sigmoid
from torch.optim            import Adam
from torch.utils.data       import DataLoader
from torchvision.datasets   import MNIST
from torchvision.transforms import Compose, ToTensor
from torchvision.utils      import make_grid

class AutoEncoder(Module):
    '''A class that implements an AutoEncoder
    '''
    @staticmethod
    def get_non_linearity(params):
        '''Determine which non linearity is to be used for both encoder and decoder'''
        def get_one(param):
            '''Determine which non linearity is to be used for either encoder or decoder'''
            param = param.lower()
            if param=='relu': return ReLU()
            if param=='sigmoid': return Sigmoid()
            return None

        decoder_non_linearity = get_one(params[0])
        encoder_non_linearity = getnl(params[a]) if len(params)>1 else decoder_non_linearity

        return encoder_non_linearity,decoder_non_linearity

    @staticmethod
    def build_layer(sizes,
                    non_linearity = None):
        '''Construct encoder or decoder as a Sequential of Linear labels, with or without non-linearities

        Positional arguments:
            sizes   List of sizes for each Linear Layer
        Keyword arguments:
            non_linearity  Object used to introduce non-linearity between layers
        '''
        linears = [Linear(m,n) for m,n in zip(sizes[:-1],sizes[1:])]
        if non_linearity==None:
            return Sequential(*linears)
        else:
            return Sequential(*[item for pair in [(layer,non_linearity) for layer in linears] for item in pair])

    def __init__(self,
                 encoder_sizes         = [28*28,400,200,100,50,25,6],
                 encoder_non_linearity = ReLU(inplace=True),
                 decoder_sizes         = [],
                 decoder_non_linearity = ReLU(inplace=True)):
        '''
        Keyword arguments:
            encoder_sizes            List of sizes for each Linear Layer in encoder
            encoder_non_linearity    Object used to introduce non-linearity between encoder layers
            decoder_sizes            List of sizes for each Linear Layer in decoder
            decoder_non_linearity    Object used to introduce non-linearity between decoder layers
        '''
        super().__init__()
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = encoder_sizes[::-1] if len(decoder_sizes)==0 else decoder_sizes
        assert self.encoder_sizes[-1] == self.decoder_sizes[0],'Encoder should match decoder'

        self.encoder_non_linearity = encoder_non_linearity
        self.decoder_non_linearity = decoder_non_linearity

        self.encoder = AutoEncoder.build_layer(self.encoder_sizes,
                                               non_linearity = self.encoder_non_linearity)
        self.decoder = AutoEncoder.build_layer(self.decoder_sizes,
                                               non_linearity = self.decoder_non_linearity)
        self.encode  = True
        self.decode  = True


    def forward(self, x):
        '''Propagate value through network

           Computation is controlled by self.encode and self.decode
        '''
        if self.encode:
            x = self.encoder(x)

        if self.decode:
            x = self.decoder(x)
        return x

    def n_encoded(self):
        return self.encoder_sizes[-1]

class Trainer:
    '''This class encapsulates criterion, optimizer, and the process of training
       Parameters:
           model
        Keyword parameters:
            lr        Learning Rate
            criterion Criterion for judging giidness of fit
            dev       device for computation
    '''
    def __init__(self,model,
                 lr        = 0.001,
                 criterion = MSELoss(),
                 dev       = 'cpu'):
                self.model     = model
                self.lr        = lr
                self.criterion = criterion
                self.optimizer = Adam(model.parameters(), lr = lr)
                self.dev       = dev
                self.reconstruction_loss = 0

    def train(self, loader, N   = 25):
        '''Train network

           Parameters:
               loader       Used to get data
               model        Model to be trained
               optimizer    Used to minimze errors
               criterion    Used to compute errors
          Keyword parameters:
              N             Number of epochs
              dev           Device - cpu or cuda
        '''
        Losses        = []

        for epoch in range(N):
            loss = 0
            for batch_features, _ in loader:
                batch_features = batch_features.view(-1, 784).to(self.dev)
                self.optimizer.zero_grad()
                outputs        = self.model(batch_features)
                train_loss     = self.criterion(outputs, batch_features)
                train_loss.backward()
                self.optimizer.step()
                loss += train_loss.item()

            Losses.append(loss / len(loader))
            print(f'epoch : {epoch+1}/{N}, loss = {Losses[-1]:.6f}')

        return Losses

    def reconstruct(self,loader):
        '''Reconstruct images from encoding

           Parameters:
               loader       Used to load images from file
        '''        self.reconstruction_loss = 0.0
        with no_grad():
            for i,(batch_features, _) in enumerate(loader):
                batch_features = batch_features.view(-1, 784).to(dev)
                outputs        = self.model(batch_features)
                test_loss      = self.criterion(outputs, batch_features)
                self.reconstruction_loss += test_loss.item()
                yield i,batch_features, outputs

    def get_params(self):
        '''Get list of lines to be displayed in legend'''
        return [f'lr = {self.lr}',
                f'encoder = {self.model.encoder}',
                f'decoder = {self.model.decoder}',
                f'encoder nonlinearity = {self.model.encoder_non_linearity}',
                f'decoder nonlinearity = {self.model.decoder_non_linearity}'
                ]

class Timer:
    '''Work out elapsed time

    This class implements the Context Manager protocol, so it can be used with a 'with' statement
    '''
    def __init__(self):
        self.start = time()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        print(self)

    def __str__(self):
        return f'Elapsed ={time() - self.start:.0f} seconds'

class Plot:
    '''Performs book keeping for plotting

    Keep track of figure and exes objects, and also save figure in plotfile.'''
    def __init__(self,show,figs,prefix,name,nrows=1,ncols=1):
        self.fig    = figure(figsize=(10,10))
        self.ax     = self.fig.subplots(nrows=nrows,ncols=ncols)
        self.show   = show
        self.figs   = figs
        self.prefix = prefix
        self.name   = name

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        savefig(join(self.figs,f'{self.prefix}-{self.name}'))
        if not self.show:
            close (self.fig)

class Displayer:
    '''Display plots'''
    def __init__(self,
                 trainer  = None,
                 prefix   = 'test',
                 show     = False,
                 figs     = './figs'):
        self.trainer = trainer
        self.prefix  = prefix
        self.show    = show
        self.figs    = figs

    def display(self):
        test_loss = self.reconstruct(test_loader,
                                     N        = args.N,
                                     n_images = args.nimages)

        print (f'Test loss={test_loss:.3f}')

        self.plot_losses(Losses,
                         N                    = args.N,
                         test_loss            = test_loss)

        self.plot_encoding(test_loader,
                      colours = [colour for colour in self.create_xkcd_colours(filter = lambda R,G,B:R<192 and max(R,G,B)>32)][::-1])

    def reconstruct(self,loader,n_images = -1,N=-1):
        '''Reconstruct images from encoding

           Parameters:
               loader
           Keyword Parameters:
               N        Number of epochs used for training (used in image title only)
        '''

        samples = [] if n_images==-1 else sample(range(len(loader)//loader.batch_size),
                                                 k = n_images)
        for i,batch_features, decoded in self.trainer.reconstruct(loader):
            if len(samples)==0 or i in samples:
                with Plot(self.show,self.figs,self.prefix,f'comparison-{i}',nrows=2) as plot:
                    plot.ax[0].imshow(make_grid(batch_features.view(-1,1,28,28)).permute(1, 2, 0))
                    plot.ax[0].set_title('Raw images')
                    scaled_decoded = decoded/decoded.max()
                    plot.ax[1].imshow(make_grid(scaled_decoded.view(-1,1,28,28)).permute(1, 2, 0))
                    plot.ax[1].set_title(f'Reconstructed images after {N} epochs')

        return trainer.reconstruction_loss


    def plot_losses(self,Losses,
                    N                    = 25,
                    test_loss            = 0):
        '''Plot curve of training losses'''
        with Plot(self.show,self.figs,self.prefix,f'losses') as plot:
            plot.ax.plot(Losses)
            plot.ax.set_ylim(bottom=0)
            plot.ax.set_title(f'Training Losses after {N} epochs')
            plot.ax.set_ylabel('MSELoss')
            plot.ax.text(0.95, 0.95, '\n'.join(self.trainer.get_params() + [f'test loss = {test_loss:.3f}'] ),
                    transform           =  plot.ax.transAxes,
                    fontsize            = 14,
                    verticalalignment   = 'top',
                    horizontalalignment = 'right',
                    bbox                = dict(boxstyle  = 'round',
                                               facecolor = 'wheat',
                                               alpha     = 0.5))


    def plot_encoding(self,loader,
                      dev     = 'cpu',
                      colours = []):
        '''Plot the encoding layer

           Since this is multi,dimensional, we will break it into 2D plots
        '''
        def extract_batch(batch_features, labels,index):
            '''Extract xs, ys, and colours for one batch'''

            batch_features = batch_features.view(-1, 784).to(dev)
            encoded        = self.trainer.model(batch_features).tolist()
            return list(zip(*([encoded[k][2*index] for k in range(len(labels))],
                              [encoded[k][2*index+1] for k in range(len(labels))],
                              [colours[labels.tolist()[k]] for k in range(len(labels))])))

        save_decode  = self.trainer.model.decode
        self.trainer.model.decode = False
        with no_grad(), Plot(self.show,self.figs,self.prefix,f'encoding',nrows=2,ncols=2) as plot:
            for i in range(2):
                for j in range(2):
                    if i==1 and j==1: break
                    index    = 2*i + j
                    if 2*index+1 < self.trainer.model.n_encoded():
                        xs,ys,cs = tuple(zip(*[xyc for batch_features, labels in loader for xyc in extract_batch(batch_features, labels,index)]))
                        plot.ax[i][j].set_title(f'{2*index}-{2*index+1}')
                        plot.ax[i][j].scatter(xs,ys,c=cs,s=1)

        plot.ax[0][0].legend(handles=[Line2D([], [],
                                        color  = colours[k],
                                        marker = 's',
                                        ls     = '',
                                        label  = f'{k}') for k in range(10)])

        self.trainer.model.decode = save_decode

    def create_xkcd_colours(self,
                            file_name = 'rgb.txt',
                            prefix    = 'xkcd:',
                            filter    = lambda R,G,B:True):
        '''  Create list of XKCD colours
             Keyword Parameters:
                file_name Where XKCD colours live
                prefix    Use to prefix each colour with "xkcd:"
                filter    Allows us to exclude some colours based on RGB values
        '''
        with open(file_name) as colours:
            for row in colours:
                parts = split(r'\s+#',row.strip())
                if len(parts)>1:
                    rgb  = int(parts[1],16)
                    B    = rgb%256
                    rest = (rgb-B)//256
                    G    = rest%256
                    R    = (rest-G)//256
                    if filter(R,G,B):
                        yield f'{prefix}{parts[0]}'

def parse_args():
    '''Extract command line arguments'''
    parser  = ArgumentParser('Autoencoder')
    parser.add_argument('--N',
                        type    = int,
                        default = 25,
                        help    = 'Number of Epochs')
    parser.add_argument('--show',
                        default = False,
                        action  = 'store_true',
                        help    = 'Display images')
    parser.add_argument('--lr',
                        default = 0.001,
                        type    = float,
                        help    = 'Learning rate')
    parser.add_argument('--figs',
                        default = './figs',
                        help    = 'path for figures')
    parser.add_argument('--encoder',
                        nargs   = '+',
                        default = [28*28, 400, 200, 100, 50, 25, 6],
                        help    = 'Sizes of each layer in encoder')
    parser.add_argument('--decoder',
                        nargs   = '*',
                        default = [],
                        help    = 'Sizes of each layer in decoder (omit if decoder is a mirror image of encoder)')
    parser.add_argument('--nonlinearity',
                        nargs   = '+',
                        default = ['relu'],
                        help    = 'Non lineraities between layers')
    parser.add_argument('--nimages',
                        type    = int,
                        default = 5,
                        help    = 'Number of Images to display')
    parser.add_argument('--prefix',
                        default = 'ae',
                        help    = 'Prefix for image file names')

    return parser.parse_args()

if __name__=='__main__':
    args          = parse_args()
    dev           = device("cuda" if is_available() else "cpu")
    enl,dnl       = AutoEncoder.get_non_linearity(args.nonlinearity)
    model         = AutoEncoder(encoder_sizes         = args.encoder,
                                encoder_non_linearity = enl,
                                decoder_non_linearity = dnl,
                                decoder_sizes         = args.decoder).to(dev)

    transform     = Compose([ToTensor()])

    train_dataset = MNIST(root="~/torch_datasets",
                          train     = True,
                          transform = transform,
                          download  = True)
    test_dataset  = MNIST(root="~/torch_datasets",
                          train     = False,
                          transform = transform,
                          download  = True)

    train_loader  = DataLoader(train_dataset,
                               batch_size  = 128,
                               shuffle     = True,
                               num_workers = 4)
    test_loader   = DataLoader(test_dataset,
                               batch_size  = 32,
                               shuffle     = False,
                               num_workers = 4)
    with Timer():
        trainer   = Trainer(model)
        Losses    = trainer.train(train_loader, N = args.N)
        displayer = Displayer(trainer = trainer,
                              show     = args.show,
                              figs     = args.figs,
                              prefix   = args.prefix)

        displayer.display()



    if args.show:
        show()

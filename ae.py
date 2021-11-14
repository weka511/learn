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
from re                     import split
from torch                  import device, no_grad
from torch.cuda             import is_available
from torch.nn               import Linear, Module, MSELoss, ReLU, Sequential
from torch.optim            import Adam
from torch.utils.data       import DataLoader
from torchvision.datasets   import MNIST
from torchvision.transforms import Compose, ToTensor
from torchvision.utils      import make_grid

class AutoEncoder(Module):
    '''A class that implements an AutoEncoder
    '''
    @staticmethod
    def build_layer(sizes,
                    non_linearity = None):
        '''Construct encoder or decoder as a Sequential of Linear labels, with or without non-linearities

        Positional arguments:
            sizes   List of sizes for each Linear Layer
        Keyword arguments:
            non_linearity
        '''
        linears = [Linear(m,n) for m,n in zip(sizes[:-1],sizes[1:])]
        if non_linearity==None:
            return Sequential(*linears)
        else:
            return Sequential(*[item for pair in [(layer,non_linearity) for layer in linears] for item in pair])

    def __init__(self,
                 encoder_sizes         = [28*28,400,200,100,50,25,6],
                 encoder_non_linearity = None,
                 decoder_sizes         = [],
                 decoder_non_linearity = ReLU(inplace=True)):
        '''
        Keyword arguments:
            encoder_sizes            List of sizes for each Linear Layer in encoder
            encoder_non_linearity
            decoder_sizes            List of sizes for each Linear Layer in decoder
            decoder_non_linearity
        '''
        super().__init__()
        if len(decoder_sizes)==0:
            decoder_sizes = encoder_sizes[::-1]

        self.encoder = AutoEncoder.build_layer(encoder_sizes,
                                               non_linearity = encoder_non_linearity)
        self.decoder = AutoEncoder.build_layer(decoder_sizes,
                                               non_linearity = decoder_non_linearity)
        self.encode  = True
        self.decode  = True


    def forward(self, x):
        if self.encode:
            x = self.encoder(x)

        if self.decode:
            x = self.decoder(x)
        return x


def train(loader,model,optimizer,criterion,
          N   = 25,
          dev = 'cpu'):
    '''Train network'''
    Losses        = []

    for epoch in range(N):
        loss = 0
        for batch_features, _ in loader:
            batch_features = batch_features.view(-1, 784).to(dev)
            optimizer.zero_grad()
            outputs        = model(batch_features)
            train_loss     = criterion(outputs, batch_features)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        Losses.append(loss / len(loader))
        print(f'epoch : {epoch+1}/{args.N}, loss = {Losses[-1]:.6f}')

    return Losses
def reconstruct(loader,model,
                N    = 25,
                name = 'test',
                show = False,
                figs = './figs'):
    '''Reconstruct images from encoding'''
    with no_grad():
        for i,(batch_features, _) in enumerate(loader):
            fig            = figure(figsize=(10,10))
            ax             = fig.subplots(nrows=2)
            batch_features = batch_features.view(-1, 784).to(dev)
            images         = batch_features.view(-1,1,28,28)
            ax[0].imshow(make_grid(images).permute(1, 2, 0))
            ax[0].set_title('Raw images')
            outputs        = model(batch_features)
            images         = outputs.view(-1,1,28,28)
            ax[1].imshow(make_grid(images).permute(1, 2, 0))
            ax[1].set_title(f'Reconstructed images after {N} epochs')
            savefig(join(figs,f'{name}-{i}'))
            if not show:
                close (fig)

def plot_losses(Losses,
                N    = 25,
                show = False,
                figs = './figs'):
    '''Plot curve of training losses'''
    fig = figure(figsize=(10,10))
    ax  = fig.subplots()
    ax.plot(Losses)
    ax.set_ylim(bottom=0)
    ax.set_title(f'Losses after {N} epochs')
    ax.set_ylabel('MSELoss')
    savefig(join(figs,'autoencoder-losses'))
    if not args.show:
        close (fig)

def plot_encoding(loader,model,
                figs    = './figs',
                dev     = 'cpu',
                colours = []):
    '''Plot the encoding layer

       Since this is multi,dimensional, we will break it into 2D plots
    '''
    def extract_batch(batch_features, labels):
        '''Extract xs, ys, and colours for one batch'''

        batch_features = batch_features.view(-1, 784).to(dev)
        encoded        = model(batch_features).tolist()
        return list(zip(*([encoded[k][2*index] for k in range(len(labels))],
                          [encoded[k][2*index+1] for k in range(len(labels))],
                          [colours[labels.tolist()[k]] for k in range(len(labels))])))

    save_decode  = model.decode
    model.decode = False
    with no_grad():
        fig     = figure(figsize=(10,10))
        ax      = fig.subplots(nrows=2,ncols=2)
        for i in range(2):
            for j in range(2):
                if i==1 and j==1: break
                index    = 2*i + j
                xs,ys,cs = tuple(zip(*[xyc for batch_features, labels in loader for xyc in extract_batch(batch_features, labels)]))
                ax[i][j].set_title(f'{2*index}-{2*index+1}')
                ax[i][j].scatter(xs,ys,c=cs,s=1)

    ax[0][0].legend(handles=[Line2D([], [],
                                    color  = colours[k],
                                    marker = 's',
                                    ls     = '',
                                    label  = f'{k}') for k in range(10)])
    savefig(join(figs,'encoding'))
    if not args.show:
        close (fig)

    model.decode = save_decode

def parse_args():
    '''Extract command line arguments'''
    parser        = ArgumentParser('Autoencoder')
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
                        nargs = '+',
                        default = [28*28,400,200,100,50,25,6])
    parser.add_argument('--decoder',
                        nargs = '*',
                        default = [])
    return parser.parse_args()


def create_xkcd_colours(file_name = 'rgb.txt',
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

if __name__=='__main__':
    args          = parse_args()
    dev           = device("cuda" if is_available() else "cpu")
    model         = AutoEncoder(encoder_sizes=args.encoder,
                                decoder_sizes=args.decoder).to(dev)
    optimizer     = Adam(model.parameters(), lr=args.lr)
    criterion     = MSELoss()
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
                               num_workers = 4,
                               pin_memory  = True)
    test_loader   = DataLoader(test_dataset,
                               batch_size  = 32,
                               shuffle     = False,
                               num_workers = 4)

    Losses = train(train_loader,model,optimizer,criterion,
                   N   = args.N,
                   dev = dev)

    plot_losses(Losses,
                N    = args.N,
                show = args.show,
                figs = args.figs)

    plot_encoding(test_loader,model,
                  colours = [colour for colour in create_xkcd_colours(filter = lambda R,G,B:R<192 and max(R,G,B)>32)][::-1])

    reconstruct(test_loader,model,
                N    = args.N,
                show = args.show,
                name = 'test',
                figs = args.figs)

    if args.show:
        show()

# snarfed from https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1

from argparse               import ArgumentParser
from matplotlib.pyplot      import close, figure, imshow, savefig, show, title
from os.path                import join
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
        if self.decoder:
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
                        default = [784,1000,500,250,30])
    parser.add_argument('--decoder',
                        nargs = '*',
                        default = [])
    return parser.parse_args()

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
    reconstruct(test_loader,model,
                N=args.N,
                show = args.show,
                name = 'test',
                figs = args.figs)

    if args.show:
        show()

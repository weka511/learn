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
    Tune hyperparameters.
    Train while training error and validation error both reduce.
'''

from argparse          import ArgumentParser
from AutoEncoder       import AutoEncoder
from matplotlib.pyplot import close, figure, legend, plot, savefig, show, title
from multiprocessing   import cpu_count
from os.path           import join
from torch             import load, no_grad, save
from torch.nn          import MSELoss
from torch.optim       import AdamW
from torch.utils.data  import DataLoader

class Trainer(object):
    '''Train network'''
    def __init__(self,model,loader,validation_loader,
                 criterion        = MSELoss(),
                 lr               = 0.001,
                 weight_decay     = 0.01,
                 path             = './'):
        super().__init__()
        self.model              = model
        self.loader             = loader
        self.validation_loader  = validation_loader
        self.Losses             = [float('inf')]
        self.ValidationLosses   = [float('inf')]
        self.criterion          = criterion
        self.optimizer          = AdamW(model.parameters(),
                                       lr           = lr,
                                       weight_decay = weight_decay)
        self.path               = path

    def train(self,
              N_EPOCHS    = 25,
              N_BURN      = 5,
              args_dict   = {}):
        '''
            Adjust weights until overtraining starts.

            The weights are saved each iteration, so the best set of weights will be preserved.
        '''
        for epoch in range(N_EPOCHS):
            self.train_step()
            self.validation_step()
            print(f'epoch : {epoch + 1}/{N_EPOCHS}, losses = {self.Losses[-1]:.6f}, {self.ValidationLosses[-1]:.6f}')
            if epoch>N_BURN and self.ValidationLosses[-1]>self.ValidationLosses[-2]:
                return self.ValidationLosses[-2]
            else:
                self.save_model(args_dict)

        return self.ValidationLosses[-1]

    def train_step(self):
        '''
            Compute gradients, adjust weights, and compute training loss
        '''
        loss = 0
        for batch_features, _ in self.loader:
            batch_features = batch_features.view(-1, self.model.get_input_length())
            self.optimizer.zero_grad()
            outputs        = self.model(batch_features)
            train_loss     = self.criterion(outputs, batch_features)
            train_loss.backward()
            self.optimizer.step()
            loss += train_loss.item()

        self.Losses.append(loss / len(self.loader))

    def validation_step(self):
        '''
            Computer validation loss
        '''
        loss = 0.0
        with no_grad():
            for i,(batch_features, _) in enumerate(self.validation_loader):
                batch_features        = batch_features.view(-1, self.model.get_input_length())
                outputs               = self.model(batch_features)
                validation_loss       = self.criterion(outputs, batch_features)
                loss   += validation_loss.item()

        self.ValidationLosses.append(loss / len(self.validation_loader))

    def save_model(self,args_dict):
        '''
            Save current state of model
        '''
        save({
            'model_state_dict'     : self.model.state_dict(),
            'args_dict'            : args_dict
            },
             join(self.path,self.get_file_name()))

    def get_file_name(self,
                      name = 'saved',
                      ext  = 'pt'):
        '''
            Used to assign names to files, including hyperparameter values
        '''

        return f'{get_file_name(name,args.dimension,args.lr,weight_decay=args.weight_decay)}.{ext}'

def parse_args():
    '''
        Extract command line arguments
    '''
    parser = ArgumentParser(__doc__)
    parser.add_argument('--encoder',
                        nargs   = '+',
                        type    = int,
                        default = [28*28, 400, 200, 100, 50, 25],
                        help    = 'Sizes of each layer in encoder')
    parser.add_argument('--dimension',
                        type    = int,
                        default = 6,
                        help    = 'Dimension of encoded vectors')
    parser.add_argument('--decoder',
                        nargs   = '*',
                        type    = int,
                        default = [],
                        help    = 'Sizes of each layer in decoder (omit if decoder is a mirror image of encoder)')
    parser.add_argument('--nonlinearity',
                        nargs   = '+',
                        default = ['relu'],
                        help    = 'Non linearities between layers (default relu)')
    parser.add_argument('--batch',
                        default = 128,
                        type    = int,
                        help    = 'Training batch size')
    parser.add_argument('--lr',
                        default = 0.001,
                        type    = float,
                        help    = 'Learning rate')
    parser.add_argument('--weight_decay',
                        default = 0.01,
                        type    = float,
                        help    = 'Weight decay')
    parser.add_argument('--show',
                        default = False,
                        action  = 'store_true',
                        help    = 'Display images (default is to only save them)')
    parser.add_argument('--N',
                        default = 100,
                        type    = int,
                        help    = 'Maximum number of epochs')
    parser.add_argument('--data',
                        default = './data',
                        help    = 'Path for storing intermediate data, such as training and validation and saved weights')
    parser.add_argument('--figs',
                        default = './figs',
                        help    = 'Path for storing plots')
    return parser.parse_args()

def get_file_name(name,dimension,lr,
                  seq          = None,
                  weight_decay = 0.0):
    '''
    File name for plots and saved model
    '''
    base      = f'{name}-dim({dimension})-lr({lr})-wd({weight_decay})'
    return base if seq==None else f'{base}-{seq:04d}'


class Plotter:
    '''
       A Context Manager that wraps matplotlib. Create figure and display title on entry,
       save figure on exit
       Parameters:
           name
           args
           loss
       Keyword Parameters:
          seq
          ext
    '''
    def __init__(self,name,args,loss,
                 seq  = None,
                 ext  = 'png'):
        self.args = args
        self.name = name
        self.seq  = seq
        self.ext  = ext
        self.loss = loss
        self.path = args.data

    def __enter__(self):
        '''
           Create figure with title when we enter context
        '''
        self.fig = figure(figsize=(10,10))
        title(f'{self.name.title()}: dimension = {self.args.dimension}, lr={self.args.lr}, weight_decay={self.args.weight_decay}, loss={self.loss:.6f}')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        '''
           Save figure when we exit. Also close figure unless user has requested show
        '''
        savefig(join(self.path,
                     f'{get_file_name(self.name,self.args.dimension,self.args.lr,weight_decay=self.args.weight_decay)}.{self.ext}'))
        if not args.show:
            close(self.fig)

if __name__=='__main__':
    args    = parse_args()
    enl,dnl = AutoEncoder.get_non_linearity(args.nonlinearity)
    trainer = Trainer(AutoEncoder(encoder_sizes         = args.encoder,
                                  encoding_dimension    = args.dimension,
                                  encoder_non_linearity = enl,
                                  decoder_non_linearity = dnl,
                                  decoder_sizes         = args.decoder) ,
                      DataLoader(load(join(args.data,
                                           'train.pt')),
                                 batch_size  = args.batch,
                                 shuffle     = True,
                                 num_workers = cpu_count()),
                      DataLoader(load(join(args.data,
                                           'validation.pt')),
                                 batch_size  = 32,
                                 shuffle     = False,
                                 num_workers = cpu_count()),
                      lr           = args.lr,
                      weight_decay = args.weight_decay,
                      path         = args.data)
    loss = trainer.train(N_EPOCHS  = args.N,
                         args_dict = {
                             'nonlinearity' : args.nonlinearity,
                             'encoder'      : args.encoder,
                             'decoder'      : args.decoder,
                             'dimension'    : args.dimension,
                         })

    with Plotter('training', args, loss):
        plot(trainer.Losses, 'bo',
             label = 'Training Losses')
        plot(trainer.ValidationLosses, 'r+',
             label = 'Validation Losses')
        legend()

    if args.show:
        show()

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
from torch             import load, no_grad, save
from torch.nn          import MSELoss
from torch.optim       import Adam
from torch.utils.data  import DataLoader

class Trainer(object):
    '''Train network'''
    def __init__(self,model,loader,validation_loader,
                 criterion        = MSELoss(),
                 lr               = 0.001):
        super().__init__()
        self.model              = model
        self.loader             = loader
        self.validation_loader  = validation_loader
        self.Losses             = [float('inf')]
        self.ValidationLosses   = [float('inf')]
        self.criterion          = criterion
        self.optimizer          = Adam(model.parameters(),
                                       lr = lr)

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
                return True
            else:
                self.save_model(args_dict)

        return False

    def train_step(self):
        '''
            Compute gradiets, adjust weights, and compute training loss
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
             self.get_file_name())

    def get_file_name(self,
                      name = 'saved',
                      ext  = 'pt'):
        '''
            Used to assign names to files, including hyerparameter values
        '''

        return f'{name}-lr({args.lr}).{ext}'

def parse_args():
    '''
        Extract command line arguments
    '''
    parser = ArgumentParser(__doc__)
    parser.add_argument('--encoder',
                        nargs   = '+',
                        type    = int,
                        default = [28*28, 400, 200, 100, 50, 25, 6],
                        help    = 'Sizes of each layer in encoder')
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
    parser.add_argument('--show',
                        default = False,
                        action  = 'store_true',
                        help    = 'Display images (default is to only save them)')
    return parser.parse_args()




class Plotter:
    '''
    A Context Manager that wraps matplotlib. Create figure and display title on entry,
    save figure on exit
    '''
    def __init__(self,name,args,seq=None,ext='png'):
        self.args = args
        self.name = name
        self.seq  = seq
        self.ext  = ext

    def __enter__(self):
        self.fig = figure(figsize=(10,10))
        title(f'{self.name.title()}: lr={self.args.lr}')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        base      = f'{self.name}-lr({self.args.lr})'
        file_name = base if self.seq==None else f'{base}-{self.seq:04d}'
        savefig(f'{file_name}.{self.ext}')
        if not args.show:
            close(self.fig)

if __name__=='__main__':
    args    = parse_args()
    enl,dnl = AutoEncoder.get_non_linearity(args.nonlinearity)
    trainer = Trainer(AutoEncoder(encoder_sizes         = args.encoder,
                                  encoder_non_linearity = enl,
                                  decoder_non_linearity = dnl,
                                  decoder_sizes         = args.decoder) ,
                      DataLoader(load('train.pt'),
                               batch_size  = args.batch,
                               shuffle     = True,
                               num_workers = cpu_count()),
                      DataLoader(load('validation.pt'),
                                 batch_size  = 32,
                                 shuffle     = False,
                                 num_workers = cpu_count()),
                      lr = args.lr)
    trainer.train(N_EPOCHS  = 100,
                  args_dict = {
                                'nonlinearity' : args.nonlinearity,
                                'encoder'      : args.encoder,
                                'decoder'      : args.decoder
                              })

    with Plotter('training',args):
        plot(trainer.Losses, 'bo', label='Training Losses')
        plot(trainer.ValidationLosses, 'r+', label='Validation Losses')
        legend()

    if args.show:
        show()

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
from matplotlib.pyplot import figure, legend, plot, savefig, show, title
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
        self.optimizer          = Adam(model.parameters(), lr = lr)

    def train(self,
              N_EPOCHS    = 25,
              N_BURN      = 5):

        for epoch in range(N_EPOCHS):
            self.train_step()
            self.validation_step()
            print(f'epoch : {epoch + 1}/{N_EPOCHS}, losses = {self.Losses[-1]:.6f}, {self.ValidationLosses[-1]:.6f}')
            self.save_model()
            if epoch>N_BURN and self.ValidationLosses[-1]>self.ValidationLosses[-2]: return True

        return False

    def train_step(self):
        loss = 0
        for batch_features, _ in self.loader:
            batch_features = batch_features.view(-1, 784)
            self.optimizer.zero_grad()
            outputs        = self.model(batch_features)
            train_loss     = self.criterion(outputs, batch_features)
            train_loss.backward()
            self.optimizer.step()
            loss += train_loss.item()

        self.Losses.append(loss / len(self.loader))

    def validation_step(self):
        loss = 0.0
        with no_grad():
            for i,(batch_features, _) in enumerate(self.validation_loader):
                batch_features        = batch_features.view(-1, 784)
                outputs               = self.model(batch_features)
                validation_loss       = self.criterion(outputs, batch_features)
                loss   += validation_loss.item()

        self.ValidationLosses.append(loss / len(self.validation_loader))

    def save_model(self):
        pass

def parse_args():
    '''Extract command line arguments'''
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
    return parser.parse_args()

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
                                 num_workers = cpu_count()))
    trainer.train(N_EPOCHS=100)

    figure(figsize=(10,10))
    plot(trainer.Losses, 'bo', label='Training Losses')
    plot(ValidationLosses.Losses, 'r+', label='Validation Losses')
    legend()
    show()

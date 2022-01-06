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
    Use Autoencoder to compress data
'''

from argparse          import ArgumentParser
from AutoEncoder       import AutoEncoder
from multiprocessing   import cpu_count
from torch             import load, no_grad
from torch.utils.data  import DataLoader

def create_model(loaded):
    old_args = loaded['args_dict']
    enl,dnl  = AutoEncoder.get_non_linearity(old_args['nonlinearity'])
    return AutoEncoder(encoder_sizes         = old_args['encoder'],
                       encoder_non_linearity = enl,
                       decoder_non_linearity = dnl,
                       decoder_sizes         = old_args['decoder'])

def extract(model,data_loader,output,
            separator = ','):
    model.decode = False
    with no_grad(),open(output,'w') as out:
        for i,(batch_features, target) in enumerate(data_loader):
            batch_features = batch_features.view(-1, 784)
            encoded        = model(batch_features).tolist()
            for xs,y in zip(encoded,target.tolist()):
                out.write(f'{separator.join([str(x) for x in xs])},{y}\n')



def parse_args():
    '''
        Extract command line arguments
    '''
    parser = ArgumentParser(__doc__)
    parser.add_argument('--load',
                        default = 'saved-lr(0.001).pt')
    return parser.parse_args()

if __name__=='__main__':
    args     = parse_args()
    loaded   = load(args.load)
    model    = create_model(loaded)
    model.load_state_dict(loaded['model_state_dict'])
    extract(model,
            DataLoader(load('train.pt'),
                       batch_size  = 2,#32,
                       shuffle     = False,
                       num_workers = cpu_count()),
            'train.csv')


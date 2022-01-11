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
from matplotlib.lines  import Line2D
from matplotlib.pyplot import axes, figure, savefig, show
from mpl_toolkits      import mplot3d
from multiprocessing   import cpu_count
from os.path           import splitext
from torch             import load, no_grad
from torch.utils.data  import DataLoader

def create_model(loaded):
    '''
    Create Autoencoder from data that has previously been saved

    Parameters:
        loaded   A model that has been loaded from a file

    Returns:
        newly created Autoencoder
    '''
    old_args = loaded['args_dict']
    enl,dnl  = AutoEncoder.get_non_linearity(old_args['nonlinearity'])
    return AutoEncoder(encoder_sizes         = old_args['encoder'],
                       encoding_dimension    = old_args['dimension'],
                       encoder_non_linearity = enl,
                       decoder_non_linearity = dnl,
                       decoder_sizes         = old_args['decoder'])

def extract(model,data_loader):
    '''
    A generator to iterate through the dataset, providing the encoding and target value for each data item.
    '''
    model.decode = False

    for i,(batch_features, target) in enumerate(data_loader):
        batch_features = batch_features.view(-1, model.get_input_length())
        encoded        = model(batch_features).tolist()
        for xs,y in zip(encoded,target.tolist()):
            yield xs,y



def parse_args():
    '''
        Extract command line arguments
    '''
    parser = ArgumentParser(__doc__)
    parser.add_argument('--load',
                        default = 'saved-dim(3)-lr(0.001).pt',
                        help    = 'Network saved by tune.py')
    parser.add_argument('--output',
                        default = '',
                        help    = 'Output file to store extracted data')
    parser.add_argument('--batch',
                        default = 128,
                        type    = int,
                        help    = 'Training batch size')
    parser.add_argument('--data',
                        default = 'validation.pt',
                        help    = 'file to encode')
    parser.add_argument('--plot3d',
                        default = False,
                        action  = 'store_true',
                        help    = 'Prepare 3D plot of encoded data')
    parser.add_argument('--show',
                        default = False,
                        action  = 'store_true',
                        help    = 'Controls whether plot shown (default is just to save)')
    return parser.parse_args()

if __name__=='__main__':
    args     = parse_args()
    loaded   = load(args.load)
    model    = create_model(loaded)
    model.load_state_dict(loaded['model_state_dict'])
    output_file = f'{splitext(args.data)[0]}.csv' if len(args.output)==0 else args.output

    xs  = []
    ys  = []
    zs  = []
    cs  = []

    Colours = [
        'xkcd:purple',
        'xkcd:green',
        'xkcd:blue',
        'xkcd:pink',
        'xkcd:brown',
        'xkcd:red',
        'xkcd:teal',
        'xkcd:orange',
        'xkcd:magenta',
        'xkcd:yellow'
    ]
    with no_grad(),open(output_file,'w') as out:
        for encoded,target in extract(model,
                            DataLoader(load(args.data),
                                       batch_size  = args.batch,
                                       shuffle     = False,
                                       num_workers = cpu_count())):
            out.write(f'{",".join([str(x) for x in encoded])},{target}\n')

            if args.plot3d:
                xs.append(encoded[0])
                ys.append(encoded[1])
                zs.append(encoded[2])
                cs.append(Colours[target])

        if args.plot3d:
            fig = figure()
            ax  = axes(projection='3d')
            ax.scatter3D(xs,ys,zs, c=cs,s=1)
            ax.legend(handles=[Line2D([], [],
                                color  = Colours[k],
                                marker = 's',
                                ls     = '',
                                label  = f'{k}') for k in range(len(Colours))])
            ax.set_title(args.data)
            savefig(f'{splitext(output_file)[0]}.png')
            if args.show:
                show()

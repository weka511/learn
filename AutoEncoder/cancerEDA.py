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

from os.path           import join
from matplotlib.pyplot import figure, savefig, show, title
from pandas            import read_csv
from seaborn           import heatmap

def read_and_split(path      = r'D:\data\cancer_mutations',
                   file_name = 'cancer_mutations',
                   ext       = 'txt'):
    '''
    Read cancer mutations dataset, and split into cholangiocarcinoma and other
    '''
    def extract(cancer_type=0):
        return df.loc[df.cancer_type==cancer_type,].drop(['cancer_type'],
                                               axis    = 1,
                                               inplace = False)

    df = read_csv(join(path,f'{file_name}.{ext}'), sep='\t')
    return (extract(cancer_type=0),
            extract(cancer_type=1))

def plot_samples_and_mutations(df,
                               ax    = None,
                               title = 'Cancer'):
    '''
    Plot mutations (horizontal) for samples (vertical)
    '''
    ax.set_title(title)
    ax.tick_params(which       = 'both',
                   bottom      = False,
                   top         = False,
                   labelleft   = False,
                   labelbottom = False )
    heatmap(df,
            ax   = ax,
            vmin = 0,
            vmax = 1,
            cbar = False)
    ax.set_xlabel('Mutations')
    ax.set_ylabel('Samples')


if __name__=='__main__':
    other_cancer,cholangiocarcinoma              = read_and_split()
    mutation_counts_other_cancer                 = other_cancer.sum(axis=0)
    mutation_counts_other_cancer_descending      = mutation_counts_other_cancer.sort_values(ascending=False)
    mutation_counts_cholangiocarcinoma           = cholangiocarcinoma.sum(axis=0)
    mutation_codes_sorted_by_counts_other_cancer = mutation_counts_other_cancer_descending.keys().tolist()

    fig                                          = figure(figsize=(20,20))
    axs                                          = fig.subplots(2)
    plot_samples_and_mutations(other_cancer[mutation_codes_sorted_by_counts_other_cancer],
                               ax    = axs[0],
                               title = f'Other Cancer {other_cancer.shape[0]} samples')
    plot_samples_and_mutations(cholangiocarcinoma[mutation_codes_sorted_by_counts_other_cancer],
                               ax    = axs[1],
                               title = f'Cholangiocarcinoma {cholangiocarcinoma.shape[0]} samples'         )

    fig.subplots_adjust(left   = 0.1,
                        right  = 0.9,
                        top    = 0.9,
                        bottom = 0.1)
    savefig('cancerEDA.png')
    show()

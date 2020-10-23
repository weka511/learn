#    Copyright (C) 2020 Greenweaves Software Limited

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import random
import gibbs2 as gibbs

def FindMotifs(k,Dna,
               cromwell   = 1,
               E          = 5200,
               BURN_IN    = 0,
               frequency  = 100,
               N_CHAINS   = 1):
    Bases = {'A':0, 'C':1, 'G':2, 'T':3}
    BaseList= list(Bases.keys())
    
    def getCounts(sample,excluded=None,cromwell=0):
        Counts = np.full((4,k),cromwell)
        for i in range(len(Dna)):
            if i != excluded:
                for j in range(k):
                    c = Dna[i][sample[i]+j]
                    Counts[Bases[c],j]+=1
        return Counts

    def getProfile(sample,excluded=None,cromwell=1):
        return np.divide(getCounts(sample,excluded,cromwell=cromwell),4*(1+cromwell))
        
    def initialize_motifs():
        return [random.randrange(0,l+1) for l in max_starts]
    
    def get_probability(motif_index,start,Profile):
        P = 1
        for i in range(k):
            row = Bases[Dna[motif_index][start+i]]
            P *= Profile[row][i]
            x=0
        return P
    
    def modify_one_motif(motif_index,sample):
        Profile = getProfile(sample,excluded=motif_index)
        probabilities = [get_probability(motif_index,i,Profile) for i in range(max_starts[motif_index])]
        return np.argmax(probabilities)

    
    def get_distance(consensus,motif):
        return sum(consensus[i]!=motif[i] for i in range(k))
    
    def get_score(motifs): #FIXME - need consensus
        Counts     = np.transpose(getCounts(motifs))
        consensus  = [np.argmax(row) for row in Counts]
        return sum([get_distance(Dna[i][motifs[i]:motifs[i]+k], ''.join([BaseList[c] for c in consensus])) for i in range(len(motifs)) ])
                
    max_starts = [len(s) - k for s in Dna]
    
    def create_chain():
        chain   = gibbs.gibbs([],
                              E          = E,
                              BURN_IN    = BURN_IN,
                              frequency  = frequency,
                              init       = initialize_motifs,
                              move       = modify_one_motif,
                              dtype      = np.dtype(int),
                              allindices = False)
        scores = [get_score(link) for link in chain]
        index  = np.argmin(scores)
 
        return (scores[index], [Dna[i][chain[index][i]:chain[index][i]+k] for i in range(len(chain[index]))])
                
    return [create_chain() for _ in range(N_CHAINS)]
    
if __name__=='__main__':
    import argparse
    import logomaker as lm
    import matplotlib.pyplot as plt
    import os
 
    def get_name_with_extension(name,ext='txt'):
        if len(os.path.splitext(name)[1])==0:
            return f'{name}.{ext}'
        else:
            return name
        
    result = None
    expected = []
    
    parser = argparse.ArgumentParser('Find motifs using Gibbs sampler')
    parser.add_argument('data',         type=str, nargs='?')
    parser.add_argument('--path',       type=str, default = './datasets')
    parser.add_argument('--starts',     type=int, default=20)
    parser.add_argument('--frequency',  type=int, default=0)
    parser.add_argument('--expected',   type=str)
    parser.add_argument('--output',     type=str)
    args   = parser.parse_args()
    if args.data==None:
        result = FindMotifs(8,
                            [
                                'CGCCCCTCTCGGGGGTGTTCAGTAAACGGCCA',
                                'GGGCGAGGTATGTGTAAGTGCCAAGGTGCCAG',
                                'TAGTACCGAGACCGAAAGAAGTATACAGGCGT',
                                'TAGATCAAGTTTCAGGTGCACGTCGGTGAACC',
                                'AATCCACCAGCTCCACGTGCAATGTTGGCCTA'    
                                ],
                            E         = 1000,
                            BURN_IN   = 0,
                            frequency = args.frequency,
                            N_CHAINS  = args.starts)
    else:
        with open(os.path.join(args.path,
                               get_name_with_extension(args.data))) as input:
            state    = 0
            k,t,N    = 0,0,0
            Dna      = []
            for line in input:
                if state==0:
                    if line.strip()=='Input': continue
                    data  = line.strip().split()
                    k     = int(data[0])
                    t     = int(data[1])
                    N     = int(data[2])
                    state = 2
                elif state==2:
                    if len(Dna)<t:
                        Dna.append(line.strip())
                    else:
                        state = 3
                elif state==3:
                    if line.strip()=='Output': continue
                    expected.append(line.strip())
        
        if args.expected!=None:
            with open(os.path.join(args.path,
                                   get_name_with_extension(args.expected))) as expected_file: 
                expected = [line.strip() for line in expected_file]
                
        expected.sort()            
        result = FindMotifs(k,
                            Dna,
                            E         = N,
                            frequency = 0,
                            N_CHAINS  = args.starts)
    
    scores = [score for score,_ in result]
    best = np.argmin(scores)
    score,motifs = result[best] 
    
    if args.output:
        with open(os.path.join(args.path,
                               get_name_with_extension(args.output)),
                  'w') as output_file:
            for motif in motifs:
                output_file.write(f'{motif}\n')
            
    f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))
    f.tight_layout(pad=1.0)

    ax1.hist(scores)
    ax1.set_xlabel('Score')
    ax1.set_title('Scores')
    motifs.sort()
 
    print ("Motifs")
    for motif in motifs:
        print (motif)
        
    if len(expected)>0:  
        differences = 0
        for e,m in zip(expected,motifs):
            if e!=m:
                if differences==0:
                    print ('Differences')
                print (e,m)
                differences+=1
        if differences==0:
            print ('No differences detected') 
            
    alignment_matrix    = lm.alignment_to_matrix(motifs)
    most_frequent_bases = alignment_matrix.idxmax(axis=1)
    consensus           = ''.join(most_frequent_bases.tolist())
    ax1.set_title(consensus)
    lm.Logo(alignment_matrix,ax=ax3)
    
    if len(expected)>0:
        ax4.set_title("Expected")
        lm.Logo(lm.alignment_to_matrix(expected),ax=ax4)
    plt.savefig(os.path.basename(__file__).split('.')[0] )
    
    plt.show()


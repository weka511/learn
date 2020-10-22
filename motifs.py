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
               BURN_IN    = 200,
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
                        frequency = 0,
                        N_CHAINS  = 25)
    
    for s,motifs in result:
        print (s)
        for m in motifs:
            print (m)

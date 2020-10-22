import numpy as np
import random
import gibbs2 as gibbs

def FindMotifs(k,Dna,
               eps=1,
               E          = 5200,
               BURN_IN    = 200,
               frequency  = 100,
               N_CHAINS   = 1):
    Bases = {'A':0, 'C':1, 'G':2, 'T':3}
    
    def getProfile(excluded=None):
        Counts = np.full((4,k),eps)
        for i in range(len(Dna)):
            if i != excluded:
                for j in range(k):
                    c = Dna[i][max_starts[i]+j]
                    Counts[Bases[c],j]+=1
        return np.divide(Counts,4*(1+eps))
        
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
        Profile = getProfile(excluded=motif_index)
        probabilities = [get_probability(motif_index,i,Profile) for i in range(max_starts[motif_index])]
        return np.argmax(probabilities)

    
    def get_distance(S1,S2):
        return sum(s1!=s2 for (s1,s2) in zip(S1,S2))
    
    def get_score(motifs):
        return max([get_distance(Dna[i][motifs[i]:motifs[i]+k],
                                 Dna[j][motifs[j]:motifs[j]+k]) for i in range(len(motifs)) for j in range(i)])
                
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
                
    return [create_chain()    for _ in range(N_CHAINS)]
    
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

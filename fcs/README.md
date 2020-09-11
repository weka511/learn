# fcs
Apply machine learning to FCS data

## General

| File  | Remarks |Depends|
|---------------|--------------------------------------------|----------|
| doublets.py |Remove doublets from GCPs and denoise data|em|
| em.py | Expectation Maximization--[Notes on the EM Algorithm for Gaussian Mixtures: CS 274A, Probabilistic Learning, Padhraic Smyth ](https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf)||
| fcs.py | Investigate fitting normal curves to scattering. Also serves as a repositiory for common code |
| gcps.py | Model GCPs with Gaussian Mixture Model |em standards|
|partition.py|Partition data using G12/H12|em fcs gcps standards|
| rsq.py | Plot statistics for r squared from gcps.py||
| sizes.py | Group data into size clusters|fcs em|
| standards.py | Functions to look up reference standards ||

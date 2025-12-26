# learn
Experiments in machine learning

## Neural Networks

 File  | Remarks
---------------|--------------------------------------------
hopfield.py | Hopfield network
chainsaw.py|Cut up logs from mnisp.py, and process them
mnist.py|Testbed for deep neural networks, using ideas from [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow and Yoshua Bengio and Aaron Courville. This program learns to recognize images from the NIST dataset
torch-nn.py|[train](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)
visualize.py|Visualize neural network used by mnisp.py

## Variational Inference

Programs written to understand Variational Inference, based on the following references:
 * [David M. Blei, Alp Kucukelbir & Jon D. McAuliffe (2017)--Variational Inference: A Review for Statisticians](http://www.cs.columbia.edu/~blei/fogm/2018F/materials/BleiKucukelbirMcAuliffe2017.pdf)
 * [Padhraic Smyth--Notes on the EM Algorithm for Gaussian Mixtures: CS 274A, Probabilistic Learning](https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf)

 File  | Remarks |
---------------|-------------------------------------------------------------------------------------------
CAVI.tex|Doco for VI programs
cavi1.py|CAVI for Univariate Gaussian from [Univariate Gaussian Example](https://suzyahyah.github.io/bayesian%20inference/machine%20learning/2019/03/20/CAVI.html)
cavi3.py|The Coordinate Ascent Mean-Field Variational Inference (CAVI) example from Section 3 of Blei et al
cavi.py|The Coordinate Ascent Mean-Field Variational Inference (CAVI) example from Section 3 of Blei et al
cavi_nd.py|A version of cavi.py that works in multiple dimensions
em.py|Expectation Maximization
gmm.py|Generate data in accordance with Gaussian Mixture Model for cavi_nd.py
UGMM.py|CAVI code snarfed from [Zhiya Zuo](https://zhiyzuo.github.io/VI)

## Bayesian Inference: Markov Chain Monte Carlo

 File  | Remarks
---------------|--------------------------------------------
gibbs.py|[Gibbs Sampling](http://www2.bcs.rochester.edu/sites/jacobslab/cheat_sheets.html)
mhast.py|[Metropolis-Hastings](http://www2.bcs.rochester.edu/sites/jacobslab/cheat_sheets.html)

## supporting code

 File  | Remarks
---------------|--------------------------------------------
bgr.txt|List of [XKCD colours](https://xkcd.com/color/rgb/), in reverse order, so they are organized by most widely recognized first
learn.bib|Bibliography
learn.wpr | Project file for Wing IDE
utils.py|Utilities: logfiles, random number seed, stopfile, [XKCD colours](https://xkcd.com/color/rgb/)

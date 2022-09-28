# learn
Experiments in machine learning

## Projects

File  | Remarks
---------------|--------------------------------------------
Autoencoder|Use of autoencoder for training

## General

 File  | Remarks
---------------|--------------------------------------------
ae.py|Simple autoencoder tutorial
gibbs.py| A generic Gibbs sampler, based on [gibbs.py](http://www2.bcs.rochester.edu/sites/jacobslab/cheat_sheets.html)
hopfield.py | Hopfield network
learn.bib|Bibliography
learn.wpr | Project file for Wing IDE



## Imported from elsewhere

 File  | Remarks
-------------------------|------------------------------------------------------------------------------------
 bayes1.py |Simple  demo for pymc3-Code from [Estimating Probabilities with Bayesian Modeling in Python](https://towardsdatascience.com/estimating-probabilities-with-bayesian-modeling-in-python-7144be007815)
 gibbs.py |  [Bayesian Inference: Gibbs Sampling](http://www2.bcs.rochester.edu/sites/jacobslab/cheat_sheets.html)
 mhast.py | [Bayesian Inference: Metropolis-Hastings](http://www2.bcs.rochester.edu/sites/jacobslab/cheat_sheets.html)
 naive.py | Code from [How to Develop a Naive Bayes Classifier from Scratch in Python](https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/)
 UGMM.py|CAVI code snarfed from [Zhiya Zuo](https://zhiyzuo.github.io/VI)


## Keras and Tensorflow explorations

 File  | Remarks |
---------------|--------------------------------------------
LeNet5.py|LeNet-5 CNN in keras
losses.R|Plot loss and accuracy for Training and Validation data from logfiles
tf1.py|[Tensorflow Tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner)
tf2.py|Modification of tf1 to use Convolutional layer


## Pytorch Learnings

 File  | Remarks |
---------------|--------------------------------------------
torch-nn.py|[train](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)

## Variational Inference

Programs written to understand Variational Inference, based on the following references:
 * [David M. Blei, Alp Kucukelbir & Jon D. McAuliffe (2017)--Variational Inference: A Review for Statisticians](http://www.cs.columbia.edu/~blei/fogm/2018F/materials/BleiKucukelbirMcAuliffe2017.pdf)
 * [Padhraic Smyth--Notes on the EM Algorithm for Gaussian Mixtures: CS 274A, Probabilistic Learning](https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf)

 File  | Remarks |
---------------|-------------------------------------------------------------------------------------------
CAVI.tex|Doco for VI programs
cavi1.py|CAVI for Univariate Gaussian from [Univariate Gaussian Example](https://suzyahyah.github.io/bayesian%20inference/machine%20learning/2019/03/20/CAVI.html)
cavi3.py|The Coordinate Ascent Mean-Field Variational Inference (CAVI) example from Section 3 of Blei et al
em.py|Expectation Maximization
gmm.py|Generate data in accordance with Gaussian Mixture Model
motifs.py|Gibbs sampler for finding motifs--[Implement GibbsSampler](http://rosalind.info/problems/ba2g/)


## Free Energy

Programs based on [A tutorial on the free-energy framework for modelling perception
and learning, by Rafal Bogacz](https://www.sciencedirect.com/science/article/pii/S0022249615000759)

 File  | Remarks |
---------------|-------------------------------------------------------------------------------------------
feex1.py| Exercise 1--posterior probabilities
feex2.py| Exercise 2--most likely size
feex3.py| Exercise 3--neural implementation
feex5.py| Exercise 5--learn variance

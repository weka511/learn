# learn
Experiments in machine learning



## Neural Networks

 File  | Remarks
------------------|--------------------------------------------
aetrain.py|Train autoencoder
autoencoder.py|Autoenclude class
hopfield.py | Hopfield network
chainsaw.py|Cut up logs from mnisp.py, and process them
classify_names.py|[NLP From Scratch: Classifying Names with a Character-Level RNN](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html). Predict which language a name is from based on the spelling.
generate_names.py|[NLP From Scratch: Generating Names with a Character-Level RNN](https://docs.pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
mnist.py|Testbed for deep neural networks, using ideas from [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow and Yoshua Bengio and Aaron Courville. This program learns to recognize images from the NIST dataset
torch-nn.py|[train](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)
translate.py|[NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
visualize.py|Visualize neural network used by mnisp.py

## Variational Inference

Programs written to understand Variational Inference, based on the following references:
 * [David M. Blei, Alp Kucukelbir & Jon D. McAuliffe (2017)--Variational Inference: A Review for Statisticians](http://www.cs.columbia.edu/~blei/fogm/2018F/materials/BleiKucukelbirMcAuliffe2017.pdf)
 * [Padhraic Smyth--Notes on the EM Algorithm for Gaussian Mixtures: CS 274A, Probabilistic Learning](https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf)

 File  | Remarks |
---------------|-------------------------------------------------------------------------------------------
cavi.py|The Coordinate Ascent Mean-Field Variational Inference (CAVI) example from Section 3 of Blei et al
cavi3.py|Coordinate Ascent Mean-Field Variational Inference (in 1D with 3 peaks
cavi_mp|Multiprocessing version of Coordinate Ascent Mean-Field Variational Inference (CAVI) for multiple dimensions
cavi_nd.py|Coordinate Ascent Mean-Field Variational Inference (CAVI) for multiple dimensions
em.py|Fit Gaussian to  data using Expectation Maximization
gmm.py|Generate data in accordance with Gaussian Mixture Model for cavi_nd.py
UGMM.py|CAVI code snarfed from [Zhiya Zuo](https://zhiyzuo.github.io/VI)

## Bayesian Inference: Markov Chain Monte Carlo

 File  | Remarks
---------------|--------------------------------------------
gibbs.py|[Gibbs Sampling](http://www2.bcs.rochester.edu/sites/jacobslab/cheat_sheets.html)
mhast.py|[Metropolis-Hastings](http://www2.bcs.rochester.edu/sites/jacobslab/cheat_sheets.html)

## Miscellaneous
 File  | Remarks
------------------|--------------------------------------------
bt.py|  Generate test data from a Bradley-Terry model, then try to fit to the data. Compare fitted parameters to original. How large does dataset need to be?
sumo.py|Establish Bradley-Terry Parameters from Sumo basho results

## supporting code

 File  | Remarks
---------------|--------------------------------------------
bgr.txt|List of [XKCD colours](https://xkcd.com/color/rgb/), in reverse order, so they are organized by most widely recognized first
learn.wpr | Project file for Wing IDE

## Documentation, stored in docs folder

 File  | Remarks
---------------|--------------------------------------------
CAVI.tex|Documentation for CAVI
learn.bib|Bibliography

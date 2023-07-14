#!/usr/bin/env python

#   Copyright (C) 2023 Simon Crase

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

'''Template for python script using pytorch'''

# https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/

from argparse import ArgumentParser
from os.path import join
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='./data')
    return parser.parse_args()

if __name__=='__main__':
    start  = time()
    args = parse_args()
    dataset = np.loadtxt(join(args.data,'pima-indians-diabetes.csv'), delimiter=',')
    X = dataset[:,:-1]
    y = dataset[:,-1]
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    model = nn.Sequential(
        nn.Linear(8, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid())
    print (model)

    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    batch_size = 10
    n_epochs   = 100
    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%5==0:
            print(f'Finished epoch {epoch}, latest loss {loss}')


    with torch.no_grad():
        y_pred = model(X)
    accuracy = (y_pred.round() == y).float().mean()
    print(f"Accuracy {accuracy}")
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

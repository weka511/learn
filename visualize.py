#!/usr/bin/env python

#   Copyright (C) 2025 Simon Crase

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

'''
	Visualize layers within nueral network
'''

from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from time import time
import numpy as np
from matplotlib.pyplot import figure, show, cm
from matplotlib import rc
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as tr
from mnist import ModelFactory

class Visualizer:
	'''
	This class extracts layers from the model for display

	Based on https://www.geeksforgeeks.org/deep-learning/visualizing-feature-maps-using-pytorch/
	'''

	def __init__(self):
		self.conv_weights = []
		self.conv_layers = []
		self.feature_maps = []
		self.layer_names = []

	def is_layer_of_interest(self, module, layer_types=[nn.Conv2d]):
		for layer_type in layer_types:
			if isinstance(module, layer_type):
				return True
		return False

	def extract_layers(self, model, layer_types=[nn.Conv2d]):
		'''
		Extract those layers that we want to visualize

		Parameters:
		    model
		    layer_types

		'''
		for module in model.children():
			if self.is_layer_of_interest(module, layer_types=layer_types):
				self.conv_weights.append(module.weight)
				self.conv_layers.append(module)
				m,_,_,_ = module.weight.shape
				weights = module.weight.squeeze()
				for i in range(m):
					print (weights[i,:,:])

	def get_n(self):
		return len(self.conv_weights)

	def build_feature_maps(self, input_image):
		'''
		Pass an image through the layers and construct feature maps

		Parameters:
		    input_image   Image to be used
		'''
		self.feature_maps = []
		self.layer_names = []
		for layer in self.conv_layers:
			input_image = layer(input_image) # 1st iteration: 3x1 1x28x28 -> 3x1 32x24x24
			self.feature_maps.append(input_image)
			self.layer_names.append(str(layer))

	def generate_feature_maps(self):
		for feature_map in self.feature_maps:
			yield feature_map

	def get_n_maps(self,layer=None):
		if layer == None:
			return max(fm.shape[0] for fm in self.generate_feature_maps())
		else:
			return self.feature_maps[layer].shape[0]

	def prepare_feature_maps_for_display(self):
		'''
		Remove batch dimension and normalize for display
		'''
		self.normalized_feature_maps = []
		for feature_map in self.feature_maps:
			feature_map = feature_map.squeeze(0)
			mean_feature_map = torch.sum(feature_map, 0) / feature_map.shape[0]
			self.normalized_feature_maps.append(mean_feature_map.data.cpu().numpy())

	def generate_normalized_maps(self):
		for feature_map in self.normalized_feature_maps:
			yield feature_map

def parse_args():
	parser = ArgumentParser(description=__doc__)
	parser.add_argument('--file', default=None, help='Used to load weights')
	parser.add_argument('--data', default='./data', help='Location of data files')
	parser.add_argument('--logfiles', default='./logfiles', help='Location of log files')
	parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
	parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
	parser.add_argument('--image_number', default=0, type=int, help='')
	parser.add_argument('--layer', default=None, type=int,  help='')
	return parser.parse_args()

if __name__ == '__main__':
	rc('font', **{'family': 'serif',
	              'serif': ['Palatino'],
	              'size': 8})
	rc('text', usetex=True)
	fig = figure(figsize=(24, 12))
	start = time()
	args = parse_args()
	model = ModelFactory.create_from_file_name(args.file)
	model.load(args.file)
	dataset = MNIST(root=args.data, download=True, train=False, transform=tr.ToTensor())
	visualizer = Visualizer()
	visualizer.extract_layers(model)
	n = visualizer.get_n()

	input_image, label = dataset[args.image_number]
	visualizer.build_feature_maps(input_image)

	if args.layer == None:
		m = visualizer.get_n_maps()
		image_index = 1
		ax = fig.add_subplot(n+1, m, image_index)
		ax.imshow(input_image[0, :, :], cmap='gray')
		ax.axis('off')

		for i, feature_map in enumerate(visualizer.generate_feature_maps()):
			image_index = i * m + m + 1
			for j in range(feature_map.shape[0]):
				ax = fig.add_subplot(n + 1, m, image_index)
				ax.imshow(feature_map[j, :, :].detach().numpy(), cmap='gray')
				ax.axis('off')
				image_index += 1
	else:
		m = visualizer.get_n_maps(args.layer)
		ax = fig.add_subplot(2, m, 1)
		ax.imshow(input_image[0, :, :], cmap='gray')
		ax.axis('off')
		feature_map = visualizer.feature_maps[args.layer]
		for j in range(m):
			ax = fig.add_subplot(2, m, m+ j+1)
			ax.imshow(feature_map[j, :, :].detach().numpy(), cmap='gray')
			ax.axis('off')

	fig.suptitle(f'{args.image_number} {args.layer}')
	fig.tight_layout(pad=0, h_pad=0, w_pad=0)
	fig.savefig(join(args.figs, Path(args.file).stem.replace('train', 'visualize')),dpi=1024)

	elapsed = time() - start
	minutes = int(elapsed / 60)
	seconds = elapsed - 60 * minutes
	print(f'Elapsed Time {minutes} m {seconds:.2f} s')

	if args.show:
		show()

#!/usr/bin/env python

#   Copyright (C) 2025-2026 Simon Crase

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
	Shared utilities classes
'''

from pathlib import Path
from re import split
from time import strftime
import numpy as np
import torch

class Logger(object):
	'''
	This class records text in a logfile
	'''

	def __init__(self, name):
		self.name = name + strftime('%Y%m%d%H%M%S')+ '.log'
		self.file = None

	def __enter__(self):
		self.file = open(self.name, 'w')
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self.file != None:
			self.file.close()

	def log(self, line):
		'''
		Output one line of text to console and file, flushing as we go

		Parameters:
		    line     A string to be logged
		'''
		print(line, flush=True)
		self.file.write(line + '\n')
		self.file.flush()


def get_seed(seed, notify=lambda s: print(f'Created new seed {s}')):
	'''
	Used to generate a new seed for random number generation if none specified

	Parameters:
	    seed       The specified seed (may be None)
		notify     A function used to notify that a new seed has been generated
	Returns:
	    The original seed (if not None), or a newly generated seed
	'''
	if seed != None:
		return seed
	rng = np.random.default_rng()
	max_int64_value = np.iinfo(np.int64).max
	new_seed = int(rng.integers(max_int64_value))
	notify(new_seed)
	return new_seed


def user_has_requested_stop(stopfile='stop'):
	'''
	Used to verify that there is a stopfile, so the program can shut down gracefully

	Parameters:
	    stopfile    Name of file used as token to stop program

	Returns:
	    True iff stopfile detected
	'''
	stop_path = Path(stopfile)
	stopfile_detected = stop_path.is_file()
	if stopfile_detected:
		print(f'{stopfile} detected')
		stop_path.unlink()
	return stopfile_detected


def generate_xkcd_colours(file_name='bgr.txt', filter=lambda R, G, B: True):
	'''
	Generate XKCD colours.

	Parameters:
		file_name Where XKCD colours live. The default organizes colours so
				  most widely recognized ones (as defined in XKCD colour
				  survey) come first.
		filter    Allows us to exclude some colours based on RGB values
	'''
	with open(file_name) as colours:
		for row in colours:
			parts = split(r'\s+#', row.strip())
			if len(parts) > 1:
				rgb = int(parts[1], 16)
				B = rgb % 256
				rest = (rgb - B) // 256
				G = rest % 256
				R = (rest - G) // 256
				if filter(R, G, B):
					yield f'xkcd:{parts[0]}'

def create_xkcd_colours(n,file_name='bgr.txt', filter=lambda R, G, B: True):
	'''
	Create a list of XKCD colours

	Parameters:
		n      Number of colours in list
		file_name Where XKCD colours live. The default organizes colours so
	              most widely recognized ones (as defined in XKCD colour
			      survey) come first.
        filter    Allows us to exclude some colours based on RGB values
	'''
	colour_iterator = generate_xkcd_colours(file_name=file_name,filter=filter)
	return [next(colour_iterator) for _ in range(n)]

def ensure_we_can_save(checkpoint_file_name):
	'''
	If there is already a checkpoint file, we need to make it
	into a backup. But if there is already a backup, delete it first

	Parameters:
	    checkpoint_file_name    Name of checkpoint file
	'''
	checkpoint_path = Path(checkpoint_file_name).with_suffix('.pth')
	if checkpoint_path.is_file():
		checkpoint_path_bak = Path(checkpoint_file_name).with_suffix('.bak')
		if checkpoint_path_bak.is_file():
			checkpoint_path_bak.unlink()
		checkpoint_path.rename(checkpoint_path_bak)

def get_device(notify=lambda device: print(f'Using device = {device}')):
	'''
	Use  CUDA if available

	Parameters:
	    notify      Used to notify user which device will be used
	'''
	torch.set_default_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
	notify(torch.get_default_device())
	return torch.get_default_device()

def get_moving_average(xs, ys, window_size=11):
	'''
	Calculate a moving average

	Parameters:
	     xs            Indices of data for plotting
	     ys            Data to be plotted
	     window_size   Number of points to be included

	Returns:
	     x1s    A subset of xs, chosen so average can be plotted on the same scale as xs,ys
	     y1s    The moving average
	'''
	kernel = np.ones(window_size) / window_size
	y1s = np.convolve(ys, kernel, mode='valid')
	skip = (len(ys) - len(y1s)) // 2
	x1s = xs[skip:]
	tail_count = len(x1s) - len(y1s)
	x1s = x1s[:-tail_count]
	return x1s, y1s

def sort_labels(ax):
	'''
	Used to sort labels for legend

	Parameters:
	    ax      The axis on which things are being plotted

	Returns:
	    sorted_handles, sorted_labels
	'''
	legend_handles, legend_labels = ax.get_legend_handles_labels()
	sorted_pairs = sorted(zip(legend_labels, legend_handles))
	sorted_labels = [label for label, handle in sorted_pairs]
	sorted_handles = [handle for label, handle in sorted_pairs]
	return sorted_handles, sorted_labels

if __name__ == '__main__':
	for colour in generate_xkcd_colours():
		print(colour)

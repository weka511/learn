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
	Shared utilities classes
'''

from re import split
import numpy as np

class Logger(object):
	'''
	This class records text in a logfile
	'''

	def __init__(self, name):
		self.name = name + '.log'
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
		'''
		print(line, flush=True)
		self.file.write(line + '\n')
		self.file.flush()

def get_seed(seed,notify=lambda s:print(f'Created new seed {s}')):
	'''
	Used to generate a new seed for random number generation if none specified

	Parameters:
	    seed
	'''
	if seed != None:
		return seed
	rng = np.random.default_rng()
	max_int64_value = np.iinfo(np.int64).max
	new_seed = rng.integers(max_int64_value)
	notify(new_seed)
	return new_seed

def user_has_requested_stop(stopfile='stop'):
	'''
	Used to verify that there is a stopfile, so the program can shut down gracefully

	Parameters:
	    stopfile    Name of file used as token to stop program
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

	    Keyword Parameters:
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

if __name__== '__main__':
	for colour in generate_xkcd_colours():
		print (colour)

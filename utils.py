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

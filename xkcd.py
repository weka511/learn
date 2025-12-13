#!/usr/bin/env python

#    Copyright (C) 2020-2025 Greenweaves Software Limited

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
    Create list of XKCD colours
'''

from re import split


def generate_xkcd_colours(file_name='bgr.txt', filter=lambda R, G, B: True):
    '''
        Generate XKCD colours.

        Keyword Parameters:
            file_name Where XKCD colours live. The default organizes colours so
                      most easily distinguished ones come first.
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



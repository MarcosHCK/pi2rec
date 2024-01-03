# Copyright (c) 2023-2025
# This file is part of pi2rec.
#
# pi2rec is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pi2rec is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pi2rec. If not, see <http://www.gnu.org/licenses/>.
#
from common import pic_height
from common import pic_width
from PIL import Image
import argparse
import keras
import matplotlib.pyplot as plt
import numpy

def process ():

  parser = argparse.ArgumentParser (description = 'pi2rec')

  parser.add_argument ('--input',
      default = 'input.jpg',
      help = 'input image to process',
      metavar = 'file',
      type = str)
  parser.add_argument ('--model',
      default = 'pi2rec.keras',
      help = 'input image to process',
      metavar = 'file',
      type = str)
  parser.add_argument ('--output',
      default = 'output.jpg',
      help = 'where to output the result image',
      metavar = 'file',
      type = str)

  args = parser.parse_args ()

  model = keras.models.load_model (args.model)
  image = keras.preprocessing.image.load_img (args.input)
  image_width = image.width
  image_height = image.height

  image = image.resize ((pic_width, pic_height))
  image = keras.preprocessing.image.img_to_array (image)
  image = (image - 127.0) / 127.0
  image = numpy.expand_dims (image, axis = 0)

  image = model.predict (image) [0]

  image = (image * 127.0) + 127.0
  image = image.astype (numpy.uint8)
  image = Image.fromarray (image)

  image = image.resize ((image_width, image_height))
  image.save (args.output)

process ()

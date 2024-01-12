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
from common import denormalize_to_256
from common import normalize_from_256
import io, math
import tensorflow as tf

#
# This is a dirty trick but sincerely I didn't find out how to do
# an image rotation at arbitrary angles with tensorflow operators
# without implementing it on a lower level
#
from PIL import Image

def rota (image, angle):

  image = image.numpy ()
  angle = (angle * -360.0) * (2 * math.pi)
  stream = io.BytesIO (image)

  image = Image.open (stream)
  image = image.rotate (angle, expand = True)

  stream = io.BytesIO ()
  image.save (stream, format = 'PNG')
  return stream.getvalue ()

@tf.function
def rotate (image, angle):

  image = denormalize_to_256 (image)
  image = tf.cast (image, dtype = tf.uint8)
  image = tf.image.encode_png (image)

  image = tf.py_function (rota, [image, angle], tf.string)

  image = tf.image.decode_png (image, channels = 4)
  image = tf.cast (image, dtype = tf.float32)
  image = normalize_from_256 (image)

  return image

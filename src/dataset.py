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
from common import normalize_from_256
from common import pic_height, pic_width
from paste import paste
from rotate import rotate
from PIL import Image
import io, keras, math, numpy, os
import tensorflow as tf

a_cone = math.pi / 8
ideal_x_for = 488.0
ideal_x_size = 333.0
ideal_y_for = 406.0
ideal_y_size = 87.0

@tf.function
def ellipse (zeta : float, a : float, b : float) -> float:

  asin = (a ** 2) * (tf.sin (zeta) ** 2)
  bcos = (b ** 2) * (tf.cos (zeta) ** 2)
  root = tf.sqrt (asin + bcos)
  return (a * b) / root

@tf.function
def normal (minval : float = 0.0, maxval : float = 1.0) -> float:

  mean = (minval + maxval) / tf.constant (2.0)
  stddev = (minval - maxval) / tf.constant (2.0 * 1.645)

  return tf.random.normal ((), dtype = tf.float32, mean = mean, stddev = stddev)

@tf.function
def uniform (minval : float = 0.0, maxval : float = 1.0) -> float:

  return tf.random.uniform ((), dtype = tf.float32, minval = minval, maxval = maxval)

@tf.function
def blend (canvas, mask):

  angle_cone = tf.constant (a_cone)
  ideal_x_ratio = tf.constant (ideal_x_size / ideal_x_for)
  ideal_y_ratio = tf.constant (ideal_y_size / ideal_y_for)
  pi_2 = tf.constant (math.pi / 2)

  canvas_width = tf.cast (tf.shape (canvas) [0], dtype = tf.float32)
  canvas_height = tf.cast (tf.shape (canvas) [1], dtype = tf.float32)
  mask_width = tf.cast (tf.shape (mask) [0], dtype = tf.float32)
  mask_height = tf.cast (tf.shape (mask) [1], dtype = tf.float32)

  zeta = uniform (0, tf.constant (2 * math.pi))
  rho = normal (0, ellipse (zeta, canvas_width, canvas_height))
  angle = normal (zeta - angle_cone, zeta + angle_cone) - pi_2

  mask_x = (rho * tf.cos (zeta)) + (mask_width / 2)
  mask_y = (rho * tf.sin (zeta)) + (mask_height / 2)

  if mask_x > pic_width or mask_y > pic_height:

    return canvas
  else:

    mask_x = tf.cast (mask_x, tf.int32)
    mask_y = tf.cast (mask_y, tf.int32)

    mask = rotate (mask, angle)
    canvas = paste (canvas, mask, mask_x, mask_y)

    return canvas

@tf.function
def load (path):

  image = tf.io.read_file (path)
  image = tf.io.decode_jpeg (image, channels = 3)
  image = tf.cast (image, dtype = tf.float32)
  image = normalize_from_256 (image)
  image = tf.image.resize (image, [ pic_width, pic_height ])
  return image

@tf.function
def targetize (path, mask):

  after = load (path)
  before = blend (after, mask)
  return (before, after)

def Dataset (root : str, mask_file : str = 'mask.svg', use_svg : bool = True) -> "DatasetV2":

  mask_width = (int) (pic_width * (ideal_x_size / ideal_x_for))
  mask_height = (int) (pic_height * (ideal_y_size / ideal_y_for))

  if not use_svg:

    mask = Image.open (mask_file, mode = 'r')
    mask = mask.resize ((mask_width, mask_height))
    mask = keras.preprocessing.image.img_to_array (mask, dtype = numpy.float32)
    mask = normalize_from_256 (mask)
    mask = tf.constant (mask)
  else:

    import cairosvg

    stream = io.BytesIO ()
    cairosvg.svg2png (url = mask_file, output_width = mask_width, output_height = mask_height, write_to = stream)

    mask = Image.open (stream)
    mask = keras.preprocessing.image.img_to_array (mask, dtype = numpy.float32)
    mask = normalize_from_256 (mask)
    mask = tf.constant (mask)

  images = tf.data.Dataset.list_files (os.path.join (root, '*.JPG'))
  images = images.map (lambda x: targetize (x, mask), num_parallel_calls = tf.data.AUTOTUNE)
  return images

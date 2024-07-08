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
mask_width = pic_width * (333.0 / 488.0)
mask_height = pic_height * (87.0 / 406.0)

@tf.function
def ellipse (zeta, a, b):

  asin = (a ** 2) * (tf.sin (zeta) ** 2)
  bcos = (b ** 2) * (tf.cos (zeta) ** 2)
  root = tf.sqrt (asin + bcos)

  return (a * b) / root

@tf.function
def normal (minval = 0.0, maxval = 1.0):

  mean = (minval + maxval) / 2
  stddev = (minval - maxval) / tf.constant (2 * 1.645)

  return tf.random.normal ((), dtype = tf.float32, mean = mean, stddev = stddev)

@tf.function
def uniform (minval = 0.0, maxval = 1.0):

  return tf.random.uniform ((), dtype = tf.float32, minval = minval, maxval = maxval)

@tf.function
def blend (canvas, mask):

  angle_cone = tf.constant (a_cone)
  pi_2 = tf.constant (math.pi / 2)

  angle = tf.constant (0.)
  mask_x = tf.constant (pic_width, dtype = tf.float32)
  mask_y = tf.constant (pic_height, dtype = tf.float32)

  while mask_x >= pic_width or mask_y >= pic_height or \
    (mask_width * tf.cos (angle) + mask_height * tf.sin (angle) + mask_x) <= pic_width * .5 or \
    (mask_height * tf.cos (angle) + mask_width * tf.sin (angle) + mask_y) <= pic_height * .5:

    zeta = uniform (0, tf.constant (2 * math.pi))
    rho = normal (0, ellipse (zeta, pic_width, pic_height))
    angle = normal (zeta - angle_cone, zeta + angle_cone) - pi_2

    mask_x = (rho * tf.cos (zeta)) + (mask_width / 2)
    mask_y = (rho * tf.sin (zeta)) + (mask_height / 2)

  mask_x = tf.cast (mask_x, dtype = tf.int32)
  mask_y = tf.cast (mask_y, dtype = tf.int32)

  return paste (canvas, rotate (mask, angle), mask_x, mask_y)

@tf.function
def cropseed ():

  return tf.random.uniform (shape = (2,), maxval = tf.int32.max, minval = tf.int32.min, dtype = tf.int32)

@tf.function
def random_jitter (image1, image2):

  addx = uniform (0, pic_width)
  addy = uniform (0, pic_height)
  seed = cropseed ()

  image1 = tf.image.resize (image1, [ pic_width + addx, pic_height + addy ], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  image2 = tf.image.resize (image2, [ pic_width + addx, pic_height + addy ], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  image1 = tf.image.stateless_random_crop (image1, seed = seed, size = (pic_width, pic_height, 3))
  image2 = tf.image.stateless_random_crop (image2, seed = seed, size = (pic_width, pic_height, 3))

  if tf.random.uniform (()) > 0.5:

    image1 = tf.image.flip_left_right (image1)
    image1 = tf.image.flip_left_right (image2)

  return image1, image2

@tf.function
def load (path):

  image = tf.io.read_file (path)
  image = tf.io.decode_jpeg (image, channels = 3)
  image = tf.image.resize (image, [ pic_width, pic_height ])
  image = tf.cast (image, dtype = tf.float32)
  image = normalize_from_256 (image)
  return image

def Dataset (root: str, mask_file: str = 'mask.svg', use_svg: bool = True, use_jitter: bool = True) -> "tf.data.Dataset":

  if not use_svg:

    mask = Image.open (mask_file, mode = 'r')
    mask = mask.resize ((int (mask_width), int (mask_height)))
    mask = keras.preprocessing.image.img_to_array (mask, dtype = numpy.float32)
    mask = normalize_from_256 (mask)
    mask = tf.constant (mask)
  else:

    import cairosvg

    stream = io.BytesIO ()
    cairosvg.svg2png (url = mask_file, output_width = int (mask_width), output_height = int (mask_height), write_to = stream)

    mask = Image.open (stream)
    mask = keras.preprocessing.image.img_to_array (mask, dtype = numpy.float32)
    mask = normalize_from_256 (mask)
    mask = tf.constant (mask)

  @tf.function
  def targetize (path):

    after = load (path)
    before = blend (after, mask)

    if use_jitter:

      after, before = random_jitter (after, before)

    return (before, after)

  images = tf.data.Dataset.list_files (os.path.join (root, '*'))
  images = images.map (targetize, num_parallel_calls = tf.data.AUTOTUNE)
  return images

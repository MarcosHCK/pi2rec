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
def ellipse (zeta : float, a : float, b : float) -> float:

  asin = (a ** 2) * (tf.sin (zeta) ** 2)
  bcos = (b ** 2) * (tf.cos (zeta) ** 2)
  root = tf.sqrt (asin + bcos)

  return (a * b) / root

@tf.function
def normal (minval : float = 0.0, maxval : float = 1.0) -> float:

  mean = (minval + maxval) / 2
  stddev = (minval - maxval) / tf.constant (2 * 1.645)

  return tf.random.normal ((), dtype = tf.float32, mean = mean, stddev = stddev)

@tf.function
def uniform (minval : float = 0.0, maxval : float = 1.0) -> float:

  return tf.random.uniform ((), dtype = tf.float32, minval = minval, maxval = maxval)

@tf.function
def blend (canvas, mask):

  angle_cone = tf.constant (a_cone)
  pi_2 = tf.constant (math.pi / 2)

  zeta = uniform (0, tf.constant (2 * math.pi))
  rho = normal (0, ellipse (zeta, pic_width, pic_height))
  angle = normal (zeta - angle_cone, zeta + angle_cone) - pi_2

  mask_x = tf.cast ((rho * tf.cos (zeta)) + (mask_width / 2), dtype = tf.int32)
  mask_y = tf.cast ((rho * tf.sin (zeta)) + (mask_height / 2), dtype = tf.int32)

  mask_x = tf.cast (pic_width / 2 + 40, dtype = tf.int32)
  mask_y = tf.cast (pic_height / 2, dtype = tf.int32)

  if mask_x > pic_width or mask_y > pic_height:

    return canvas
  else:

    mask = rotate (mask, angle)
    canvas = paste (canvas, mask, mask_x, mask_y)

    return canvas

@tf.function
def random_jitter (image):

  image = tf.image.resize (image, [ pic_width + 32, pic_height + 32 ])
  image = tf.image.random_crop (image, size = [pic_width, pic_height, 3])
  image = tf.image.random_flip_left_right (image)
  image = tf.image.random_flip_up_down (image)
  return image

@tf.function
def load (path):

  image = tf.io.read_file (path)
  image = tf.io.decode_jpeg (image, channels = 3)
  image = tf.cast (image, dtype = tf.float32)
  image = normalize_from_256 (image)
  image = tf.image.resize (image, [ pic_width, pic_height ])
  image = random_jitter (image)
  return image

def Dataset (root : str, mask_file : str = 'mask.svg', use_svg : bool = True) -> "DatasetV2":

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
    return (before, after)

  images = tf.data.Dataset.list_files (os.path.join (root, '*.JPG'))
  images = images.map (targetize, num_parallel_calls = 1)
  return images

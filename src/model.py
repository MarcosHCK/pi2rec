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
from common import loss_ssim
from common import metric_psnr, metric_ssim
from common import pic_height, pic_width
from paste import paste
from rotate import rotate
from PIL import Image
import argparse, io, keras, math, numpy, os, random
import tensorflow as tf

a_cone = math.pi / 8
batches = 32
ideal_x_for = 488.0
ideal_y_for = 406.0
ideal_x_size = 333.0
ideal_y_size = 87.0

def ellipse (zeta : float, a : float, b : float) -> float:

  asin = (a ** 2) * (tf.sin (zeta) ** 2)
  bcos = (b ** 2) * (tf.cos (zeta) ** 2)
  root = tf.sqrt (asin + bcos)
  return (a * b) / root

def normal (minval : float = 0.0, maxval : float = 1.0) -> float:

  mean = (minval + maxval) / tf.constant (2.0)
  stddev = (minval - maxval) / tf.constant (2.0 * 1.645)

  return tf.random.normal ((), dtype = tf.float32, mean = mean, stddev = stddev)

def uniform (minval : float = 0.0, maxval : float = 1.0) -> float:

  return tf.random.uniform ((), dtype = tf.float32, minval = minval, maxval = maxval)

def blend (inputs, mask):

  angle_cone = tf.constant (a_cone)
  ideal_x_ratio = tf.constant (ideal_x_size / ideal_x_for)
  ideal_y_ratio = tf.constant (ideal_y_size / ideal_y_for)
  pi_2 = tf.constant (math.pi / 2)

  canvas = load (inputs)
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

def load (inputs):

  image = tf.io.read_file (inputs)
  image = tf.image.decode_jpeg (image, channels = 3)
  image = tf.cast (image, dtype = tf.float32)
  image = (image - 127.0) / 127.0
  image = tf.image.resize (image, [ pic_width, pic_height ])
  return image

def prepare (root, mask_file = 'mask.png', use_svg = False):

  mask_width = (int) (pic_width * (ideal_x_size / ideal_x_for))
  mask_height = (int) (pic_height * (ideal_y_size / ideal_y_for))

  if not use_svg:

    mask = Image.open (mask_file, mode = 'r')
    mask = mask.resize ((mask_width, mask_height))
    mask = keras.preprocessing.image.img_to_array (mask, dtype = numpy.float32)
    mask = tf.constant ((mask - 127.0) / 127.0)
  else:

    import cairosvg

    stream = io.BytesIO ()
    cairosvg.svg2png (url = mask_file, output_width = mask_width, output_height = mask_height, write_to = stream)

    mask = Image.open (stream)
    mask = keras.preprocessing.image.img_to_array (mask, dtype = numpy.float32)
    mask = tf.constant ((mask - 127.0) / 127.0)

  images = tf.data.Dataset.list_files (os.path.join (root, '*.JPG'))
  images = images.map (lambda x: (x, x), num_parallel_calls = tf.data.AUTOTUNE)
  images = images.map (lambda x1, x2: (blend (x1, mask), load (x2)), num_parallel_calls = tf.data.AUTOTUNE)
  return images

def train (dataset, epochs = 10):

  if pic_width % 4 != 0:
    raise ValueError ("pic_width is not a power of 4")
  if pic_height % 4 != 0:
    raise ValueError ("pic_width is not a power of 4")

  loss = loss_ssim
  metrics = [ keras.metrics.MeanSquaredError (), metric_psnr, metric_ssim ]

  model = keras.models.Sequential ()
  model.add (keras.layers.Conv2D (16, (3, 3), activation = 'relu', padding = 'same'))
  model.add (keras.layers.MaxPooling2D ((2, 2)))
  model.add (keras.layers.Conv2D (32, (3, 3), activation = 'relu', padding = 'same'))
  model.add (keras.layers.MaxPooling2D ((2, 2)))
  model.add (keras.layers.BatchNormalization ())
  model.add (keras.layers.Conv2DTranspose (32, (3, 3), strides = (2, 2), padding = 'same'))
  model.add (keras.layers.Conv2DTranspose (3, (3, 3), strides = (2, 2), padding = 'same',
    activation = keras.activations.tanh))

  model.compile (loss = loss, metrics = metrics, optimizer = 'adam')
  model.fit (dataset, epochs = epochs, steps_per_epoch = len (dataset))
  return model

def take_sample (dataset, size, directory):

  if not os.path.exists (directory):
    os.mkdir (directory)
  for i, (image, target) in enumerate (dataset.take (10)):

    image = keras.preprocessing.image.array_to_img ((image * 127.0) + 127.0)
    image.save (os.path.join (directory, f'input_{i}.jpg'))
    image = keras.preprocessing.image.array_to_img ((target * 127.0) + 127.0)
    image.save (os.path.join (directory, f'target_{i}.jpg'))

def program ():

  parser = argparse.ArgumentParser (description = 'pi2rec')

  parser.add_argument ('dataset',
      help = 'dataset root',
      metavar = 'directory',
      type = str)
  parser.add_argument ('--epochs',
      help = 'epoch number to train in',
      metavar  = 'N',
      type = int)
  parser.add_argument ('--mask',
      help = 'epoch number to train in',
      metavar  = 'N',
      type = str)
  parser.add_argument ('--output',
      default = 'pi2rec.keras',
      help = 'take dataset sample',
      metavar  = 'directory',
      type = str)
  parser.add_argument ('--sample',
      help = 'take dataset sample',
      metavar  = 'N',
      type = int)
  parser.add_argument ('--sample-at',
      default = 'dataset_sample',
      help = 'take dataset sample',
      metavar  = 'directory',
      type = str)
  parser.add_argument ('--use-svg',
      action = 'store_true',
      help = 'use SVG masks (instead of PNG)')

  args = parser.parse_args ()
  dataset = prepare (args.dataset, args.mask, args.use_svg)

  if (args.sample != None):
    size = args.sample
    take_sample (dataset, size, args.sample_at)
  else:
    model = train (dataset.batch (batches), args.epochs)
    model.save (args.output)

program ()

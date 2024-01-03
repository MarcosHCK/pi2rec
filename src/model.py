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
from PIL import Image
import argparse, io, keras, math, os, random
import tensorflow as tf

a_cone = math.pi / 8
batches = 6
ideal_x_for = 488.0
ideal_y_for = 406.0
ideal_x_size = 333.0
ideal_y_size = 87.0

def ellipse (zeta : float, a : float, b : float) -> float:

  asin = (a ** 2) * (math.sin (zeta) ** 2)
  bcos = (b ** 2) * (math.cos (zeta) ** 2)
  root = math.sqrt (asin + bcos)

  return (a * b) / root

def normal (n_min : float, n_max : float) -> float:

  u1 = random.random ()
  u2 = random.random ()

  z = math.sqrt (-2.0 * math.log (u1)) * math.cos (2.0 * math.pi * u2)

  mean = (n_min + n_max) / 2.0
  dev = (n_min - n_max) / (2.0 * 1.645)

  return z * dev + mean

def blender (inputs):

  data = inputs.numpy ()
  source = Image.open ('mask.png')
  canvas = Image.open (io.BytesIO (data))

  source_width = (int) (canvas.width * (ideal_x_size / ideal_x_for))
  source_height = (int) (canvas.height * (ideal_y_size / ideal_y_for))
  source = source.resize ((source_width, source_height))

  zeta = random.random () * math.pi * 2
  rho = normal (0, ellipse (zeta, source_width, source_height))
  angle = normal (zeta - a_cone, zeta + a_cone) - (math.pi / 2)

  source_xpos = (int) ((rho * math.cos (zeta)) + (source_width / 2))
  source_ypos = (int) ((rho * math.sin (zeta)) + (source_height / 2))

  source = source.rotate ((angle * -360.0) / (math.pi * 2), expand = True)

  canvas.paste (source, (source_xpos, source_ypos), source)

  output = io.BytesIO ()
  canvas.save (output, format = 'JPEG')
  return output.getvalue ()

def blend (inputs):

  image = tf.io.read_file (inputs)
  image = tf.py_function (blender, [image], tf.string)
  image = tf.image.decode_jpeg (image, channels = 3)
  image = tf.image.convert_image_dtype (image, tf.float32)
  image = tf.image.resize (image, [pic_width, pic_height])
  return image

def load (inputs):

  image = tf.io.read_file (inputs)
  image = tf.image.decode_jpeg (image, channels = 3)
  image = tf.image.convert_image_dtype (image, tf.float32)
  image = tf.image.resize (image, [pic_width, pic_height])
  return image

def prepare (root):

  images = tf.data.Dataset.list_files (os.path.join (root, '*.JPG'))
  images = images.map (lambda x: (x, x), num_parallel_calls = tf.data.AUTOTUNE)
  images = images.map (lambda x1, x2: (blend (x1), load (x2)), num_parallel_calls = tf.data.AUTOTUNE)
  return images

def train (dataset):

  if pic_width % 4 != 0:
    raise ValueError ("pic_width is not a power of 4")
  if pic_height % 4 != 0:
    raise ValueError ("pic_width is not a power of 4")

  loss = loss_ssim
  metrics = [ keras.metrics.MeanSquaredError (), metric_psnr, metric_ssim ]

  model = keras.models.Sequential ()
  model.add (keras.layers.Conv2D (64, (3, 3), activation = 'relu', padding = 'same'))
  model.add (keras.layers.MaxPooling2D ((2, 2)))
  model.add (keras.layers.Conv2D (128, (3, 3), activation = 'relu', padding = 'same'))
  model.add (keras.layers.MaxPooling2D ((2, 2)))
  model.add (keras.layers.BatchNormalization ())
  model.add (keras.layers.Conv2DTranspose (128, (3, 3), strides = (2, 2), padding = 'same'))
  model.add (keras.layers.Conv2DTranspose (3, (3, 3), strides = (2, 2), padding = 'same',
    activation = 'tanh'))

  model.compile (loss = loss, metrics = metrics, optimizer = 'adam')
  model.fit (dataset, epochs = 10, steps_per_epoch = len (dataset))
  return model

def program ():

  parser = argparse.ArgumentParser (description = 'pi2rec')

  parser.add_argument ('dataset',
      help = 'dataset root',
      metavar = 'dataset',
      type = str)

  args = parser.parse_args ()
  dataset = prepare (args.dataset)
  model = train (dataset.batch (batches))

  model.save ('pi2rec.keras')

program ()

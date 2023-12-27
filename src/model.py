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
from PIL import Image
import io
import keras
import math
import random
import subprocess, tempfile
import tensorflow as tf

pic_width = 64
pic_height = 64
a_cone = math.pi / 8
ideal_x_for = 488.0
ideal_y_for = 406.0
ideal_x_size = 333.0
ideal_y_size = 87.0

def normal (n_min : float, n_max : float) -> float:

  u1 = random.random ()
  u2 = random.random ()

  z = math.sqrt (-2.0 * math.log (u1)) * math.cos (2.0 * math.pi * u2)

  mean = (n_min + n_max) / 2.0
  dev = (n_min - n_max) / (2.0 * 1.645)

  return z * dev + mean

def ellipse (zeta : float, a : float, b : float) -> float:

  zsin = math.sin (zeta)
  zcos = math.cos (zeta)

  asin = (a * a) * (zsin * zsin)
  bcos = (b * b) * (zcos * zcos)
  root = math.sqrt (asin + bcos)

  return (a * b) / root

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

  output = io.BytesIO (data)
  canvas.save (output, format = 'PNG')
  return output.getvalue ()

def blend (inputs):

  image = tf.io.read_file (inputs)

  image = tf.py_function (blender, [image], tf.string)
  image = tf.image.decode_png (image, channels = 3)
  image = tf.image.resize (image, [pic_width, pic_height])
  image = tf.image.convert_image_dtype (image, tf.float32)
  return image

def load (inputs):

  image = tf.io.read_file (inputs)
  image = tf.image.decode_jpeg (image, channels = 3)
  image = tf.image.resize (image, [pic_width, pic_height])
  image = tf.image.convert_image_dtype (image, tf.float32)
  return image

def prepare ():

  dataset = tf.data.Dataset.list_files ('../dataset2/*.JPG')
  dataset = dataset.map (blend, num_parallel_calls = tf.data.AUTOTUNE)
  dataset = dataset.batch (32)
  target = tf.data.Dataset.list_files ('../dataset2/*.JPG')
  target = target.map (load, num_parallel_calls = tf.data.AUTOTUNE)
  target = target.batch (32)
  return tf.data.Dataset.zip ((dataset, target))

def train (dataset):

  if pic_width % 4 != 0:
    raise ValueError ("pic_width is not a power of 4")
  if pic_height % 4 != 0:
    raise ValueError ("pic_width is not a power of 4")

  quarter_width = (int) (pic_width / 4)
  quarter_height = (int) (pic_height / 4)

  model = keras.models.Sequential ()
  model.add (keras.layers.Conv2D (32, (3, 3), activation = 'relu', input_shape = (pic_width, pic_height, 3)))
  model.add (keras.layers.MaxPooling2D ((2, 2)))
  model.add (keras.layers.Conv2D (64, (3, 3), activation = 'relu'))
  model.add (keras.layers.MaxPooling2D ((2, 2)))
  model.add (keras.layers.Flatten ())
  model.add (keras.layers.Dense (64, activation = 'relu'))
  model.add (keras.layers.Dense (quarter_width * quarter_height * 128, activation = 'relu'))
  model.add (keras.layers.Reshape ((quarter_width, quarter_height, 128)))
  model.add (keras.layers.Conv2DTranspose (3, (3, 3), strides = (4, 4), activation = 'sigmoid', padding = 'same'))

  model.compile (optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
  model.fit (dataset, steps_per_epoch = len (dataset), epochs = 10)
  return model

dataset = prepare ()
model = train (dataset)
model.save_weights ('pi2rec.h5')

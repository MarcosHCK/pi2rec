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
import keras
import tensorflow as tf

units = 64

def downsampler (filters, size, normalize = True):

  initzr = keras.initializers.RandomNormal (mean = 0.0, stddev = 0.02)
  result = keras.models.Sequential ()

  if True:
    result.add (keras.layers.Conv2D (filters, size, strides = 2, padding = 'same', kernel_initializer = initzr, use_bias = False))
  if normalize:
    result.add (keras.layers.BatchNormalization ())

  result.add (keras.layers.LeakyReLU ())
  return result

def upsampler (filters, size, dropout = False):

  initzr = keras.initializers.RandomNormal (mean = 0.0, stddev = 0.02)
  result = keras.models.Sequential ()

  if True:
    result.add (keras.layers.Conv2DTranspose (filters, size, strides = 2, padding = 'same', kernel_initializer = initzr, use_bias = False))
    result.add (keras.layers.BatchNormalization ())
  if dropout:
    result.add (keras.layers.Dropout (0.5))

  result.add (keras.layers.ReLU ())
  return result

def Generator (pic_width, pic_height, channels):

  downstack = \
    [
      downsampler (units * 1, 4, normalize = False),
      downsampler (units * 2, 4, normalize = True),
      downsampler (units * 4, 4, normalize = True),
      downsampler (units * 8, 4, normalize = True),
      downsampler (units * 8, 4, normalize = True),
      downsampler (units * 8, 4, normalize = True),
      downsampler (units * 8, 4, normalize = True),
      downsampler (units * 8, 4, normalize = True),
    ]

  upstack = \
    [
      upsampler (units * 8, 4, dropout = True),
      upsampler (units * 8, 4, dropout = True),
      upsampler (units * 8, 4, dropout = True),
      upsampler (units * 8, 4, dropout = False),
      upsampler (units * 4, 4, dropout = False),
      upsampler (units * 2, 4, dropout = False),
      upsampler (units * 1, 4, dropout = False),
    ]

  initzr = keras.initializers.RandomNormal (mean = 0.0, stddev = 0.02)
  inputs = keras.layers.Input (shape = [ pic_width, pic_height, channels ])
  last = keras.layers.Conv2DTranspose (channels, 4, strides = 2, padding = 'same', kernel_initializer = initzr, activation = 'tanh')

  x = inputs
  skips = []

  for down in downstack:

    x = down (x)
    skips.append (x)

  skips = reversed (skips [:-1])

  for up, skip in zip (upstack, skips):

    x = up (x)
    x = keras.layers.Concatenate () ([x, skip])

  x = last (x)

  return keras.Model (inputs = inputs, outputs = x)

def GeneratorLoss (lambda_: float = 100):

  cross_entropy = keras.losses.BinaryCrossentropy (from_logits = True)

  @tf.function
  def loss (y_disc, y_pred, y_true):

    loss1 = cross_entropy (tf.ones_like (y_disc), y_disc)
    loss2 = tf.reduce_mean (tf.abs (y_true - y_pred))
    total = loss1 + (lambda_ * loss2)

    return total, loss1, loss2

  return loss

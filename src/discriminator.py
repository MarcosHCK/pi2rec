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
from generator import downsampler, units
import keras
import tensorflow as tf

def Discriminator (pic_width, pic_height, channels):

  initzr1 = keras.initializers.RandomNormal (mean = 0.0, stddev = 0.02)
  initzr2 = keras.initializers.RandomNormal (mean = 0.0, stddev = 0.02)

  inputs_true = keras.layers.Input (shape = [pic_width, pic_height, channels], name = 'y_true')
  inputs_pred = keras.layers.Input (shape = [pic_width, pic_height, channels], name = 'y_pred')
  x = keras.layers.concatenate ([ inputs_true, inputs_pred ])

  down1 = downsampler (units * 1, 4, normalize = False) (x)
  down2 = downsampler (units * 2, 4, normalize = False) (down1)
  down3 = downsampler (units * 4, 4, normalize = False) (down2)

  zeropad1 = keras.layers.ZeroPadding2D () (down3)
  conv = keras.layers.Conv2D (units * 8, 4, strides = 1, kernel_initializer = initzr1, use_bias = False) (zeropad1)
  normalize = keras.layers.BatchNormalization () (conv)
  activation = keras.layers.LeakyReLU () (normalize)
  zeropad2 = keras.layers.ZeroPadding2D () (activation)
  last = keras.layers.Conv2D (1, 4, strides = 1, kernel_initializer = initzr2) (zeropad2)

  return keras.Model (inputs = [ inputs_true, inputs_pred ], outputs = last)

def DiscriminatorLoss ():

  cross_entropy = keras.losses.BinaryCrossentropy (from_logits = True)

  def loss (y_true, y_disc):

    loss1 = cross_entropy (tf.ones_like (y_true), y_true)
    loss2 = cross_entropy (tf.zeros_like (y_disc), y_disc)

    return loss1 + loss2

  return loss

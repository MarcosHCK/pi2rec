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
import tensorflow as tf

@tf.function
def paste (image, mask, x, y):

  image_width = tf.shape (image) [0]
  image_height = tf.shape (image) [1]
  mask_width = tf.shape (mask) [0]
  mask_height = tf.shape (mask) [1]

  off_x = tf.where (tf.greater_equal (x, 0), 0, -x)
  off_y = tf.where (tf.greater_equal (y, 0), 0, -y)

  x = tf.where (tf.less (x, 0), 0, x)
  y = tf.where (tf.less (y, 0), 0, y)

  tf.assert_less (x, image_width,
    message = 'paste position must be inside of image (or outside but just at the top-left sides)')
  tf.assert_less (y, image_height,
    message = 'paste position must be inside of image (or outside but just at the top-left sides)')

  bound_x = tf.maximum (0, tf.minimum (mask_width - off_x, image_width - x))
  bound_y = tf.maximum (0, tf.minimum (mask_height - off_y, image_height - y))
  mask = mask [off_x : off_x + bound_x, off_y : off_y + bound_y, :]

  pad_x = [ x, image_width - (x + bound_x) ]
  pad_y = [ y, image_height - (y + bound_y) ]
  pad = [ pad_x, pad_y, [0, 0] ]

  mask = tf.pad (mask, pad, constant_values = -1, mode = 'CONSTANT')
  alpha = mask [:, :, 3:]

  #
  # This is a derivation of the original alpha blending formula:
  #
  #   R = A * M + (1 - A) * I
  # where:
  #   - R: result image
  #   - A: alpha channel vector (WxHx1 matrix)
  #   - M: mask RGB vector (WxHx3 matrix)
  #   - I: image RGB vector (WxHx3 matrix)
  #
  # All values of R, A, M, I should be normalized in the range
  # of [0, 1], but since input image is the range of [-1, 1]
  # it has to be transformed: R = (R' + 1) / 2, [-1, 1] => [0, 1]
  #
  # So the new formula is:
  #
  #   R = (2*A-1)*M + (1-A)*2*I + 1/2
  #

  return (2*alpha - 1) * mask [:, :, :3] + (1 - alpha) * 2 * image + 0.5

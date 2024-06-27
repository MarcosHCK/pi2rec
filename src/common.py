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
from tensorflow.image import psnr
from tensorflow.image import ssim
from tensorflow import function
from keras.saving import register_keras_serializable

pic_width = 2048
pic_height = 2048

if pic_width % 4 != 0:
  raise Exception ('pic_width is not a power of 4')

if pic_height % 4 != 0:
  raise Exception ('pic_width is not a power of 4')

@function
def normalize_from_256 (picture):

  return (picture / 127.5) - 1

@function
def normalize_from_1 (picture):

  return (picture * 2) - 1

@function
def denormalize_to_256 (picture):

  return (picture + 1) * 127.5

@function
def denormalize_to_1 (picture):

  return (picture + 1) / 2

@register_keras_serializable ()
def metric_psnr (y_true, y_pred):
  return psnr (y_true, y_pred, max_val = 1.0)

@register_keras_serializable ()
def metric_ssim (y_true, y_pred):
  return ssim (y_true, y_pred, max_val = 1.0)

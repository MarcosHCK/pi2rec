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
from common import denormalize_to_256
from common import normalize_from_256
from common import pic_height, pic_width
from dataset import Dataset
from model import Pi2REC
import argparse, keras, numpy, os

def program ():

  parser = argparse.ArgumentParser (description = 'pi2rec')

  # Options
  parser.add_argument ('--checkpoint-dir', default = 'checkpoints/', help = 'checkpoint\'s root directory', metavar = 'DIRECTORY', type = str)
  parser.add_argument ('--checkpoint-prefix', default = 'chkp', help = 'checkpoint\'s prefix', metavar = 'VALUE', type = str)
  parser.add_argument ('--output', default = None, help = 'place result at FILE', metavar  = 'FILE', type = str)
  parser.add_argument ('--log-dir', default = 'logs/', help = 'place logs at DIRECTORY', metavar = 'DIRECTORY', type = str)
  parser.add_argument ('--mask', default = 'mask.svg', help = 'use mask FILE', metavar  = 'FILE', type = str)
  parser.add_argument ('--model', default = 'pi2rec.keras', help = 'use serialized model FILE', metavar = 'FILE', type = str)
  parser.add_argument ('--sample-at', default = 'sample/', help = 'output dataset sample at DIRECTORY', metavar  = 'DIRECTORY', type = str)
  parser.add_argument ('--sample-size', default = 10, help = 'take N dataset samples', metavar  = 'N', type = int)
  parser.add_argument ('--use-svg', default = True, help = 'use SVG masks (needs CairoSVG)', metavar = '<Y/N>', type = bool)

  # Subsystems
  parser.add_argument ('--freeze', help = 'use Pi2REC model to process FILE', action = 'store_true')
  parser.add_argument ('--process', help = 'use Pi2REC model to process FILE', metavar  = 'FILE', type = str)
  parser.add_argument ('--sample', help = 'take samples from dataset at DIRECTORY', metavar  = 'DIRECTORY', type = str)
  parser.add_argument ('--train', help = 'train using dataset at DIRECTORY', metavar = 'DIRECTORY', type = str)

  args = parser.parse_args ()

  if args.process != None:

    model = keras.models.load_model (args.model)
    image = keras.preprocessing.image.load_img (args.process)
    shape = (image.width, image.height)

    image = image.resize ((pic_width, pic_height))
    image = keras.preprocessing.image.img_to_array (image)
    image = normalize_from_256 (image)

    image = model.predict (numpy.expand_dims (image, axis = 0)) [0]
    image = denormalize_to_256 (image)

    image = keras.preprocessing.image.array_to_img (image)
    image = image.resize (shape)

    image.save ('output.jpg' if args.output == None else args.output)

  elif args.sample != None:

    dataset = Dataset (args.sample, args.mask, args.use_svg)

    if not os.path.exists (args.sample_at):

      os.mkdir (args.sample_at)

    for i, (image, target) in enumerate (dataset.take (args.sample_size)):

      image = keras.preprocessing.image.array_to_img (denormalize_to_256 (image))
      image.save (os.path.join (args.sample_at, f'input_{i}.jpg'))

      image = keras.preprocessing.image.array_to_img (denormalize_to_256 (target))
      image.save (os.path.join (args.sample_at, f'target_{i}.jpg'))

  elif args.train != None or args.freeze != None:

    model = Pi2REC (args.checkpoint_dir, args.checkpoint_prefix)

    if args.freeze:

      model.freeze ()
      model.generator.save (args.model)

    else:

      dataset = Dataset (args.train, args.mask, args.use_svg)

      dataset = dataset.shuffle (400)
      dataset = dataset.batch (1)

      model.train (dataset, args.log_dir)
      model.generator.save (args.model)

program ()

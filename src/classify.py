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
import argparse

def program ():

  parser = argparse.ArgumentParser ('py2rec_classify')

  # Options
  parser.add_argument ('--checkpoint-dir', default = 'checkpoints/', help = 'checkpoint\'s root directory', metavar = 'DIRECTORY', type = str)
  parser.add_argument ('--checkpoint-prefix', default = 'chkp', help = 'checkpoint\'s prefix', metavar = 'VALUE', type = str)
  parser.add_argument ('--log-dir', default = 'logs/', help = 'place logs at DIRECTORY', metavar = 'DIRECTORY', type = str)
  parser.add_argument ('--mask', default = 'mask.svg', help = 'use mask FILE', metavar  = 'FILE', type = str)
  parser.add_argument ('--model', default = 'pi2rec.keras', help = 'use serialized model FILE', metavar = 'FILE', type = str)
  parser.add_argument ('--use-svg', default = True, help = 'use SVG masks (needs CairoSVG)', metavar = '<Y/N>', type = bool)

  # Subsystems
  parser.add_argument ('--freeze', help = 'use Pi2REC model to process FILE', action = 'store_true')
  parser.add_argument ('--process', help = 'use Pi2REC model to process FILE', metavar  = 'FILE', type = str)
  parser.add_argument ('--train', help = 'train using dataset at DIRECTORY', metavar = 'DIRECTORY', type = str)

  args = parser.parse_args ()

  if args.process != None:

    from common import normalize_from_256
    from common import pic_height, pic_width
    import keras, numpy

    model = keras.models.load_model (args.model)
    image = keras.preprocessing.image.load_img (args.process)

    image = image.resize ((pic_width, pic_height))
    image = keras.preprocessing.image.img_to_array (image)
    image = normalize_from_256 (image)

    score = model.predict (numpy.expand_dims (image, axis = 0)) [0]

    print (f'score: {score [0]:.4f}')

  elif args.train != None or args.freeze:

    from classifier import Classifier
    from pi2rec import load_dataset

    model = Classifier (args.checkpoint_dir, args.checkpoint_prefix, args.log_dir)

    if args.freeze:

      model.freeze ()
      model.classifier.save (args.model)

    else:

      test, train = load_dataset (args.train, args.mask, args.use_svg)

      test = test.batch (1)
      train = train.shuffle (400)
      train = train.batch (1)

      model.train (train, test, 3333333)
      model.classifier.save (args.model)

if __name__ == "__main__":

  program ()

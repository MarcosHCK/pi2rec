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

batch_size = 16
epochs = 25

def Classifier ():

  from common import pic_height, pic_width
  import keras

  model = keras.Sequential \
    ([
      keras.layers.Conv2D (32, (3, 3), padding = 'same', activation = 'relu', input_shape = (pic_height, pic_width, 3)),
      keras.layers.MaxPooling2D ((2, 2), strides = 2),
      keras.layers.Conv2D (32, (3, 3), padding = 'same', activation = 'relu'),
      keras.layers.MaxPooling2D ((2, 2), strides = 2),
      keras.layers.Dropout (0.5),
      keras.layers.Flatten (),
      keras.layers.Dense (128, activation = 'relu'),
      keras.layers.Dense (2, activation = 'softmax'),
    ])

  return model

def program ():

  parser = argparse.ArgumentParser ('py2rec_classify')

  # Options
  parser.add_argument ('--model', default = 'pi2rec_classifier.keras', help = 'use serialized model FILE', metavar = 'FILE', type = str)

  # Subsystems
  parser.add_argument ('--process', help = 'use Pi2REC model to process FILE', metavar  = 'FILE', type = str)
  parser.add_argument ('--train', help = 'train using dataset at DIRECTORY', metavar = 'DIRECTORY', type = str)

  args = parser.parse_args ()

  if args.process != None:

    from dataset import load
    import keras

    model = keras.models.load_model (args.model)
    image = load (args.process)
    score = model.predict (image)

    print (f'score: {score:.02f}')

  elif args.train != None:

    from common import pic_height, pic_width
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    import numpy as np
    import keras, os, random
    import tensorflow as tf

    CATEGORIES = [ 'with_rule', 'without_rule' ]

    model = Classifier ()

    def load (path: str):

      image = tf.io.read_file (path)
      image = tf.io.decode_jpeg (image, channels = 3)
      image = tf.image.resize (image, [ pic_width, pic_height ])
      image = tf.cast (image, dtype = tf.float32)

      return image / 255.

    root = Path (args.train)
    training = [ [ load (str (root / cat / image)), n ] for n, cat in enumerate (CATEGORIES) for image in os.listdir (root / cat) ]

    random.shuffle (training)

    features, labels = zip (*training)

    X = np.array (features).reshape (-1, pic_height, pic_width, 3).astype (dtype = np.float32)

    Y = keras.utils.to_categorical (labels, num_classes = len (CATEGORIES))

    X_train, X_test, y_train, y_test = train_test_split (X, Y, test_size = 0.2, random_state = 4)

    model.compile (optimizer = keras.optimizers.Adam (), loss = 'categorical_crossentropy', metrics = [ 'accuracy' ])
    model.fit (X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (X_test, y_test))
    model.save (args.model)

    score = model.evaluate (X_test, y_test, verbose = 0)

    print("Test accuracy: ", score[1])

if __name__ == "__main__":

  program ()

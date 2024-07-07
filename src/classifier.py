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
from common import pic_height, pic_width
from datetime import datetime
import tensorflow as tf
import keras, numpy, os, time

class Classifier:

  def __init__ (self, checkpoint_dir: str = 'checkpoints/', checkpoint_prefix: str = 'ckp', log_dir: str = 'logs/'):

    self.classifier = keras.Sequential \
      ([
        keras.layers.Conv2D (4, 3, padding = 'same', activation = 'relu', input_shape = (pic_height, pic_width, 3)),
        keras.layers.MaxPooling2D (),
        keras.layers.Conv2D (16, 3, padding = 'same', activation = 'relu'),
        keras.layers.MaxPooling2D (),
        keras.layers.Conv2D (32, 3, padding = 'same', activation = 'relu'),
        keras.layers.MaxPooling2D (),
        keras.layers.Flatten (),
        keras.layers.Dense (64, activation = 'relu'),
        keras.layers.Dense (1, activation = 'sigmoid'),
      ])

    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_prefix = os.path.join (checkpoint_dir, checkpoint_prefix)
    self.log_dir = log_dir

  def freeze (self):

    latest = tf.train.latest_checkpoint (self.checkpoint_dir)

    if latest != None:

      optimizer = keras.optimizers.Adam (2e-4, beta_1 = 0.5)
      checkpoint = tf.train.Checkpoint (classifier = self.classifier, classifier_optimizer = optimizer)

      checkpoint.restore (latest)

  def train (self, train: "tf.data.Dataset", test: "tf.data.Dataset", steps: int, cyclesz: int = -1):

    checkpoint_dir = self.checkpoint_dir
    checkpoint_prefix = self.checkpoint_prefix

    classifier = self.classifier
    classifier_loss = keras.losses.BinaryCrossentropy (from_logits = False)
    classifier_optimizer = keras.optimizers.Adam (2e-4, beta_1 = 0.5)

    log_name = datetime.now ().strftime ('%Y%m%d-%H%M%S')
    log_name = os.path.join (self.log_dir, log_name)

    checkpoint = tf.train.Checkpoint (classifier = classifier, classifier_optimizer = classifier_optimizer)

    summary_writer = tf.summary.create_file_writer (log_name)

    if not os.path.exists (checkpoint_dir):

      os.makedirs (checkpoint_dir)

    else:

      if (latest := tf.train.latest_checkpoint (checkpoint_dir)) != None:

        checkpoint.restore (latest)

    @tf.function
    def fit_step (image, y_true, n):

      with tf.GradientTape () as tape:

        y_pred = classifier (image, training = True)
        loss = classifier_loss (y_true, y_pred)

      classifier_grads = tape.gradient (loss, classifier.trainable_variables)
      classifier_optimizer.apply_gradients (zip (classifier_grads, classifier.trainable_variables))

      with summary_writer.as_default ():

        tf.summary.scalar ('loss', loss, step = n)

    def fit (train : "tf.data.Dataset", test : "tf.data.Dataset", steps : int, cyclesz: int = -1):

      begin = time.time ()
      cyclesz = cyclesz if cyclesz > 0 else len (train)

      checkpoint_rate = cyclesz * 6
      sample_rate = int (cyclesz / 2)

      for (_, __) in train.take (1):

        batchsz = len (_)
        break

      for step, (image, target) in train.repeat ().take (steps).enumerate ():

        if step % cyclesz == 0:

          if step > 0:

            print (f'')
            print (f'Time taken for {cyclesz} steps: {time.time () - begin} seconds')

          begin = time.time ()

        if (1 + step) % checkpoint_rate == 0:

          print ('Time for a checkpoint, right?')
          checkpoint.save (file_prefix = checkpoint_prefix)

        fit_step (image, numpy.array ([[ 1 ]] * batchsz), step)
        fit_step (target, numpy.array ([[ 0 ]] * batchsz), step)

        if (1 + step) % sample_rate != 0:

          print ('.', end = '', flush = True)
        else:
          print ('x', end = '', flush = True)

    try:

      fit (train, test, steps, cyclesz)

    except KeyboardInterrupt:

      print ('k')
      checkpoint.save (file_prefix = checkpoint_prefix)

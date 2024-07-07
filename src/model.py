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
from common import denormalize_to_1
from common import metric_psnr, metric_ssim
from common import pic_width, pic_height
from datetime import datetime
from discriminator import Discriminator, DiscriminatorLoss
from generator import Generator, GeneratorLoss
import keras, os, time
import tensorflow as tf

class Pi2REC ():

  def __init__ (self, checkpoint_dir: str = 'checkpoints/', checkpoint_prefix: str = 'ckp', log_dir: str = 'logs/'):

    self.discriminator = Discriminator (pic_width, pic_height, channels = 3)
    self.generator = Generator (pic_width, pic_height, channels = 3)
    
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_prefix = os.path.join (checkpoint_dir, checkpoint_prefix)
    self.log_dir = log_dir

  def freeze (self):

    latest = tf.train.latest_checkpoint (self.checkpoint_dir)

    if latest != None:

      checkpoint = tf.train.Checkpoint (discriminator = self.discriminator, generator = self.generator)
      checkpoint.restore (latest)

  def train (self, train: "tf.data.Dataset", test: "tf.data.Dataset", steps : int, cyclesz: int = -1):

    checkpoint_dir = self.checkpoint_dir
    checkpoint_prefix = self.checkpoint_prefix

    discriminator = self.discriminator
    discriminator_loss = DiscriminatorLoss ()
    discriminator_optimizer = keras.optimizers.Adam (2e-4, beta_1 = 0.5)

    generator = self.generator
    generator_loss = GeneratorLoss ()
    generator_optimizer = keras.optimizers.Adam (2e-4, beta_1 = 0.5)

    log_name = datetime.now ().strftime ('%Y%m%d-%H%M%S')
    log_name = os.path.join (self.log_dir, log_name)

    checkpoint = tf.train.Checkpoint (
      discriminator_optimizer = discriminator_optimizer,
      generator_optimizer = generator_optimizer,
      discriminator = discriminator,
      generator = generator)

    metrics = { 'psnr' : metric_psnr, 'ssim' : metric_ssim }

    summary_writer = tf.summary.create_file_writer (log_name)

    if not os.path.exists (checkpoint_dir):

      os.makedirs (checkpoint_dir)

    elif (latest := tf.train.latest_checkpoint (checkpoint_dir)) != None:

      checkpoint.restore (latest)

    @tf.function
    def fit_step (input_image, target, n):

      with tf.GradientTape () as gen_tape, tf.GradientTape () as dis_tape:

        y_pred = generator (input_image, training = True)
        y_true = discriminator ([input_image, target], training = True)
        y_disc = discriminator ([input_image, y_pred], training = True)

        loss_total, loss_gan, loss_l1 = generator_loss (y_disc, y_pred, target)
        loss_disc = discriminator_loss (y_true, y_disc)

      generator_grads = gen_tape.gradient (loss_total, generator.trainable_variables)
      discriminator_grads = dis_tape.gradient (loss_disc, discriminator.trainable_variables)

      generator_optimizer.apply_gradients (zip (generator_grads, generator.trainable_variables))
      discriminator_optimizer.apply_gradients (zip (discriminator_grads, discriminator.trainable_variables))

      with summary_writer.as_default ():

        tf.summary.scalar ('loss_total', loss_total, step = n)
        tf.summary.scalar ('loss_gan', loss_gan, step = n)
        tf.summary.scalar ('loss_l1', loss_l1, step = n)
        tf.summary.scalar ('loss_disc', loss_disc, step = n)

    def fit (train: "tf.data.Dataset", test: "tf.data.Dataset", steps : int, cyclesz: int = -1):

      begin = time.time ()
      cyclesz = cyclesz if cyclesz > 0 else len (train)

      checkpoint_rate = cyclesz * 8

      sample_rate = cyclesz // 2
      sample_input, sample_target = next (iter (test.take (1)))

      with summary_writer.as_default ():

        tf.summary.image ('sample', [ denormalize_to_1 (sample_input [0]) ], step = 0)

      for step, (image, target) in train.repeat ().take (steps).enumerate ():

        if step % cyclesz == 0:

          if step > 0:

            print (f'')
            print (f'Time taken for {cyclesz} steps: {time.time () - begin:.2f} seconds')

          begin = time.time ()

        if (1 + step) % checkpoint_rate == 0 and step > 0:

          print ('Time for a checkpoint, right?')
          checkpoint.save (file_prefix = checkpoint_prefix)

        fit_step (image, target, step)

        if (1 + step) % sample_rate != 0:

          print ('.', end = '', flush = True)
        else:

          pred = generator (sample_input, training = False)

          with summary_writer.as_default ():

            tf.summary.image ('sample', [ denormalize_to_1 (pred [0]) ], step = step)

            for name, value in [ (name, func (sample_target, pred) [0]) for name, func in metrics.items () ]:

              tf.summary.scalar (name, value, step = step)

          print ('x', end = '', flush = True)

    try:

      fit (train, test, steps, cyclesz)

    except KeyboardInterrupt:

      print ('k')
      checkpoint.save (file_prefix = checkpoint_prefix)

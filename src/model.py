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
from common import pic_width
from common import pic_height
from dataset import Dataset
from datetime import datetime
from discriminator import Discriminator, DiscriminatorLoss
from generator import Generator, GeneratorLoss
from paste import paste
from rotate import rotate
import argparse, io, keras, numpy, os, random, time
import tensorflow as tf

def train (dataset, checkpoint_dir = 'checkpoints/', checkpoint_prefix = 'ckp', log_dir = 'logs/'):

  if pic_width % 4 != 0:
    raise Exception ('pic_width is not a power of 4')
  if pic_height % 4 != 0:
    raise Exception ('pic_width is not a power of 4')

  metrics = [ keras.metrics.MeanSquaredError (), metric_psnr, metric_ssim ]

  generator = Generator (pic_width, pic_height, channels = 3)
  discriminator = Discriminator (pic_width, pic_height, channels = 3)

  generator_loss = GeneratorLoss ()
  discriminator_loss = DiscriminatorLoss ()

  generator_optimizer = keras.optimizers.Adam (2e-4, beta_1 = 0.5)
  discriminator_optimizer = keras.optimizers.Adam (2e-4, beta_1 = 0.5)

  log_name = datetime.now ().strftime ('%Y%m%d-%H%M%S')
  log_name = os.path.join (log_dir, log_name)

  summary_writer = tf.summary.create_file_writer (log_name)

  checkpoint_prefix = os.path.join (checkpoint_dir, checkpoint_prefix)
  checkpoint = tf.train.Checkpoint (discriminator_optimizer = discriminator_optimizer,
                                    generator_optimizer = generator_optimizer,
                                    discriminator = discriminator,
                                    generator = generator)

  # Actual training loop

  @tf.function
  def fit_step (inputs, target, n : int, summary_writer : tf.summary.SummaryWriter):

    with tf.GradientTape () as gen_tape, tf.GradientTape () as dis_tape:

      y_pred = generator (inputs, training = True)
      y_true = discriminator ([inputs, target], training = True)
      y_disc = discriminator ([inputs, y_pred], training = True)

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

  def fit (dataset : "DatasetV2", steps : int, summary_writer : tf.summary.SummaryWriter):

    begin = time.time ()
    cyclesz = len (dataset)

    for step, (image, target) in dataset.repeat ().take (steps).enumerate ():

      if step % cyclesz == 0 and step > 0:

        print (f'')
        print (f'Time taken for {steps} steps: {time.time () - begin} seconds, checkpoint hit')

        begin = time.time ()
        checkpoint.save (file_prefix = checkpoint_prefix)

      fit_step (image, target, step, summary_writer)

      print ('.', end = '', flush = True)

  if not os.path.exists (checkpoint_dir):

    os.makedirs (checkpoint_dir)

  else:

    latest = tf.train.latest_checkpoint (checkpoint_dir)

    if latest != None:

      checkpoint.restore (latest)

  fit (dataset, 33, summary_writer = summary_writer)
  return generator

def take_sample (dataset, size, directory):

  if not os.path.exists (directory):
    os.mkdir (directory)
  for i, (image, target) in enumerate (dataset.take (size)):

    image = keras.preprocessing.image.array_to_img ((image * 127.0) + 127.0)
    image.save (os.path.join (directory, f'input_{i}.jpg'))
    image = keras.preprocessing.image.array_to_img ((target * 127.0) + 127.0)
    image.save (os.path.join (directory, f'target_{i}.jpg'))

def program ():

  parser = argparse.ArgumentParser (description = 'pi2rec')

  parser.add_argument ('dataset',
      default = 'dataset/',
      help = 'dataset root',
      metavar = 'directory',
      type = str)
  parser.add_argument ('--checkpoint-dir',
      default = 'checkpoints/',
      help = 'checkpoints root',
      metavar = 'directory',
      type = str)
  parser.add_argument ('--checkpoint-prefix',
      default = 'chkp',
      help = 'checkpoints prefix',
      metavar = 'prefix',
      type = str)
  parser.add_argument ('--log-dir',
      default = 'logs/',
      help = 'logs root',
      metavar = 'directory',
      type = str)
  parser.add_argument ('--mask',
      default = 'mask.png',
      help = 'epoch number to train in',
      metavar  = 'file',
      type = str)
  parser.add_argument ('--output',
      default = 'pi2rec.keras',
      help = 'take dataset sample',
      metavar  = 'directory',
      type = str)
  parser.add_argument ('--sample',
      help = 'take dataset sample',
      metavar  = 'N',
      type = int)
  parser.add_argument ('--sample-at',
      default = 'dataset_sample/',
      help = 'take dataset sample',
      metavar  = 'directory',
      type = str)
  parser.add_argument ('--use-svg',
      action = 'store_true',
      help = 'use SVG masks (instead of PNG)')

  args = parser.parse_args ()

  dataset = Dataset (args.dataset, args.mask, args.use_svg)

  if args.sample != None:

    take_sample (dataset, args.sample, args.sample_at)

  else:

    dataset = dataset.shuffle (400)
    dataset = dataset.batch (1)

    model = train (dataset, args.checkpoint_dir, args.checkpoint_prefix, args.log_dir)

    model.save (args.output)

program ()

"""Generates a stylized image for every image in a directory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import glob
import math
import os

import numpy as np
import tensorflow as tf

from magenta.models.image_stylization import image_utils
from magenta.models.image_stylization import model


flags = tf.flags
flags.DEFINE_integer('num_styles', 1,
                     'Number of styles the model was trained on.')
flags.DEFINE_string('checkpoint', None, 'Checkpoint to load the model from')
flags.DEFINE_string('input_dir', None, 'Directory containing the input images')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_string('output_basename', None, 'Output base name.')
flags.DEFINE_string('which_styles', '[0]',
                    'Which styles to use. This is either a Python list or a '
                    'dictionary. If it is a list then a separate image will be '
                    'generated for each style index in the list. If it is a '
                    'dictionary which maps from style index to weight then a '
                    'single image with the linear combination of style weights '
                    'will be created. [0] is equivalent to {0: 1.0}.')
flags.DEFINE_integer('batch_size', 10, 'The number of images to stylize per batch.')
FLAGS = flags.FLAGS


def _load_checkpoint(sess, checkpoint):
  """Loads a checkpoint file into the session."""
  model_saver = tf.train.Saver(tf.global_variables())
  checkpoint = os.path.expanduser(checkpoint)
  if tf.gfile.IsDirectory(checkpoint):
    checkpoint = tf.train.latest_checkpoint(checkpoint)
    tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))
  model_saver.restore(sess, checkpoint)


def main(unused_argv=None):
  input_dir = FLAGS.input_dir
  output_dir = FLAGS.output_dir
  batch_size = FLAGS.batch_size
  which_styles = ast.literal_eval(FLAGS.which_styles)
  checkpoint = FLAGS.checkpoint
  num_styles = FLAGS.num_styles

  input_files = glob.glob(input_dir + '/*.png')
  iterations = int(math.ceil(len(input_files) / batch_size))

  with tf.Graph().as_default(), tf.Session() as sess:
    images = tf.placeholder(tf.float32, shape=(None, 1280, 1920, 3))
    stylizer = model.transform(
        images,
        normalizer_params={
            'labels': tf.constant(which_styles),
            'num_categories': num_styles,
            'center': True,
            'scale': True})
    _load_checkpoint(sess, checkpoint)

    for i in range(iterations):
        batch_files = input_files[batch_size * i:batch_size * (i + 1)]
        batch_images = []
        for file in batch_files:
            batch_images.append(image_utils.load_np_image(file))

        stylized_images = sess.run(stylizer, {images: batch_images})

        for input_file, stylized_image in zip(batch_files, stylized_images):
            image_utils.save_np_image(
                stylized_image[None, ...],
                input_file.replace(input_dir, output_dir))


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()

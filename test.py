#!/usr/bin/python3

from absl import app, flags;
import numpy as np;
import cv2;
from models import LAPGAN;

FLAGS = flags.FLAGS;
flags.DEFINE_enum('device', default = 'cpu', enum_values = ['cpu', 'gpu'], help = 'device to run');

def main(unused_argv):
  lapgan = LAPGAN(device = FLAGS.device);
  for i in range(3):
    cv2.namedWindow(str(i), cv2.WINDOW_NORMAL);
  while True:
    imgs = lapgan.generate(batch_size = 1); # img.shape = (batch, channels, height, width)
    imgs = [np.transpose(img, (0, 2, 3, 1)) for img in imgs]; # img.shape = (batch, height, width, channels)
    imgs = [np.squeeze(img, axis = 0) for img in imgs]; # img.shape = (height, width, channels)
    # NOTE: img range in [-1, +1]
    imgs = [(255. * (img * 0.5 + 0.5)).astype(np.uint8) for img in imgs];
    for i in range(3):
      cv2.imshow(str(i), imgs[i]);
    cv2.waitKey();

if __name__ == "__main__":
  app.run(main)


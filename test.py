#!/usr/bin/python3

from absl import app, flags;
import numpy as np;
import cv2;
from models import LAPGAN;

FLAGS = flags.FLAGS;
flags.DEFINE_enum('device', default = 'cpu', enum_values = ['cpu', 'gpu'], help = 'device to run');

def main(unused_argv):
  lapgan = LAPGAN(device = FLAGS.device);
  cv2.namedWindow('generate', cv2.WINDOW_NORMAL);
  while True:
    imgs = lapgan.generate(batch_size = 1); # imgs.shape = (batch, channels, height, width)
    imgs = np.transpose(imgs, (0, 2, 3, 1));
    img = np.squeeze(imgs, axis = 0); # img.shape = (height, width, 3)
    # NOTE: img range in [-1, +1]
    img = (255. * (img * 0.5 + 0.5)).astype(np.uint8);
    cv2.imshow('generate', img);
    cv2.waitKey();

if __name__ == "__main__":
  app.run(main)


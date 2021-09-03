#!/usr/bin/python3

import argparse;
from os import mkdir;
from os.path import exists, join;
from torch import save;
import pytorch_lightning as pl;
from pytorch_lightning.callbacks import ModelCheckpoint;
from models import Trainer;
from create_dataset import CIFAR10Dataset;

def main():
  pl.seed_everything(1234);
  parser = argparse.ArgumentParser();
  parser = pl.Trainer.add_argparse_args(parser);
  parser = Trainer.add_model_specific_args(parser);
  parser.add_argument('--checkpoint', type = str, default = None);
  args = parser.parse_args();

  model = Trainer.load_from_checkpoint(args = args, checkpoint_path = args.checkpoint);
  if not exists('models'): mkdir('models');
  save(model.generators[0], join('models', 'gen_0.pth'));
  save(model.generators[1], join('models', 'gen_1.pth'));
  save(model.generators[2], join('models', 'gen_2.pth'));

if __name__ == "__main__":
  main();

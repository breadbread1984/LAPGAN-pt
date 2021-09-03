#!/usr/bin/python3

import argparse;
import pytorch_lightning as pl;
from pytorch_lightning.callbacks import ModelCheckpoint;
from models import Trainer;
from create_dataset import CIFAR10Dataset;

def main():
  pl.seed_everything(1234);
  parser = argparse.ArgumentParser();
  parser = pl.Trainer.add_argparse_args(parser);
  parser = Trainer.add_model_specific_args(parser);
  parser.add_argument('--batch_size', type = int, default = 256);
  parser.add_argument('--num_workers', type = int, default = 8);
  parser.add_argument('--download', type = bool, default = False);
  args = parser.parse_args();

  dataset = CIFAR10Dataset(args);
  model = Trainer(args);
  
  callbacks = [ModelCheckpoint(every_n_train_steps = 10)];
  kwargs = dict();
  trainer = pl.Trainer.from_argparse_args(args, callbacks = callbacks, max_steps = 25 * 50000 / args.batch_size, **kwargs);
  trainer.fit(model, dataset);

if __name__ == "__main__":
  main();

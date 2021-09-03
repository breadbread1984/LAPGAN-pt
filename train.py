#!/usr/bin/python3

import argparse;
import pytorch_lightning as pl;
from pytorch_lightning.callbacks import ModelCheckpoint;
from pytorch_lightning.loggers import TensorBoardLogger;
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
  parser.add_argument('--checkpoint', type = str, default = None);
  args = parser.parse_args();

  dataset = CIFAR10Dataset(args);
  model = Trainer(args) if args.checkpoint is None else Trainer.load_from_checkpoint(args = args, checkpoint_path = args.checkpoint);
  
  callbacks = [ModelCheckpoint(every_n_train_steps = 10)];
  logger = TensorBoardLogger('tb_logs', name = 'LAPGAN');
  kwargs = dict();
  trainer = pl.Trainer.from_argparse_args(args, callbacks = callbacks, max_steps = 25 * 50000 / args.batch_size, logger = logger, **kwargs);
  trainer.fit(model, dataset);

if __name__ == "__main__":
  main();

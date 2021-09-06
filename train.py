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
  parser.add_argument('--epochs', type = int, default = 25);
  args = parser.parse_args();

  dataset = CIFAR10Dataset(args);
  model = Trainer(args) if args.checkpoint is None else Trainer.load_from_checkpoint(args = args, checkpoint_path = args.checkpoint);
  
  callbacks = [ModelCheckpoint(every_n_train_steps = 10)];
  logger = TensorBoardLogger('tb_logs', name = 'LAPGAN', log_graph = True);
  kwargs = dict();
  trainer = pl.Trainer(max_epochs = args.epochs, check_val_every_n_epoch = 1, callbacks = callbacks, logger = logger, gpus = args.gpus, **kwargs);
  trainer.fit(model, dataset);

if __name__ == "__main__":
  main();

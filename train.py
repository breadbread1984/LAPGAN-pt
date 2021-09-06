#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
import argparse;
from torch import save;
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
  trainer = pl.Trainer(max_epochs = args.epochs, val_check_interval = 0.1, callbacks = callbacks, logger = logger, gpus = args.gpus, **kwargs);
  trainer.fit(model, dataset);
  if not exists('models'): mkdir('models');
  save(model.generators[0], join('models', 'gen_0.pth'));
  save(model.generators[1], join('models', 'gen_1.pth'));
  save(model.generators[2], join('models', 'gen_2.pth'));

if __name__ == "__main__":
  main();

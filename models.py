#!/usr/bin/python3

import torch;
from torch import nn;
import pytorch_lightning as pl;
from pytorch_functional import Input, FunctionalModel, layers;

def DiscriminatorZero():
  laplacian = Input(shape = (3,32,32)); # laplacian response
  coarse = Input(shape = (3,32,32)); # coarsed high resolution image
  results = laplacian(layers.AddOpLayer(), coarse); # results.shape = (batch, 3, 32, 32)
  results = results(nn.Conv2d(results.channels, 128, kernel_size = (5,5)))(nn.LeakyReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 128, kernel_size = (5,5), stride = (2,2)))(nn.LeakyReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Flatten());
  results = results(nn.Linear(results.channels, 1));
  results = results(nn.Sigmoid())
  return FunctionalModel(inputs = (laplacian, coarse), outputs = results);

def DiscriminatorOne():
  laplacian = Input(shape = (3,16,16)); # laplacian response
  coarse = Input(shape = (3,16,16)); # coarsed high resolution image
  results = laplacian(layers.AddOpLayer(), coarse); # results.shape = (batch, 3, 16, 16)
  results = results(nn.Conv2d(results.channels, 64, kernel_size = (5,5)))(nn.LeakyReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 64, kernel_size = (5,5), stride = (2,2)))(nn.LeakyReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Flatten());
  results = results(nn.Linear(results.channels, 1));
  results = results(nn.Sigmoid());
  return FunctionalModel(inputs = (laplacian, coarse), outputs = results);

def DiscriminatorTwo():
  coarse = Input(shape = (3,8,8)); # low_pass.shape = (batch, 3, 8, 8)
  results = coarse(nn.Flatten());
  results = results(nn.Linear(results.channels, 600))(nn.LeakyReLU());
  results = results(nn.Linear(results.channels, 600))(nn.LeakyReLU());
  results = results(nn.Linear(results.channels, 1));
  results = results(nn.Sigmoid());
  return FunctionalModel(inputs = coarse, outputs = results);

def GeneratorZero():
  noise = Input(shape = (1,32,32));
  coarse = Input(shape = (3,32,32));
  results = coarse(layers.ConcatLayer(dim = 1), noise);
  results = results(nn.Conv2d(results.channels, 128, kernel_size = (3,3), padding = 1))(nn.ReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 128, kernel_size = (3,3), padding = 1))(nn.ReLU());
  results = results(nn.BatchNorm2d(results.channels));
  laplacian = results(nn.Conv2d(results.channels, 3, kernel_size = (3,3), padding = 1));
  return FunctionalModel(inputs = (noise, coarse), outputs = laplacian);

def GeneratorOne():
  noise = Input(shape = (1,16,16));
  coarse = Input(shape = (3,16,16));
  results = coarse(layers.ConcatLayer(dim = 1), noise);
  results = results(nn.Conv2d(results.channels, 64, kernel_size = (3,3), padding = 1))(nn.ReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 64, kernel_size = (3,3), padding = 1))(nn.ReLU());
  results = results(nn.BatchNorm2d(results.channels));
  laplacian = results(nn.Conv2d(results.channels, 3, kernel_size = (3,3), padding = 1));
  return FunctionalModel(inputs = (noise, coarse), outputs = laplacian);

def GeneratorTwo():
  noise = Input(shape = (100,));
  results = noise(nn.Linear(noise.channels, 1200))(nn.ReLU());
  results = results(nn.Linear(results.channels, 1200));
  results = results(nn.Sigmoid());
  results = results(nn.Linear(results.channels, 3*8*8));
  coarse = results(layers.ReshapeLayer((3,8,8)));
  return FunctionalModel(inputs = noise, outputs = coarse);

class Trainer(pl.LightningModule):
  def __init__(self, args):
    super(Trainer, self).__init__();
    self.args = args;
    self.generators = nn.ModuleList([GeneratorZero(), GeneratorOne(), GeneratorTwo()]);
    self.discriminators = nn.ModuleList([DiscriminatorZero(), DiscriminatorOne(), DiscriminatorTwo()]);
    self.criterion = nn.BCELoss();
  def forward(self, x):
    # NOTE: x = (input0, input1, dummy_input2, true_input0, true_input1, true_input2)
    # NOTE: real batch size is not always equals to self.args.batch_size, for example, the last batch of an epochs always less than self.args.batch_size
    batch_size = x[0].shape[0];
    gen_losses = list();
    disc_losses = list();
    for idx, generator in enumerate(self.generators):
      # 1) noise = noise[, condition = coarsed] -> fake_residual
      noise = torch.normal(mean = torch.zeros([batch_size,] + list(generator.inputs[0].shape[1:])), std = 0.1 * torch.ones([batch_size,] + list(generator.inputs[0].shape[1:])));
      if self.args.gpus > 0: noise = noise.cuda();
      coarse = x[idx];
      if len(generator.inputs) == 2:
        fake = generator([noise, coarse]); # fake laplacian
      else:
        fake = generator(noise); # fake coarse image
      # 2) sample = (fake_residual, real_residual)[, condition = (coarsed, coarsed)] -> (true|false)
      real = x[idx + 3];
      samples = torch.cat([fake, real]);
      if len(self.discriminators[idx].inputs) == 2:
        conditions = torch.cat([coarse, coarse]);
        predictions = self.discriminators[idx]([samples, conditions]);
      else:
        predictions = self.discriminators[idx](samples);
      # 3) generator loss
      gen_labels = torch.ones([batch_size,]);
      if self.args.gpus > 0: gen_labels = gen_labels.cuda();
      gen_loss = self.criterion(predictions[:batch_size,0], gen_labels);
      # 4) discriminator loss
      disc_labels = torch.cat([torch.zeros([batch_size,]), torch.ones([batch_size,])]);
      if self.args.gpus > 0: disc_labels = disc_labels.cuda();
      disc_loss = self.criterion(predictions[:,0], disc_labels);
      # 5) save loss
      gen_losses.append(gen_loss);
      disc_losses.append(disc_loss);
    return tuple(gen_losses + disc_losses);
  def training_step(self, batch, batch_idx):
    from functools import reduce;
    samples, labels = batch;
    losses = self.forward(samples);
    loss = reduce(torch.add, losses);
    batch_dictionary = {
      'loss': loss,
      'log': {
        'train/gen0_loss': losses[0].cpu().detach().numpy(),
        'train/gen1_loss': losses[1].cpu().detach().numpy(),
        'train/gen2_loss': losses[2].cpu().detach().numpy(),
        'train/disc0_loss': losses[3].cpu().detach().numpy(),
        'train/disc1_loss': losses[4].cpu().detach().numpy(),
        'train/disc2_loss': losses[5].cpu().detach().numpy()
      }
    };
    return batch_dictionary;
  def validation_step(self, batch, batch_idx):
    from functools import reduce;
    samples, labels = batch;
    losses = self.forward(samples);
    self.log('val/gen0_loss', losses[0], prog_bar = True);
    self.log('val/gen1_loss', losses[1], prog_bar = True);
    self.log('val/gen2_loss', losses[2], prog_bar = True);
    self.log('val/disc0_loss', losses[3], prog_bar = True);
    self.log('val/disc1_loss', losses[4], prog_bar = True);
    self.log('val/disc2_loss', losses[5], prog_bar = True);
    self.log('val/total_loss', reduce(torch.add, losses), prog_bar = True);
  def configure_optimizers(self):
    return torch.optim.Adam([{'params': self.generators[0].parameters(), 'lr': 0.0003},
                             {'params': self.generators[1].parameters(), 'lr': 0.0005},
                             {'params': self.generators[2].parameters(), 'lr': 0.003},
                             {'params': self.discriminators[0].parameters(), 'lr': 0.0001},
                             {'params': self.discriminators[1].parameters(), 'lr': 0.0001},
                             {'params': self.discriminators[2].parameters(), 'lr': 0.0001}], lr = 3e-4, betas = (0.5, 0.999));
  @staticmethod
  def add_model_specific_args(parent_parser):
    import argparse;
    parser = argparse.ArgumentParser(parents = [parent_parser], add_help = False);
    return parser;

class LAPGAN(object):
  def __init__(self, model_dir = 'models', device = 'gpu'):
    from os.path import join;
    from torch import load;
    self.models = list();
    with open(join(model_dir, 'gen_0.pth'), 'rb') as f:
      gen_0 = load(f); gen_0.eval(); self.models.append(gen_0);
    with open(join(model_dir, 'gen_1.pth'), 'rb') as f:
      gen_1 = load(f); gen_1.eval(); self.models.append(gen_1);
    with open(join(model_dir, 'gen_2.pth'), 'rb') as f:
      gen_2 = load(f); gen_2.eval(); self.models.append(gen_2);
    self.device = device;
    if self.device == 'gpu':
      for model in self.models:
        model = model.cuda();
  def generate(self, batch_size = 4):
    import numpy as np;
    import cv2;
    for model in reversed(self.models):
      noise = torch.from_numpy(np.random.normal(loc = 0, scale = 0.1, size = [batch_size,] + list(model.inputs[0].shape[1:])).astype(np.float32));
      if self.device == 'gpu':
        noise = noise.cuda();
      if len(model.inputs) == 1:
        imgs = model(noise);
        imgs = imgs.cpu().detach().numpy();
      else:
        imgs = np.array([[cv2.pyrUp(imgs[b, c,...]) for c in range(3)] for b in range(batch_size)]);
        inputs = torch.from_numpy(imgs);
        if self.device == 'gpu':
          inputs = inputs.cuda();
        residual = model([noise, inputs]);
        residual = residual.cpu().detach().numpy();
        imgs = residual + imgs;
    return imgs;

if __name__ == "__main__":

  import numpy as np;
  disc_0 = DiscriminatorZero(); disc_0 = disc_0.eval();
  results = disc_0.forward([torch.from_numpy(np.random.normal(size = (4,3,32,32)).astype(np.float32)), torch.from_numpy(np.random.normal(size = (4,3,32,32)).astype(np.float32))]);
  assert results is not None; print(results.detach().numpy().shape);
  disc_1 = DiscriminatorOne(); disc_1 = disc_1.eval();
  results = disc_1.forward([torch.from_numpy(np.random.normal(size = (4,3,16,16)).astype(np.float32)), torch.from_numpy(np.random.normal(size = (4,3,16,16)).astype(np.float32))]);
  assert results is not None; print(results.detach().numpy().shape);
  disc_2 = DiscriminatorTwo(); disc_2 = disc_2.eval();
  results = disc_2.forward(torch.from_numpy(np.random.normal(size = (4,3,8,8)).astype(np.float32)));
  assert results is not None; print(results.detach().numpy().shape);
  gen_0 = GeneratorZero(); gen_0 = gen_0.eval();
  results = gen_0.forward([torch.from_numpy(np.random.normal(size = (4,1,32,32)).astype(np.float32)), torch.from_numpy(np.random.normal(size = (4,3,32,32)).astype(np.float32))]);
  assert results is not None; print(results.detach().numpy().shape);
  gen_1 = GeneratorOne(); gen_1 = gen_1.eval();
  results = gen_1.forward([torch.from_numpy(np.random.normal(size = (4,1,16,16)).astype(np.float32)), torch.from_numpy(np.random.normal(size = (4,3,16,16)).astype(np.float32))]);
  assert results is not None; print(results.detach().numpy().shape);
  gen_2 = GeneratorTwo(); gen_2 = gen_2.eval();
  results = gen_2.forward(torch.from_numpy(np.random.normal(size = (4,100)).astype(np.float32)));
  assert results is not None; print(results.detach().numpy().shape);

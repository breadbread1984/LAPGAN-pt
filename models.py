#!/usr/bin/python3

import torch;
from torch import nn;
import pytorch_lightning as pl;
from pytorch_functional import Input, FunctionalModel, layers;

def DiscriminatorZero():
  high_pass = Input(shape = (3,32,32)); # laplacian response
  low_pass = Input(shape = (3,32,32)); # coarsed high resolution image
  results = high_pass(layers.AddOpLayer(), low_pass); # results.shape = (batch, 3, 32, 32)
  results = results(nn.Conv2d(results.channels, 128, kernel_size = (5,5)))(nn.LeakyReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 128, kernel_size = (5,5), stride = (2,2)))(nn.LeakyReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Flatten());
  results = results(nn.Linear(results.channels, 1));
  results = results(nn.Sigmoid())
  return FunctionalModel(inputs = (high_pass, low_pass), outputs = results);

def DiscriminatorOne():
  high_pass = Input(shape = (3,16,16)); # laplacian response
  low_pass = Input(shape = (3,16,16)); # coarsed high resolution image
  results = high_pass(layers.AddOpLayer(), low_pass); # results.shape = (batch, 3, 16, 16)
  results = results(nn.Conv2d(results.channels, 64, kernel_size = (5,5)))(nn.LeakyReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 64, kernel_size = (5,5), stride = (2,2)))(nn.LeakyReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Flatten());
  results = results(nn.Linear(results.channels, 1));
  results = results(nn.Sigmoid());
  return FunctionalModel(inputs = (high_pass, low_pass), outputs = results);

def DiscriminatorTwo():
  low_pass = Input(shape = (3,8,8)); # low_pass.shape = (batch, 3, 8, 8)
  results = low_pass(nn.Flatten());
  results = results(nn.Linear(results.channels, 600))(nn.LeakyReLU());
  results = results(nn.Linear(results.channels, 600))(nn.LeakyReLU());
  results = results(nn.Linear(results.channels, 1));
  results = results(nn.Sigmoid());
  return FunctionalModel(inputs = low_pass, outputs = results);

def GeneratorZero():
  noise = Input(shape = (1,32,32));
  low_pass = Input(shape = (3,32,32));
  results = low_pass(layers.ConcatLayer(dim = 1), noise);
  results = results(nn.Conv2d(results.channels, 128, kernel_size = (3,3), padding = 1))(nn.ReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 128, kernel_size = (3,3), padding = 1))(nn.ReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 3, kernel_size = (3,3), padding = 1));
  return FunctionalModel(inputs = (noise, low_pass), outputs = results);

def GeneratorOne():
  noise = Input(shape = (1,16,16));
  low_pass = Input(shape = (3,16,16));
  results = low_pass(layers.ConcatLayer(dim = 1), noise);
  results = results(nn.Conv2d(results.channels, 64, kernel_size = (3,3), padding = 1))(nn.ReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 64, kernel_size = (3,3), padding = 1))(nn.ReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 3, kernel_size = (3,3), padding = 1));
  return FunctionalModel(inputs = (noise, low_pass), outputs = results);

def GeneratorTwo():
  noise = Input(shape = (100,));
  results = noise(nn.Linear(noise.channels, 1200))(nn.ReLU());
  results = results(nn.Linear(results.channels, 1200));
  results = results(nn.Sigmoid());
  results = results(nn.Linear(results.channels, 3*8*8));
  results = results(layers.ReshapeLayer((-1,3,8,8)));
  return FunctionalModel(inputs = noise, outputs = results);

class Trainer(pl.LightningModule):
  def __init__(self, args):
    super(Trainer, self).__init__();
    self.args = args;
  def forward(self, x):
    
  def training_step(self, batch, batch_idx):
    
  def validation_step(self, batch, batch_idx):
    
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr = 3e-4, betas = (0.9, 0.999));
  @staticmethod
  def add_model_specific_args(parent_parser):
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
        imgs = model.forward(noise);
        imgs = imgs.cpu().detach().numpy();
      else:
        imgs = np.array([[cv2.pyrUp(imgs[b, c,...]) for c in range(3)] for b in range(batch_size)]);
        inputs = torch.from_numpy(imgs);
        if self.device == 'gpu':
          inputs = inputs.cuda();
        residual = model.forward([noise, inputs]);
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

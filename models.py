#!/usr/bin/python3

import torch;
from torch import nn;
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

class LAPGAN(object):
  def __init__(self, model_dir = 'models', device = 'gpu'):
    from os.path import join;
    from torch import load;
    with open(join(model_dir, 'gen_0.pth'), 'rb') as f:
      self.gen_0 = load(f);
      self.gen_0.eval();
    with open(join(model_dir, 'gen_1.pth'), 'rb') as f:
      self.gen_1 = load(f);
      self.gen_1.eval();
    with open(join(model_dir, 'gen_2.pth'), 'rb') as f:
      self.gen_2 = load(f);
      self.gen_2.eval();
    self.device = device;
    if self.device == 'gpu':
      self.gen_0.cuda();
      self.gen_1.cuda();
      self.gen_2.cuda();
  def generate(self,):
    import numpy as np;
    noise_2 = np.random.normal(loc = 0, scale = 0.1, size = (4,100,));
    noise_1 = np.random.normal(loc = 0, scale = 0.1, size = (4, 1, 16, 16));
    noise_0 = np.random.normal(loc = 0, scale = 0.1, size = (4, 1, 32, 32));


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

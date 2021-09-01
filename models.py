#!/usr/bin/python3

import torch;
from torch import nn;
from pytorch_functional import Input, FunctionalModel, layers;

def DiscriminatorZero():
  high_pass = Input(shape = (3,32,32)); # laplacian response
  low_pass = Input(shape = (3,32,32)); # coarsed high resolution image
  results = high_pass(layers.AddOpLayer(), low_pass); # results.shape = (batch, 3, 32, 32)
  results = results(nn.Conv2d(results.channels, 128, kernel_size = (5,5)));
  results = results(nn.LeakyReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 128, kernel_size = (5,5), stride = (2,2)));
  results = results(nn.LeakyReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Flatten());
  results = results(nn.Linear(results.channels, 1));
  results = results(nn.Sigmoid())
  return FunctionalModel(inputs = (high_pass, low_pass), outputs = results);

def DiscriminatorOne():
  high_pass = Input(shape = (3,16,16)); # laplacian response
  low_pass = Input(shape = (3,16,16)); # coarsed high resolution image
  results = high_pass(layers.AddOpLayer(), low_pass); # results.shape = (batch, 3, 16, 16)
  results = results(nn.Conv2d(results.channels, 64, kernel_size = (5,5)));
  results = results(nn.LeakyReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 64, kernel_size = (5,5), stride = (2,2)));
  results = results(nn.LeakyReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Flatten());
  results = results(nn.Linear(results.channels, 1));
  results = results(nn.Sigmoid());
  return FunctionalModel(inputs = (high_pass, low_pass), outputs = results);

def DiscriminatorTwo():
  low_pass = Input(shape = (3,8,8)); # low_pass.shape = (batch, 3, 8, 8)
  results = low_pass(nn.Flatten());
  results = results(nn.Linear(results.channels, 600));
  results = results(nn.LeakyReLU());
  results = results(nn.Linear(results.channels, 600));
  results = results(nn.LeakyReLU());
  results = results(nn.Linear(results.channels, 1));
  results = results(nn.Sigmoid());
  return FunctionalModel(inputs = low_pass, outputs = results);

def GeneratorZero():
  noise = Input(shape = (1,32,32));
  low_pass = Input(shape = (3,32,32));
  results = low_pass(layers.ConcatLayer(dim = 1), noise);
  results = results(nn.Conv2d(results.channels, 128, kernel_size = (3,3), padding = 1));
  results = results(nn.ReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 128, kernel_size = (3,3), padding = 1));
  results = results(nn.ReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 3, kernel_size = (3,3), padding = 1));
  return FunctionalModel(inputs = (noise, low_pass), outputs = results);

def GeneratorOne():
  noise = Input(shape = (1,16,16));
  low_pass = Input(shape = (3,16,16));
  results = low_pass(layers.ConcatLayer(dim = 1), noise);
  results = results(nn.Conv2d(results.channels, 64, kernel_size = (3,3), padding = 1));
  results = results(nn.ReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 64, kernel_size = (3,3), padding = 1));
  results = results(nn.ReLU());
  results = results(nn.BatchNorm2d(results.channels));
  results = results(nn.Conv2d(results.channels, 3, kernel_size = (3,3), padding = 1));
  return FunctionalModel(inputs = (noise, low_pass), outputs = results);

class ReshapeLayer(nn.Module):
  def __init__(self, dim):
    super().__init__();
    self.dim = dim;
  def forward(self, *elements):
    return torch.reshape(input = elements[0], shape = self.dim);

def GeneratorTwo():
  noise = Input(shape = (100,));
  results = noise(nn.Linear(noise.channels, 1200));
  results = results(nn.ReLU());
  results = results(nn.Linear(results.channels, 1200));
  results = results(nn.Sigmoid());
  results = results(nn.Linear(results.channels, 3*8*8));
  results = results(ReshapeLayer((3,8,8)));
  return FunctionalModel(inputs = noise, outputs = results);

if __name__ == "__main__":

  disc_0 = DiscriminatorZero();
  disc_1 = DiscriminatorOne();
  disc_2 = DiscriminatorTwo();
  gen_0 = GeneratorZero();
  gen_1 = GeneratorOne();
  gen_2 = GeneratorTwo();


#!/usr/bin/python3

import torch;
from torch.utils.data import Dataset, DataLoader;
from torchvision import transforms, utils;
from torchvision.datasets import CIFAR10;

class MultiScale(object):
  def __init__(self, n_scale = 3):
    self.n_scale = n_scale;
  def __call__(self, sample):
    # NOTE: force opencv to use single thread to enable multiple worker for dataloader
    cv2.setNumThreads(0);
    img, label = sample;
    img = img.numpy(); # img.shape = (channel, h, w)
    outputs = list();
    for i in range(n_scale):
      if i == n_scale - 1:
        outputs.append(img);
      else:
        downsampled = np.array([cv2.pyrDown(img[c,...]) for c in range(3)]);
        coarsed = np.array([cv2.pyrUp(downsampled[c,...]) for c in range(3)]);
        residual = img - coarsed;
        outputs.append(residual);
        # update img for next scale
        img = downsampled;
    return tuple(outputs);

def load_cifar10(batch_size, download = False, num_workers = 2):
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), MultiScale(n_scale = 3)]);
  trainset = CIFAR10(root = 'cifar10', train = True, download = download, transform = transform);
  testset = CIFAR10(root = 'cifar10', train = False, download = download, transform = transform);
  trainset = DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = num_workers);
  testset = DataLoader(testset, batch_size = batch_size, shuffle = True, num_workers = num_workers);
  return trainset, testset;

if __name__ == "__main__":
  
  trainset, testset = load_cifar10(4, download = True);
  trainset_iter = iter(trainset);
  sample0, sample1, sample2 = next(trainset_iter);
  print(sample0.shape, sample1.shape, sample2.shape);

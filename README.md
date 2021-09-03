# LAPGAN

this project implements algorithm LAPGAN for high resolution image generation introduced in paper [Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](https://arxiv.org/abs/1506.05751).

## install prerequisite packages

install with command

```shell
pip3 install -r requirements.txt
```

## how to train

train with the following command

```shell
python3 train.py --batch_size=<batch size> --num_workers=<workers> --download=(True|False) [--gpus=<gpu number>]
```


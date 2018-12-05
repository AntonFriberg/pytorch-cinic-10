# Train CINIC-10 with PyTorch

An example of how to train different deep neural networks on the [CINIC-10]
dataset based on the [pytorch-cifar] repository.

[CINIC-10]: https://github.com/BayesWatch/cinic-10
[pytorch-cifar]: https://github.com/kuangliu/pytorch-cifar/

## Development history
This example was constructed from kuangliu's excellent [pytorch-cifar] example.
Instead of utilizing the [CIFAR-10] dataset this example use [CINIC-10] which
is a drop in replacement to [CIFAR-10] which increases the difficulty of the
image classification task.

In addition to the dataset change this example also contains a number of other
changes including:

- Cosine annealing learning rate scheduler.
- Training time output.
- Dockerfile for simple dependency management

## Accuracy

This implementation achieves the following accuracy after training for 300
epochs.

| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 76.54%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | -      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | -      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | -      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | -      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | -      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | -      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | -      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | -      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | -      |

## Learning rate adjustment
Instead of manually adjusting the learning rate a cosine annealing learning
rate scheduler is used.

## Useful links

- https://github.com/BayesWatch/cinic-10
- https://github.com/kuangliu/pytorch-cifar/
- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
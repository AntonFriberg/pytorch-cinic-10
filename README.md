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
epochs on a single Nvidia GTX 1080TI.

| Model             | Acc.        | Training Time |
| ----------------- | ----------- | ------------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 84.740% | 3 hours 12 minutes |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | -      | - |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | -      | - |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | -      | - |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | -      | - |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | -      | - |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | -      | - |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | -      | - |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | -      | - |
| [DPN92](https://arxiv.org/abs/1707.01629)             | -      | - |

## Learning rate adjustment
Instead of manually adjusting the learning rate a cosine annealing learning
rate scheduler is used.

## Useful links

- https://github.com/BayesWatch/cinic-10
- https://github.com/kuangliu/pytorch-cifar/
- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

## Samples
Below are samples randomly selected from CINIC-10 and from CIFAR-10 for comparison. It is clear that CINIC-10 is a more noisy dataset because the *Imagenet constituent samples were not vetted*.


### Airplane

##### CIFAR-10

![CIFAR airplane](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cifar-airplane.png)
##### CINIC-10
![CINIC airplane](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cinic-airplane.png)

---


### Automobile

##### CIFAR-10

![CIFAR automobile](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cifar-automobile.png)
##### CINIC-10
![CINIC automobile](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cinic-automobile.png)

---

### Bird

##### CIFAR-10

![CIFAR bird](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cifar-bird.png)
##### CINIC-10
![CINIC bird](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cinic-bird.png)

---

### Cat

##### CIFAR-10

![CIFAR cat](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cifar-cat.png)
##### CINIC-10
![CINIC cat](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cinic-cat.png)

---

### Deer

##### CIFAR-10

![CIFAR deer](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cifar-deer.png)
##### CINIC-10
![CINIC deer](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cinic-deer.png)

---

### Dog

##### CIFAR-10

![CIFAR dog](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cifar-dog.png)
##### CINIC-10
![CINIC dog](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cinic-dog.png)

---

### Frog

##### CIFAR-10

![CIFAR frog](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cifar-frog.png)
##### CINIC-10
![CINIC frog](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cinic-frog.png)

---
### Horse

##### CIFAR-10

![CIFAR horse](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cifar-horse.png)
##### CINIC-10
![CINIC horse](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cinic-horse.png)

---

### Ship

##### CIFAR-10

![CIFAR ship](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cifar-ship.png)
##### CINIC-10
![CINIC ship](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cinic-ship.png)

---

### Truck

##### CIFAR-10

![CIFAR truck](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cifar-truck.png)
##### CINIC-10
![CINIC truck](https://raw.githubusercontent.com/BayesWatch/cinic-10/master/images/cinic-truck.png)

---

## References

Darlow L.N., Crowley E.J., Antoniou A., and A.J. Storkey (2018) CINIC-10 is not ImageNet or CIFAR-10. Report EDI-INF-ANC-1802 (arXiv:1810.03505).](https://arxiv.org/abs/1810.03505
2018.

Patryk Chrabaszcz, Ilya Loshchilov, and Hutter Frank. A downsampled variant of ImageNet as an alternative
to the CIFAR datasets. arXiv preprint arXiv:1707.08819, 2017.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Master’s thesis, Toronto University,
2009.

Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436–444, 2015.
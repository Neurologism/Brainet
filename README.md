
# Brainet

Brainet is a deep learning engine developed in C++, without relying on external libraries. Its primary goal is to offer a transparent framework that allows users to fully understand the underlying mechanisms. Unlike alternatives such as TensorFlow, PyTorch, or Caffe, which often feel like black boxes due to their high level of abstraction, Brainet wants to focus on explaining how everything works under the hood.

Currently, the best place to start is the example.cpp file, which can be found in the tests folder. This file contains an example of how to train a 2-layer Neural Network on the MNIST dataset.

## Overview 
The following image shows a high-level overview of the Brainet architecture:


![alt text](image.png)


## Performance

- [MNIST](https://yann.lecun.com/exdb/mnist/): 

    | Model | Test Error Rate | Training Time |
    |-------|-----------------|---------------|
    | 2-layer NN, 300 hidden units, cross-entopy | 5.9%            | 8 min         |
    

## Installation

You'll need a C++ compiler that supports at least C++20.
I recommend using the [GNU Compiler Collection](https://gcc.gnu.org/).
(And git to download the source code.)
For easy compilation, you can use [CMake](https://cmake.org/).

To download, run the following command in your terminal:

```bash
  git clone https://github.com/Neurologism/brainet
```

## Usage
To use Brainet, you need to include the Brainet header file in your project. 
```cpp
#include "brainet.h"
```

An example of how to train a 2-layer Neural Network on the MNIST dataset can be found in the example.cpp file.

To compile the example file, navigate to the build folder and run the following command in your terminal:
```bash
cmake --build ./
```

To run the compiled file, navigate to the bin folder and run the following command in your terminal:

```bash
./example
```

## Extensions
The only dataset that comes with Brainet is the MNIST dataset. 
The following contains a list of download links for other datasets that can be used with Brainet:
- [EMNIST](https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip)
- Fashion MNIST: 
    - [Train](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)
    - [Test](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz)
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)

To use them, you will also need a tool to extract gzip and tar files. I recommend using [7-Zip](https://www.7-zip.org/).

## Authors

- [@Servant-of-Scietia](https://github.com/Servant-of-Scietia)

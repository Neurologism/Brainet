
# Brainet

Brainet is a Deep Learning Engine build within C++ and CUDA without use of external libraries.
The vision of Brainet is to provide a transparent Framework that allows full understanding of the underlying mechanisms of Deep Learning. Most of the current 
alternatives like TensorFlow, PyTorch or Caffe have a high level of abstraction that feels like a black box. Brainet isn't competing with these frameworks regarding performance, but rather aims to explain how everything works under the hood.
To reach this goal Brainet relies on a simple codebase that is well documented. Additionally there will be a small book that 
explains the design choices and the mechanisms of Brainet and deep learning in general.

However, please note that the Project is still in a very early stage of development. So the codebase might be hard to understand and the supporting book isn't yet written. However I'm working on it and happy to provide you with explanations 
how everything works, if you dare to reach out to me. :)

Currently the example.cpp file is the best place to start. It contains a simple example of how to train a 2-layer Neural Network on the MNIST dataset.



## Performance

- [MNIST](https://yann.lecun.com/exdb/mnist/): 

    | Model | Test Error Rate | Training Time |
    |-------|----------|---------------|
    | 2-layer NN, 300 hidden units, cross-entopy | 8.66% | 30 min |
    

## Installation

<!-- To use Brainet, you need to have a CUDA compatible GPU and the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed. -->

You'll need a C++ compiler that supports at least C++17. I recommend using the [GNU Compiler Collection](https://gcc.gnu.org/). (And git to download the source code.)

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

To compile the example file run the following command in your terminal:
```bash
g++ -o example example.cpp -std=c++17 -O3 -lcudart -lcublas -lcurand
```

To run the compiled file, run the following command in your terminal:

```bash
./example
```

The algorithm will print logs to the console, showing the training progress. After the training is done, the test error rate will be printed to the console.

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

## Feedback
Feedback is greatly appreciated. If you have any questions or suggestions, please feel free to reach out to me.
If you find a bug or have a feature request, please open an issue on GitHub.

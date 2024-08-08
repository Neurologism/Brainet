#pragma GCC optimize("O3")

#include "brainet.h"

using namespace std;


/*
    This is a simple example of how to use the library
    The dataset used is the MNIST dataset
    The model is a simple feedforward neural network with 4 hidden layers
    The activation function used is ReLU for the hidden layers and Sigmoid for the output layer
    The cost function used is Mean Squared Error
    The model is trained for 25 epochs with a batch size of 100 and a learning rate of 0.01
    The model is then tested on the first 1000 samples of the training set
 */

std::int32_t main()
{
    
    typedef std::vector<std::vector<double>> data_type;
    data_type input = read_idx("datasets/mnist/train-images.idx3-ubyte");
    data_type target = read_idx("datasets/mnist/train-labels.idx1-ubyte");

    data_type test_input = read_idx("datasets/mnist/t10k-images.idx3-ubyte");
    data_type test_target = read_idx("datasets/mnist/t10k-labels.idx1-ubyte");

    // suported modules to be used in sequential can be found in the module folder or just look at the model variant

    // suported operations to be used with modules can be found in the operation folder; regarding activation functions and cost functions just look at the corresponding variants

    // feel free to contact me via email : samsun2006@outlook.com if you have any questions, suggestions or if you want to know how the project works

    MODEL model;
    model.sequential({INPUT(input[0].size()), DENSE(ReLU(),300), DENSE(Sigmoid(),10), COST(MSE(),10)});   

    model.train(input,target,250,250,0.01);

    model.test(test_input,test_target);
    return 0; 
}
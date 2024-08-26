#pragma GCC optimize("O3")

#include "brainet.hpp"

using namespace std;

/**
 * Hey there! And thank you for checking out brainet!
 * This displays an example of how to use brainet to train a simple neural network on the MNIST dataset.
 * We start by reading the dataset which is stored in the idx format. We then normalize the pixel values to be between 0 and 1.
 * We then create a SequentialModel with an input layer of size 784, a hidden layer of 200 neurons with ReLU activation function and an output layer of 10 neurons with Softmax activation function.
 * We then train the model using the train function, specifying the input and target data, the number of epochs, the batch size, the optimizer, the early stopping patience and the train/validation split ratio.
 * Finally, we test the model using the test function and the test data.
 * 
 * If you want to write your own models, please check out the documentation of the various classes and functions in the code.
 * There you can find more information about what is available and how to use it. If you have any questions, feel free to contact me via email (samsun2006@icloud.com).
 */
std::int32_t main()
{
    typedef std::vector<std::vector<double>> dataType;
    dataType train_input = read_idx("mnist/train-images.idx3-ubyte");
    dataType train_target = read_idx("mnist/train-labels.idx1-ubyte");

    dataType test_input = read_idx("mnist/t10k-images.idx3-ubyte");
    dataType test_target = read_idx("mnist/t10k-labels.idx1-ubyte");

    // 200 neurons in the hidden layer, ReLU activation function
    // 10 output neurons, Softmax activation function, ErrorRate as loss function
    SequentialModel model(Input(train_input[0].size()), { Dense(ReLU(),200)}, Output(Softmax(), 10, ErrorRate()));

    train_input = preprocessing::normalize(train_input);
    test_input = preprocessing::normalize(test_input);

    // train the model
    model.train( train_input, train_target, 10, 100, SGD(0.1,150), 20, 0.996 );

    // test the model
    model.test(test_input,test_target);


    return 0; 
}
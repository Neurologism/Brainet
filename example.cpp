#pragma GCC optimize("O3")

#include "brainet.hpp"

using namespace std;


/*
    Heyo there!
    I don't know how you got here, but I'm happy to see you. Let me show you around a bit.

    The following code is a simple example of how to use Brainet.
    The model is the same as in the Example in the Performance Section of the README.

    The following is an quick overview of the structure of the Interface:

    - The Model class is the main class that is used to build a neural network.
    - To build things in a Network you can use child classes of the Module class.
    - The current Modules are:
        - Input
        - Dense
        - Cost
    
    - Each Module provides certain configuration options. For example the Dense Module provides the activation function.
    - Those configuration options are basically just functions with a derivative and can be found in the operation folder.

    - There are some additional features like Optimizers and Preprocessing functions.
    - Just look into the folders to see what is available.

    - regarding datasets: The only supported format is the idx format currently. Feel free to write your own reader if you need another format. All you have to do is turn the data into a 2D vector of doubles, where the first dimension is the number of samples and the second dimension is the number of features.

    - And I should probably talk about the Model class. The Model class provides the following functions:
        - sequential: This function is used to build a neural network. It takes a vector of Modules.
        - train: This function is used to train the model. It takes the training data, the validation data, the number of epochs, the batch size, the optimizer and the number of training steps of early stopping.
        - test: This function is used to test the model. It takes the test data and the test target.
        - load: This function is more like a template function. It is used if you want to have multiple models in one file. To do modifications to a model you have to call the load function first.

    If you have any questions or want to learn about the features I left out in this short Introduction, feel free to ask me on github or via email(samsun2006@icloud.com). I'm happy to show you arround. Sorry for the bad writing btw. :)

    Have fun using Brainet!
 */

std::int32_t main()
{
    typedef std::vector<std::vector<double>> dataType;
    dataType input = read_idx("datasets/mnist/train-images.idx3-ubyte");
    dataType target = read_idx("datasets/mnist/train-labels.idx1-ubyte");

    dataType train_input, validation_input;
    dataType train_target, validation_target;

    dataType test_input = read_idx("datasets/mnist/t10k-images.idx3-ubyte");
    dataType test_target = read_idx("datasets/mnist/t10k-labels.idx1-ubyte");

    SequentialModel model(Input(input[0].size()), {Dense(ReLU(),300)}, Output(Softmax(),10, CrossEntropy()));

    model.train( train_input, train_target, 15, 100, SGD(0.5), 20, 0.99 );

    model.test(test_input,test_target);
    return 0; 
}
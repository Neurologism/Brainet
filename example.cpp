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
    dataType input = read_idx("datasets/mnist/train-images.idx3-ubyte"); // read the data from the file
    dataType target = read_idx("datasets/mnist/train-labels.idx1-ubyte"); // read the target from the file

    dataType train_input, validation_input;
    dataType train_target, validation_target;

    preprocessing::split(input, target, 0.98, train_input, validation_input, train_target, validation_target); // split the data into training and validation data with a ratio of 0.98

    dataType test_input = read_idx("datasets/mnist/t10k-images.idx3-ubyte"); // read the test data from the file
    dataType test_target = read_idx("datasets/mnist/t10k-labels.idx1-ubyte"); // read the test target from the file

    Model model; // interface to Brainet
    model.sequential({Input(input[0].size()), Dense(ReLU(),300), Dense(Sigmoid(),10), Cost(MSE(),10)}); // simple sequential model with 2 Dense Layers and a Mean Squared Error Cost Function   

    model.train( train_input, train_target, validation_input, validation_target, 1500, 200, PrimitiveSGD(1,0.995), 20); // train the model with the training data and validate it with the validation data. Console output will be shown.

    model.test(test_input,test_target); // test the model with the test data and the test target. Console output will be shown.
    return 0; 
}
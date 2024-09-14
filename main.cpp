#pragma GCC optimize("O3")
#include "brainet.hpp"

/**
 * file used for testing purposes
 */
std::int32_t main()
{
    typedef std::vector<std::vector<double>> dataType;
    dataType train_input = read_idx("../mnist/train-images.idx3-ubyte");
    dataType train_target = read_idx("../mnist/train-labels.idx1-ubyte");

    dataType test_input = read_idx("../mnist/t10k-images.idx3-ubyte");
    dataType test_target = read_idx("../mnist/t10k-labels.idx1-ubyte");

    Model model;

    train_input = preprocessing::normalize(train_input);
    test_input = preprocessing::normalize(test_input);

    model.addModule(Dense(ReLU(),100, "dense1"));
    model.addModule(Dense(Softmax(),10, "output"));
    model.addModule(Loss(ErrorRate(), "loss"));

    model.connectModules("dense1", "output");
    model.connectModules("output", "loss");

    Dataset dataset(train_input, train_target, 0.998, test_input, test_target);

    model.train(dataset, "dense1", "loss", 10, 100, SGD(0.1,500), 10);
    model.test( dataset, "dense1", "loss");

    return 0; 
}
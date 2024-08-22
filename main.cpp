#pragma GCC optimize("O3")
#include "brainet.hpp"

/**
 * file used for testing purposes
 */
std::int32_t main()
{
    typedef std::vector<std::vector<double>> dataType;
    dataType train_input = read_idx("mnist/train-images.idx3-ubyte");
    dataType train_target = read_idx("mnist/train-labels.idx1-ubyte");

    dataType test_input = read_idx("mnist/t10k-images.idx3-ubyte");
    dataType test_target = read_idx("mnist/t10k-labels.idx1-ubyte");

    SequentialModel model(Input(train_input[0].size()), { Dense(ReLU(),100)}, Output(Softmax(), 10, ErrorRate()));

    train_input = preprocessing::normalize(train_input);
    test_input = preprocessing::normalize(test_input);

    model.train( train_input, train_target, 10, 100, SGD(0.1,500), 20, 0.998 );

    model.test(test_input,test_target);


    return 0; 
}
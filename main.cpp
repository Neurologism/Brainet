#pragma GCC optimize("O3")
#include "brainet.hpp"


std::int32_t main()
{
    typedef std::vector<std::vector<double>> dataType;
    dataType train_input = read_idx("datasets/mnist/train-images.idx3-ubyte");
    dataType train_target = read_idx("datasets/mnist/train-labels.idx1-ubyte");

    dataType test_input = read_idx("datasets/mnist/t10k-images.idx3-ubyte");
    dataType test_target = read_idx("datasets/mnist/t10k-labels.idx1-ubyte");

    SequentialModel model(Input(train_input[0].size()), { Dense(ReLU(),800)}, Output(Softmax(), 10, CrossEntropy()));

    model.train( train_input, train_target, 20, 100, SGD(1), 200, 0.997 );

    model.test(test_input,test_target);
    model.test(train_input,train_target);


    return 0; 
}
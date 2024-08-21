#pragma GCC optimize("O3")
#include "brainet.hpp"


std::int32_t main()
{
    typedef std::vector<std::vector<double>> dataType;
    dataType train_input = read_idx("mnist/train-images.idx3-ubyte");
    dataType train_target = read_idx("mnist/train-labels.idx1-ubyte");

    dataType test_input = read_idx("mnist/t10k-images.idx3-ubyte");
    dataType test_target = read_idx("mnist/t10k-labels.idx1-ubyte");

    SequentialModel model(Input(train_input[0].size()), { Dense(ReLU(),100)}, Output(Softmax(), 10, ErrorRate()));

    for (std::uint32_t i = 0; i < train_input.size(); i++)
    {
        for (std::uint32_t j = 0; j < train_input[i].size(); j++)
        {
            train_input[i][j] /= 255;
        }
    }
    for (std::uint32_t i = 0; i < test_input.size(); i++)
    {
        for (std::uint32_t j = 0; j < test_input[i].size(); j++)
        {
            test_input[i][j] /= 255;
        }
    }

    model.train( train_input, train_target, 10, 100, SGD(0.1,500), 20, 0.998 );

    model.test(test_input,test_target);


    return 0; 
}
#include "brainet.hpp"

std::int32_t main()
{
    typedef std::vector<std::vector<double>> dataType;

    dataType train_input = Reader::read_idx("../data/mnist/train-images.idx3-ubyte");
    dataType train_target = Reader::read_idx("../data/mnist/train-labels.idx1-ubyte");

    dataType test_input = Reader::read_idx("../data/mnist/t10k-images.idx3-ubyte");
    dataType test_target = Reader::read_idx("../data/mnist/t10k-labels.idx1-ubyte");

    train_input = Preprocessing::normalize(train_input);
    test_input = Preprocessing::normalize(test_input);

    Model model;

    model.addSequential({
        Dense(ReLU(), 100, "dense0"),
        Dense(Softmax(), 10, "output"),
        Loss(ErrorRate(), "loss")
    });

    Dataset dataset(train_input, train_target, 0.998, test_input, test_target);

    model.train(dataset, "dense0", "loss", 1, 32, SGD(0.1,100), 10);
    model.test( dataset, "dense0", "loss");

    return 0;
}



























#include "brainet.h"

using namespace std;

// implementin general tests
std::int32_t main()
{
    typedef std::vector<std::vector<double>> data_type;
    data_type input = read_idx("datasets/train-images.idx3-ubyte");
    data_type target = read_idx("datasets/train-labels.idx1-ubyte");


    MODEL model;
    model.sequential({INPUT(input[0].size()), DENSE(ReLU(),2), DENSE(Sigmoid(),1), COST(MSE())});   

    model.train(input,target,10,100,2);
    return 0; 
}
#include "brainet.h"

using namespace std;

// implementin general tests
int main()
{
    std::shared_ptr<TENSOR<double>> input = std::make_shared<TENSOR<double>>(read_idx("datasets/train-images.idx3-ubyte"));
    std::shared_ptr<TENSOR<double>> target = std::make_shared<TENSOR<double>>(read_idx("datasets/train-labels.idx1-ubyte"));

    input->reshape({input->shape()[0],input->shape()[1]*input->shape()[2]}); // flatten the input

    MODEL model;
    model.sequential({INPUT(input,input->shape()[1]), DENSE(HyperbolicTangent(),2), DENSE(Sigmoid(),1), COST(MSE(),target)});   

    model.train(100,2);
    return 0; 
}
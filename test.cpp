#include "brainet.h"

using namespace std;

// implementin general tests
int main()
{
    MODEL model;
    TENSOR<double> input = TENSOR<double>({2,2});

    model.sequential({INPUT(input,2), DENSE(ReLU(),2), DENSE(ReLU(),1)});   

    model.train(0,0);
    return 0; 
}
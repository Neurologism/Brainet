#include "brainet.h"

using namespace std;

// implementin general tests
int main()
{
    MODEL model;
    shared_ptr<TENSOR<double>> input = make_shared<TENSOR<double>>(TENSOR<double>({4,2}));
    input->set({0,0},0);
    input->set({0,1},0);
    input->set({1,0},0);
    input->set({1,1},1);
    input->set({2,0},1);
    input->set({2,1},0);
    input->set({3,0},1);
    input->set({3,1},1);
    input = input->transpose();

    shared_ptr<TENSOR<double>> target = make_shared<TENSOR<double>>(TENSOR<double>({4,1}));
    target->set({0,0},0);
    target->set({1,0},1);
    target->set({2,0},1);
    target->set({3,0},0);
    target = target->transpose();
    
    model.sequential({INPUT(input,2), DENSE(ReLU(),2), DENSE(ReLU(),1), COST(MSE(),target)});   

    model.train(0,0);
    return 0; 
}
#ifndef MODEL_INCLUDE_GUARD
#define MODEL_INCLUDE_GUARD

#include "dependencies.h"
#include "graph.h"
#include "builders/layer_builder.h"

class MODEL
{
    GRAPH __graph;
    VARIABLE * __sequential_head;
    LAYER_BUILDER __layer_builder = LAYER_BUILDER(&__graph);
public:
    void add_input(std::vector<std::vector<double>> & input);
    void add_dense(OPERATION * op, int units);
    void train(std::vector<std::vector<double>> & data, std::vector<std::vector<double>> & target, int epochs, double learning_rate);
};

void MODEL::add_dense(OPERATION * op, int units)
{
    std::vector<double> weights;
    std::vector<int> shape = {__sequential_head->get_shape().back(), units};
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(-0.1,0.1); // initialize weights to small random values
    for(int i = 0;i < shape[0]*shape[1];i++) 
    {
        weights.push_back(distribution(generator));
    }
    __sequential_head = __layer_builder.add_linear_transformation(__sequential_head,weights,shape);
    __sequential_head = __layer_builder.add_activation_function(__sequential_head, op);
}

void MODEL::train(std::vector<std::vector<double>> & data, std::vector<std::vector<double>> & target, int epochs, double learning_rate)
{
    __graph.forward();
    std::vector<bool> v(__graph.get_variables().size(),true);
    __graph.backprop(v,10);
}

#endif // MODEL_INCLUDE_GUARD
#ifndef MODEL_INCLUDE_GUARD
#define MODEL_INCLUDE_GUARD

#include "dependencies.h"
#include "graph.h"
#include "cluster/input.h"
#include "cluster/cluster.h"
#include "cluster/dense.h"

class MODEL
{
    GRAPH __graph;
        
public:
    void sequential(std::vector<CLUSTER> layers);
    void train(TENSOR<double> & data, TENSOR<double> & target, int epochs, double learning_rate);
};

void MODEL::sequential(std::vector<CLUSTER> layers)
{
    GRAPH * CLUSTER::__graph = &__graph; // set graph in cluster class
    for(int i = 0; i < layers.size() - 1; i++)
    {
        layers[i].add_output(layers[i+1].input());
        layers[i+1].add_input(layers[i].output());
    }
}

void MODEL::train(TENSOR<double> & data, TENSOR<double> & target, int epochs, double learning_rate)
{
    __graph.forward();
    std::vector<bool> v(__graph.get_variables().size(),true);
    __graph.backprop(v,10);
}

#endif // MODEL_INCLUDE_GUARD
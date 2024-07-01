#ifndef MODEL_INCLUDE_GUARD
#define MODEL_INCLUDE_GUARD

#include "dependencies.h"
#include "graph.h"
#include "cluster/cluster.h"
#include "operation/activation_function/activation_function.h"

class MODEL
{
    GRAPH __graph;
    std::vector<VARIABLE *> __to_be_differentiated;
public:
    MODEL(){CLUSTER::set_graph(&__graph);};
    void load();
    void sequential(std::vector<CLUSTER_VARIANT> layers, bool add_backprop = true);
    void train(int epochs, double learning_rate);
};

void MODEL::load()
{
    CLUSTER::set_graph(&__graph);
}

void MODEL::sequential(std::vector<CLUSTER_VARIANT> layers, bool add_backprop)
{
    std::vector<CLUSTER*> clusters;
    for (CLUSTER_VARIANT& layer : layers) {
        clusters.push_back(std::visit([](auto&& arg) -> CLUSTER* {
        // Assuming all types in the variant can be dynamically casted to OPERATION*
        return dynamic_cast<CLUSTER*>(&arg);
    }, layer));
    }
    
    for(int i = 0; i < layers.size() - 1; i++)
    {
        clusters[i]->add_output(clusters[i+1]->input());
        clusters[i+1]->add_input(clusters[i]->output(),clusters[i]->size());
    }

    if(add_backprop)__to_be_differentiated.push_back(clusters.back()->output());
}

void MODEL::train(int epochs, double learning_rate)
{
    __graph.forward();
    std::vector<bool> v(__graph.get_variables().size(),true);
    std::vector<TENSOR<double>> grad_table = __graph.backprop(v, __to_be_differentiated);
}

#endif // MODEL_INCLUDE_GUARD
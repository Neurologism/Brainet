#ifndef MODEL_INCLUDE_GUARD
#define MODEL_INCLUDE_GUARD

#include "dependencies.h"
#include "graph.h"
#include "cluster/cluster.h"
#include "operation/activation_function/activation_function.h"

CLUSTER* convertToClusterPointer(CLUSTER_VARIANT& variant) {
    return std::visit([](auto&& arg) -> CLUSTER* {
        // Attempt dynamic_cast for polymorphic types
        auto ptr = dynamic_cast<CLUSTER*>(&arg);
        if (ptr) {
            return ptr; // Successful cast
        } else {
            // Handle the case where arg is not a CLUSTER* (e.g., INPUT)
            // This might involve calling a specific method like arg.toClusterPointer()
            // if such a method exists.
            return nullptr; // or some other appropriate handling
        }
    }, variant);
}

class MODEL
{
    GRAPH __graph;
public:
    MODEL(){CLUSTER::set_graph(&__graph);};
    void load();
    void sequential(std::vector<CLUSTER_VARIANT> layers);
    void train(int epochs, double learning_rate);
};

void MODEL::load()
{
    CLUSTER::set_graph(&__graph);
}

void MODEL::sequential(std::vector<CLUSTER_VARIANT> layers)
{
    std::vector<CLUSTER*> clusters;
    for (CLUSTER_VARIANT& layer : layers) {
        clusters.push_back(convertToClusterPointer(layer));
    }
    
    for(int i = 0; i < layers.size() - 1; i++)
    {
        clusters[i]->add_output(clusters[i+1]->input());
        clusters[i+1]->add_input(clusters[i]->output());
    }
}

void MODEL::train(int epochs, double learning_rate)
{
    __graph.forward();
    std::vector<bool> v(__graph.get_variables().size(),true);
    __graph.backprop(v,10);
}

#endif // MODEL_INCLUDE_GUARD
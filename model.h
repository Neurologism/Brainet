#ifndef MODEL_INCLUDE_GUARD
#define MODEL_INCLUDE_GUARD

#include "dependencies.h"
#include "graph.h"
#include "cluster/cluster.h"
#include "operation/activation_function/activation_function.h"

class MODEL
{
    std::shared_ptr<GRAPH> __graph = std::make_shared<GRAPH>();
    std::vector<std::shared_ptr<VARIABLE>> __to_be_differentiated;
public:
    MODEL(){CLUSTER::set_graph(__graph);};
    ~MODEL(){};
    void load();
    void sequential(std::vector<CLUSTER_VARIANT> layers, bool add_backprop = true);
    void train(int epochs, double learning_rate);
};

void MODEL::load()
{
    CLUSTER::set_graph(__graph);
}

void MODEL::sequential(std::vector<CLUSTER_VARIANT> layers, bool add_backprop)
{
    std::vector<std::shared_ptr<CLUSTER>> clusters;
    for (CLUSTER_VARIANT& layer : layers) {
        std::shared_ptr<CLUSTER> cluster_ptr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to OPERATION*
        return std::shared_ptr<CLUSTER>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, CLUSTER_VARIANT{layer});
        clusters.push_back(cluster_ptr);
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
    for(int epoch = 0; epoch < epochs; epoch++)
    {
        __graph->forward();
        std::shared_ptr<TENSOR<double>> loss = __to_be_differentiated[0]->get_data();
        std::cout << "Epoch: " << epoch << " Loss: " << loss->data()[0] << std::endl;
        std::vector<std::shared_ptr<TENSOR<double>>> v = __graph->backprop(CLUSTER::get_learnable_parameters(), __to_be_differentiated);
        for(int i = 0; i < CLUSTER::get_learnable_parameters().size(); i++)
        {
            std::cout << "Parameter " << i << " ";
            for(int j = 0; j < CLUSTER::get_learnable_parameters()[i]->get_data()->size(); j++)
            {
                CLUSTER::get_learnable_parameters()[i]->get_data()->data()[j] += learning_rate * v[i]->data()[j];
                std::cout << CLUSTER::get_learnable_parameters()[i]->get_data()->data()[j] << " ";
            }
            std::cout << std::endl;
        }
    }
}

#endif // MODEL_INCLUDE_GUARD
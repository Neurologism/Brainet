#ifndef MODEL_INCLUDE_GUARD
#define MODEL_INCLUDE_GUARD

#include "dependencies.h"
#include "graph.h"
#include "module/module.h"
#include "operation/activation_function/activation_function.h"

/**
 * @brief The model class is the main interface for the user to create a neural network. It is used to build the network and to train it.
 * The model class combines the graph, the module and the operation classes to create a neural network.
 */
class MODEL
{
    std::shared_ptr<GRAPH> __graph = std::make_shared<GRAPH>(); // the computational graph
    std::vector<std::shared_ptr<VARIABLE>> __to_be_differentiated; // the variables that are to be differentiated
public:
    /**
     * @brief Construct a new MODEL object.
     */
    MODEL()
    {
        MODULE::set_graph(__graph); // set the static graph pointer in the module class
    };
    ~MODEL(){};
    /**
     * @brief This function loads the graph of the model into the module class.
     */
    void load();
    /**
     * @brief This function creates a sequential neural network.
     * @param layers The layers of the neural network.
     * @param add_backprop If true the output of the last layer is added to the differentiation vector.
     */
    void sequential(std::vector<CLUSTER_VARIANT> layers, bool add_backprop = true);
    /**
     * @brief This function trains the model. It uses the backpropagation algorithm to update the learnable parameters. 
     * @param epochs The number of epochs.
     * @param learning_rate The learning rate.
     */
    void train(int epochs, double learning_rate);
};

void MODEL::load()
{
    MODULE::set_graph(__graph);
}

void MODEL::sequential(std::vector<CLUSTER_VARIANT> layers, bool add_backprop)
{
    std::vector<std::shared_ptr<MODULE>> clusters;
    for (CLUSTER_VARIANT& layer : layers) {
        std::shared_ptr<MODULE> cluster_ptr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to OPERATION*
        return std::shared_ptr<MODULE>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, CLUSTER_VARIANT{layer});
        clusters.push_back(cluster_ptr);
    }
    
    for(int i = 0; i < layers.size() - 1; i++)
    {
        clusters[i]->add_output(clusters[i+1]->input());
        clusters[i+1]->add_input(clusters[i]->output(),clusters[i]->getUnits());
    }

    if(add_backprop)__to_be_differentiated.push_back(clusters.back()->output());
}

void MODEL::train(int epochs, double learning_rate)
{
    // this should be replaced by a more sophisticated training algorithm
    for(int epoch = 0; epoch < epochs; epoch++)
    {
        __graph->forward();
        std::shared_ptr<TENSOR<double>> loss = __to_be_differentiated[0]->get_data();
        std::cout << "Epoch: " << epoch << " Loss: " << loss->data()[0] << std::endl; // print loss
        std::vector<std::shared_ptr<TENSOR<double>>> v = __graph->backprop(MODULE::get_learnable_parameters(), __to_be_differentiated); // backpropagation
        for(int i = 0; i < MODULE::get_learnable_parameters().size(); i++)
        {
            std::cout << "Parameter " << i << " "; // debug
            for(int j = 0; j < MODULE::get_learnable_parameters()[i]->get_data()->size(); j++)
            {
                MODULE::get_learnable_parameters()[i]->get_data()->data()[j] += learning_rate * v[i]->data()[j];
                std::cout << MODULE::get_learnable_parameters()[i]->get_data()->data()[j] << " ";
            }
            std::cout << std::endl;
        }
    }
}

#endif // MODEL_INCLUDE_GUARD
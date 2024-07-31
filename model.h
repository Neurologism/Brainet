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
    std::map<std::uint32_t, std::pair<std::shared_ptr<VARIABLE>, std::shared_ptr<VARIABLE>>> __data_label_pairs; // the data/label pairs
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
     * @param ID The ID of the data/label pair.
     */
    void sequential(std::vector<MODULE_VARIANT> layers, int ID = 0);
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

void MODEL::sequential(std::vector<MODULE_VARIANT> layers, int ID)
{
    std::vector<std::shared_ptr<MODULE>> clusters;
    for (MODULE_VARIANT& layer : layers) {
        std::shared_ptr<MODULE> cluster_ptr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to OPERATION*
        return std::shared_ptr<MODULE>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, MODULE_VARIANT{layer});
        clusters.push_back(cluster_ptr);
    }
    
    for(int i = 0; i < layers.size() - 1; i++)
    {
        clusters[i]->add_output(clusters[i+1]->input());
        clusters[i+1]->add_input(clusters[i]->output(),clusters[i]->getUnits());
    }

    __to_be_differentiated.push_back(clusters.back()->output());
    if (__data_label_pairs.find(ID) != __data_label_pairs.end())
    {
        throw std::runtime_error("ID already exists");
    }
    // add error checks in the future
    std::shared_ptr<INPUT> input = std::dynamic_pointer_cast<INPUT>(clusters.front());
    std::shared_ptr<COST> output = std::dynamic_pointer_cast<COST>(clusters.back());
    __data_label_pairs[ID] = std::make_pair(input->data(), output->target());
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
            //std::cout << "Parameter " << i << " "; // debug
            for(int j = 0; j < MODULE::get_learnable_parameters()[i]->get_data()->size(); j++)
            {
                MODULE::get_learnable_parameters()[i]->get_data()->data()[j] += learning_rate * v[i]->data()[j];
                //std::cout << MODULE::get_learnable_parameters()[i]->get_data()->data()[j] << " "; // debug
            }
            //std::cout << std::endl; // debug
        }
    }
}

#endif // MODEL_INCLUDE_GUARD
#ifndef MODEL_HPP
#define MODEL_HPP

#include "graph.hpp"
#include "module/module.hpp"
#include "operation/activation_function/activation_function.hpp"
#include "optimizers/optimizer.hpp"

/**
 * @brief The model class is the main interface for the user to create a neural network. It is used to build the network and to train it.
 * The model class combines the graph, the module and the operation classes to create a neural network.
 */
class Model
{
    typedef std::vector<std::vector<double>> data_type;
    typedef std::vector<std::vector<double>> label_type;

    std::uint32_t __loss_index; // the index of the loss function in the graph


    std::shared_ptr<Graph> __graph = std::make_shared<Graph>(); // the computational graph
    std::map<std::uint32_t, std::pair<std::shared_ptr<Variable>, std::shared_ptr<Variable>>> __data_label_pairs; // the data/label pairs


public:
    /**
     * @brief Construct a new Model object.
     */
    Model()
    {
        Module::set_graph(__graph); // set the static graph pointer in the module class
    };
    ~Model(){};
    /**
     * @brief This function loads the graph of the model into the module class.
     */
    void load();
    /**
     * @brief This function creates a sequential neural network.
     * @param layers The layers of the neural network.
     * @param ID The ID of the data/label pair.
     */
    void sequential(std::vector<Module_VARIANT> layers, std::uint32_t ID = 0);

    /**
     * @brief This function creates a sequential neural network.
     * @param layers The layers of the neural network.
     * @param norm The norm to use for regularization.
     * @param ID The ID of the data/label pair.
     */
    void sequential(std::vector<Module_VARIANT> layers, NormVariant norm, std::uint32_t ID = 0);

    /**
     * @brief This function trains the model. It uses the backpropagation algorithm to update the learnable parameters. 
     * @param data_label_pairs Map distributing the data/label pairs to the input/output nodes according to their ID. ID : (data, label)
     * @param epochs The number of epochs.
     * @param batch_size The batch size.
     * @param learning_rate The learning rate.
     */
    void train(std::map<std::uint32_t, std::pair<data_type, label_type>> const & data_label_pairs, std::uint32_t const epochs, std::uint32_t const batch_size, OptimizerVariant optimizer);
    /**
     * @brief Shortcut for training a model with only one data/label pair. Assumes the ID is 0.
     * @param data The data.
     * @param label The label.
     * @param epochs The number of epochs.
     * @param batch_size The batch size.
     * @param learning_rate The learning rate.
     */
    void train(data_type const data, label_type const label, std::uint32_t const epochs, std::uint32_t const batch_size, OptimizerVariant optimizer);

    /**
     * @brief This function gets the test error of the model.
     * @param data_label_pairs Map distributing the data/label pairs to the input/output nodes according to their ID. ID : (data, label)
     * @param max_test_size takes the first test_size elements of the data/label pairs, if max_test_size is default, it takes all elements
     */
    void test(std::map<std::uint32_t, std::pair<data_type, label_type>> const & data_label_pairs, std::uint32_t const max_test_size = std::numeric_limits<std::uint32_t>::max());
    /**
     * @brief Shortcut for testing a model with only one data/label pair. Assumes the ID is 0.
     * @param data The data.
     * @param label The label.
     * @param max_test_size takes the first test_size elements of the data/label pairs, if max_test_size is default, it takes all elements
     */
    void test(data_type const data, label_type const label, std::uint32_t const max_test_size = std::numeric_limits<std::uint32_t>::max());
};

void Model::load()
{
    Module::set_graph(__graph);
}

void Model::sequential(std::vector<Module_VARIANT> layers, std::uint32_t ID)
{
    std::vector<std::shared_ptr<Module>> clusters;
    for (Module_VARIANT& layer : layers) {
        std::shared_ptr<Module> cluster_ptr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to OPERATION*
        return std::shared_ptr<Module>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, Module_VARIANT{layer});
        clusters.push_back(cluster_ptr);
    }
    
    for(std::uint32_t i = 0; i < layers.size() - 1; i++)
    {
        clusters[i]->add_output(clusters[i+1]->input());
        clusters[i+1]->add_input(clusters[i]->output(),clusters[i]->getUnits());
    }

    __loss_index = __graph->add_output(clusters.back()->output());
    if (__data_label_pairs.find(ID) != __data_label_pairs.end())
    {
        throw std::runtime_error("ID already exists");
    }
    // add error checks in the future
    std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(clusters.front());
    std::shared_ptr<Cost> output = std::dynamic_pointer_cast<Cost>(clusters.back());
    __data_label_pairs[ID] = std::make_pair(input->data(), output->target());
}

void Model::sequential(std::vector<Module_VARIANT> layers, NormVariant norm, std::uint32_t ID)
{
    Dense::set_default_norm(norm);
    sequential(layers, ID);
}


void Model::train(std::map<std::uint32_t, std::pair<data_type, label_type>> const & data_label_pairs, std::uint32_t const epochs, std::uint32_t const batch_size, OptimizerVariant optimizer)
{
    if ( data_label_pairs.size() != 1)
    {
        throw std::runtime_error("Only one data/label pair is currently supported");
    }

    const std::uint32_t trainingIterations = epochs; // adjust this value later to train for epochs

    // this should be replaced by a more sophisticated training algorithm
    for(std::uint32_t iteration = 0; iteration < trainingIterations; iteration++)
    {
        std::vector<std::vector<double>> batch_data;
        std::vector<std::vector<double>> batch_label;

        data_type data = data_label_pairs.at(0).first;
        label_type label = data_label_pairs.at(0).second;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, data.size() - 1);
        for (std::uint32_t i = 0; i < batch_size; i++)
        {
            std::uint32_t random_index = dis(gen);
            batch_data.push_back(data[random_index]);
            batch_label.push_back(label[random_index]);
            // Use batch_data and batch_label for training
        }
        

        __data_label_pairs[0].first->get_data() = std::make_shared<Tensor<double>>(Matrix<double>(batch_data));
        __data_label_pairs[0].second->get_data() = std::make_shared<Tensor<double>>(Matrix<double>(batch_label));

        __graph->forward();
        std::shared_ptr<Tensor<double>> loss = __graph->get_output(__loss_index);
        std::cout << "Batch: " << iteration << " Loss: " << loss->at(0) << std::endl; // print loss
        std::vector<std::shared_ptr<Tensor<double>>> __gradients = __graph->backprop(Module::get_learnable_parameters()); // backpropagation
        
        std::visit([__gradients, batch_size](auto&& arg) {
            arg.update(__gradients, batch_size);}, optimizer);
    }
}


void Model::train(data_type const data, label_type const label, std::uint32_t const epochs, std::uint32_t const batch_size, OptimizerVariant optimizer)
{
    std::map<std::uint32_t, std::pair<data_type, label_type>> data_label_pairs;
    if (__data_label_pairs.find(0) == __data_label_pairs.end())
    {
        throw std::runtime_error("Assumed ID 0, but no data/label pair with ID 0 found");
    }
    data_label_pairs[0] = std::make_pair(data, label);
    train(data_label_pairs, epochs, batch_size, optimizer);
}


void Model::test(std::map<std::uint32_t, std::pair<data_type, label_type>> const & data_label_pairs, std::uint32_t const max_test_size)
{
    for (auto const & data_label_pair : data_label_pairs)
    {
        __data_label_pairs[data_label_pair.first].first->get_data() = std::make_shared<Tensor<double>>(Matrix<double>(data_label_pair.second.first));
        __data_label_pairs[data_label_pair.first].second->get_data() = std::make_shared<Tensor<double>>(Matrix<double>(data_label_pair.second.second));

        __graph->forward();
        std::shared_ptr<Tensor<double>> loss = __graph->get_output(__loss_index);
        std::cout << "Test Loss: " << loss->at(0) << std::endl; // print loss
    }
}


void Model::test(data_type const data, label_type const label, std::uint32_t const max_test_size)
{
    std::map<std::uint32_t, std::pair<data_type, label_type>> data_label_pairs;
    if (__data_label_pairs.find(0) == __data_label_pairs.end())
    {
        throw std::runtime_error("Assumed ID 0, but no data/label pair with ID 0 found");
    }
    data_label_pairs[0] = std::make_pair(data, label);
    test(data_label_pairs, max_test_size);
}

#endif // MODEL_HPP
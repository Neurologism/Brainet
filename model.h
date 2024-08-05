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
    typedef std::vector<std::vector<double>> data_type;
    typedef std::vector<std::vector<double>> label_type;

    std::uint32_t __loss_index; // the index of the loss function in the graph


    std::shared_ptr<GRAPH> __graph = std::make_shared<GRAPH>(); // the computational graph
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
    void sequential(std::vector<MODULE_VARIANT> layers, std::uint32_t ID = 0);
    /**
     * @brief This function trains the model. It uses the backpropagation algorithm to update the learnable parameters. 
     * @param data_label_pairs Map distributing the data/label pairs to the input/output nodes according to their ID. ID : (data, label)
     * @param epochs The number of epochs.
     * @param batch_size The batch size.
     * @param learning_rate The learning rate.
     */
    void train(std::map<std::uint32_t, std::pair<data_type, label_type>> const & data_label_pairs, std::uint32_t const epochs, std::uint32_t const batch_size, double const learning_rate);
    /**
     * @brief Shortcut for training a model with only one data/label pair. Assumes the ID is 0.
     * @param data The data.
     * @param label The label.
     * @param epochs The number of epochs.
     * @param batch_size The batch size.
     * @param learning_rate The learning rate.
     */
    void train(data_type const data, label_type const label, std::uint32_t const epochs, std::uint32_t const batch_size, double const learning_rate);

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

void MODEL::load()
{
    MODULE::set_graph(__graph);
}

void MODEL::sequential(std::vector<MODULE_VARIANT> layers, std::uint32_t ID)
{
    std::vector<std::shared_ptr<MODULE>> clusters;
    for (MODULE_VARIANT& layer : layers) {
        std::shared_ptr<MODULE> cluster_ptr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to OPERATION*
        return std::shared_ptr<MODULE>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, MODULE_VARIANT{layer});
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
    std::shared_ptr<INPUT> input = std::dynamic_pointer_cast<INPUT>(clusters.front());
    std::shared_ptr<COST> output = std::dynamic_pointer_cast<COST>(clusters.back());
    __data_label_pairs[ID] = std::make_pair(input->data(), output->target());
}

void MODEL::train(std::map<std::uint32_t, std::pair<data_type, label_type>> const & data_label_pairs, std::uint32_t const epochs, std::uint32_t const batch_size, double const learning_rate)
{
    // this should be replaced by a more sophisticated training algorithm
    for(std::uint32_t epoch = 0; epoch < epochs; epoch++)
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
        std::shared_ptr<TENSOR<double>> data_tensor = std::make_shared<TENSOR<double>>(TENSOR<double>({(std::uint32_t) batch_data.size(), (std::uint32_t) batch_data[0].size()}, 0.0, false));
        std::shared_ptr<TENSOR<double>> label_tensor = std::make_shared<TENSOR<double>>(TENSOR<double>({(std::uint32_t) batch_label.size(),(std::uint32_t) batch_label[0].size()}, 0.0, false));

        for (std::uint32_t i = 0; i < batch_data.size(); i++)
        {
            for (std::uint32_t j = 0; j < batch_data[i].size(); j++)
            {
                data_tensor->data()[i * batch_data[i].size() + j] = batch_data[i][j];
            }
            for (std::uint32_t j = 0; j < batch_label[i].size(); j++)
            {
                label_tensor->data()[i * batch_label[i].size() + j] = batch_label[i][j];
            }
        }

        __data_label_pairs[0].first->get_data() = data_tensor;
        __data_label_pairs[0].second->get_data() = label_tensor;




        __graph->forward();
        std::shared_ptr<TENSOR<double>> loss = __graph->get_output(__loss_index);
        std::cout << "Epoch: " << epoch << " Loss: " << loss->data()[0] << std::endl; // print loss
        std::vector<std::shared_ptr<TENSOR<double>>> v = __graph->backprop(MODULE::get_learnable_parameters()); // backpropagation
        for(std::uint32_t i = 0; i < MODULE::get_learnable_parameters().size(); i++)
        {
            //std::cout << "Parameter " << i << " "; // debug
            for(std::uint32_t j = 0; j < MODULE::get_learnable_parameters()[i]->get_data()->size(); j++)
            {
                MODULE::get_learnable_parameters()[i]->get_data()->data()[j] += learning_rate * v[i]->data()[j];
                //std::cout << MODULE::get_learnable_parameters()[i]->get_data()->data()[j] << " "; // debug
            }
            //std::cout << std::endl; // debug
        }
    }
}


void MODEL::train(data_type const data, label_type const label, std::uint32_t const epochs, std::uint32_t const batch_size, double const learning_rate)
{
    std::map<std::uint32_t, std::pair<data_type, label_type>> data_label_pairs;
    if (__data_label_pairs.find(0) == __data_label_pairs.end())
    {
        throw std::runtime_error("Assumed ID 0, but no data/label pair with ID 0 found");
    }
    data_label_pairs[0] = std::make_pair(data, label);
    train(data_label_pairs, epochs, batch_size, learning_rate);
}


void MODEL::test(std::map<std::uint32_t, std::pair<data_type, label_type>> const & data_label_pairs, std::uint32_t const max_test_size)
{
    for (auto const & data_label_pair : data_label_pairs)
    {
        std::shared_ptr<TENSOR<double>> data_tensor = std::make_shared<TENSOR<double>>(TENSOR<double>({std::min((std::uint32_t)data_label_pair.second.first.size(), max_test_size), (std::uint32_t) data_label_pair.second.first[0].size()}, 0.0, false));
        std::shared_ptr<TENSOR<double>> label_tensor = std::make_shared<TENSOR<double>>(TENSOR<double>({std::min((std::uint32_t)data_label_pair.second.second.size(), max_test_size), (std::uint32_t) data_label_pair.second.second[0].size()}, 0.0, false));

        for (std::uint32_t i = 0; i < std::min((std::uint32_t)data_label_pair.second.first.size(), max_test_size); i++)
        {
            for (std::uint32_t j = 0; j < data_label_pair.second.first[i].size(); j++)
            {
                data_tensor->data()[i * data_label_pair.second.first[i].size() + j] = data_label_pair.second.first[i][j];
            }
            for (std::uint32_t j = 0; j < data_label_pair.second.second[i].size(); j++)
            {
                label_tensor->data()[i * data_label_pair.second.second[i].size() + j] = data_label_pair.second.second[i][j];
            }
        }

        __data_label_pairs[data_label_pair.first].first->get_data() = data_tensor;
        __data_label_pairs[data_label_pair.first].second->get_data() = label_tensor;

        __graph->forward();
        std::shared_ptr<TENSOR<double>> loss = __graph->get_output(__loss_index);
        std::cout << "Test Loss: " << loss->data()[0] << std::endl; // print loss
    }
}


void MODEL::test(data_type const data, label_type const label, std::uint32_t const max_test_size)
{
    std::map<std::uint32_t, std::pair<data_type, label_type>> data_label_pairs;
    if (__data_label_pairs.find(0) == __data_label_pairs.end())
    {
        throw std::runtime_error("Assumed ID 0, but no data/label pair with ID 0 found");
    }
    data_label_pairs[0] = std::make_pair(data, label);
    test(data_label_pairs, max_test_size);
}

#endif // MODEL_INCLUDE_GUARD